// load model in .safetensor format
// https://huggingface.co/docs/safetensors/index

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "l_util.h"
#include "mem_alloc.h"
#include "json.h"
#include "model.h"
#include "matmul.h"
#include "omp_numa.h"
#include "load_transformer.h"

// ----------------------------------------------
// init config with config.json
// ----------------------------------------------

// load config in config.json
void load_checkpoint_config(void)
{
  enum e_model_id model_id = model.config.e_model_id;
  const char *checkpoint_path = model.config.load.model_path;
  struct transformer_config_t *p = &model.transformer.config;
  struct h_json_t *h;
  char conf_name[256];
  
  int l = _snprintf(conf_name, sizeof(conf_name), "%s/%s", checkpoint_path, "config.json");
  if ((l < 0) || (l == sizeof(conf_name)))
    msg_error("checkpoint path too long");
  
  msg_info("read model config in: %s\n", conf_name);
  h = js_load_file(conf_name, false);

  #define CHECK_KEY(key_id, value) js_find_key_list_check(h, key_id); js_check_key_value_str(h, value)
  #define GET_KEY_I32(key_id)  (js_find_key_list_check(h, key_id), js_get_num_value_i32(h))
  #define GET_KEY_F32(key_id)  (js_find_key_list_check(h, key_id), js_get_num_value_f32(h))
  #define GET_KEY_BOOL(key_id) (js_find_key_list_check(h, key_id), js_get_num_value_bool(h))

  // check some key values
  if (    (model_id == model_id_mistral) 
       || (model_id == model_id_mathstral)
       || (model_id == model_id_zephyr))
  {
    CHECK_KEY("architectures", "MistralForCausalLM");
    CHECK_KEY("model_type", "mistral");
  }
  else
  if (model_id == model_id_mixtral)
  {
    CHECK_KEY("architectures", "MixtralForCausalLM");
    CHECK_KEY("model_type", "mixtral");
    p->moe.num_experts = GET_KEY_I32("num_local_experts");   // 8 (mixtral 8x7b)
    p->moe.top_k       = GET_KEY_I32("num_experts_per_tok"); // 2 (mixtral 8x7b)
  }
  else
  if (model_id == model_id_qwen2)
  {
    CHECK_KEY("architectures", "Qwen2ForCausalLM");
    CHECK_KEY("model_type", "qwen2");
  }
  else
  {
    CHECK_KEY("architectures", "LlamaForCausalLM");
    CHECK_KEY("model_type", "llama");
  }

  CHECK_KEY("hidden_act", "silu");

  // load used value                                        llama2 7B values
  p->dim          = GET_KEY_I32("hidden_size");          // 4096
  p->hidden_dim   = GET_KEY_I32("intermediate_size");    // 11008
  p->n_layers     = GET_KEY_I32("num_hidden_layers");    // 32
  p->n_heads      = GET_KEY_I32("num_attention_heads");  // 32
  
  if (js_find_key_list(h, "num_key_value_heads"))        // 32
    p->n_kv_heads = js_get_num_value_i32(h);
  else
  {
    p->n_kv_heads = p->n_heads;
    msg_info("n_kv_heads undefined, assumed = n_heads (%d)\n", p->n_heads);  // for llama1
  }
  p->seq_len      = GET_KEY_I32("max_position_embeddings"); // 2048
  p->rms_norm_eps = GET_KEY_F32("rms_norm_eps");            // 1e-05

  // rope_theta is optional (then .safetensors must contain rope data)
  if (js_find_key_list(h, "rope_theta"))                 // 10000.0
    p->rope_theta = js_get_num_value_f32(h);
  else
    msg_info("rope_theta undefined, expect rotary_emb.inv_freq contained in .safetensors\n");

  p->vocab_size = GET_KEY_I32("vocab_size");             // 32000

  // get torch type for weights
  js_find_key_list_check(h, "torch_dtype");              // float16/bfloat16
  if      (js_cmp_key_value_str(h, "float16"))  p->torch_type = w_type_f16;
  else if (js_cmp_key_value_str(h, "bfloat16")) p->torch_type = w_type_bf16;
  else if (js_cmp_key_value_str(h, "float32"))  p->torch_type = w_type_f32;
  else
    msg_error("unsupported torch_dtype = %s", js_get_key_value_str_tmp(h));

  js_close(h);

  // complete config                                     llama2 7B values
  p->head_size = p->dim / p->n_heads;                    // 4096 / 32 = 128
  p->kv_dim    = (p->dim * p->n_kv_heads) / p->n_heads;  // (4096 * 32) / 32 = 4096
  p->kv_mul    = p->n_heads / p->n_kv_heads;             // 32 / 32 = 1
  p->sqrt_head_size = sqrtf((float)p->head_size);        // sqrt(128) = 11.31..

  // display config infos
  msg_info("torch float type: %s\n", w_type_name[p->torch_type]);

  // if rope_theta is not defined in config.json, then expect rotary_emb datas in .safetentors files.
  // to avoid loading these datas (or if not present), rope_theta can be defined in arg_conf_t
  if (model.config.rope_set)
  {
    if (p->rope_theta && (model.config.rope_set != p->rope_theta))
      msg_info("rope_theta user changed from %.2f to %.2f\n", p->rope_theta, model.config.rope_set);
    else
      msg_info("rope_theta user set to %.2f\n", model.config.rope_set);
    p->rope_theta = model.config.rope_set;
  }
}

// ----------------------------------------------
// load .safetensor weights
// ----------------------------------------------

// tensor infos coded in json header
struct tens_inf_t
{
  const char *name;                    // name
  enum e_w_type d_type;                // data type, F16, BF16, F32
  int shape[2];                        // array shape
  int64_t data_ofs[2];                 // data offset in file
};

// temporary buffer used to transpose or convert float format
static struct
{
  void *w;
  int64_t sz_alloc;
} tmp_buff = { 0 };

// realloc buffer if too small
static void tmp_realloc(int64_t sz)
{
  if (sz > tmp_buff.sz_alloc)
  {
    tmp_buff.sz_alloc = sz;
    tmp_buff.w = realloc_check(tmp_buff.w, sz);
  }
}

// ---------------------------------------------------------------
// safetensors q/k weights/bias data are permuted in python using:
// w.view(dim1, dim2).reshape(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
// here operation is reversed for format used in transformer.c math
static void inv_reshape_4_transpose_12(char *wd, const char *ws, size_t sz_wx, int wy, int n_heads)
{
  int a, b, c;
  int na = n_heads;
  int nb = (wy / n_heads) / 2;
  int nc = 2;
  
  for (a=0; a<na; a++)
    for (b=0; b<nb; b++)
      for (c=0; c<nc; c++)
      {
        //size_t id = (size_t)a*nb*nc + c*nb + b;
        size_t id = ((size_t)a*nc + c)*nb + b;
        memcpy(wd, ws + id * sz_wx, sz_wx);
        wd += sz_wx;
      }
}

// load weights datas in file format and convert to expected memory format, transpose if tr != NULL
static void load_weights_cvt(file_t *file, const struct tens_inf_t *ti, int layer_id, struct w_dat_t *wd, bool optional, int tr_n_heads)
{
  bool cvt = ti->d_type != wd->d_type;           // test if convert type required
  size_t ne = (size_t)ti->shape[1] * ti->shape[0];
  size_t sz_ld = ne * w_type_sizeof[ti->d_type]; // data size in file
  
  // skip data if optional and not required (mem not allocated) (ex: rope freq)
  if (!wd->wx)
  {
    if (!optional)
      msg_error("weight mem not allocated");     // is assert
    return;
  }

  // check shape and seek to datas in file
  if ((ti->shape[0] != wd->wx) || (ti->shape[1] != wd->wy))
    msg_error("%s: w sizes: [%d, %d], expected [%d, %d]\n", ti->name, ti->shape[0], ti->shape[1], wd->wx, wd->wy);

  // check binary size match with expected format
  if (sz_ld != (ti->data_ofs[1] - ti->data_ofs[0]))
    msg_error("tensor binary size missmatch");

  // seek to datas in file
  f_seek(file, file->seek_ofs + ti->data_ofs[0], SEEK_SET);

  // load and convert datas
  if (!cvt && !tr_n_heads)                       // direct load without conversion possible
    numa_cpy_wd_z(wd, layer_id, NULL, file);
  else
  {
    // get weight pointer, can be used directly only if single node used (single continuous buffer)
    char *w = (wd->nn == 1) ? (char *)wd->lp[0].p + wd->lp[0].sz_l * layer_id : NULL;
    char *d_cvt, *s_tr, *d_tr, *res;             // convert/transpose source/dest buffers and final result
    
    // get working buffers
    void *tmp0, *tmp1;
    tmp_realloc(sz_ld * 2);
    tmp0 = (char *)tmp_buff.w;
    tmp1 = (char *)tmp_buff.w + sz_ld;

    // load data in tmp0
    f_read(tmp0, sz_ld, file);

    if (cvt && tr_n_heads)                       // convert + transpose
    {
      d_cvt = tmp1;
      s_tr = tmp1;
      d_tr = w ? w : tmp0;
      res = d_tr;
    }
    else
    if (cvt)                                     // convert only
    {
      d_cvt = w ? w : tmp1;                      // copy result in w
      res = d_cvt;
    }
    else                                         // transpose only
    {
      s_tr = tmp0;                               // source is load buffer
      d_tr = w ? w : tmp1;
      res = d_tr;
    }

    // convert load format to memory format
    if (cvt)
      cvt_w_data(d_cvt, wd->d_type, tmp0, ti->d_type, ne);

    // reverse permutation
    if (tr_n_heads)
    {
      if (wd->wy == 1)      // bias datas
        inv_reshape_4_transpose_12(d_tr, s_tr, w_type_sizeof[wd->d_type], wd->wx, tr_n_heads);
      else                  // weights datas
        inv_reshape_4_transpose_12(d_tr, s_tr, wd_ne_sizeof(wd, wd->wx), wd->wy, tr_n_heads);
    }

    if (res == w)
      wd->ne += ne;
    else
      numa_cpy_wd_z(wd, layer_id, res, NULL);    // copy result
  }
}

// load layer weights
static void load_layer_weights(file_t *file, const struct tens_inf_t *t_inf, const char *js_key, 
                               struct transformer_weights_t *w, int n_heads, int n_kv_heads, int n_experts)
{
  char *e;
  int layer_id = strtol(js_key, &e, 10);   // get layer id in key name
  int l = (int)(e - js_key);               // layer number len
  js_key += l;                             // pass layer number

  // test name and load w if name match and return
  #define W_TST_LD(_key, _w, _opt, _tr)\
    if (!strcmp(js_key, _key)) { load_weights_cvt(file, t_inf, layer_id, &_w, _opt, _tr); return; }

  // debug: display json keys
  // msg_info("load layer %d:tensort name \"%s\"\n", layer_id, js_key);

  // attention weights
  W_TST_LD(".input_layernorm.weight"         , w->rms_att, false, 0);
  W_TST_LD(".self_attn.rotary_emb.inv_freq"  , w->rope_if, true , 0);
  
  // qkv weights
  W_TST_LD(".self_attn.q_proj.weight"        , w->wq     , false, n_heads);
  W_TST_LD(".self_attn.k_proj.weight"        , w->wk     , false, n_kv_heads);
  W_TST_LD(".self_attn.v_proj.weight"        , w->wv     , false, 0);

  // (optional) qkv bias
  W_TST_LD(".self_attn.q_proj.bias"          , w->bq     , false, n_heads);
  W_TST_LD(".self_attn.k_proj.bias"          , w->bk     , false, n_kv_heads);
  W_TST_LD(".self_attn.v_proj.bias"          , w->bv     , false, 0);

  W_TST_LD(".self_attn.o_proj.weight"        , w->wo     , false, 0);
  W_TST_LD(".post_attention_layernorm.weight", w->rms_ffn, false, 0);

  // w1/w2/w3
  if (!n_experts)  // not MoE
  {
    W_TST_LD(".mlp.gate_proj.weight" , w->w1, false, 0);
    W_TST_LD(".mlp.down_proj.weight" , w->w2, false, 0);
    W_TST_LD(".mlp.up_proj.weight"   , w->w3, false, 0);
  }
  else             // MoE
  {
    W_TST_LD(".block_sparse_moe.gate.weight" , w->moe_gate, false, 0);

    // experts w1/w2/w3
    if (!memcmp(js_key, ".block_sparse_moe.experts.", 26))
    {
      struct w_dat_t *wx;
      char *e;
      int exp_id = strtol((js_key += 26), &e, 10);  // get expert id
      if ((js_key == e) || (exp_id >= n_experts))
        msg_error("MoE invalid expert id");

      if      (!strcmp(e, ".w1.weight")) wx = &w->w1;
      else if (!strcmp(e, ".w2.weight")) wx = &w->w2;
      else if (!strcmp(e, ".w3.weight")) wx = &w->w3;
      else
        msg_error("MoE invalid weight identifier");

      load_weights_cvt(file, t_inf, layer_id * n_experts + exp_id, wx, false, 0);
      return;
    }
  }
  // other ?
  msg_info("layer %d: ignored tensort name \"%s\"\n", layer_id, js_key);
}

// get dtype, shape and data_offsets in file
static void load_weights_info(struct h_json_t *h, struct js_read_inf_t *j_inf, struct tens_inf_t *t_inf)
{
  do
  {
    // js_print_param(h, &j_inf);

    if (js_cmp_key_ident(h, "dtype", 1))
    {
      // https://huggingface.co/docs/safetensors/v0.3.2rc1/metadata_parsing
      // "F64" | "F32" | "F16" | "BF16" | "I64" | "I32" | "I16" | "I8" | "U8" | "BOOL"
      if      (js_cmp_key_value_str(h, "F16"))  t_inf->d_type = w_type_f16;
      else if (js_cmp_key_value_str(h, "BF16")) t_inf->d_type = w_type_bf16;
      else if (js_cmp_key_value_str(h, "F32"))  t_inf->d_type = w_type_f32;
      else
        msg_error("unsupported torch load format %s", js_get_key_value_str_tmp(h));
    }
    else
    if (js_cmp_key_ident(h, "shape", 1))
    {
      int n = js_get_num_value_i32(h);
      // note: want raw size in shape[0], need to swap [y][x] json format to [x][y]
      if      (j_inf->arr_id == 0) { t_inf->shape[0] = n; t_inf->shape[1] = 1; }
      else if (j_inf->arr_id == 1) { t_inf->shape[1] = t_inf->shape[0]; t_inf->shape[0] = n; }
      else
        msg_error("tensor shapes > 2");
    }
    else
    if (js_cmp_key_ident(h, "data_offsets", 1))
    {
      if ((unsigned int)j_inf->arr_id < 2)
        t_inf->data_ofs[j_inf->arr_id] = js_get_num_value_i64(h);
      else
        msg_error("data_offsets > 2");
    }
    else
      msg_error("unexpected key value");
  }
  while (!j_inf->is_lev_end && js_read_param(h, j_inf));  // read next
}

static void load_file_st(const char *file_name, struct transformer_weights_t *weights, int n_heads, int n_kv_heads, int n_experts)
{
  file_t f = { 0 };
  struct js_read_inf_t j_inf = { 0 };
  struct h_json_t *h;
  char *json_text;
  int64_t json_len;
  
  msg_info("load: %s\n", file_name);
  f_open(&f, file_name, "rb");

  // get json part size
  f_read(&json_len, sizeof(json_len), &f);   // get header len (8 bytes)
  if ((json_len > 1024*1024) || (json_len >= f.size)) // check if size is realistic
    msg_error("json invalid header size");

  // copy json part in a buffer
  json_text = (char *)malloc_check(json_len + 1);  // +1 required to terminate text
  f_read(json_text, json_len, &f);
  json_text[json_len] = 0;

  // init json parser
  h = js_get_handle(json_text, false);
  js_seek_origin(h);

  // save weights datas seek origin
  f.seek_ofs = f_tell(&f);

  // load tensors
  while (js_read_param(h, &j_inf))
  {
    int l;
    struct tens_inf_t t_inf = { 0 };
    char t_name[256];

    // js_print_param(h, &j_inf);

    // compare part of key at start, if equal return len (= sizeof - 1)
    #define KEY_EQ(a, b) (memcmp(a, b, sizeof(b)-1) ? 0 : sizeof(b)-1)

    t_inf.name = js_copy_key_ident_str(h, t_name, sizeof(t_name), 0);
    if ((l = KEY_EQ(t_inf.name, "model.layers."))) // layer data
    {
      load_weights_info(h, &j_inf, &t_inf);
      load_layer_weights(&f, &t_inf, t_inf.name + l, weights, n_heads, n_kv_heads, n_experts);
    }
    else
    if (KEY_EQ(t_inf.name, "model.embed_tokens.weight"))
    {
      load_weights_info(h, &j_inf, &t_inf);
      load_weights_cvt(&f, &t_inf, 0, &weights->token_emb, false, 0);
    }
    else
    if (KEY_EQ(t_inf.name, "lm_head.weight"))
    {
      load_weights_info(h, &j_inf, &t_inf);
      load_weights_cvt(&f, &t_inf, 0, &weights->wcls, false, 0);
    }
    else
    if (KEY_EQ(t_inf.name, "model.norm.weight"))
    {
      load_weights_info(h, &j_inf, &t_inf);
      load_weights_cvt(&f, &t_inf, 0, &weights->rms_final, false, 0);
    }
    else
    if (!KEY_EQ(t_inf.name, "__metadata__"))     // is ignored
      msg_info("ignored json name: \"%s\"\n", t_inf.name);
  }

  f_close(&f);
  js_close(h);
}

// check all weight datas loaded
static void chk_ne_loaded(struct w_dat_t *wd)
{
  if (wd->ne != (size_t)wd->nz * wd->wy * wd->wx)
    msg_error("uncomplete weight data load");
}

// check transformer all datas loaded
static void check_load(struct transformer_weights_t *w, struct transformer_config_t *p)
{
  if (p->moe.num_experts)
    chk_ne_loaded(&w->moe_gate);

  chk_ne_loaded(&w->token_emb);
  chk_ne_loaded(&w->rms_att);
  chk_ne_loaded(&w->wq);
  chk_ne_loaded(&w->wk);
  chk_ne_loaded(&w->wv);
  chk_ne_loaded(&w->wo);
  chk_ne_loaded(&w->rms_ffn);
  chk_ne_loaded(&w->w1);
  chk_ne_loaded(&w->w2);
  chk_ne_loaded(&w->w3);
  chk_ne_loaded(&w->rms_final);

  // rope
  if (!p->rope_theta)
  {
    if (w->rope_if.ne)
      chk_ne_loaded(&w->rope_if);
    else
    // Found some models where rope_theta not defined in config.json and also rotary_emb.inv_freq datas not defined in .safetensors files.
    // In this case rope_theta must be defined in run config .json file.
    msg_error("rope_theta is undefined in config.json and rotary_emb.inv_freq not found in .safetensors files.\n"
              "please define rope_theta using rope_set in run config .json file to run model.");
  }

  // (optional) classifier weights for the logits, on the last layer
  // if lm_head.weight not contained in .safetensors, model.embed_tokens.weight is used (qwen2 model)
  if (w->wcls.ne)
    chk_ne_loaded(&w->wcls);
  else
  {
    msg_info("info: classifier use embed_tokens.weight.\n");
    free_wd(&w->wcls);
    w->wcls = w->token_emb;
  }

  // optional bias
  if (w->bq.ne)
  {
    chk_ne_loaded(&w->bq);
    chk_ne_loaded(&w->bk);
    chk_ne_loaded(&w->bv);
  }
  else
  {
    // free reserved mem if unused
    CHECK(!w->bk.ne && !w->bv.ne);
    free_wd(&w->bq);
    free_wd(&w->bk);
    free_wd(&w->bv);
  }
}

// load checkpoint all weights datas using list of .safetensors files
void load_checkpoint_weights(void)
{
  struct transformer_config_t *config = &model.transformer.config;
  struct transformer_weights_t *weights = &model.transformer.weights;
  const char *checkpoint_path = model.config.load.model_path;
  int file_count = model.config.load.model_num_safetensors;
  const char *mame_fmt = (file_count == 1) ? "model.safetensors" : "model-%.5d-of-%.5d.safetensors";
  char path_name[256];       // file path + name
  int i;
 
  // def path
  int lp = _snprintf(path_name, sizeof(path_name), "%s/", checkpoint_path);
  if ((lp < 0) || (lp == sizeof(path_name)))
    msg_error("model_path path too long or invalid format");

  // load .safetensors file list
  for (i=1; i<=file_count; i++)
  {
    // add file name to path
    int l = _snprintf(path_name + lp, sizeof(path_name) - lp, mame_fmt, i, file_count);
    if ((l < 0) || (l == sizeof(path_name)))
      msg_error(".safetensors path + name too long or invalid format");
    
    // load part
    load_file_st(path_name, weights, config->n_heads, config->n_kv_heads, config->moe.num_experts);
  }

  // check all loaded
  check_load(weights, config);

  // free temporary buffer
  free_check(tmp_buff.w);
  tmp_buff.w = NULL;
  tmp_buff.sz_alloc = 0;
}
