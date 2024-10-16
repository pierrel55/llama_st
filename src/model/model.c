#include <stdio.h>
#include <omp.h>
#include "l_util.h"
#include "mem_alloc.h"
#include "json.h"
#include "term_utf8.h"
#include "model.h"
#include "matmul.h"
#include "load_tokenizer.h"

struct model_t model = { 0 };

// names of models, must match enum e_model_id
const char *model_id_names[model_id_count] = 
{
  "tinyllama",
  "llama1",
  "llama2",
  "codellama",
  "llama3",
  "llama31",
  "mistral",
  "mathstral",
  "zephyr",
  "mixtral",
  "vigogne2",
  "qwen2",
};

// get model id from string
static enum e_model_id get_model_id(const char *str)
{
  enum e_model_id m;
  for (m=0; m<model_id_count; m++)
    if (!strcmp(str, model_id_names[m]))
      return m;
  return -1;
}

// read run configuration from json file
static void load_run_config(const char *file_name)
{
  struct run_conf_t *conf = &model.config;
  struct sampler_conf_t *sconf = &model.sampler.conf;
  struct h_json_t *h = js_load_file(file_name, true);
  msg_info("read file %s\n", file_name);

  #define GET_KEY_STR(name)  name = (js_find_key_list_check(h, #name), js_get_key_value_str_alloc(h))
  #define GET_KEY_I32(name)  name = (js_find_key_list_check(h, #name), js_get_num_value_i32(h))
  #define GET_KEY_F32(name)  name = (js_find_key_list_check(h, #name), js_get_num_value_f32(h))
  #define GET_KEY_BOOL(name) name = (js_find_key_list_check(h, #name), js_get_num_value_bool(h))
  #define GET_KEY_RGB(name, id) term_def_color(id, term_get_color((js_find_key_list_check(h, #name), js_get_key_value_str_tmp(h))))

  // model identifier
  conf->GET_KEY_STR(model_ident);
  conf->e_model_id = get_model_id(conf->model_ident);
  if (conf->e_model_id < 0)
    msg_error("undefined model_ident: %s", conf->model_ident);

  // model load
  conf->load.GET_KEY_I32(model_num_safetensors);
  conf->load.GET_KEY_STR(model_path);
  conf->load.GET_KEY_STR(tokenizer_name);

  // set or override rope freq
  conf->GET_KEY_F32(rope_set);

  // sampler
  sconf->GET_KEY_F32(temperature);
  sconf->GET_KEY_F32(topp);
  sconf->GET_KEY_I32(topk);
  sconf->GET_KEY_F32(topp_minp);
  sconf->GET_KEY_BOOL(topp_eos);
  sconf->GET_KEY_F32(repeat_penalty);
  sconf->GET_KEY_I32(repeat_penalty_n);
  sconf->GET_KEY_F32(eos_amp);
  sconf->GET_KEY_I32(eos_amp_n);
  sconf->GET_KEY_I32(rand_seed);
  // optional 
  if (js_find_key_list(h, "ch_restrict"))
    sconf->ch_restrict = js_get_key_value_str_alloc(h);
  // checks
  conf->GET_KEY_BOOL(test_nan_logits);
  
  // load parameters
  conf->GET_KEY_BOOL(cvt_sf16);
  conf->GET_KEY_BOOL(cvt_f12);
  conf->GET_KEY_BOOL(cvt_f8);

  // hardware parameters
  conf->GET_KEY_I32(num_procs);
  conf->GET_KEY_I32(numa_nodes);
  conf->GET_KEY_I32(simd_mode);

  // run mode
  conf->GET_KEY_I32(run_mode);
  conf->GET_KEY_I32(gen_run_steps);
  conf->GET_KEY_STR(token_eos_str);
  conf->GET_KEY_STR(token_eot_str);

  // token display option
  conf->GET_KEY_BOOL(tok_disp_raw);
  conf->GET_KEY_BOOL(tok_disp_split);
  conf->GET_KEY_BOOL(tok_disp_prob);

  if (conf->run_mode == run_mode_generate)
    conf->GET_KEY_STR(gen_mode_prompt);
  else
  if (conf->run_mode == run_mode_chat)
  {
    // init colors
    conf->chat.GET_KEY_BOOL(chat_use_colors);
    if (conf->chat.chat_use_colors)
    {
      GET_KEY_RGB(chat_col_msg, 0);
      GET_KEY_RGB(chat_col_user, 1);
      GET_KEY_RGB(chat_col_assistant, 2);
    }
    conf->chat.GET_KEY_I32(fwd_disp_mode);

    // chat mode
    conf->chat.GET_KEY_I32(chat_prompt_mode);

    // prompt generation
    conf->chat.GET_KEY_STR(chat_assistant_name);
    conf->chat.GET_KEY_STR(chat_user_name);

    if (conf->chat.chat_prompt_mode == 0)
    {
      conf->chat.GET_KEY_STR(cm0_sys_prompt);
      conf->chat.GET_KEY_STR(cm0_user_prompt);
    }
    else
    if (conf->chat.chat_prompt_mode == 1)
    {
      conf->chat.GET_KEY_STR(cm1_sys_prompt);
      conf->chat.GET_KEY_STR(cm1_sys_template);
      conf->chat.GET_KEY_STR(cm1_user_first_template);
      conf->chat.GET_KEY_STR(cm1_end_template);
      conf->chat.GET_KEY_STR(cm1_user_template);
      conf->chat.GET_KEY_STR(cm1_user_prompt);
    }
    else
    if (conf->chat.chat_prompt_mode == 2)
    {
      conf->chat.GET_KEY_STR(cm2_sys_template);
      conf->chat.GET_KEY_STR(cm2_user_template);
      conf->chat.GET_KEY_STR(cm2_user_name_sw);
      conf->chat.GET_KEY_STR(cm2_sys_prompt);
      conf->chat.GET_KEY_STR(cm2_user_prompt);
    }
    else
      msg_error("chat_prompt_mode = %d is undefined", conf->chat.chat_prompt_mode);
  }
  else
    msg_error("undefined run_mode = %d", conf->run_mode);

  js_close(h);
}

// free all allocated strings
static void free_run_config(void)
{
  struct run_conf_t *conf = &model.config;
  free_check(conf->model_ident);
  free_check(conf->load.model_path);
  free_check(conf->load.tokenizer_name);
  free_check(conf->token_eos_str);
  free_check(conf->token_eot_str);
  free_check(conf->gen_mode_prompt);
  free_check(model.sampler.conf.ch_restrict);

  // chat strings
  free_check(conf->chat.chat_assistant_name);
  free_check(conf->chat.chat_user_name);

  free_check(conf->chat.cm0_sys_prompt);
  free_check(conf->chat.cm0_user_prompt);

  free_check(conf->chat.cm1_sys_template);
  free_check(conf->chat.cm1_user_first_template);
  free_check(conf->chat.cm1_user_template);
  free_check(conf->chat.cm1_end_template);
  free_check(conf->chat.cm1_sys_prompt);
  free_check(conf->chat.cm1_user_prompt);

  free_check(conf->chat.cm2_sys_template);
  free_check(conf->chat.cm2_user_template);
  free_check(conf->chat.cm2_user_name_sw);
  free_check(conf->chat.cm2_sys_prompt);
  free_check(conf->chat.cm2_user_prompt);
}

// alloc/load/init model datas
void build_model(const char *conf_file_name)
{
  struct run_conf_t *conf = &model.config;
  struct tokenizer_t *tokenizer = &model.tokenizer;
  struct transformer_t *transformer = &model.transformer;

  // read config file
  load_run_config(conf_file_name);

  // init matmul and conversions fonctions
  matmul_init(conf->simd_mode);

  // build tokenizer
  build_tokenizer();

  // get eos/eot
  conf->token_eos = tokenizer_find_sp_token_id(conf->token_eos_str);
  conf->token_eot = tokenizer_find_sp_token_id(conf->token_eot_str);

  // build transformer
  msg_info("load transformer..\n");
#if 1
  build_transformer();
#else
  // debug usage
  transformer->config.seq_len = 2048;
  transformer->config.vocab_size = 32000;
#endif

  // check token count tokenizer/transformer match
  if (tokenizer->tok_index_list_size != transformer->config.vocab_size)  // occur with qwen2.5
    msg_info("info: tokenizer/transformer vocab_size missmatch (%d/%d)\n", tokenizer->tok_index_list_size, transformer->config.vocab_size);

  // init the sampler
  build_sampler();

  // adjust run_steps
  if (conf->gen_run_steps <= 0)
    conf->gen_run_steps = transformer->config.seq_len;
}

// free memory
void free_model(void)
{
  free_sampler();
  free_tokenizer();
  free_transformer();
  free_run_config();
  matmul_exit();
}
