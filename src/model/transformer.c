#include <math.h>
#include <stdlib.h>
#include "l_util.h"
#include "mem_alloc.h"
#include "model.h"
#include "load_transformer.h"
#include "omp_numa.h"
#include "matmul.h"

#ifdef USE_SA_SIMD
#include "tr_opt_simd.h"
#endif

// ------------------------------------
// allocate transformer run state mem

// alloc memory for state
static void *alloc_smem(size_t ne, int type_sizeof)
{
  // check ne can be divided by SIMD_LV for simd code
  if (ne & (SIMD_LV-1)) 
    msg_error("state alloc size %d modulus SIMD_LV is not 0", ne);
  return numa_alloc(ne * type_sizeof, numa_map.tid_to_node_id[0]);
}

static void alloc_run_state(void)
{
  const struct transformer_config_t *p = &model.transformer.config;
  struct transformer_runstate_t *s = &model.transformer.state;
  int64_t ne_kv = (int64_t)p->n_layers * p->seq_len * p->kv_dim;

  #define ST_ALLOC(typ, st, ne) st = (typ *)alloc_smem(ne, sizeof(typ))

  // alloc mem                                        llama2 7B values
  ST_ALLOC(float, s->x       , p->dim);                   // 4096
  ST_ALLOC(float, s->xb      , p->dim);                   // 4096
  ST_ALLOC(float, s->xb2     , p->dim);                   // 4096
  ST_ALLOC(float, s->hb      , p->hidden_dim);            // 11008
  ST_ALLOC(float, s->hb2     , p->hidden_dim);            // 11008
  ST_ALLOC(float, s->q       , p->dim);                   // 4096
  ST_ALLOC(float, s->k_cache , ne_kv);                    // 32 * 2048 * 4096
  ST_ALLOC(float, s->v_cache , ne_kv);                    // 32 * 2048 * 4096
  ST_ALLOC(float, s->att     , p->n_heads * p->seq_len);  // 32 * 2048
  if (p->rope_theta)                          // else expect defined in transformer_weights_t.rope_if
    ST_ALLOC(float, s->rope_freq, p->head_size / 2);      // 64
  ST_ALLOC(float, s->rope_sin_cos, p->head_size);         // 128
  ST_ALLOC(struct ctoken_t, s->cache.tokens, p->seq_len); // 2048
  if (p->moe.num_experts)    // MeO mixtral
  {
    s->moe.exp_logits = malloc_check(p->moe.num_experts*sizeof(float));
    s->moe.exp_probs  = malloc_check(p->moe.num_experts*sizeof(struct exp_prob_t));
  }
  // single threaded main process
  s->logits = malloc_check(p->vocab_size * sizeof(float)); // 32000
}

static void free_run_state(struct transformer_runstate_t *s)
{
  numa_free(s->x);
  numa_free(s->xb);
  numa_free(s->xb2);
  numa_free(s->hb);
  numa_free(s->hb2);
  numa_free(s->q);
  numa_free(s->k_cache);
  numa_free(s->v_cache);
  numa_free(s->att);
  numa_free(s->rope_freq);
  numa_free(s->rope_sin_cos);
  numa_free(s->cache.tokens);
  free_check(s->moe.exp_logits);
  free_check(s->moe.exp_probs);
  free_check(s->logits);
}

// ------------------------------------
// allocate transformer weight datas

// alloc memory for transformer weights
// - single dim 1 (rms_att, rms_ffn, rms_final) are converted once to float 32.
// - single dim > 1 (token_emb, wcls) are keept in torch load format or 16 bits minimal format.
// - layers dim > 1 (wq/wk/..) are keept in torch load format or can be converted.
// - 2D tensor used with matmul are splitted for multi threaded operation.
static void alloc_transformer(void)
{
  const struct transformer_config_t *p = &model.transformer.config;
  struct transformer_weights_t *w = &model.transformer.weights;
  
  int nl = p->n_layers;
  int nw = nl;                         // num w1/w2/w3
  if (p->moe.num_experts)              // MoE used (mixtral)
  {
    nw *= p->moe.num_experts;          // alloc num_experts w1/w2/w3 per layer
    numa_alloc_wd(&w->moe_gate, nl , p->moe.num_experts           , p->dim , p->lw_type, true);
  }

  // size[nz][wy][wx]:          nz                  wy                wx (raw)  type
  numa_alloc_wd(&w->token_emb ,  1 , p->vocab_size                , p->dim , p->em_type, true);
  numa_alloc_wd(&w->rms_att   , nl , 1                            , p->dim , w_type_f32, false);
  numa_alloc_wd(&w->wq        , nl , p->n_heads    * p->head_size , p->dim , p->lw_type, true);
  numa_alloc_wd(&w->wk        , nl , p->n_kv_heads * p->head_size , p->dim , p->lw_type, true);
  numa_alloc_wd(&w->wv        , nl , p->n_kv_heads * p->head_size , p->dim , p->lw_type, true);
  
  numa_alloc_wd(&w->wo        , nl , p->dim ,    p->n_heads * p->head_size , p->lw_type, true);
  numa_alloc_wd(&w->rms_ffn   , nl , 1                            , p->dim , w_type_f32, false);
  numa_alloc_wd(&w->w1        , nw , p->hidden_dim                , p->dim , p->lw_type, true);
  numa_alloc_wd(&w->w2        , nw , p->dim ,                p->hidden_dim , p->lw_type, true);
  numa_alloc_wd(&w->w3        , nw , p->hidden_dim                , p->dim , p->lw_type, true);
  numa_alloc_wd(&w->rms_final ,  1 , 1                            , p->dim , w_type_f32, false);
  if (!p->rope_theta)
    numa_alloc_wd(&w->rope_if , nl , 1                   , p->head_size / 2, w_type_f32, false);

  // optional classifier if not same as token_emb
  numa_alloc_wd(&w->wcls      ,  1 , p->vocab_size                , p->dim , p->em_type, true);

  // optional qkv bias
  numa_alloc_wd(&w->bq        , nl , 1,       p->n_heads    * p->head_size , w_type_f32, false);
  numa_alloc_wd(&w->bk        , nl , 1,       p->n_kv_heads * p->head_size , w_type_f32, false);
  numa_alloc_wd(&w->bv        , nl , 1,       p->n_kv_heads * p->head_size , w_type_f32, false);
}

// free datas allocated using numa_alloc_wd
void free_wd(struct w_dat_t *wd)
{
  int i;
  for (i=0; i<wd->nn; i++)
    numa_free(wd->p_node[i]);
}

// free transformer datas
void free_transformer(void) 
{
  struct transformer_t *t = &model.transformer;
  struct transformer_weights_t *w = &t->weights;

  free_wd(&w->moe_gate);
  free_wd(&w->token_emb);
  free_wd(&w->rms_att);
  free_wd(&w->wq);
  free_wd(&w->wk);
  free_wd(&w->wv);
  free_wd(&w->wo);
  free_wd(&w->rms_ffn);
  free_wd(&w->w1);
  free_wd(&w->w2);
  free_wd(&w->w3);
  free_wd(&w->rms_final);

  free_wd(&w->rope_if);

  // optional classifier
  if (w->wcls.lp[0].p != w->token_emb.lp[0].p)  // test if not w->token_emb
    free_wd(&w->wcls);
  
  // optional qkv bias
  if (w->bq.ne)
  {
    free_wd(&w->bq);
    free_wd(&w->bk);
    free_wd(&w->bv);
  }

  // free the struct transformer_runstate_t buffers
  free_run_state(&t->state);
}

// ------------------------------------
// RoPE relative positional encoding:
// complex-valued rotate q and k in each head.

// init RoPE freq
static void init_RoPE(float *freq, float rope_theta, int head_size)
{
  int i;
  for (i=0; i<head_size; i+=2, freq++)
    *freq = (float)(1.0 / pow(rope_theta, (double)i / head_size));
}

// define rope sin/cos for pos
void set_RoPE_pos(float *sin_cos, int pos, const float *freq, int n_freq)
{
  int i;
  for (i=0; i<n_freq; i++, freq++, sin_cos+=2)
  {
    float f = *freq * pos;
    sin_cos[0] = sinf(f);
    sin_cos[1] = cosf(f);
  }
}

// apply RoPE on a, b vectors list
void RoPE(float *a, float *b, const float *sin_cos, int head_size, int a_dim, int b_dim)
{
  int i = 0;
  CHECK(b_dim <= a_dim);
  do
  {
    int j;
    for (j=0; j != head_size; j+=2, i+=2) 
    {
      float x = a[i];
      float y = a[i+1];               // rotate a (query)
      float s = sin_cos[j];           // get sin
      float c = sin_cos[j+1];         // get cos
      a[i]   = x*c - y*s;
      a[i+1] = x*s + y*c;
      if (i < b_dim)
      {
        x = b[i];
        y = b[i+1];                   // rotate b (key)
        b[i]   = x*c - y*s;
        b[i+1] = x*s + y*c;
      }
    }
  }
  while (i != a_dim);
}

// ------------------------------------
// neural net block functions

// add vectors a + b => a
static void vec_add(float *a, const float *b, int n)
{
  int i;
  for (i=0; i<n; i++)
    a[i] += b[i];
}

// calculate sum of squares
static float vec_get_sq_sum(const float *a, int n)
{
  float sq_sum = 0.0f;
  int i;
  for (i=0; i<n; i++)
    sq_sum += a[i] * a[i];
  return sq_sum;
}

// add vectors a + b => a + get sq_sum
static float vec_add_get_sq_sum(float *a, const float *b, int n)
{
  float sq_sum = 0;
  int i;
  for (i=0; i<n; i++)
  {
    float sum = a[i] + b[i];
    a[i] = sum;
    sq_sum += sum * sum;
  }
  return sq_sum;
}

// normalize and scale
static void norm_scale(float *o, const float *a, float sq_sum, float rms_norm_eps, const float *weight, int size)
{
  // scale factor
  float k = 1.0f / sqrtf((sq_sum/size) + rms_norm_eps);
  int i;

  // normalize and scale
  for (i=0; i<size; i++)
    o[i] = (a[i] * k) * weight[i];
}

// single head attention.
static void head_attention(int h, uint64_t s_kv_ofs)
{
  const struct transformer_config_t *p = &model.transformer.config;
  struct transformer_runstate_t *s = &model.transformer.state;

  float *xb = s->xb + h * p->head_size;           // result in xb
  float *att = s->att + h * p->seq_len;           // attention scores for this head

  // get the query vector for this head
  const float *q = s->q + h * p->head_size;

  // get cache k and v for this head
  // s_kv_ofs  = layer_id * p->seq_len * p->kv_dim;      // layer_id * 2048 * [1024..4096] (8*128..32*128)
  // p->kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;  // (4096 * [8..32]) / 32 = [1024..4096]
  // p->kv_mul = p->n_heads / p->n_kv_heads;             // 32 / [8..32] = 4..1
  // h_kv_ofs  = ((layer_id * seq_len * dim + h * head_size) * n_kv_heads) / n_heads;

  uint64_t h_kv_ofs = s_kv_ofs + (h / p->kv_mul) * p->head_size;  // head offset in caches
  const float *k = s->k_cache + h_kv_ofs;
  const float *v = s->v_cache + h_kv_ofs;

#ifdef USE_SA_SIMD
  // simd optimized
  head_att_opt(xb, s->cache.n_tokens, att, q, k, v, p);
#else
  // iterate over all timesteps, including the current one
  // calculate the attention score as the dot product of q and k
  int t;
  for (t=0; t<s->cache.n_tokens; t++, k += p->kv_dim)
  {
    // 1 line matrix is used for dot product
    matmul_procs.matmul_f32_f32(&att[t], q, k, p->head_size, 1); 
    att[t] /= p->sqrt_head_size;
  }

  // softmax the scores to get attention weights, from 0..pos inclusively
  softmax(att, s->cache.n_tokens);

  // weighted sum of the values, accumulate xb for t = 0..pos inclusively
  for (t=0; t<s->cache.n_tokens; t++, v += p->kv_dim)
  {
    int j;
    float a = att[t];
    if (!t)
      for (j=0; j<p->head_size; j++) xb[j]  = a * v[j];  // t = 0, init xb
    else
      for (j=0; j<p->head_size; j++) xb[j] += a * v[j];  // t > 0, accumulate xb
  }
#endif
}

// multihead attention. iterate over all heads
static void multihead_attention(uint64_t s_kv_ofs)
{
  int n_heads = model.transformer.config.n_heads;
#if 0
  int h;
  #pragma omp parallel for
  for (h=0; h<n_heads; h++)
    head_attention(h, s_kv_ofs);
#else
  // run only threads in node that contain states data
  int h0 = 0;
  while (h0 < n_heads)
  {
    int nt = (h0 + numa_map.nt_mp) <= n_heads ? numa_map.nt_mp : n_heads - h0;
    int tid;
    #pragma omp parallel for
    for (tid=0; tid<nt; tid++)
      head_attention(h0 + tid, s_kv_ofs);
    h0 += nt;
  }
#endif
}

// splitted multi threaded matmul
static void lw_matmul(float *d, const float *s, const struct w_dat_t *wd, int layer_id, mm_proc_t mm_proc)
{
  int i, n_thrd = wd->wy < numa_map.n_threads ? wd->wy : numa_map.n_threads;

  #pragma omp parallel for
  for (i=0; i<n_thrd; i++)
  {
    const struct w_part_t *lp = &wd->lp[i];
    const char *p = (const char *)lp->p + (size_t)layer_id * lp->sz_l;
    int y = i * wd->dy;
    int dy = WD_GET_DY(y, wd->dy, wd->wy);
    mm_proc(d + y, s, p, wd->wx, dy);
  }
}

// MoE qsort prob index
static int moe_compare(const void *_a, const void *_b)
{
  const struct exp_prob_t *a = (const struct exp_prob_t *)_a;
  const struct exp_prob_t *b = (const struct exp_prob_t *)_b;
  if (a->prob > b->prob) return -1;
  if (a->prob < b->prob) return 1;
  return 0;
}

// SwiGLU non-linearity x * (1.0f / (1.0f + expf(-x)));
static _inline float swiglu(float x)
{
  return (x / (1.0f + expf(-x)));
}

// token cache, save list of injected + generated tokens history.
// compact kv cache if max context size reached.
static int update_token_cache(int token, bool is_sampled)
{
  struct transformer_t *t = &model.transformer;
  const struct transformer_config_t *p = &t->config;
  struct transformer_runstate_t *s = &t->state;
  int pos = s->cache.n_tokens++;
  s->cache.tokens[pos].token_id = token;
  s->cache.tokens[pos].sampled = is_sampled;
  if (is_sampled)
    s->cache.n_tokens_samp++;   // count last sampled tokens count (used by sampler for eos_amp option)
  else
    s->cache.n_tokens_samp = 0; // reset (injected user defined token)
  return pos;
}

#ifdef USE_THRD_BATCH
#define INC_THRD_BATCH
#include "tr_opt_inc.c"
#endif

// define pointer to y raw in weight datas and x raw size
#define WDL_Y(id, y) (const void *)((char *)w->id.lp[0].p + (size_t)(y) * w->id.wx * w_type_sizeof[w->id.d_type]), w->id.wx

void forward(int token, bool is_sampled, bool def_logits)
{
  struct transformer_t *transformer = &model.transformer;
  const struct transformer_config_t *p = &transformer->config;
  const struct transformer_weights_t *w = &transformer->weights;
  struct transformer_runstate_t *s = &transformer->state;
  int id_exit = def_logits ? -1 : (p->n_layers - 1);
  int pos, layer_id;
  float sq_sum;

  if (s->cache.n_tokens == p->seq_len)                 // context max size reached
#ifdef PACK_KV_CACHE
    reserve_kv_cache(p->seq_len/20);                   // forget some tokens
#else
  {
    memset(s->logits, 0, p->vocab_size * sizeof(float));
    s->logits[model.config.token_eot] = 1.0;           // force return eot
    return;
  }
#endif

  // save token in token cache and get current pos
  pos = update_token_cache(token, is_sampled);
  CHECK(pos < p->seq_len); 

  // update rope freqs for pos
  if (p->rope_theta)
    set_RoPE_pos(s->rope_sin_cos, pos, s->rope_freq, p->head_size/2);
  
  // ----------------------------------
  // define the token embedding into x
  if (w->token_emb.nn == 1)                            // single node (contiguous buffer)
    p->def_embeddings(s->x, WDL_Y(token_emb, token));  // token * p->dim, p->dim
  else
  {
    // mem splited in numa nodes
    int t = token / w->token_emb.dy;
    int r = token % w->token_emb.dy;
    const void *p_emb = (char *)w->token_emb.lp[t].p + (size_t)(r) * w->token_emb.wx * w_type_sizeof[w->token_emb.d_type];
    p->def_embeddings(s->x, p_emb, w->token_emb.wx);
  }

  sq_sum = vec_get_sq_sum(s->x, w->token_emb.wx);  // get x sum square for first norm

  // ----------------------------------
  // forward all the layers
  for (layer_id=0; layer_id<p->n_layers; layer_id++)
  {
    size_t s_kv_ofs;                             // layer kv offset in states
    float *k, *v;                                // key and value in cache
    bool def_q = layer_id != id_exit;
    
    // ----------------------------------
    // attention rmsnorm
    norm_scale(s->xb, s->x, sq_sum, p->rms_norm_eps, WDL_Y(rms_att, layer_id)); // p->dim

    // get key and value point to the kv cache to update
    s_kv_ofs = (size_t)layer_id * p->seq_len * p->kv_dim;   // kv cache layer offset for convenience
    k = s->k_cache + s_kv_ofs + pos * p->kv_dim;
    v = s->v_cache + s_kv_ofs + pos * p->kv_dim;

#ifdef USE_THRD_BATCH
    opt_compute_qkv(def_q ? s->q : NULL, k, v, s->xb, w, layer_id, p->matmul_lw);
#else
    // qkv matmuls for this position
    lw_matmul(k, s->xb, &w->wk, layer_id, p->matmul_lw); // p->dim, p->kv_dim
    lw_matmul(v, s->xb, &w->wv, layer_id, p->matmul_lw); // p->dim, p->kv_dim
    if (def_q)
      lw_matmul(s->q, s->xb, &w->wq, layer_id, p->matmul_lw); // p->dim, p->dim
#endif

    // optional qkv bias
    if (w->bq.ne)
    {
      vec_add(k, WDL_Y(bk, layer_id));
      vec_add(v, WDL_Y(bv, layer_id));
      if (def_q)
        vec_add(s->q, WDL_Y(bq, layer_id));
    }

    if (!p->rope_theta)                          // freq contained in layers datas
      set_RoPE_pos(s->rope_sin_cos, pos, WDL_Y(rope_if, layer_id));  // n_freq = p->head_size/2

#if 1
    // if logits not needed (tokens injection), on last layer update only k and exit
    if (!def_q)
    {
      RoPE(k, NULL, s->rope_sin_cos, p->head_size, p->kv_dim, 0);
      return;
    }
#endif

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    RoPE(s->q, k, s->rope_sin_cos, p->head_size, p->dim, p->kv_dim);

    // multihead attention. iterate over all heads, result stored in s->xb
    multihead_attention(s_kv_ofs);

    // final matmul to get the output of the attention
    lw_matmul(s->xb2, s->xb, &w->wo, layer_id, p->matmul_lw); // p->dim, p->dim

    // residual connection back into x + sq_sum
    sq_sum = vec_add_get_sq_sum(s->x, s->xb2, p->dim);

    // ffn rmsnorm
    norm_scale(s->xb, s->x, sq_sum, p->rms_norm_eps, WDL_Y(rms_ffn, layer_id)); // p->dim

    // !MoE
    if (!p->moe.num_experts)
    {
#ifdef USE_THRD_BATCH
      opt_compute_w1_w3_swiglu(s->hb, s->hb2, s->xb, w, layer_id, p->matmul_lw);
#else
      int i;
      // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
      lw_matmul(s->hb,  s->xb, &w->w1, layer_id, p->matmul_lw); // p->dim, p->hidden_dim
      lw_matmul(s->hb2, s->xb, &w->w3, layer_id, p->matmul_lw); // p->dim, p->hidden_dim

      // SwiGLU non-linearity
      for (i=0; i<p->hidden_dim; i++)
        s->hb[i] = swiglu(s->hb[i]) * s->hb2[i];
#endif
      // final p->matmul_lw to get the output of the ffn
      lw_matmul(s->xb, s->hb, &w->w2, layer_id, p->matmul_lw);  // p->hidden_dim, p->dim

      // residual connection + sq_sum
      sq_sum = vec_add_get_sq_sum(s->x, s->xb, p->dim);
    }
    else // MoE
    {
      int i, n_experts = p->moe.num_experts;
      float sum_prob = 0.0f;

      lw_matmul(s->moe.exp_logits, s->xb, &w->moe_gate, layer_id, p->matmul_lw);
      softmax(s->moe.exp_logits, n_experts);
      for (i=0; i<n_experts; i++)
      {
        s->moe.exp_probs[i].exp_id = i;
        s->moe.exp_probs[i].prob = s->moe.exp_logits[i];
      }
      qsort(s->moe.exp_probs, n_experts, sizeof(struct exp_prob_t), moe_compare);

      // Calculates the sum of probabilities for the top_k elements
      for (i=0; i<p->moe.top_k; i++)
        sum_prob += s->moe.exp_probs[i].prob;

      for (i=0; i<p->moe.top_k; i++)
      {
        int j, index = layer_id * n_experts + s->moe.exp_probs[i].exp_id;
        float k;

#ifdef USE_THRD_BATCH
        opt_compute_w1_w3_swiglu(s->hb, s->hb2, s->xb, w, index, p->matmul_lw);
#else
        lw_matmul(s->hb, s->xb, &w->w1, index, p->matmul_lw);
        lw_matmul(s->hb2, s->xb, &w->w3, index, p->matmul_lw);

        // SwiGLU non-linearity
        for (j=0; j<p->hidden_dim; j++)
          s->hb[j] = swiglu(s->hb[j]) * s->hb2[j];
#endif
        // final p->matmul_lw to get the output of the ffn
        lw_matmul(s->xb2, s->hb, &w->w2, index, p->matmul_lw);  // p->hidden_dim, p->dim

        // residual connection
        k = s->moe.exp_probs[i].prob / sum_prob;
        for (j=0; j<p->dim; j++)
          s->x[j] += s->xb2[j] * k;
      }

      // sq_sum
      sq_sum = vec_get_sq_sum(s->x, p->dim);
    }
  }

  // final rmsnorm
  norm_scale(s->x, s->x, sq_sum, p->rms_norm_eps, w->rms_final.lp[0].p, w->rms_final.wx); // p->dim

  // classifier into logits
  lw_matmul(s->logits, s->x, &w->wcls, 0, p->matmul_em); // p->dim, p->vocab_size
#if 0
  omp_proc_bind_numa_check();                 // debug check
#endif
}

#ifdef _GCC_BLD
// GCC produce "-incompatible-pointer-types" warning if no cast used.
#define FN_MUL (mm_proc_t)
#define FN_CVT (void (*)(float *, const void *, size_t))
#else
// VS do not warn. (that seem to be correct behavehour because function arguments are compatible).
#define FN_MUL
#define FN_CVT
#endif

// dummy convert function, only copy
static void data_cvt_buff_f32_to_f32(float *f32, const float *_f32, size_t ne)
{
  memcpy(f32, _f32, ne*sizeof(float));
}

// init math/convert functions depending of weights data types
static void init_wd_types_procs(void)
{
  struct transformer_config_t *p = &model.transformer.config;

  // set the type we want for memory weights, default keep .safetensors files type
  p->em_type = p->torch_type;
  p->lw_type = p->torch_type;

  if (p->torch_type == w_type_f16)
  {
    p->matmul_lw = FN_MUL matmul_procs.matmul_f32_f16;
    p->matmul_em = FN_MUL matmul_procs.matmul_f32_f16;
    p->def_embeddings = FN_CVT matmul_procs.cvt_f16_to_f32;
  }
  else
  if (p->torch_type == w_type_bf16)
  {
    p->matmul_lw = FN_MUL matmul_procs.matmul_f32_bf16;
    p->matmul_em = FN_MUL matmul_procs.matmul_f32_bf16;
    p->def_embeddings = FN_CVT matmul_procs.cvt_bf16_to_f32;
  }
  else
  if (p->torch_type == w_type_f32)
  {
    // note: using float for weights double model size, support added to test TinyLlama version 0.4 (v1.0 && 1.1 are in bfloat)
    p->matmul_lw = FN_MUL matmul_procs.matmul_f32_f32;
    p->matmul_em = FN_MUL matmul_procs.matmul_f32_f32;
    p->def_embeddings = FN_CVT data_cvt_buff_f32_to_f32; // this is useless, but keep general convert code structure
  }
  else
    msg_error("unsupported model torch type %d", p->torch_type);

  // convert data options
  if (!matmul_procs.cpu_f16c && (p->torch_type == w_type_f16))
  {
    msg_info("model is float16 but CPU have no F16C support. sf16 conversion is used.\n");
    model.config.cvt_sf16 = true;
  }

  if (model.config.cvt_sf16)
  {
    if (p->torch_type != w_type_f16)
      msg_error("model conversion to sf16 require f16 model");  // else is useless

    p->em_type = w_type_sf16;
    p->lw_type = w_type_sf16;
    p->matmul_lw = FN_MUL matmul_procs.matmul_f32_sf16;
    p->matmul_em = FN_MUL matmul_procs.matmul_f32_sf16;
    p->def_embeddings = FN_CVT matmul_procs.cvt_sf16_to_f32;
    msg_info("model converted to small float16.\n");
  }

  // convert to sf8 option
  // note: can combine with cvt_sf16, then embeddings will be sf16 and layer weights f8
  if (model.config.cvt_f8)
  {
    p->lw_type = w_type_f8;
    p->matmul_lw = FN_MUL matmul_procs.matmul_f32_f8;
    msg_info("model weights converted to float8.\n");
  }
  else
  if (model.config.cvt_f12)
  {
    p->lw_type = w_type_f12;
    p->matmul_lw = FN_MUL matmul_procs.matmul_f32_f12;
    msg_info("model weights converted to float12.\n");
  }
}

void build_transformer(void)
{
  struct transformer_config_t *p = &model.transformer.config;

  // load config
  load_checkpoint_config();

  // init math/convert functions depending of weights data types
  init_wd_types_procs();

  // init numa config and omp 
  numa_init_omp(model.config.num_procs, model.config.numa_nodes);

  // numa_disp_mem();                     // mem in nodes before allocs

  // alloc mem to load weights
  alloc_transformer();

  // load weight datas
  load_checkpoint_weights();

  // alloc struct transformer_runstate_t buffers
  alloc_run_state();

  // numa_disp_mem();                     // mem in nodes after allocs

  // init RoPE freqs
  if (p->rope_theta)
    init_RoPE(model.transformer.state.rope_freq, p->rope_theta, p->head_size);

#ifdef USE_SA_SIMD
  init_head_att_opt(matmul_procs.simd_set);    // sse/avx select code
#endif
}
