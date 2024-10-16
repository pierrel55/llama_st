#include "numa.h"                  // need MAX_NUMA_PROCS / MAX_NUMA_NODES
#include "w_types.h"

// matmul function, used one depend of weights data type (f32/f16/bf16/f8)
typedef void (*mm_proc_t)(float *res, const float *vec, const void *mat, int len_vec, int y_mat);

// transformer config from config.json
struct transformer_config_t
{
  // loaded parameters in config.json
  int dim;                         // transformer dimension
  int hidden_dim;                  // for ffn layers
  int n_layers;                    // number of layers
  int n_heads;                     // number of query heads
  int n_kv_heads;                  // number of key/value heads (can be < query heads because of multiquery)
  int seq_len;                     // max sequence length
  float rms_norm_eps;
  float rope_theta;                // optional if .safetensors contain rope freqs array
  int vocab_size;                  // vocabulary size, usually 256 (byte-level)

  enum e_w_type torch_type;        // float format (float32/float16/bfloat16)

  // defined using previous loaded parameters
  int head_size;                   // dim / n_heads
  int kv_dim;                      // (dim * n_kv_heads) / n_heads
  int kv_mul;                      // n_heads / n_kv_heads
  float sqrt_head_size;            // sqrt(head_size)

  // matmul and conversion functions depend of loaded/mem weights format (f32/f16/bf16/f8)
  enum e_w_type em_type;           // embeddings data type
  enum e_w_type lw_type;           // layer weights data type
  void (* def_embeddings)(float *f32, const void *emb, size_t ne);
  mm_proc_t matmul_em;             // matmul function used for embeddings weights
  mm_proc_t matmul_lw;             // matmul function used for layer weights

  // MoE/mixtral specific
  struct
  {
    int num_experts;               // number of experts in the expert model
    int top_k;                     // expert num per token
  } moe;
};

// weight data part
struct w_part_t
{
  void *p;                         // weight part data pointer
  size_t sz_l;                     // layer byte stride for node ptr
};

// memory weight datas
struct w_dat_t
{
  enum e_w_type d_type;            // data type in p (f32/f16/bf16/sf16/f8)
  int wx;                          // width x (raw size, mem contiguous elements)
  int wy;                          // width y (raw count)
  int nz;                          // count of wx * wy in p
  int dy;                          // splitted wy size
  int nn;                          // num different nodes used to store weights
  size_t ne;                       // num element total nz*wy*wx, used to check load
  void *p_node[MAX_NUMA_NODES];    // allocated mem base in nodes
  struct w_part_t lp[MAX_NUMA_PROCS]; // layer 0 weight part list
};

// copy weight datas
void copy_w_dat(struct w_dat_t *wd, void *d);

// weights datas
struct transformer_weights_t
{
  // token embedding table
  struct w_dat_t token_emb;        // (vocab_size, dim)
  // attention weights
  struct w_dat_t rms_att;          // (layer, dim, 1) rmsnorm weights
  struct w_dat_t rope_if;          // (layer, head_size/2, 1)
  struct w_dat_t wq;               // (layer, dim, n_heads * head_size)
  struct w_dat_t wk;               // (layer, dim, n_kv_heads * head_size)
  struct w_dat_t wv;               // (layer, dim, n_kv_heads * head_size)
  struct w_dat_t wo;               // (layer, n_heads * head_size, dim)
  // optional qkv bias (used in qwen2)
  struct w_dat_t bq;               // (layer, n_heads * head_size)
  struct w_dat_t bk;               // (layer, n_kv_heads * head_size)
  struct w_dat_t bv;               // (layer, n_kv_heads * head_size)
  // ffn weights
  struct w_dat_t rms_ffn;          // (layer, dim, 1)
  struct w_dat_t w1;               // (layer, hidden_dim, dim)  (layer * n_experts if MoE)
  struct w_dat_t w2;               // (layer, dim, hidden_dim)  (layer * n_experts if MoE)
  struct w_dat_t w3;               // (layer, hidden_dim, dim)  (layer * n_experts if MoE)
  // final rmsnorm
  struct w_dat_t rms_final;        // (dim, 1)
  // (optional) classifier weights for the logits, on the last layer
  struct w_dat_t wcls;
  // MoE
  struct w_dat_t moe_gate;         // (layer, dim, num_experts)
};

// MoE specific, used to sort experts prob
struct exp_prob_t
{
  float prob;
  int exp_id;
};

// cache saved token
struct ctoken_t
{
  int token_id;                    // token id
  bool sampled;                    // 0 if injected (user defined), 1 if sampled (LLM defined)
};

struct transformer_runstate_t
{
  // current wave of activations
  float *x;                        // activation at current time stamp (dim,)
  float *xb;                       // same, but inside a residual branch (dim,)
  float *xb2;                      // an additional buffer just for convenience (dim,)
  float *hb;                       // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;                      // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q;                        // query (dim,)
  float *k_cache;                  // key cache (layer, seq_len, dim)
  float *v_cache;                  // value cache (layer, seq_len, dim)
  float *att;                      // buffer for scores/attention values (n_heads, seq_len)
  float *logits;                   // output logits

  // RoPE
  float *rope_freq;                // inv freq (NULL if contained in .safetensors)
  float *rope_sin_cos;             // sin/cos values for positions

  // tokens cache/history
  struct
  {
    struct ctoken_t *tokens;       // current list of tokens that produces kv cache state
    int n_tokens;                  // num tokens encoded in cache (= pos)
    int n_tokens_samp;             // num sampled tokens at tokens list end
    int n_tokens_sys;              // num tokens to keep in sys prompt if context compacted
    int n_tokens_del;              // num tokens deleted in cache (user info)
  } cache;

  // moe mixtral
  struct
  {
    float *exp_logits;             // num_experts
    struct exp_prob_t *exp_probs;  // num_experts
  } moe;
};

struct transformer_t
{
  struct transformer_config_t config;     // the hyperparameters of the architecture (the blueprint)
  struct transformer_weights_t weights;   // the weights of the model
  struct transformer_runstate_t state;    // buffers for the "wave" of activations in the forward pass
};

// init
void build_transformer(void);

// free mem
void free_wd(struct w_dat_t *wd);
void free_transformer(void);

// forward, update cache and return logits if def_logits set as true
void forward(int token, bool is_sampled, bool def_logits);

#ifdef PACK_KV_CACHE

// private, for kv_cache.c
void set_RoPE_pos(float *sin_cos, int pos, const float *freq, int n_freq);
void RoPE(float *a, float *b, const float *sin_cos, int head_size, int a_dim, int b_dim);

// in kv_cache.c
int reserve_kv_cache(int min_token_reserve);

#endif