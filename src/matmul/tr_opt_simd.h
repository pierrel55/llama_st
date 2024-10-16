// -----------------------------------------------------
// simd optimized head attention (code in tr_opt_simd.c)

typedef void (* head_att_opt_t)(float *xb, int n_tok, float *att, const float *q, const float *k, const float *v, const struct transformer_config_t *p);

// head attention simd (defined by matmul_init())
extern head_att_opt_t head_att_opt;

// init
void init_head_att_opt(enum e_simd_typ simd_typ);
