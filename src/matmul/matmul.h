
// simd code type
enum e_simd_typ
{
  simd_fpu = 0,
  simd_sse,
  simd_avx1,
  simd_avx2,
  simd_n,
};

#define SIMD_LV 32                     // max used len_vec/or ne stride in matmul_xx/cvt_xx functions

// ---------------------------
// data conversion to float 32

// float16 to float32 conversion
typedef void (* cvt_f16_to_f32_t)(float *f32, const f16_t *f16, size_t ne);

// bfloat16 to float32 conversion
typedef void (* cvt_bf16_to_f32_t)(float *f32, const bf16_t *bf16, size_t ne);

// sfloat16 to float32 conversion
typedef void (* cvt_sf16_to_f32_t)(float *f32, const sf16_t *sf16, size_t ne);

// float12 to float32 conversion
typedef void (* cvt_f12_to_f32_t)(float *f32, const f12_t *f12, size_t ne);

// float12 and float8 to float32 conversion
// => not coded because this conversion in used only in matmul_f32_f12_procs / matmul_f32_f8_procs

// -----------------------------------
// vector to matrix multiply functions

// float32 * float32 => float32
typedef void (* matmul_f32_f32_t)(float *res, const float *vec, const float *mat, int len_vec, int y_mat);

// float32 * float16 => float32
typedef void (* matmul_f32_f16_t)(float *res, const float *vec, const f16_t *mat, int len_vec, int y_mat);

// float32 * bfloat16 => float32
typedef void (* matmul_f32_bf16_t)(float *res, const float *vec, const bf16_t *mat, int len_vec, int y_mat);

// float32 * sfloat16 => float32
typedef void (* matmul_f32_sf16_t)(float *res, const float *vec, const sf16_t *mat, int len_vec, int y_mat);

// float32 * float12 => float32
typedef void (* matmul_f32_f12_t)(float *res, const float *vec, const f12_t *mat, int len_vec, int y_mat);

// float32 * float8 => float32
typedef void (* matmul_f32_f8_t)(float *res, const float *vec, const f8_t *mat, int len_vec, int y_mat);

// list of functions
struct matmul_procs_t
{
  // convert
  cvt_f16_to_f32_t  cvt_f16_to_f32;
  cvt_bf16_to_f32_t cvt_bf16_to_f32;
  cvt_sf16_to_f32_t cvt_sf16_to_f32;
  cvt_f12_to_f32_t  cvt_f12_to_f32;

  // matmul
  matmul_f32_f32_t  matmul_f32_f32;
  matmul_f32_f16_t  matmul_f32_f16;
  matmul_f32_bf16_t matmul_f32_bf16;
  matmul_f32_sf16_t matmul_f32_sf16;
  matmul_f32_f12_t  matmul_f32_f12;
  matmul_f32_f8_t   matmul_f32_f8;

  // infos
  enum e_simd_typ simd_set;    // initialized mode
  int cpu_f16c;                // 1: f16c support
};

// interface
extern struct matmul_procs_t matmul_procs;

// generic data types conversions
void cvt_w_data(void *d, enum e_w_type d_type, const void *s, enum e_w_type s_type, size_t ne);

// init
void matmul_init(enum e_simd_typ simd_typ);

// free some memory
void matmul_exit(void);
