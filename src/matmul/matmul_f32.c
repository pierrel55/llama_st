#include <intrin.h>
#include "mm_hsum.h"
#include "w_types.h"
#include "matmul.h"

// ------------------------------------------------------------------
// f32 * f32 => f32
// ------------------------------------------------------------------

static void matmul_f32_f32_fpu(float *res, const float *vec, const float *mat, int len_vec, int y_mat)
{
  const float *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    float acc = 0;
    int i;
    for (i=0; i!=len_vec; i++)
      acc += vec[i] * m[i];
    *res++ = acc;
  }
}

static void matmul_f32_f32_sse(float *res, const float *vec, const float *mat, int len_vec, int y_mat)
{
  const float *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m128 acc = _mm_setzero_ps();              // init 0 in sum
    int i;
    for (i=0; i!=len_vec; i+=4)
      acc = _mm_fmadd_ps(_mm_load_ps(vec + i), _mm_load_ps(m + i), acc);
    *res++ = hsum_ps_sse(acc);
  }
}

static void matmul_f32_f32_avx1(float *res, const float *vec, const float *mat, int len_vec, int y_mat)
{
  const float *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m256 acc = _mm256_setzero_ps();
    int i;
    for (i=0; i!=len_vec; i+=8)
      acc = _mm256_fmadd_ps(_mm256_load_ps(vec + i), _mm256_load_ps(m + i), acc);
    *res++ = hsum_ps_avx1(acc);
  }
}

// init functions list
const matmul_f32_f32_t matmul_f32_f32_procs[simd_n] =
{
  matmul_f32_f32_fpu,
  matmul_f32_f32_sse,
  matmul_f32_f32_avx1,
  NULL,
};
