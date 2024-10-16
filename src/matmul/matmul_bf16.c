#include <intrin.h>
#include "mm_hsum.h"
#include "w_types.h"
#include "matmul.h"

// ------------------------------------------------------------------
// conversion bf16 => f32
// ------------------------------------------------------------------

static void cvt_bf16_to_f32_fpu(float *f32, const bf16_t *bf16, size_t ne)
{
  int *ps = (int *)f32;
  size_t i;
  for (i=0; i<ne; i++)
    ps[i] = (unsigned int)bf16[i] << 16;
}

static void cvt_bf16_to_f32_sse(float *f32, const bf16_t *bf16, size_t ne)
{
  size_t i;
  for (i=0; i!=ne; i+=4)
  {
    __m128i pi = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i *)(bf16 + i)));
    _mm_store_ps(f32 + i, _mm_castsi128_ps(_mm_slli_epi32(pi, 16)));
  }
}

#define GET_8BF16_AVX1(d) _mm256_castsi256_ps(_mm256_set_m128i(_mm_unpackhi_epi16(_mm_setzero_si128(), d), _mm_unpacklo_epi16(_mm_setzero_si128(), d)))

// todo: use shuffle ?
static void cvt_bf16_to_f32_avx1(float *f32, const bf16_t *bf16, size_t ne)
{
  size_t i;
  for (i=0; i!=ne; i+=8)
  {
    __m128i d = _mm_load_si128((__m128i *)(bf16 + i));
    _mm256_store_ps(f32 + i, GET_8BF16_AVX1(d));
  }
}

#define GET_8BF16_AVX2(d) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(d), 16))

static void cvt_bf16_to_f32_avx2(float *f32, const bf16_t *bf16, size_t ne)
{
  size_t i;
  for (i=0; i!=ne; i+=8)
  {
    __m128i d = _mm_load_si128((__m128i *)(bf16 + i));
    _mm256_store_ps(f32 + i, GET_8BF16_AVX2(d));
  }
}

const cvt_bf16_to_f32_t cvt_bf16_to_f32_procs[simd_n] =
{
  cvt_bf16_to_f32_fpu,
  cvt_bf16_to_f32_sse,
  cvt_bf16_to_f32_avx1,
  cvt_bf16_to_f32_avx2,
};

// ------------------------------------------------------------------
// f32 * bf16 => f32
// ------------------------------------------------------------------

static void matmul_f32_bf16_fpu(float *res, const float *vec, const bf16_t *mat, int len_vec, int y_mat)
{
  const bf16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    float acc = 0;
    int i;
    for (i=0; i!=len_vec; i++)
    {
      unsigned int _f = m[i] << 16;
      acc += vec[i] * *(float *)&_f;
    }
    *res++ = acc;
  }
}

static void matmul_f32_bf16_sse(float *res, const float *vec, const bf16_t *mat, int len_vec, int y_mat)
{
  const bf16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();
    int i;
    for (i=0; i!=len_vec; i+=16)
    {
      __m128i d0 = _mm_load_si128((__m128i *)(m + i));
      __m128 ps_l0 = _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), d0));
      __m128 ps_h0 = _mm_castsi128_ps(_mm_unpackhi_epi16(_mm_setzero_si128(), d0));
      __m128i d1 = _mm_load_si128((__m128i *)(m + i + 8));
      __m128 ps_l1 = _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), d1));
      __m128 ps_h1 = _mm_castsi128_ps(_mm_unpackhi_epi16(_mm_setzero_si128(), d1));
      acc0 = _mm_fmadd_ps(ps_l0, _mm_load_ps(vec + i     ), acc0);
      acc1 = _mm_fmadd_ps(ps_h0, _mm_load_ps(vec + i + 4 ), acc1);
      acc2 = _mm_fmadd_ps(ps_l1, _mm_load_ps(vec + i + 8 ), acc2);
      acc3 = _mm_fmadd_ps(ps_h1, _mm_load_ps(vec + i + 12), acc3);
    }
    *res++ = hsum_ps_sse_4x(acc0,acc1,acc2,acc3);
  }
}

static void matmul_f32_bf16_avx1(float *res, const float *vec, const bf16_t *mat, int len_vec, int y_mat)
{
  const bf16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i;
    for (i=0; i!=len_vec; i+=16)
    {
      __m128i d0 = _mm_load_si128((__m128i *)(m + i    ));
      __m128i d1 = _mm_load_si128((__m128i *)(m + i + 8));
      acc0 = _mm256_fmadd_ps(GET_8BF16_AVX1(d0), _mm256_load_ps(vec + i    ), acc0);
      acc1 = _mm256_fmadd_ps(GET_8BF16_AVX1(d1), _mm256_load_ps(vec + i + 8), acc1);
    }
    *res++ = hsum_ps_avx_2x(acc0, acc1);
  }
}

static void matmul_f32_bf16_avx2(float *res, const float *vec, const bf16_t *mat, int len_vec, int y_mat)
{
  const bf16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i;
    for (i=0; i!=len_vec; i+=16)
    {
      __m128i d0 = _mm_load_si128((__m128i *)(m + i    ));
      __m128i d1 = _mm_load_si128((__m128i *)(m + i + 8));
      acc0 = _mm256_fmadd_ps(GET_8BF16_AVX2(d0), _mm256_load_ps(vec + i    ), acc0);
      acc1 = _mm256_fmadd_ps(GET_8BF16_AVX2(d1), _mm256_load_ps(vec + i + 8), acc1);
    }
    *res++ = hsum_ps_avx_2x(acc0, acc1);
  }
}

// init functions list
const matmul_f32_bf16_t matmul_f32_bf16_procs[simd_n] =
{
  matmul_f32_bf16_fpu,
  matmul_f32_bf16_sse,
  matmul_f32_bf16_avx1,
  matmul_f32_bf16_avx2
};
