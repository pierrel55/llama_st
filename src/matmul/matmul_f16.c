#include <intrin.h>
#include "mm_hsum.h"
#include "w_types.h"
#include "matmul.h"
#include "matmul_priv.h"

// ------------------------------------------------------------------
// conversion f16 => f32
// ------------------------------------------------------------------

static float *lut_f16_to_f32 = NULL;

// must be used to create conversion lut only
static void cvt_f16_to_f32_fpu(float *f32, const f16_t *f16, size_t ne)
{
  size_t i;
  for (i=0; i!=ne; i++)
    f32[i] = lut_f16_to_f32[f16[i]];
}

static void cvt_f16_to_f32_sse(float *f32, const f16_t *f16, size_t ne)
{
  size_t i;
  for (i=0; i!=ne; i+=4)
    _mm_store_ps(f32 + i, _mm_cvtph_ps(_mm_loadl_epi64((__m128i *)(f16 + i))));
}

static void cvt_f16_to_f32_avx1(float *f32, const f16_t *f16, size_t ne)
{
  size_t i;
  for (i=0; i!=ne; i+=8)
    _mm256_store_ps(f32 + i, _mm256_cvtph_ps(_mm_load_si128((__m128i *)(f16 + i))));
}

const cvt_f16_to_f32_t cvt_f16_to_f32_procs[simd_n] =
{
  cvt_f16_to_f32_fpu,
  cvt_f16_to_f32_sse,
  cvt_f16_to_f32_avx1,
  NULL,
};

// ------------------------------------------------------------------
// matmul f32 * f16 => f32
// ------------------------------------------------------------------

// is very slow, usable for very small models
static void matmul_f32_f16_fpu(float *res, const float *vec, const f16_t *mat, int len_vec, int y_mat)
{
  const f16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    float acc = 0;
    int i;
    for (i=0; i!=len_vec; i++)
      acc += vec[i] * lut_f16_to_f32[m[i]];
    *res++ = acc;
  }
}

static void matmul_f32_f16_sse(float *res, const float *vec, const f16_t *mat, int len_vec, int y_mat)
{
  const f16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m128 acc = _mm_setzero_ps();              // init 0 in sum
    int i;
    for (i=0; i!=len_vec; i+=4)
      acc = _mm_fmadd_ps(_mm_cvtph_ps(_mm_loadl_epi64((__m128i *)(m + i))), _mm_load_ps(vec + i), acc);
    *res++ = hsum_ps_sse(acc);
  }
}

static void matmul_f32_f16_avx1(float *res, const float *vec, const f16_t *mat, int len_vec, int y_mat)
{
  const f16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m256 acc = _mm256_setzero_ps();
    int i;
    for (i=0; i!=len_vec; i+=8)
      acc = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_load_si128((__m128i *)(m + i))), _mm256_load_ps(vec + i), acc);
    *res++ = hsum_ps_avx1(acc);
  }
}

// init functions list
const matmul_f32_f16_t matmul_f32_f16_procs[simd_n] =
{
  matmul_f32_f16_fpu,
  matmul_f32_f16_sse,
  matmul_f32_f16_avx1,
  NULL,
};

// ------------------------------------------------------------------
// F16 conversions
// ------------------------------------------------------------------

#include "l_util.h"
#include "mem_alloc.h"

// --------------------------------------------------------
// software conversion f16 to f32 if no CPU support of F16C
// (opterons 62xx, xeon E55xx, x56xx, xeon E5 v1, ..)
// used for data conversion to sf16 only.

// software convert if no F16C support
static f16_t sw_cvt_f32_to_f16(float f32)
{ 
  const uint32_t b = (*(uint32_t*)&f32) + 0x00001000;
  const uint32_t e = (b & 0x7F800000) >> 23;
  uint32_t r = (b & 0x80000000) >> 16;
  if (e > 101)
  {
    const uint32_t m = b & 0x007FFFFF;
    if (e < 113) r |= (((0x007FF000 + m) >> (125-e)) + 1) >> 1;
    else
    {
      r |= (((e - 112) << 10) & 0x7C00) | m >> 13;
      if (e > 143) r |= 0x7FFF;
    }
  }
  return (f16_t)r;
}

// convert buffer f32 to f16
void cvt_f32_to_f16(f16_t *f16, const float *f32, size_t ne)
{
  size_t i;
  if (matmul_procs.cpu_f16c)
  {
    for (i=0; i!=ne; i+=4)
    {
      __m128i h4 = _mm_cvtps_ph(_mm_loadu_ps(f32 + i), _MM_FROUND_TO_NEAREST_INT);  // convert to 4 float 16
      _mm_storel_epi64((__m128i *)(f16 + i), h4);
    }
  }
  else
  {
    for (i=0; i!=ne; i++)
      f16[i] = sw_cvt_f32_to_f16(f32[i]);
  }
}

static float sw_cvt_f16_to_f32(f16_t f16) 
{ 
  const uint32_t e = (f16 & 0x7C00) >> 10;   // exponent
  const uint32_t m = (f16 & 0x03FF) << 13;   // mantissa
  uint32_t r = (f16 & 0x8000) << 16;

  if (e) r |= ((e + 112) << 23 | m);
  else if (m)
  {
    const float f = (float)m;
    const uint32_t v = (*(uint32_t*)&f)>>23;
    r |= ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000));
  }
  return *(float *)&r;
}

void init_sw_f16c(void)
{
  int i;
  lut_f16_to_f32 = malloc_check(N_64K*sizeof(float));
  for (i=0; i<N_64K; i++)
    lut_f16_to_f32[i] = sw_cvt_f16_to_f32(i);
}

void free_sw_f16c(void)
{
  free_check(lut_f16_to_f32);
  lut_f16_to_f32 = NULL;
}