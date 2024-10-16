#include <intrin.h>
#include "mm_hsum.h"
#include "w_types.h"
#include "matmul.h"
#include "matmul_priv.h"

// --------------------------------
// float SF16 conversions

#define SF16_CVT_MSK 0xfffc7fff
#define SF16_CVT_LSL 13
#define SF16_CVT_ADD 0x18800           // sf16 range 1.8626451e-009 to 7.9960938

#define SF16_CVT_MAX 8.0f              // max +/- converted value
#define SF16_ERR_MAX 0.0039062f        // max convert error for SF16_CVT_MAX value

#define F16_8_00 18432                 // 8.00 in float 16
#define F16_TO_SF16_MAX F16_8_00       // max F16 value that can be converted to SF16

// ------------------------------------------------------------------
// conversion sf16 => f32
// ------------------------------------------------------------------

static void cvt_sf16_to_f32_fpu(float *f32, const sf16_t *sf16, size_t ne)
{
  int *ps = (int *)f32;
  size_t i;
  for (i=0; i<ne; i++)
    ps[i] = (((short)sf16[i] & SF16_CVT_MSK) + SF16_CVT_ADD) << SF16_CVT_LSL;
}

// convert 4 SF16 to 4 FP32
#define CVT_4SF16(a) _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(_mm_and_si128(\
  _mm_cvtepi16_epi32(a), _mm_set1_epi32(SF16_CVT_MSK)), _mm_set1_epi32(SF16_CVT_ADD)), SF16_CVT_LSL))

static void cvt_sf16_to_f32_sse(float *f32, const sf16_t *sf16, size_t ne)
{
  size_t i;
  for (i=0; i!=ne; i+=4)
    _mm_store_ps(f32 + i, CVT_4SF16(_mm_loadl_epi64((__m128i *)(sf16 + i))));
}

// convert 8 SF16 to 8 FP32
#define CVT_8SF16(a) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_and_si256(\
  _mm256_cvtepi16_epi32(a), _mm256_set1_epi32(SF16_CVT_MSK)), _mm256_set1_epi32(SF16_CVT_ADD)), SF16_CVT_LSL))

static void cvt_sf16_to_f32_avx2(float *f32, const sf16_t *sf16, size_t ne)
{
  size_t i;
  for (i=0; i!=ne; i+=8)
    _mm256_store_ps(f32 + i, CVT_8SF16(_mm_load_si128((__m128i *)(sf16 + i))));
}

const cvt_sf16_to_f32_t cvt_sf16_to_f32_procs[simd_n] =
{
  cvt_sf16_to_f32_fpu,
  cvt_sf16_to_f32_sse,
  NULL,
  cvt_sf16_to_f32_avx2,
};

// ------------------------------------------------------------------
// f32 * sf16 => f32
// ------------------------------------------------------------------

static void matmul_f32_sf16_fpu(float *res, const float *vec, const sf16_t *mat, int len_vec, int y_mat)
{
  const sf16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    float acc = 0;
    int i;
    for (i=0; i!=len_vec; i++)
    {
      unsigned int f32i = (((short)m[i] & SF16_CVT_MSK) + SF16_CVT_ADD) << SF16_CVT_LSL;
      acc += vec[i] * *(float *)&f32i;
    }
    *res++ = acc;
  }
}

static void matmul_f32_sf16_sse(float *res, const float *vec, const sf16_t *mat, int len_vec, int y_mat)
{
  const sf16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();
    int i;
    for (i=0; i!=len_vec; i+=16)
    {
      acc0 = _mm_fmadd_ps(_mm_load_ps(vec + i     ), CVT_4SF16(_mm_loadl_epi64((__m128i *)(m + i     ))), acc0);
      acc1 = _mm_fmadd_ps(_mm_load_ps(vec + i +  4), CVT_4SF16(_mm_loadl_epi64((__m128i *)(m + i +  4))), acc1);
      acc2 = _mm_fmadd_ps(_mm_load_ps(vec + i +  8), CVT_4SF16(_mm_loadl_epi64((__m128i *)(m + i +  8))), acc2);
      acc3 = _mm_fmadd_ps(_mm_load_ps(vec + i + 12), CVT_4SF16(_mm_loadl_epi64((__m128i *)(m + i + 12))), acc3);
    }
    *res++ = hsum_ps_sse_4x(acc0,acc1,acc2,acc3);
  }
}

static void matmul_f32_sf16_avx2(float *res, const float *vec, const sf16_t *mat, int len_vec, int y_mat)
{
  const sf16_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m256 acc = _mm256_setzero_ps();
    int i;
    for (i=0; i!=len_vec; i+=8)
      acc = _mm256_fmadd_ps(_mm256_load_ps(vec + i), CVT_8SF16(_mm_load_si128((__m128i *)(m + i))), acc);
    *res++ = hsum_ps_avx1(acc);
  }
}

// init functions list
const matmul_f32_sf16_t matmul_f32_sf16_procs[simd_n] =
{
  matmul_f32_sf16_fpu,
  matmul_f32_sf16_sse,
  NULL,
  matmul_f32_sf16_avx2,
};

// ------------------------------------------------------------------
// SF16 conversions
// ------------------------------------------------------------------

#include "l_util.h"
#include "mem_alloc.h"

// lut to convert model weights
static sf16_t lut_f16_to_sf16[N_64K] = { 0 };

// f32 to sf16 (using e_ofs = 98)
static int f32_to_sf16(float f32)
{
  int a = *(int *)&f32;
  int e = (a >> 23) & 0xff;
  int m = (a >> (23 - 10)) & ((1 << 10) - 1);
  int f16 = m + ((e - 98) << 10);
  return f16;
}

// init lookup table.
void init_conv_sf16(void)
{
  // alloc temporary AVX aligned arrays
  VAR_ALLOC(f16_list, f16_t, N_64K/2);
  VAR_ALLOC(f16_to_f32, float, N_64K/2);
  int i;

  for (i=0; i<N_64K/2; i++)
    f16_list[i] = (f16_t)i;

  matmul_procs.cvt_f16_to_f32(f16_to_f32, f16_list, N_64K/2);
  if (f16_to_f32[F16_TO_SF16_MAX] != SF16_CVT_MAX)   // error in config constants
    msg_error("init_conv_sf16 failed");

  for (i=0; i<N_64K/2; i++)
  {
    float f32 = f16_to_f32[i];
    int k = f32_to_sf16(f32);
    if      (k < 0)          k = 0;
    else if (k >= (N_64K/2)) k = (N_64K/2)-1;

    // note: there is no rounding required, except for F16 0.0 replaced
    // by +/-1.8626451e-009, all other values < SF16_TO_F16_MAX match exactly.
    lut_f16_to_sf16[i] = k;
    lut_f16_to_sf16[i+(N_64K/2)] = 0x8000 | k;
  }
  free_check(f16_to_f32);
  free_check(f16_list);
}

// convert buffer f16 to sf16
void cvt_f16_to_sf16(sf16_t *sf16, const f16_t *f16, size_t ne)
{
  const f16_t *f16_end = f16 + ne;
  while (f16 < f16_end)
  {
    f16_t _a = *f16++;
    if (ABS_F16(_a) > F16_TO_SF16_MAX)    // check _a can be converted with small error
      msg_error("conversion F16 to SF16 out of range");
    *sf16++ = lut_f16_to_sf16[_a];
  }
}