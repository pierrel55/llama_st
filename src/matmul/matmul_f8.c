#include <intrin.h>
#include "mm_hsum.h"
#include "w_types.h"
#include "matmul.h"
#include "matmul_priv.h"

#define F8_CVT_MSK 0xfffff87f
#define F8_CVT_LSL 20 

#if 1
#define F8_CVT_ADD 0x380               // f8_t range 3.0517578e-005 to 1.8750000
#define F8_CVT_MAX 2.0f                // max +/- converted value
#define F8_ERR_MAX 0.125f              // max convert error for F8_CVT_MAX value

#define F16_2_00 16384                 // 2.00 in F16
#define F16_TO_F8_MAX F16_2_00         // max F16 value that can be converted to F8

#define BF16_2_00 16384                // 2.00 in BF16
#define BF16_TO_F8_MAX BF16_2_00       // max BF16 value that can be converted to F8
#else
#define F8_CVT_ADD 0x388               // f8_t range 6.1035156e-005 to 3.7500000
#define F8_CVT_MAX 4.0f
#define F8_ERR_MAX 0.25f

#define F16_4_00 17408                 // 4.00
#define F16_TO_F8_MAX F16_4_00         // max F16 value that can be converted to F8

#define BF16_4_00 16512                // 4.00
#define BF16_TO_F8_MAX BF16_4_00       // max BF16 value that can be converted to F8
#endif

// ------------------------------------------------------------------
// f32 * f8 => f32
// ------------------------------------------------------------------

static void matmul_f32_f8_fpu(float *res, const float *vec, const f8_t *mat, int len_vec, int y_mat)
{
  const f8_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    float acc = 0;
    int i;
    for (i=0; i!=len_vec; i++)
    {
      unsigned int f32i = (((char)m[i] & F8_CVT_MSK) + F8_CVT_ADD) << F8_CVT_LSL;
      acc += vec[i] * *(float *)&f32i;
    }
    *res++ = acc;
  }
}

static void matmul_f32_f8_sse(float *res, const float *vec, const f8_t *mat, int len_vec, int y_mat)
{
  const f8_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();
    int i;

    // convert 4 F8 to 4 FP32
    #define CVT_4F8(a) _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(_mm_and_si128(\
      _mm_cvtepi8_epi32(a), _mm_set1_epi32(F8_CVT_MSK)), _mm_set1_epi32(F8_CVT_ADD)), F8_CVT_LSL))
  
    // copy bytes 4..7 to 0..3, no change for bytes 8..15
    #define LSR_32(r) _mm_shufflelo_epi16(r, 2 | (3 << 2) | (2 << 4) | (3 << 6))

    for (i=0; i!=len_vec; i+=16)
    {
      __m128i b8_l = _mm_load_si128((__m128i *)(m + i)); // 8 bytes (8 FP8) 0..7
      __m128i b8_h = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(b8_l), _mm_castsi128_ps(b8_l))); // 8 bytes (8 FP8) 8..15

      acc0 = _mm_fmadd_ps(CVT_4F8(b8_l)        , _mm_load_ps(vec + i     ), acc0);  // convert 4 FP8 (byte 0..3)
      acc1 = _mm_fmadd_ps(CVT_4F8(LSR_32(b8_l)), _mm_load_ps(vec + i +  4), acc1);  // convert 4 FP8 (byte 4..7)
      acc2 = _mm_fmadd_ps(CVT_4F8(b8_h)        , _mm_load_ps(vec + i +  8), acc2);  // convert 4 FP8 (byte 8..11)
      acc3 = _mm_fmadd_ps(CVT_4F8(LSR_32(b8_h)), _mm_load_ps(vec + i + 12), acc3);  // convert 4 FP8 (byte 12..15)
    }
    *res++ = hsum_ps_sse_4x(acc0,acc1,acc2,acc3);
  }
}

static void matmul_f32_f8_avx2(float *res, const float *vec, const f8_t *mat, int len_vec, int y_mat)
{
  const f8_t *m, *m_end = mat + y_mat * len_vec;
  for (m=mat; m!=m_end; m+=len_vec)
  {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i;

    // convert 8 FP8 to 8 FP32
    #define CVT_8F8(a) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_and_si256(\
      _mm256_cvtepi8_epi32(a), _mm256_set1_epi32(F8_CVT_MSK)), _mm256_set1_epi32(F8_CVT_ADD)), F8_CVT_LSL))

    for (i=0; i!=len_vec; i+=32)
    {
      // convert 8 FP8
      acc0 = _mm256_fmadd_ps(CVT_8F8(_mm_loadl_epi64((__m128i *)(m + i     ))), _mm256_load_ps(vec + i     ), acc0);
      acc1 = _mm256_fmadd_ps(CVT_8F8(_mm_loadl_epi64((__m128i *)(m + i + 8 ))), _mm256_load_ps(vec + i + 8 ), acc1);
      acc2 = _mm256_fmadd_ps(CVT_8F8(_mm_loadl_epi64((__m128i *)(m + i + 16))), _mm256_load_ps(vec + i + 16), acc2);
      acc3 = _mm256_fmadd_ps(CVT_8F8(_mm_loadl_epi64((__m128i *)(m + i + 24))), _mm256_load_ps(vec + i + 24), acc3);
    }
    *res++ = hsum_ps_avx_4x(acc0,acc1,acc2,acc3);
  }
}

// init functions list
const matmul_f32_f8_t matmul_f32_f8_procs[simd_n] =
{
  matmul_f32_f8_fpu,
  matmul_f32_f8_sse,
  NULL,
  matmul_f32_f8_avx2,
};

// ------------------------------------------------------------------
// F8 conversions
// ------------------------------------------------------------------

#include <float.h>
#include "l_util.h"
#include "mem_alloc.h"

static float cvt_f8_to_f32(f8_t f8)
{
  unsigned int f32i = (((char)f8 & F8_CVT_MSK) + F8_CVT_ADD) << 20;
  return *(float *)&f32i;
}

// note: slow, used only to create f16 to f8 lut.
static f8_t cvt_f32_to_f8(float f32)
{
  int f_i = *(int *)&f32;
  int neg = f_i & 0x80000000;
  int i;

  if (_isnan(f32))                     // for NAN, saturate to min/max
    i = 127;
  else
  {
    float f = neg ? -f32 : f32;        // get f >= 0 value
    float f_sup = 0.0f;
    float f_inf;

    for (i=1; i<128; i++)              // positive range - NAN
    {
      f_inf = f_sup;
      f_sup = cvt_f8_to_f32((f8_t)i);
      if (f_sup > f)
        break;
    }

    if (i == 128)
      i = 127;                         // saturate to max
    else
    if ((f - f_inf) < (f_sup - f))     // get nearest value
      i--;
  }
  return neg ? i | 128 : i;
}

static f8_t lut_f16_to_f8[N_64K] = { 0 };
static f8_t lut_bf16_to_f8[N_64K] = { 0 };

// init all lookup tables.
void init_conv_f8(void)
{
  // alloc temporary AVX aligned arrays
  VAR_ALLOC(i16_list, unsigned short, N_64K);
  VAR_ALLOC(lut_i16_to_f32, float, N_64K);
  int i;

  for (i=0; i<N_64K; i++)
    i16_list[i] = (unsigned short)i;

  // F16 lut
  matmul_procs.cvt_f16_to_f32(lut_i16_to_f32, i16_list, N_64K);
  if (lut_i16_to_f32[F16_TO_F8_MAX] != F8_CVT_MAX)     // error in config constants
    msg_error("init_conv_f8 failed");

  for (i=0; i<N_64K; i++)
    lut_f16_to_f8[i] = cvt_f32_to_f8(lut_i16_to_f32[i]);

  // BF16 lut
  matmul_procs.cvt_bf16_to_f32(lut_i16_to_f32, i16_list, N_64K);
  CHECK(lut_i16_to_f32[BF16_TO_F8_MAX] == F8_CVT_MAX);
  for (i=0; i<N_64K; i++)
    lut_bf16_to_f8[i] = cvt_f32_to_f8(lut_i16_to_f32[i]);

  free_check(lut_i16_to_f32);
  free_check(i16_list);
}

// convert buffer f16 to f8
void cvt_f16_to_f8(f8_t *f8, const f16_t *f16, size_t ne)
{
  const f16_t *f16_end = f16 + ne;
  while (f16 < f16_end)
  {
    f16_t _a = *f16++;
    if (ABS_F16(_a) > F16_TO_F8_MAX)    // check _a can be converted with small error
      msg_error("conversion F16 to F8 out of range");
    *f8++ = lut_f16_to_f8[_a];
  }
}

// convert buffer bf16 to f8
void cvt_bf16_to_f8(f8_t *f8, const bf16_t *bf16, size_t ne)
{
  const bf16_t *bf16_end = bf16 + ne;
  while (bf16 < bf16_end)
  {
    bf16_t _a = *bf16++;
    if (ABS_F16(_a) > BF16_TO_F8_MAX)  // check _a can be converted with small error
      msg_error("conversion BF16 to F8 out of range");
    *f8++ = lut_bf16_to_f8[_a];
  }
}
