#include <intrin.h>
#include "mm_hsum.h"
#include "w_types.h"
#include "matmul.h"
#include "matmul_priv.h"

// ------------------------------------------------------------------
// float F12 E4M7 conversions, range +/- 6.1035156e-5 to 3.9843750

#define F12_CVT_MSK 0xffff87ff
#define F12_CVT_LSL 16
#define F12_CVT_ADD 0x3880            // f12 range 1.8626451e-009 to 7.9960938

#define F12_CVT_MAX 4.0f              // max +/- converted value
//#define F12_ERR_MAX f               // max convert error for F12_CVT_MAX value

#define F16_4_00 17408                // 4.00 in float 16
#define F16_TO_F12_MAX F16_4_00       // max F16 value that can be converted to F12

#define BF16_4_00 16512               // 4.00
#define BF16_TO_F12_MAX BF16_4_00     // max BF16 value that can be converted to F8

#define SE12(x) ((short)((x) << 4) >> 4)   // sign extend

// SF12 to F32 convert.
static float cvt_f12_to_f32(f12_t f12)
{
  unsigned int f32i = ((SE12(f12) & F12_CVT_MSK) + F12_CVT_ADD) << F12_CVT_LSL;
  return *(float *)&f32i;
}

// ------------------------------------------------------------------
// f32 * f12 => f32
// ------------------------------------------------------------------

// sotware version. (slow, for reference only)
static void matmul_f32_f12_fpu(float *res, const float *vec, const f12_t *mat, int len_vec, int y_mat)
{
  const unsigned char *e = (const unsigned char *)mat;
  int y;
  for (y=0; y<y_mat; y++)
  {
    float acc = 0;
    const float *v, *v_end = vec + len_vec;
    for (v=vec; v!=v_end; v+=16, e+=24)
    {
      acc += cvt_f12_to_f32(((f12_t)e[0]  << 4) | (e[16] & 0xf)) * v[0];
      acc += cvt_f12_to_f32(((f12_t)e[1]  << 4) | (e[17] & 0xf)) * v[1];
      acc += cvt_f12_to_f32(((f12_t)e[2]  << 4) | (e[18] & 0xf)) * v[2];
      acc += cvt_f12_to_f32(((f12_t)e[3]  << 4) | (e[19] & 0xf)) * v[3];
      acc += cvt_f12_to_f32(((f12_t)e[4]  << 4) | (e[20] & 0xf)) * v[4];
      acc += cvt_f12_to_f32(((f12_t)e[5]  << 4) | (e[21] & 0xf)) * v[5];
      acc += cvt_f12_to_f32(((f12_t)e[6]  << 4) | (e[22] & 0xf)) * v[6];
      acc += cvt_f12_to_f32(((f12_t)e[7]  << 4) | (e[23] & 0xf)) * v[7];

      acc += cvt_f12_to_f32(((f12_t)e[8]  << 4) | (e[16] >> 4))  * v[8];
      acc += cvt_f12_to_f32(((f12_t)e[9]  << 4) | (e[17] >> 4))  * v[9];
      acc += cvt_f12_to_f32(((f12_t)e[10] << 4) | (e[18] >> 4))  * v[10];
      acc += cvt_f12_to_f32(((f12_t)e[11] << 4) | (e[19] >> 4))  * v[11];
      acc += cvt_f12_to_f32(((f12_t)e[12] << 4) | (e[20] >> 4))  * v[12];
      acc += cvt_f12_to_f32(((f12_t)e[13] << 4) | (e[21] >> 4))  * v[13];
      acc += cvt_f12_to_f32(((f12_t)e[14] << 4) | (e[22] >> 4))  * v[14];
      acc += cvt_f12_to_f32(((f12_t)e[15] << 4) | (e[23] >> 4))  * v[15];
    }
    *res++ = acc;
  }
}

// shuffle index
#define SHI(d,c,b,a) ((d << 6) | (c << 4) | (b << 2) | a)

static void matmul_f32_f12_sse(float *res, const float *vec, const f12_t *mat, int len_vec, int y_mat)
{
  const unsigned char *e = (const unsigned char *)mat;
  int y;
  for (y=0; y<y_mat; y++)
  {
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();

    const float *v, *v_end = vec + len_vec;
    for (v=vec; v!=v_end; v+=16, e+=24)
    {
      __m128i _r0h, _r0l, _r1h, _r1l, _m0h, _m0l;

      // unpack 16 f12 to 16 f32
      _r0l = _mm_loadu_si128((__m128i *)e);
      _r0h = _mm_shuffle_epi32(_r0l, SHI(3,2,1,1));
      _r1l = _mm_shuffle_epi32(_r0l, SHI(3,2,1,2));
      _r1h = _mm_shuffle_epi32(_r0l, SHI(3,2,1,3));
      
      _r0l = _mm_cvtepi8_epi32(_r0l);
      _r0h = _mm_cvtepi8_epi32(_r0h);
      _r1l = _mm_cvtepi8_epi32(_r1l);
      _r1h = _mm_cvtepi8_epi32(_r1h);

      _r0l = _mm_slli_epi32(_r0l, 4);
      _r0h = _mm_slli_epi32(_r0h, 4);
      _r1l = _mm_slli_epi32(_r1l, 4);
      _r1h = _mm_slli_epi32(_r1h, 4);

      _m0l = _mm_loadl_epi64((__m128i *)(e + 16));
      _m0h = _mm_shuffle_epi32(_m0l, SHI(3,2,1,1));
      _m0l = _mm_cvtepu8_epi32(_m0l);
      _m0h = _mm_cvtepu8_epi32(_m0h);

      _r0l = _mm_or_si128(_r0l, _mm_and_si128(_m0l, _mm_set1_epi32(0xf)));
      _r0h = _mm_or_si128(_r0h, _mm_and_si128(_m0h, _mm_set1_epi32(0xf)));
      _r1l = _mm_or_si128(_r1l, _mm_srli_epi32(_m0l, 4));
      _r1h = _mm_or_si128(_r1h, _mm_srli_epi32(_m0h, 4));

      _r0l = _mm_and_si128(_r0l, _mm_set1_epi32(F12_CVT_MSK));
      _r0h = _mm_and_si128(_r0h, _mm_set1_epi32(F12_CVT_MSK));
      _r1l = _mm_and_si128(_r1l, _mm_set1_epi32(F12_CVT_MSK));
      _r1h = _mm_and_si128(_r1h, _mm_set1_epi32(F12_CVT_MSK));

      _r0l = _mm_add_epi32(_r0l, _mm_set1_epi32(F12_CVT_ADD));
      _r0h = _mm_add_epi32(_r0h, _mm_set1_epi32(F12_CVT_ADD));
      _r1l = _mm_add_epi32(_r1l, _mm_set1_epi32(F12_CVT_ADD));
      _r1h = _mm_add_epi32(_r1h, _mm_set1_epi32(F12_CVT_ADD));
      
      _r0l = _mm_slli_epi32(_r0l, F12_CVT_LSL);
      _r0h = _mm_slli_epi32(_r0h, F12_CVT_LSL);
      _r1l = _mm_slli_epi32(_r1l, F12_CVT_LSL);
      _r1h = _mm_slli_epi32(_r1h, F12_CVT_LSL);

      acc0 = _mm_fmadd_ps(_mm_load_ps(v     ), _mm_castsi128_ps(_r0l), acc0);
      acc1 = _mm_fmadd_ps(_mm_load_ps(v +  4), _mm_castsi128_ps(_r0h), acc1);
      acc2 = _mm_fmadd_ps(_mm_load_ps(v +  8), _mm_castsi128_ps(_r1l), acc2);
      acc3 = _mm_fmadd_ps(_mm_load_ps(v + 12), _mm_castsi128_ps(_r1h), acc3);
    }
    *res++ = hsum_ps_sse_4x(acc0,acc1,acc2,acc3);
  }
}

// this is mostly sse + fma in avx (no gain vs sse only)
static void matmul_f32_f12_avx(float *res, const float *vec, const f12_t *mat, int len_vec, int y_mat)
{
  const unsigned char *e = (const unsigned char *)mat;
  int y;
  for (y=0; y<y_mat; y++)
  {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    const float *v, *v_end = vec + len_vec;
    for (v=vec; v!=v_end; v+=16, e+=24)
    {
      __m128i _r0h, _r0l, _r1h, _r1l, _m0h, _m0l;

      // unpack 16 f12 to 16 f32
      _r0l = _mm_loadu_si128((__m128i *)e);
      _r0h = _mm_shuffle_epi32(_r0l, SHI(3,2,1,1));
      _r1l = _mm_shuffle_epi32(_r0l, SHI(3,2,1,2));
      _r1h = _mm_shuffle_epi32(_r0l, SHI(3,2,1,3));
      
      _r0l = _mm_cvtepi8_epi32(_r0l);
      _r0h = _mm_cvtepi8_epi32(_r0h);
      _r1l = _mm_cvtepi8_epi32(_r1l);
      _r1h = _mm_cvtepi8_epi32(_r1h);

      _r0l = _mm_slli_epi32(_r0l, 4);
      _r0h = _mm_slli_epi32(_r0h, 4);
      _r1l = _mm_slli_epi32(_r1l, 4);
      _r1h = _mm_slli_epi32(_r1h, 4);

      _m0l = _mm_loadl_epi64((__m128i *)(e + 16));
      _m0h = _mm_shuffle_epi32(_m0l, SHI(3,2,1,1));
      _m0l = _mm_cvtepu8_epi32(_m0l);
      _m0h = _mm_cvtepu8_epi32(_m0h);

      _r0l = _mm_or_si128(_r0l, _mm_and_si128(_m0l, _mm_set1_epi32(0xf)));
      _r0h = _mm_or_si128(_r0h, _mm_and_si128(_m0h, _mm_set1_epi32(0xf)));
      _r1l = _mm_or_si128(_r1l, _mm_srli_epi32(_m0l, 4));
      _r1h = _mm_or_si128(_r1h, _mm_srli_epi32(_m0h, 4));

      _r0l = _mm_and_si128(_r0l, _mm_set1_epi32(F12_CVT_MSK));
      _r0h = _mm_and_si128(_r0h, _mm_set1_epi32(F12_CVT_MSK));
      _r1l = _mm_and_si128(_r1l, _mm_set1_epi32(F12_CVT_MSK));
      _r1h = _mm_and_si128(_r1h, _mm_set1_epi32(F12_CVT_MSK));

      _r0l = _mm_add_epi32(_r0l, _mm_set1_epi32(F12_CVT_ADD));
      _r0h = _mm_add_epi32(_r0h, _mm_set1_epi32(F12_CVT_ADD));
      _r1l = _mm_add_epi32(_r1l, _mm_set1_epi32(F12_CVT_ADD));
      _r1h = _mm_add_epi32(_r1h, _mm_set1_epi32(F12_CVT_ADD));
      
      _r0l = _mm_slli_epi32(_r0l, F12_CVT_LSL);
      _r0h = _mm_slli_epi32(_r0h, F12_CVT_LSL);
      _r1l = _mm_slli_epi32(_r1l, F12_CVT_LSL);
      _r1h = _mm_slli_epi32(_r1h, F12_CVT_LSL);

      acc0 = _mm256_fmadd_ps(_mm256_load_ps(v    ),  _mm256_setr_m128(_mm_castsi128_ps(_r0l), _mm_castsi128_ps(_r0h)), acc0);
      acc1 = _mm256_fmadd_ps(_mm256_load_ps(v + 8),  _mm256_setr_m128(_mm_castsi128_ps(_r1l), _mm_castsi128_ps(_r1h)), acc1);
    }
    *res++ = hsum_ps_avx_2x(acc0,acc1);
  }
}

static void matmul_f32_f12_avx2(float *res, const float *vec, const f12_t *mat, int len_vec, int y_mat)
{
  const unsigned char *e = (const unsigned char *)mat;
  int y;
  for (y=0; y<y_mat; y++)
  {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    const float *v, *v_end = vec + len_vec;
    for (v=vec; v!=v_end; v+=16, e+=24)
    {
      __m128i _ld0;
      __m256i _r0h, _r0l, _m;

      // unpack 16 f12 to 16 f32
      _ld0 = _mm_loadu_si128((__m128i *)e);
      _r0l = _mm256_cvtepi8_epi32(_ld0);
      _r0h = _mm256_cvtepi8_epi32(_mm_shuffle_epi32(_ld0, SHI(3,2,3,2)));

      _r0l = _mm256_slli_epi32(_r0l, 4);
      _r0h = _mm256_slli_epi32(_r0h, 4);

      _m = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)(e + 16)));

      _r0l = _mm256_or_si256(_r0l, _mm256_and_si256(_m, _mm256_set1_epi32(0xf)));
      _r0h = _mm256_or_si256(_r0h, _mm256_srli_epi32(_m, 4));

      _r0l = _mm256_and_si256(_r0l, _mm256_set1_epi32(F12_CVT_MSK));
      _r0h = _mm256_and_si256(_r0h, _mm256_set1_epi32(F12_CVT_MSK));

      _r0l = _mm256_add_epi32(_r0l, _mm256_set1_epi32(F12_CVT_ADD));
      _r0h = _mm256_add_epi32(_r0h, _mm256_set1_epi32(F12_CVT_ADD));
      
      _r0l = _mm256_slli_epi32(_r0l, F12_CVT_LSL);
      _r0h = _mm256_slli_epi32(_r0h, F12_CVT_LSL);

      acc0 = _mm256_fmadd_ps(_mm256_load_ps(v    ), _mm256_castsi256_ps(_r0l), acc0);
      acc1 = _mm256_fmadd_ps(_mm256_load_ps(v + 8), _mm256_castsi256_ps(_r0h), acc1);
    }
    *res++ = hsum_ps_avx_2x(acc0,acc1);
  }
}

// init functions list
const matmul_f32_f12_t matmul_f32_f12_procs[simd_n] =
{
  matmul_f32_f12_fpu,
  matmul_f32_f12_sse,
  matmul_f32_f12_avx,
  matmul_f32_f12_avx2,
};

// ------------------------------------------------------------------
// F12 conversions
// ------------------------------------------------------------------

#include "l_util.h"
#include "mem_alloc.h"

// lut to convert model weights
static f12_t lut_f16_to_f12[N_64K] = { 0 };
static f12_t lut_bf16_to_f12[N_64K] = { 0 };

static float _fabs(float x)          // avoid include math.h only for that
{
  return (x >= 0) ? x : -x;
}

// f32 to f12 E4M7 (using e_ofs = 113)
static int f32_to_f12(float f32)
{
  int a = *(int *)&f32;
  int e = (a >> 23) & 0xff;
  int m = (a >> (23 - 7)) & ((1 << 7) - 1);
  int k = m + ((e - 113) << 7);

  // rounding. occur for f16 to f12, exact for bf16 to f12 except for 0 rounded to 6.1035156e-5
  int ki, ks;
  float fc, fi, fs, ec, ei, es;

  if (k < 0)
    k = 0;
  else
  if (k >= (1 << 11))
    k = (1 << 11) - 1;

  ki = k > 0 ? k - 1 : k;
  ks = k < (1 << 11) - 1 ? k + 1 : k;

  fc = cvt_f12_to_f32(k);
  fi = cvt_f12_to_f32(ki);
  fs = cvt_f12_to_f32(ks);

  ec = _fabs(fc - f32);
  ei = _fabs(fi - f32);
  es = _fabs(fs - f32);

  if (ei < ec)
    k = ki;
  else
  if (es < ec)
    k = ks;
  
  return k;
}

// init lookup table.
void init_conv_f12(void)
{
  // alloc temporary AVX aligned arrays
  VAR_ALLOC(f16_list, f16_t, N_64K/2);
  VAR_ALLOC(f16_to_f32, float, N_64K/2);
  int i;

  for (i=0; i<N_64K/2; i++)
    f16_list[i] = (f16_t)i;

  // f16
  matmul_procs.cvt_f16_to_f32(f16_to_f32, f16_list, N_64K/2);
  if (f16_to_f32[F16_TO_F12_MAX] != F12_CVT_MAX)   // error in config constants
    msg_error("init conv f16 to f12 failed");

  for (i=0; i<N_64K/2; i++)
  {
    float f32 = f16_to_f32[i];
    int k = f32_to_f12(f32);
    lut_f16_to_f12[i] = k;
    lut_f16_to_f12[i+(N_64K/2)] = (1 << 11) | k;
  }

  // bf16
  matmul_procs.cvt_bf16_to_f32(f16_to_f32, f16_list, N_64K/2);
  if (f16_to_f32[BF16_TO_F12_MAX] != F12_CVT_MAX)   // error in config constants
    msg_error("init conv bf16 to f12 failed");

  for (i=0; i<N_64K/2; i++)
  {
    float f32 = f16_to_f32[i];
    int k = f32_to_f12(f32);
    lut_bf16_to_f12[i] = k;
    lut_bf16_to_f12[i+(N_64K/2)] = (1 << 11) | k;
  }

  free_check(f16_to_f32);
  free_check(f16_list);
}

// encode 16 x f12 to 24 bytes
static _inline void pack_f12_cpu(unsigned char *e, const f12_t *f12)
{
#if 0
  // reference code
  int i;
  for (i=0; i<8; i++)
  {
    e[i]    = (unsigned char)(f12[i] >> 4);
    e[i+8]  = (unsigned char)(f12[i+8] >> 4);
    e[i+16] = (unsigned char)((f12[i] & 0xf) | ((f12[i+8] & 0xf) << 4));
  }
#else
  // cpu optimized
  e[0]   = (unsigned char)(f12[0] >> 4);
  e[1]   = (unsigned char)(f12[1] >> 4);
  e[2]   = (unsigned char)(f12[2] >> 4);
  e[3]   = (unsigned char)(f12[3] >> 4);
  e[4]   = (unsigned char)(f12[4] >> 4);
  e[5]   = (unsigned char)(f12[5] >> 4);
  e[6]   = (unsigned char)(f12[6] >> 4);
  e[7]   = (unsigned char)(f12[7] >> 4);

  e[8]   = (unsigned char)(f12[8]  >> 4);
  e[9]   = (unsigned char)(f12[9]  >> 4);
  e[10]  = (unsigned char)(f12[10] >> 4);
  e[11]  = (unsigned char)(f12[11] >> 4);
  e[12]  = (unsigned char)(f12[12] >> 4);
  e[13]  = (unsigned char)(f12[13] >> 4);
  e[14]  = (unsigned char)(f12[14] >> 4);
  e[15]  = (unsigned char)(f12[15] >> 4);

  e[16] = (unsigned char)((f12[0] & 0xf) | ((f12[8]  & 0xf) << 4));
  e[17] = (unsigned char)((f12[1] & 0xf) | ((f12[9]  & 0xf) << 4));
  e[18] = (unsigned char)((f12[2] & 0xf) | ((f12[10] & 0xf) << 4));
  e[19] = (unsigned char)((f12[3] & 0xf) | ((f12[11] & 0xf) << 4));
  e[20] = (unsigned char)((f12[4] & 0xf) | ((f12[12] & 0xf) << 4));
  e[21] = (unsigned char)((f12[5] & 0xf) | ((f12[13] & 0xf) << 4));
  e[22] = (unsigned char)((f12[6] & 0xf) | ((f12[14] & 0xf) << 4));
  e[23] = (unsigned char)((f12[7] & 0xf) | ((f12[15] & 0xf) << 4));
#endif
}

// encode 16 x f12 to 24 bytes
static _inline void pack_f12_sse(unsigned char *e, const f12_t *f12)
{
  const __m128i _sh = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
  __m128i _f12l, _f12h, _l, _h, _p;

  _f12l = _mm_load_si128((__m128i *)f12);            // 8 f12
  _f12h = _mm_load_si128((__m128i *)(f12 + 8));      // 8 f12
  _l = _mm_srli_epi16(_f12l, 4);
  _h = _mm_slli_epi16(_f12h, 4);
  _h = _mm_and_si128(_h, _mm_set1_epi16(0xff00));
  _l = _mm_or_si128(_l, _h);
  _p = _mm_shuffle_epi8(_l, _sh);
  _mm_storeu_si128((__m128i *)e, _p);

  _l = _mm_and_si128(_f12l, _mm_set1_epi16(0xf));
  _h = _mm_slli_epi16(_f12h, 4);
  _l = _mm_or_si128(_l, _h);
  _p = _mm_shuffle_epi8(_l, _sh);
  _mm_storel_epi64((__m128i *)(e + 16), _p);
}

// convert buffer f16 to f12
void cvt_f16_to_f12(f12_t *f12, const f16_t *f16, size_t ne)
{
  unsigned char *e = (unsigned char *)f12;
  const f16_t *f16_end = f16 + ne;
  for (;f16 != f16_end; f16+=16, e+=24)
  {
    __m128i _cvt[2];                // need aligned buffer if sse used
    f12_t *cvt = (f12_t *)_cvt;
    int i;
    for (i=0; i<16; i++)
    {
      f16_t _a = f16[i];
      if (ABS_F16(_a) > F16_TO_F12_MAX)    // check _a can be converted with small error
        msg_error("conversion F16 to F12 out of range");
      cvt[i] = lut_f16_to_f12[_a];
    }
    if (matmul_procs.simd_set == simd_fpu)
      pack_f12_cpu(e, cvt);
    else
      pack_f12_sse(e, cvt);
  }
}

// convert buffer f16 to f12
void cvt_bf16_to_f12(f12_t *f12, const bf16_t *bf16, size_t ne)
{
  unsigned char *e = (unsigned char *)f12;
  const bf16_t *bf16_end = bf16 + ne;
  for (;bf16 != bf16_end; bf16+=16, e+=24)
  {
    __m128i _cvt[2];                // need aligned buffer
    f12_t *cvt = (f12_t *)_cvt;
    int i;
    for (i=0; i<16; i++)
    {
      bf16_t _a = bf16[i];
      if (ABS_F16(_a) > BF16_TO_F12_MAX)
        msg_error("conversion BF16 to F12 out of range");
      cvt[i] = lut_bf16_to_f12[_a];
    }
    if (matmul_procs.simd_set == simd_fpu)
      pack_f12_cpu(e, cvt);
    else
      pack_f12_sse(e, cvt);
  }
}