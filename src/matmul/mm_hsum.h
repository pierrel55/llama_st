#ifdef VS_2008
// old compiler. some SSE not defined in intrin. (_mm_cvtph_ps/_mm_cvtps_ph/_mm_fmadd_ps etc..)
// this use external linking. (slow but work)
#include "conv_ph_ps.h"
#endif

#ifndef MM_USE_FMA
#define _mm_fmadd_ps(a,b,c) _mm_add_ps(c,_mm_mul_ps(a,b))
#define _mm256_fmadd_ps(a,b,c) _mm256_add_ps(c,_mm256_mul_ps(a,b))
#endif

// ------------------------------------------------------------------
// SSE3/AVX horizontal sum
// https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
// ------------------------------------------------------------------

#if 1

static __inline float hsum_ps_sse(__m128 v)
{
  __m128 shuf = _mm_movehdup_ps(v);            // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(v, shuf);
  shuf        = _mm_movehl_ps(shuf, sums);     // high half -> low half
  sums        = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

static __inline float hsum_ps_avx1(__m256 v) 
{
  __m128 vlow  = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1);  // high 128
         vlow  = _mm_add_ps(vlow, vhigh);      // add the low 128
  return hsum_ps_sse(vlow);                    // and inline the sse3 version, which is optimal for AVX
}

#else

// FPU.
// note: can be faster than AVX/SSE versions (compiler optimized ?).

static __inline float hsum_ps_sse(__m128 v)
{
  float *sum_4 = (float *)&v;
  return sum_4[0] + sum_4[1] + sum_4[2] + sum_4[3];
}

static __inline float hsum_ps_avx1(__m256 v)
{
  float *sum_8 = (float *)&v;
  return sum_8[0] + sum_8[1] + sum_8[2] + sum_8[3] + sum_8[4] + sum_8[5] + sum_8[6] + sum_8[7];
}

#endif

#define hsum_ps_sse_2x(a,b) hsum_ps_sse(_mm_add_ps(a,b))
#define hsum_ps_sse_4x(a,b,c,d) hsum_ps_sse(_mm_add_ps(_mm_add_ps(a,b),_mm_add_ps(c,d)))

#define hsum_ps_avx_2x(a,b) hsum_ps_avx1(_mm256_add_ps(a,b))
#define hsum_ps_avx_4x(a,b,c,d) hsum_ps_avx1(_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d)))

