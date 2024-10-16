// simd optimized head attention for transformer

#ifdef USE_SA_SIMD

#include <math.h>
#include <intrin.h>
#include <stdbool.h>
#include "mm_hsum.h"
#include "transformer.h"
#include "matmul.h"
#include "tr_opt_simd.h"

static void head_att_opt_fpu(float *xb, int n_tok, float *att, const float *q, const float *k, const float *v, const struct transformer_config_t *p)
{
  float att_max = -1e10;                         // softmax max att value
  float att_e_sum = 0;                           // softmax exp diff sum
  int kv_dim = p->kv_dim;
  int head_size = p->head_size;
  float sqrt_head_size = p->sqrt_head_size;

  int t;
  for (t=0; t<n_tok; t++, k += kv_dim)
  {
    // 1 line matrix is used for dot product
    float *a = &att[t];
    matmul_procs.matmul_f32_f32(a, q, k, head_size, 1); 
    if (*a > att_max)
      att_max = *a;                              // softmax max att value
  }

  // softmax the scores to get attention weights, from 0..pos inclusively
  for (t=0; t<n_tok; t++)                        // softmax exp and sum
  {
    float e = expf((att[t] - att_max)/sqrt_head_size);
    att[t] = e;
    att_e_sum += e;
  }

  // weighted sum of the values, accumulate xb for t = 0..pos inclusively
  for (t=0; t<n_tok; t++, v += kv_dim)
  {
    int j;
    float a = att[t] / att_e_sum;
    if (!t)
      for (j=0; j<head_size; j++) xb[j]  = a * v[j];  // t = 0, init xb
    else
      for (j=0; j<head_size; j++) xb[j] += a * v[j];  // t > 0, accumulate xb
  }
}

// sse
static void head_att_opt_sse(float *xb, int n_tok, float *att, const float *q, const float *k, const float *v, const struct transformer_config_t *p)
{
  float att_max = -1e10;                         // softmax max att value
  float att_e_sum = 0;                           // softmax exp diff sum
  int kv_dim = p->kv_dim;
  int head_size = p->head_size;
  int t;
  float sqrt_head_size = p->sqrt_head_size;

  const float *m, *m_end = k + n_tok * kv_dim;
  float *a = att;
  for (m=k; m!=m_end; m+=kv_dim)
  {
    __m128 acc = _mm_setzero_ps();
    float r;
    int i;
    for (i=0; i!=head_size; i+=4)
      acc = _mm_fmadd_ps(_mm_load_ps(q + i), _mm_load_ps(m + i), acc);
    r = hsum_ps_sse(acc);
    *a++ = r;
    if (r > att_max)
      att_max = r;
  }

  // softmax the scores to get attention weights, from 0..pos inclusively
  for (t=0; t<n_tok; t++)                        // softmax exp and sum
  {
    float e = expf((att[t] - att_max)/sqrt_head_size);
    att[t] = e;
    att_e_sum += e;
  }

  // weighted sum of the values, accumulate xb for t = 0..pos inclusively
  for (t=0; t<n_tok; t++, v += kv_dim)
  {
    float a = att[t] / att_e_sum;
    __m128 _a = _mm_set1_ps(a);
    if (!t)
    {
      int j;
      for (j=0; j<head_size; j+=4)
        _mm_store_ps(xb + j, _mm_mul_ps(_a, _mm_load_ps(v + j)));
    }
    else
    {
      int j;
      for (j=0; j<head_size; j+=4)
        _mm_store_ps(xb + j, _mm_add_ps(_mm_load_ps(xb + j), _mm_mul_ps(_a, _mm_load_ps(v + j))));
    }
  }
}

// avx/avx2
static void head_att_opt_avx(float *xb, int n_tok, float *att, const float *q, const float *k, const float *v, const struct transformer_config_t *p)
{
  float att_max = -1e10;                         // softmax max att value
  float att_e_sum = 0;                           // softmax exp diff sum
  int kv_dim = p->kv_dim;
  int head_size = p->head_size;
  int t;
  float sqrt_head_size = p->sqrt_head_size;

  const float *m, *m_end = k + n_tok * kv_dim;
  float *a = att;
  for (m=k; m!=m_end; m+=kv_dim)
  {
    __m256 acc = _mm256_setzero_ps();
    float r;
    int i;
    for (i=0; i!=head_size; i+=8)
      acc = _mm256_fmadd_ps(_mm256_load_ps(q + i), _mm256_load_ps(m + i), acc);
    r = hsum_ps_avx1(acc);
    *a++ = r;
    if (r > att_max)
      att_max = r;
  }

  // softmax the scores to get attention weights, from 0..pos inclusively
  for (t=0; t<n_tok; t++)                        // softmax exp and sum
  {
    float e = expf((att[t] - att_max)/sqrt_head_size);
    att[t] = e;
    att_e_sum += e;
  }

  // weighted sum of the values, accumulate xb for t = 0..pos inclusively
  for (t=0; t<n_tok; t++, v += kv_dim)
  {
    float a = att[t] / att_e_sum;
    __m256 _a = _mm256_set1_ps(a);
    if (!t)
    {
      int j;
      for (j=0; j<head_size; j+=8)
        _mm256_store_ps(xb + j, _mm256_mul_ps(_a, _mm256_load_ps(v + j)));
    }
    else
    {
      int j;
      for (j=0; j<head_size; j+=8)
        _mm256_store_ps(xb + j, _mm256_add_ps(_mm256_load_ps(xb + j), _mm256_mul_ps(_a, _mm256_load_ps(v + j))));
    }
  }
}

head_att_opt_t head_att_opt = NULL;

void init_head_att_opt(enum e_simd_typ simd_typ)
{
  if (simd_typ >= simd_avx1)
    head_att_opt = head_att_opt_avx;
  else
  if (simd_typ == simd_sse)
    head_att_opt = head_att_opt_sse;
  else
    head_att_opt = head_att_opt_fpu;
}

#endif // USE_SA_SIMD
