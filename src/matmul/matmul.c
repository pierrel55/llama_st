// matmul inits and data types conversions
#include <math.h>
#include <intrin.h>
#include "l_util.h"
#include "mem_alloc.h"
#include "w_types.h"
#include "matmul.h"
#include "matmul_priv.h"
#ifdef VS_2008
#define __cpuidex(a,b,c) __cpuid(a,b)    // avx2 will not detect
#endif

struct matmul_procs_t matmul_procs = { 0 };

static const char *simd_typ_names[simd_n] = { "FPU", "SSE", "AVX", "AVX2" };

// types names and sizeof
const char *w_type_name[w_type_COUNT]  = { "fp32", "fp16", "bf16", "sf16", "f12", "f8" };
// const int w_type_sizeof[w_type_COUNT] = {   4,      2,      2,       2,   1.5,   1  };

// -------------------------------------------
// types convert

void cvt_w_data(void *d, enum e_w_type d_type, const void *s, enum e_w_type s_type, size_t ne)
{
  // macro for short syntax
  #define CVT_TYP(ta, tb) ((s_type == w_type_##ta) && (d_type == w_type_##tb))

  if      (CVT_TYP( f16,  f32)) matmul_procs.cvt_f16_to_f32(d, s, ne);
  else if (CVT_TYP(bf16,  f32)) matmul_procs.cvt_bf16_to_f32(d, s, ne);
  else if (CVT_TYP(sf16,  f32)) matmul_procs.cvt_sf16_to_f32(d, s, ne);
  else if (CVT_TYP( f16, sf16)) cvt_f16_to_sf16(d, s, ne);
  else if (CVT_TYP( f16,  f12)) cvt_f16_to_f12(d, s, ne);
  else if (CVT_TYP(bf16,  f12)) cvt_bf16_to_f12(d, s, ne);
  else if (CVT_TYP( f16,   f8)) cvt_f16_to_f8(d, s, ne);
  else if (CVT_TYP(bf16,   f8)) cvt_bf16_to_f8(d, s, ne);
  else
    msg_error("unsupported weight type conversion: %s to %s\n", 
      w_type_name[s_type], w_type_name[d_type]);
}

// -------------------------------------------
// check all conversions and matmul results

// check error on result lower than max
static void check_error(const float *a, const float *b, int ne, float err_max, float err_sum_max, const char *func_typ, enum e_simd_typ simd_id)
{
  int i;
  float e_max = 0;    // max error value
  float e_sum = 0;    // error sum
  for (i=0; i<ne; i++)
  {
    float err = fabsf(a[i] - b[i]);
    e_sum += err;
    if (err > e_max)
      e_max = err;
  }

#if 1
  if ((e_max > err_max) || (e_sum > err_sum_max))
    msg_error("conv_check:%s:%d  e_max:%.5f  e_sum:%.5f\n", func_typ, simd_id, e_max, e_sum);
#else
  msg_info("conv_check:%s:%d  e_max:%.5f  e_sum:%.5f\n", func_typ, simd_id, e_max, e_sum);
#endif
}

// clear result buffer
static void zero_mem(void *d, enum e_w_type typ, size_t ne)
{
  memset(d, 0, ne * w_type_sizeof[typ]);
}

// check conversions and matmuls functions for selected simd mode
static void conv_matmul_check(enum e_simd_typ simd)
{
  int i;
  int wx = 4096;
  int wy = 200; 
  int ne = wx * wy;
  VAR_ALLOC(w, float, ne);
  VAR_ALLOC(v, float, wx);
  VAR_ALLOC(res_ref, float, wy);
  VAR_ALLOC(res_f32, float, wy);
  VAR_ALLOC(res_bf16, float, wy);
  VAR_ALLOC(res_f16, float, wy);
  VAR_ALLOC(res_sf16, float, wy);
  VAR_ALLOC(res_f12, float, wy);
  VAR_ALLOC(res_f8, float, wy);
  VAR_ALLOC(w_f32, float, ne);
  VAR_ALLOC(w_f16, f16_t, ne);
  VAR_ALLOC(w_bf16, bf16_t, ne);
  VAR_ALLOC(w_sf16, sf16_t, ne);
  VAR_ALLOC(w_f12, f12_t, ne);
  VAR_ALLOC(w_f8, f8_t, ne);

  // define a reference result using fpu and float32
  rand_seed(543);
  // init vector
  for (i=0; i<wx; i++)
    v[i] = rand1s() * 2.0f;
  // init matrix
  for (i=0; i<ne; i++)
    w[i] = rand1s() * 2.0f;
  // init reference float result
  matmul_procs.matmul_f32_f32(res_ref, v, w, wx, wy);

  // ------------------------
  // f32
  matmul_procs.matmul_f32_f32(res_f32, v, w, wx, wy);
  check_error(res_ref, res_f32, wy, 0.0005f, 0.017f, "f32 mul ", simd);   // cmp f32(simd) with f32 ref(fpu)

  // ------------------------
  // bf16
  for (i=0; i<ne; i++)                                         // init bf16 weights
    w_bf16[i] = *(unsigned *)(w + i) >> 16;

  zero_mem(w_f32, w_type_f32, ne);
  cvt_w_data(w_f32, w_type_f32, w_bf16, w_type_bf16, ne);      // cvt bf16 to f32

  zero_mem(res_bf16, w_type_f32, wy);
  matmul_procs.matmul_f32_f32(res_bf16, v, w_f32, wx, wy);     // mul f32(bf16)
  check_error(res_ref, res_bf16, wy, 0.65f, 41.7f, "bf16 cvt", simd);  // cmp f32(bf16)

  zero_mem(res_bf16, w_type_f32, wy);
  matmul_procs.matmul_f32_bf16(res_bf16, v, w_bf16, wx, wy);   // mul bf16 to f32(bf16)
  check_error(res_ref, res_bf16, wy, 0.65f, 41.7f, "bf16 mul", simd);  // cmp f32(bf16)

  // ------------------------
  // f16
  zero_mem(w_f16, w_type_f16, ne);
  cvt_f32_to_f16(w_f16, w, ne);                                // init f16 weights

  zero_mem(w_f32, w_type_f32, ne);
  cvt_w_data(w_f32, w_type_f32, w_f16, w_type_f16, ne);        // cvt f16 to f32(f16)

  zero_mem(res_f16, w_type_f32, wy);
  matmul_procs.matmul_f32_f32(res_f16, v, w_f32, wx, wy);      // mul f32(f16)
  check_error(res_ref, res_f16, wy, 0.044f, 2.59f, "f16 cvt ", simd);   // cmp f32(f16)
  
  zero_mem(res_f16, w_type_f32, wy);
  matmul_procs.matmul_f32_f16(res_f16, v, w_f16, wx, wy);      // mul f16 to f32(f16)
  check_error(res_ref, res_f16, wy, 0.044f, 2.59f, "f16 mul ", simd);   // cmp f32(f16)

  // ------------------------
  // sf16
  zero_mem(w_sf16, w_type_sf16, ne);
  cvt_w_data(w_sf16, w_type_sf16, w_f16, w_type_f16, ne);      // cvt f16 to sf16

  zero_mem(w_f32, w_type_f32, ne);
  cvt_w_data(w_f32, w_type_f32, w_sf16, w_type_sf16, ne);      // cvt sf16 to f32

  zero_mem(res_sf16, w_type_f32, wy);
  matmul_procs.matmul_f32_f32(res_sf16, v, w_f32, wx, wy);     // mul f32(sf16)
  check_error(res_ref, res_sf16, wy, 0.044f, 2.59f, "sf16 cvt", simd);  // cmp res sf16

  zero_mem(res_sf16, w_type_f32, wy);
  matmul_procs.matmul_f32_sf16(res_sf16, v, w_sf16, wx, wy);   // mul sf16 to f32
  check_error(res_ref, res_sf16, wy, 0.044f, 2.59f, "sf16 mul", simd);  // cmp res f16

  // ------------------------
  // bf16 to f12
  zero_mem(w_f12, w_type_f12, ne);
  cvt_w_data(w_f12, w_type_f12, w_bf16, w_type_bf16, ne);      // cvt bf16 to f12

  zero_mem(res_f12, w_type_f32, wy);
  matmul_procs.matmul_f32_f12(res_f12, v, w_f12, wx, wy);      // mul f12 to f32
  check_error(res_ref, res_f12, wy, 0.645f, 41.63f, "f12 mul bf16", simd); // cmp res sf16

  // ------------------------
  // f16 to f12
  zero_mem(w_f12, w_type_f12, ne);
  cvt_w_data(w_f12, w_type_f12, w_f16, w_type_f16, ne);        // cvt f16 to f12

  zero_mem(res_f12, w_type_f32, wy);
  matmul_procs.matmul_f32_f12(res_f12, v, w_f12, wx, wy);      // mul f12 to f32
  check_error(res_ref, res_f12, wy, 0.45f, 21.623f, "f12 mul f16", simd); // cmp res sf16

  // ------------------------
  // bf16 to f8
  zero_mem(w_f8, w_type_f8, ne);
  cvt_w_data(w_f8, w_type_f8, w_bf16, w_type_bf16, ne);        // cvt bf16 to f8
  
  zero_mem(res_f8, w_type_f32, wy);
  matmul_procs.matmul_f32_f8(res_f8, v, w_f8, wx, wy);         // mul f8 to f32
  check_error(res_ref, res_f8, wy, 6.80f, 367.4f, "f8 bf16 ", simd);   // cmp res f32

  // ------------------------
  // f16 to f8
  zero_mem(w_f8, w_type_f8, ne);
  cvt_w_data(w_f8, w_type_f8, w_f16, w_type_f16, ne);          // cvt f16 to f8

  zero_mem(res_f8, w_type_f32, wy);
  matmul_procs.matmul_f32_f8(res_f8, v, w_f8, wx, wy);         // mul f8 to f32
  check_error(res_ref, res_f8, wy, 6.41f, 364.8f, "f8 f16  ", simd);    // cmp res f8

  free_check(w);
  free_check(v);
  free_check(res_ref);
  free_check(res_f32);
  free_check(res_bf16);
  free_check(res_f16);
  free_check(res_sf16);
  free_check(res_f12);
  free_check(res_f8);
  free_check(w_f32);
  free_check(w_f16);
  free_check(w_bf16);
  free_check(w_sf16);
  free_check(w_f12);
  free_check(w_f8);
  msg_info("conv/matmul %s checks done.\n", simd_typ_names[simd]);
}

// -----------------------------------------
// detect cpu features

struct cpu_info_t
{
  //  simd 128 bits
  int sse;
  int sse2;
  int sse3;
  int ssse3;
  int sse41;
  int sse42;
  int sse4a;

  //  simd 256 bits
  int avx;
  int avx2;

  // other
  int f16c;
  int fma3;
};

// https://learn.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?view=msvc-170
static void get_cpu_info(struct cpu_info_t *inf)
{
  unsigned int info[4];
  unsigned int nIds;

  __cpuid(info, 0);
  nIds = info[0];  // gets the number of the highest valid function ID.

  #define T_BIT(reg, bit) (info[reg] & (1ul << bit)) != 0

  if (nIds >= 0x1)
  {
    __cpuidex(info, 0x1, 0);
    inf->sse    = T_BIT(3, 25);
    inf->sse2   = T_BIT(3, 26);
    inf->sse3   = T_BIT(2,  0);
    inf->ssse3  = T_BIT(2,  9);
    inf->sse41  = T_BIT(2, 19);
    inf->sse42  = T_BIT(2, 20);
    inf->avx    = T_BIT(2, 28);
    inf->fma3   = T_BIT(2, 12);
    inf->f16c   = T_BIT(2, 29);
  }

  if (nIds >= 0x7)
  {
    __cpuidex(info, 0x7, 0);
    inf->avx2   = T_BIT(1, 5);
  }
  msg_info("CPU flags: f16c:%d fma3:%d, sse4.2:%d avx:%d avx2:%d\n", 
           inf->f16c, inf->fma3, inf->sse42, inf->avx, inf->avx2);
}

// select simd proc
static int select_simd(void *proc_list[], int max_simd, const char *func_name)
{
  int i;
  for (i=max_simd; i>=0; i--)
    if (proc_list[i])
      break;

  if (i<0)
  {
    if (!proc_list[1])  // if no fpu version sse version must exist
      msg_error("no code definition for %s", func_name);
    i = 1;
  }

  // msg_info("simd %s: %s\n", func_name, simd_typ_names[i]);
  return i;
}

// set matmul/converts procs for selected simd_typ
static void set_mm_procs(enum e_simd_typ simd_typ)
{
  #define SEL_SIMD(id) id[select_simd((void *)id, simd_typ, #id)]

  // set convert procs
  matmul_procs.cvt_f16_to_f32  = SEL_SIMD(cvt_f16_to_f32_procs);
  matmul_procs.cvt_bf16_to_f32 = SEL_SIMD(cvt_bf16_to_f32_procs);
  matmul_procs.cvt_sf16_to_f32 = SEL_SIMD(cvt_sf16_to_f32_procs);

  // set matmul procs
  matmul_procs.matmul_f32_f32  = SEL_SIMD(matmul_f32_f32_procs);
  matmul_procs.matmul_f32_f16  = SEL_SIMD(matmul_f32_f16_procs);
  matmul_procs.matmul_f32_bf16 = SEL_SIMD(matmul_f32_bf16_procs);
  matmul_procs.matmul_f32_sf16 = SEL_SIMD(matmul_f32_sf16_procs);
  matmul_procs.matmul_f32_f12  = SEL_SIMD(matmul_f32_f12_procs);
  matmul_procs.matmul_f32_f8   = SEL_SIMD(matmul_f32_f8_procs);
}

// selec matmul and convert functions depending of simd mode
void matmul_init(enum e_simd_typ simd_typ)
{
  struct cpu_info_t inf = { 0 };
  get_cpu_info(&inf);

  if (!inf.sse42)              // todo: check if can reduce
    msg_error("program need CPU SSE4.2 support");

#if 0
  inf.f16c = 0;                                  // debug: sim no F16C support
#endif

  matmul_procs.cpu_f16c = inf.f16c;              // set flag for user
  if (!inf.f16c || (simd_typ == simd_fpu))
  {
    if (!inf.f16c)
      msg_info("CPU do not support F16C\n");
    init_sw_f16c();
  }

  // adjust simd_typ
  if (simd_typ < 0) simd_typ = simd_avx2;        // auto to max
  if (simd_typ > simd_avx2) simd_typ = simd_avx2; // truncate to max
  if ((simd_typ == simd_avx2) && !inf.avx2) simd_typ--;
  if ((simd_typ == simd_avx1) && !inf.avx) simd_typ--;

#ifdef VS_2008                                   // dummy avx code used
  if (simd_typ > simd_sse) simd_typ = simd_sse;
#endif
  matmul_procs.simd_set = simd_typ;              // save configured mode

#if 0
  if (inf.avx2)                                  // debug: check all functions for all modes
  {
    enum e_simd_typ t;
    init_sw_f16c();
    for (t=simd_fpu; t<=simd_avx2; t++)
    {
      set_mm_procs(t);
      init_conv_sf16();
      init_conv_f12();
      init_conv_f8();
      conv_matmul_check(t);
    }
    msg_info("debug: all simd functions checked.\n");
  }
#endif

  // set user selected mode
  set_mm_procs(simd_typ);

  // init conversions
  init_conv_sf16();
  init_conv_f12();
  init_conv_f8();

  // check functions
  conv_matmul_check(simd_typ);
}

// free some memory
void matmul_exit(void)
{
  free_sw_f16c();
}

#if 0
int main(void)
{
  matmul_init(-1);
  matmul_exit();
  dbg_print_alloc();
}
#endif
