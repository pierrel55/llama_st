#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <intrin.h>           // for __debugbreak()
#include "l_util.h"

// ----------------------------------------------
// debug

void debug_break(void)
{
#ifdef _DEBUG
  __debugbreak();
#else
  printf("\ndebug break..\n");
  exit(-1);
#endif  
}

// -------------------------------------------
// floats checks

// return true if no NAN found in float buffer
bool check_no_nan_f32(const float *buff, size_t ne)
{
  size_t i;
  for (i=0; i<ne; i++)
    if (_isnan(buff[i]))
      return false;
  return true;
}

// ----------------------------------------------
// softmax

// convert x to probabilities 0..1 with x elements sum = 1
void softmax(float *x, int size)
{
  int i;
  float sum = 0.0f;
  float max_val = x[0];
  
  // find max value
  for (i=1; i<size; i++)
    if (x[i] > max_val)
      max_val = x[i];

  // exp and sum
  for (i=0; i<size; i++)
  {
    float e = expf(x[i] - max_val);
    x[i] = e;
    sum += e;
  }

  // normalize
  for (i=0; i<size; i++)
    x[i] /= sum;
}

// ----------------------------------------------
// information messages

void msg_info(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
}

// print some space (align text usage)
void msg_spc(int n_spc)
{
  while (n_spc-- > 0) 
    putchar(' ');
}

// ----------------------------------------------
// errors

jmp_buf error_jmp;

void __no_return msg_error(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  printf("\nERROR: ");
  vprintf(fmt, ap);
  va_end(ap);
  printf("\n");
#ifdef _DEBUG
  wait_return_exit();
#endif
  longjmp(error_jmp, -1);
}

void __no_return assert_exit(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  printf("ASSERT:");
  vprintf(fmt, ap);
  va_end(ap);
#ifdef _DEBUG
  wait_return_exit();
#endif
  exit(-1);
}

// check range of a value
void check_range_i(int a, const char *name, int min, int max)
{
  if ((a < min) || (a > max))
    msg_error("invalid value for '%s' = %d expected in [%d..%d] range", name, a, min, max);
}

// ----------------------------------------------
// rng

#if 1
// ms version
static int r_seed = -123;
#define RD_SEED (r_seed = (r_seed * 214013L + 2531011L))
#else
// glibc version
static int32_t r_seed = 123459876;
#define RD_SEED (r_seed = (r_seed * 1103515245 + 12345))
#endif

void rand_seed(int seed)
{
  r_seed = seed;
}

// [0..32767]
int rand_n(void)
{
  return (RD_SEED >> 16) & 0x7fff;
}

// [0..1[
float rand1(void)
{
  return ((uint32_t)RD_SEED >> 16)*(1.0f/0x10000);
}

// [-1..1[
float rand1s(void)
{
  return ((int)RD_SEED >> 15)*(1.0f/0x10000);
}

// ----------------------------------------------
// coarse time

int time_in_ms(void)
{
  clock_t t = clock();
#if CLOCKS_PER_SEC != 1000
  t = (t*1000)/CLOCKS_PER_SEC;
#endif
  return (unsigned int)t;
}

// ----------------------------------------------
// files, manage big files and trap errors

// ensure seek constants same as compiler
#if (f_SEEK_CUR != SEEK_CUR) || (f_SEEK_END != SEEK_END) || (f_SEEK_SET != SEEK_SET)
  #error error in SEEK constants (adjust l_util.h)
#endif

static void f_check_handle(file_t *h)
{
  if (!h || !h->handle)
    msg_error("file NULL handle");
}

void f_seek(file_t *h, int64_t ofs, int origin)
{
  f_check_handle(h);
  if (_fseeki64((FILE *)h->handle, ofs, origin))  // return 0 if success
    msg_error("seek error in file '%s'", h->name);
}

int64_t f_tell(file_t *h)
{
  f_check_handle(h);
  return _ftelli64((FILE *)h->handle);
}

void f_open(file_t *h, const char *name, const char *mode)
{
  FILE *f = fopen(name, mode);
  if (!f)
    msg_error("failed to open file '%s'", name);

  h->name = name;
  h->handle = f;
  f_seek(h, 0, f_SEEK_END);
  h->size = f_tell(h);
  f_seek(h, 0, f_SEEK_SET);
}

void f_close(file_t *h)
{
  f_check_handle(h);
  fclose((FILE *)h->handle);
  h->handle = NULL;
}

void f_read(void *p, int64_t size, file_t *h)
{
  f_check_handle(h);
  if (!fread(p, size, 1, (FILE *)h->handle))
    msg_error("read error in file '%s'", h->name);
}

void f_write(void *p, int64_t size, file_t *h)
{
  f_check_handle(h);
  if (!fwrite(p, size, 1, (FILE *)h->handle))
    msg_error("write error in file '%s'", h->name);
}

// --------------------------------------------------------
// wait return pressed and exit (debug usage)

void wait_return_exit(void)
{
  printf("press return to exit.");
  getchar();
  exit(-1);
}

// --------------------------------------------------------
// basic progress bar for long time operations

#define PB_LEN 32     // len

static struct
{
  int curr;
  int64_t max;
} pbar = { 0 };

void progress_bar_init(bool new_line, int64_t max_value)
{
  pbar.max = 0;
  pbar.curr = 0;
  if (max_value > 0)
  {
    int i;
    if (new_line)
      putchar('\n');
    pbar.max = max_value;
    putchar('[');
    for (i=0; i<PB_LEN; i++)
      putchar('.');
    putchar(']');
    // move cursor back
    for (i=0; i<PB_LEN+1; i++)
      putchar(8);    // back space
  }
}

void progress_bar_update(int64_t value)
{
  if (pbar.max)
  {
    int x = (int)((value*PB_LEN)/pbar.max);
    if (x > PB_LEN)
      x = PB_LEN;
    for (;pbar.curr < x; pbar.curr++)
      putchar('*');
  }
}

// force end
void progress_bar_done(void)
{
  if (pbar.max)
  {
    progress_bar_update(pbar.max);
    pbar.max = 0;
    msg_info("\n");
  }
}

#if 0
// test progress bar
extern void Sleep(unsigned int ms);
void main(void)
{
  int i = 0;

  msg_info("start.\n");
  progress_bar_init(1000);
  while (i < 500)
  {
    i += 1000/20;
    progress_bar_update(i);
    Sleep(200);
  }
  progress_bar_done();
  msg_info("done.\n");
}
#endif

// ajust range of float value in [x_min..x_max]
void adjust_range_f32(float *x, const char *x_name, float x_min, float x_max)
{
  float x0 = *x;
  if      (*x < x_min) *x = x_min;
  else if (*x > x_max) *x = x_max;
  else
    return;
  msg_info("parameter %s: %.4f init value adjusted to %.4f\n", x_name, x0, *x);
}

// ajust range of int x value in [x_min..x_max]
void adjust_range_int(int *x, const char *x_name, int x_min, int x_max)
{
  int x0 = *x;
  if      (*x < x_min) *x = x_min;
  else if (*x > x_max) *x = x_max;
  else
    return;
  msg_info("parameter %s: %d init value adjusted to %d\n", x_name, x0, *x);
}
