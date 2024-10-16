// generic functions for llama api

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <setjmp.h>

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

#define MBYTE (1024*1024)              // 1 megabyte
#define GBYTE (1024*1024*1024)         // 1 gigabyte

#define __no_return __declspec(noreturn)

// ------------------------------------
// debug

#if defined(_DEBUG) || defined(_CHECK)
void debug_break(void);
#define CHECK(a) ((a) ? (void)0 : debug_break())
#else
#define CHECK(a)
#endif

// ------------------------------------
// floats checks
bool check_no_nan_f32(const float *buff, size_t ne);

// ------------------------------------
// softmax

void softmax(float *x, int size);

// ------------------------------------
// information messages

void msg_info(const char *fmt, ...);

// print some space (align text usage)
void msg_spc(int n_spc);

// ------------------------------------
// errors

extern jmp_buf error_jmp;

#define APP_ERROR() setjmp(error_jmp)

// print error message and jump to error code
void __no_return msg_error(const char *fmt, ...);

// print assert error message and exit
void __no_return assert_exit(const char *fmt, ...);

// assert + position + exit
#define _ASSERT(x) if (!(x)) assert_exit("%s:%s:%d: %s\n", __FILE__, __FUNCTION__, __LINE__, #x)

// check range of an int32 value
void check_range_i(int a, const char *name, int min, int max);

// ------------------------------------
// rng

void rand_seed(int seed);
int rand_n(void);
float rand1(void);
float rand1s(void);

// ------------------------------------
// time

int time_in_ms(void);

// ------------------------------------
// files, manage big files and trap errors

typedef struct
{
  void *handle;
  const char *name;
  int64_t size;
  // user datas
  int64_t seek_ofs;                    // offset to add to f_seek 
} file_t;

#define f_SEEK_CUR 1
#define f_SEEK_END 2
#define f_SEEK_SET 0

void f_seek(file_t *h, int64_t ofs, int origin);
int64_t f_tell(file_t *h);

void f_open(file_t *h, const char *name, const char *mode);
void f_close(file_t *h);
void f_read(void *p, int64_t size, file_t *h);
void f_write(void *p, int64_t size, file_t *h);

// ------------------------------------
// wait return pressed and exit

void wait_return_exit(void);

// ------------------------------------
// display progress bar for long time operations

void progress_bar_init(bool new_line, int64_t max_value);
void progress_bar_update(int64_t value);
void progress_bar_done(void);

// ajust range of float x value in [x_min..x_max]
void adjust_range_f32(float *x, const char *x_name, float x_min, float x_max);

// ajust range of int x value in [x_min..x_max]
void adjust_range_int(int *x, const char *x_name, int x_min, int x_max);
