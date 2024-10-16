#include <stdlib.h>
#include "l_util.h"
#include "mem_alloc.h"

#define ALLOC_SZ_ALIGN (256/8)   // align for AVX

// ------------------------------------
// memory allocation + check

#if !defined(_DEBUG) && !defined(CHECK_ALLOC)

void *malloc_check(size_t size)
{
  void *p = _aligned_malloc(size, ALLOC_SZ_ALIGN);
  if (!p)
    msg_error("malloc failed to alloc %d bytes\n", size);
  return p;
}

void *calloc_check(size_t size)
{
  void *p = _aligned_malloc(size, ALLOC_SZ_ALIGN);
  if (!p)
    msg_error("calloc failed to alloc %d bytes\n", size);
  memset(p, 0, size);
  return p;
}

void *realloc_check(void *ptr, size_t size)
{
  void *p = _aligned_realloc(ptr, size, ALLOC_SZ_ALIGN);
  if (!p)
    msg_error("realloc failed to alloc %d bytes\n", size);
  return p;
}

void free_check(void *ptr)
{
  if (ptr)
    _aligned_free(ptr);
}

void dbg_print_alloc(void)
{
  // not checked in release
}

#else

// debug malloc + allocated size infos

// stats
struct
{
  size_t size_alloc_sum;              // current allocated size
  size_t size_alloc_sum_max;          // max reached allocated size
  size_t size_block_max;              // max allocated block size
  int n_header;                        // current header in use
  int n_malloc;                        // sum count
  int n_realloc;                       // sum count
} a_inf = { 0 };

// alloc header, use size that keep alignment for sse
typedef union
{
  size_t sz;
  char mem[ALLOC_SZ_ALIGN];
} a_hdr;

// return aligned address
static void *mem_align(a_hdr *h, size_t alloc_sz)
{
  void *p;
  if (alloc_sz > a_inf.size_block_max)
    a_inf.size_block_max = alloc_sz;

  if (a_inf.size_alloc_sum > a_inf.size_alloc_sum_max)
    a_inf.size_alloc_sum_max = a_inf.size_alloc_sum;

  h->sz = alloc_sz;
  p = h + 1;
  CHECK(((size_t)p & (ALLOC_SZ_ALIGN-1)) == 0);
  return p;
}

void *malloc_check(size_t size)
{
  a_hdr *h = (a_hdr *)_aligned_malloc(size + sizeof(a_hdr), ALLOC_SZ_ALIGN);
  if (!h)
    msg_error("malloc failed to alloc %d bytes\n", size);

  memset(h, 0, sizeof(a_hdr));         // 0 unused bytes
  a_inf.size_alloc_sum += size;
  a_inf.n_header++;
  a_inf.n_malloc++;
  return mem_align(h, size);
}

void *calloc_check(size_t size)
{
  void *p = malloc_check(size);
  memset(p, 0, size);
  return p;
}

void *realloc_check(void *ptr, size_t size)
{
  if (ptr)
  {
    a_hdr *h = (a_hdr *)ptr - 1;
    a_inf.size_alloc_sum -= h->sz;
    CHECK(a_inf.size_alloc_sum >= 0);
    h = (a_hdr *)_aligned_realloc(h, size + sizeof(a_hdr), ALLOC_SZ_ALIGN);
    if (!h)
      msg_error("realloc failed to alloc %d bytes\n", size);

    a_inf.size_alloc_sum += size;
    a_inf.n_realloc++;
    return mem_align(h, size);
  }
  return malloc_check(size);
}

void free_check(void *ptr)
{
  if (ptr)
  {
    a_hdr *h = (a_hdr *)ptr - 1;
    a_inf.size_alloc_sum -= h->sz;
    a_inf.n_header--;
    CHECK(a_inf.size_alloc_sum >= 0);
    CHECK(a_inf.n_header >= 0);
    _aligned_free(h);
  }
}

// debug info
void dbg_print_alloc(void)
{
  msg_info("INFO mem alloc:\n");
  msg_info(" size_alloc_sum   %.6f Mb\n", (double)a_inf.size_alloc_sum / (1024*1024));
  msg_info(" size_block_max;  %.6f Mb\n", (double)a_inf.size_block_max / (1024*1024));
  msg_info(" n_header   %d\n", a_inf.n_header); 
  msg_info(" n_malloc   %d\n", a_inf.n_malloc);
  msg_info(" n_realloc  %d\n", a_inf.n_realloc);
  msg_info(" size_alloc_sum_max %.6f Mb\n", (double)a_inf.size_alloc_sum_max / (1024*1024));
  if (a_inf.size_alloc_sum || a_inf.n_header)
    msg_info(" >some memory is still allocated: %u bytes\n", (int)a_inf.size_alloc_sum);
  else
    msg_info(" >all memory has been freed.\n");
}

#endif

// alloc string
char *str_alloc(const char *str, int len)
{
  char *s;
  CHECK(len >= 0);
  s = malloc_check(len+1);
  memcpy(s, str, len);
  s[len] = 0;
  return s;
}
