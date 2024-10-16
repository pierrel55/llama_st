// -------------------------------------------
// time eval

#include <stdint.h>
#include "time_ev.h"

#ifdef OPT_EVAL_TIMES

#include <windows.h>
#include <stdio.h>

struct op_time_t op_time[MAX_TIMES] = { 0 };

uint64_t time_ctr_freq = 0;

static void init_time_ctr_freq(void)
{
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  time_ctr_freq = freq.QuadPart;
}

uint64_t get_time_ctr(void)
{
  LARGE_INTEGER ticks;
  QueryPerformanceCounter(&ticks);
  return ticks.QuadPart;
}

// print all times
void tm_print(void)
{
  int i;
  if (!time_ctr_freq)
    init_time_ctr_freq();

  printf("\n----------\ntime list:\n");
  for (i=0; i<MAX_TIMES; i++)
  {
    struct op_time_t *t = &op_time[i];
    if (t->n_call)
      printf("time[%d]: nc:%d\t  dt:%.4f s\n", i, t->n_call, (double)t->t_sum/time_ctr_freq);
  }
  printf("----------\n");
}

#endif