// -----------------------------------------
// debug/optimization: eval operations times

// enable to eval times with debugger
// #define OPT_EVAL_TIMES

#ifdef OPT_EVAL_TIMES

#define MAX_TIMES 16

struct op_time_t
{
  uint64_t t0;
  uint64_t t_sum;
  int n_call;
};

extern struct op_time_t op_time[MAX_TIMES];

uint64_t get_time_ctr(void);

static __inline void tm_stop(int id)
{
  struct op_time_t *t = &op_time[id];
  t->t_sum += (get_time_ctr() - t->t0);
  t->n_call++;
}

void tm_print(void);

#define T_START(id) op_time[id].t0 = get_time_ctr()
#define T_STOP(id) tm_stop(id)
#define T_RESET() memset(op_time, 0, sizeof(op_time))
#define T_CLR(id) op_time[id].t_sum = 0; op_time[id].n_call = 0
#define T_PRINT() tm_print()

#else

#define T_RESET()
#define T_START(id)
#define T_STOP(id)
#define T_CLR(id)
#define T_PRINT()

#endif
