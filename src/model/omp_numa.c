// functions to support numa.
// alloc/copy memory in numa nodes

#include <omp.h>
#include "l_util.h"
#include "transformer.h"
#include "matmul.h"
#include "omp_numa.h"

// return sizeof wd ne elements in bytes (usage required where f12 can be used)
size_t wd_ne_sizeof(const struct w_dat_t *wd, size_t ne)
{
  if (wd->d_type == w_type_f12)
    return ne + (ne >> 1);                       // special case for f12, 1.5 byte per float
  return ne * w_type_sizeof[wd->d_type];
}

// split and alloc weight datas in different memory nodes for numa configurations.
void numa_alloc_wd(struct w_dat_t *wd, int nz, int wy, int wx, enum e_w_type w_type, bool mm_split)
{
  int dy_l[MAX_NUMA_NODES] = { 0 };              // weight lines stored in each nodes
  char *p_node[MAX_NUMA_NODES] = { 0 };          // mem pointer for each node
  int i, n_thrd = mm_split ? numa_map.n_threads : 1;
  size_t sz_wx;                                  // raw size in bytes
  
  // check wx size for simd matmul can be exactly divided by SIMD_LV
  if (wx & (SIMD_LV-1))
    msg_error("tensor raw size %d modulus %d (SIMD_LV) is not 0", wx, SIMD_LV);

  wd->d_type = w_type;
  wd->wx = wx;
  wd->wy = wy;
  wd->nz = nz;
  wd->dy = (wy + (n_thrd - 1)) / n_thrd;         // matrix split size for each thread
  sz_wx = wd_ne_sizeof(wd, wx);                  // raw size in bytes

  // get dy and requested mem size in each node
  for (i=0; i<n_thrd; i++)
  {
    int y = i * wd->dy;                          // y start in matrix
    int dy = WD_GET_DY(y, wd->dy, wd->wy);       // sptit part size of matrix
    if (dy > 0)                                  // can be <=0 if n_thrd > wy
    {
      int nd = mm_split ? numa_map.tid_to_node_id[i] : numa.mt_node;  // if matmul not used, store all datas in main thread node
#if 0
      // debug: test speed impact of allocation not in thread cpu node
      // nd = numa.mt_node;                         // all in main node
      nd = mm_split ? (numa_map.tid_to_node_id[i] + 1) % numa.n_nodes : numa.mt_node; // bad node for matmul only
      nd = (nd + 1) % numa.n_nodes;              // always bad node
#endif
      dy_l[nd] += dy;                            // lines sum stored in node for one z (layer)
      wd->lp[i].sz_l = 1 + nd;                   // sz_l used temporary to save node id + 1
    }
  }

  // alloc mem in nodes
  for (i=0; i<numa.n_nodes; i++)                 // note: mt_node can be > numa_map.n_nodes
  {
    if (dy_l[i])
    {
      size_t sz_node = (size_t)nz * dy_l[i] * sz_wx;
      p_node[i] = numa_alloc(sz_node, i);
      wd->p_node[i] = p_node[i];                 // save base address (for free)
      wd->nn++;
    }
  }

  // set threads data pointers
  for (i=0; i<n_thrd; i++)
  {
    int nd = (int)wd->lp[i].sz_l - 1;            // retreive used node id saved in sz_l
    if (nd >= 0)
    {
      wd->lp[i].p = p_node[nd];                  // set node pointer
      wd->lp[i].sz_l = (size_t)dy_l[nd] * sz_wx; // layer byte stride for node ptr
      p_node[nd] += (size_t)wd->dy * sz_wx; 
    }
  }
}

// copy or load datas to weights for one z unit in weight node memory.
void numa_cpy_wd_z(struct w_dat_t *wd, int z_id, const void *s, file_t *f)
{
  size_t sz_wx = wd_ne_sizeof(wd, wd->wx);
  size_t sz_data = (size_t)wd->wy * sz_wx;
  CHECK((s && !f) || (!s && f));
  
  if (wd->nn == 1)    // single data pointer (single node or single thread used)
  {
    char *p = (char *)wd->lp[0].p + (size_t)z_id * wd->lp[0].sz_l;
    if (s)
      memcpy(p, s, sz_data);
    else
      f_read(p, sz_data, f);
    wd->ne += (size_t)wd->wy * wd->wx;
  }
  else
  {
    int i;
    const char *_s = s;
    for (i=0; i<numa_map.n_threads; i++)
    {
      char *p = (char *)wd->lp[i].p + (size_t)z_id * wd->lp[i].sz_l;
      int y = i * wd->dy;
      int dy = WD_GET_DY(y, wd->dy, wd->wy);
      if (dy > 0)
      {
        size_t sz_bloc = (size_t)dy * sz_wx;
        if (_s)
        {
          memcpy(p, _s, sz_bloc);
          _s += sz_bloc;
        }
        else
          f_read(p, sz_bloc, f);
        wd->ne += (size_t)dy * wd->wx;
      }
    }
  }
}

// ------------------------------------
// thread list definition

struct numa_thread_map_t numa_map = { 0 };

// def thread list to spread the procs into different nodes
// (OMP_PROC_BIND = spread), result in numa_map
static void numa_def_thread_map(int n_procs, int n_nodes)
{
  int i, j, k, tpn;

  // adjust user config
  if ((n_nodes <= 0) || (n_nodes > numa.n_nodes))
    n_nodes = numa.n_nodes;
  if ((n_procs <= 0) || (n_procs > numa.n_procs))
    n_procs = numa.n_procs;
  if (n_nodes > n_procs)                         // cannot use more nodes than procs
    n_nodes = n_procs;

  tpn = n_procs / n_nodes;                       // used threads (procs) per node 

  for (i=0, j=0, k=0; i<n_nodes; i++)
  {
    int nt = numa.node_nprocs[i] >= tpn ? tpn : numa.node_nprocs[i];
    memcpy(numa_map.tid_to_proc_id + j, numa.proc_list + k, nt);
    memcpy(numa_map.tid_to_node_id + j, numa.proc_node + k, nt);
    j += nt;
    k += numa.node_nprocs[i];
    if (!i)
      numa_map.nt_mp = nt;                       // num threads in main process
  }
  numa_map.n_threads = j;                        // rounded total threads count
  msg_info("processor(s) used: %d in %d node(s).\n", j, n_nodes);
}

// display map nodes/cores
static void numa_disp_thread_map(void)
{
  int n;
  for (n=0; n<numa.n_nodes; n++)
  {
    int i;
    msg_info("node %d procs: ", n);
    for (i=0; i<numa_map.n_threads; i++)
    {
      if (numa_map.tid_to_node_id[i] == n)
        msg_info("%d,", numa_map.tid_to_proc_id[i]);
    }
    msg_info("\n");
  }
}

// ------------------------------------
// OMP inits

// note: OMP_PLACES and OMP_PROC_BIND not supported on OpenMP 2.0 vs compiler.
// need OMP_PROC_BIND = spread defined by numa_map config.
// bind is done here using system calls, this seem to cause no problems.

// bind OMP procs to numa_map config
static void omp_proc_bind_numa(void)
{
  int i, n_thrd = numa_map.n_threads;

  // bind procs
  #pragma omp parallel for
  for (i=0; i<n_thrd; i++)
    numa_set_thread_proc(numa_map.tid_to_proc_id[i]);
}

// check omp thread proc match numa_map configuration
void omp_proc_bind_numa_check(void)
{
  int nt, n_thrd = numa_map.n_threads;
  for (nt=1; nt<=n_thrd; nt++)
  {
    int i;
    bool set_res[MAX_NUMA_PROCS] = { 0 };

    #pragma omp parallel for
    for (i=0; i<nt; i++)
      set_res[i] = (numa_get_thread_proc() == numa_map.tid_to_proc_id[i]);

    // check omp proc result
    for (i=0; i<nt; i++)
      if (!set_res[i])
        msg_error("omp_proc_bind_numa failed.");
  }
}

// init OMP for numa configuration
void numa_init_omp(int cfg_n_procs, int cfg_n_nodes)
{
  init_numa_info();                 // get numa hardware config
  numa_def_thread_map(cfg_n_procs, cfg_n_nodes);  // init numa run configuration
  numa_disp_thread_map();          // display map

  omp_set_dynamic(0);              // want fixed threads count, one per physical core
  omp_set_num_threads(numa_map.n_threads);
  omp_proc_bind_numa();            // bind omp thread id to numa map
  omp_proc_bind_numa_check();      // check
}

#if 0
int main(void)
{
  numa_init_omp(10, 1);
}
#endif