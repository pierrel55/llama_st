// numa informations.
#define MAX_NUMA_PROCS 64    // max supported procs (need to manage processor group if more needed, or can also disable hyperthreading in BIOS)
#define MAX_NUMA_NODES 8     // max supported nodes (can be increased with current code)

// numa informations
struct numa_inf_t
{
  int mt_node;               // main thread node
  int mt_procs;              // main thread node proc count
  int n_nodes;               // nodes count
  int n_procs;               // physical processors count
  unsigned char proc_list[MAX_NUMA_PROCS];   // procs list batched with same node id
  unsigned char proc_node[MAX_NUMA_PROCS];   // node id for each processor in proc_list
  unsigned char node_nprocs[MAX_NUMA_NODES]; // proc count in each node
};

// global numa informations, use as read only
extern struct numa_inf_t numa;

// init numa struct
void init_numa_info(void);

// display mem available in nodes
void numa_disp_mem(void);

// set proc for current thread
bool numa_set_thread_proc(int proc_id);

// return proc for current thread
int numa_get_thread_proc(void);

// --------------------------
// memory alloc/free

// reserve physical memory in node
void *numa_alloc(size_t sz, int node);

// free memory allocated with numa_alloc
void numa_free(void *p);
