// omp for numa support

// thread list numa map
struct numa_thread_map_t
{
  int nt_mp;                   // num threads in main process
  int n_threads;
  unsigned char tid_to_proc_id[MAX_NUMA_PROCS];
  unsigned char tid_to_node_id[MAX_NUMA_PROCS];
};

extern struct numa_thread_map_t numa_map;

// get weight data dy split size for y
#define WD_GET_DY(y, dy, wy) ((y + dy) <= wy) ? dy : wy - y

// return sizeof wd ne elements in bytes (usage required where f12 can be used)
size_t wd_ne_sizeof(const struct w_dat_t *wd, size_t ne);

// split and alloc weight datas in different memory nodes for numa configurations.
void numa_alloc_wd(struct w_dat_t *wd, int nz, int wy, int wx, enum e_w_type w_type, bool mm_split);

// copy or load datas to weights for one z unit (layer).
void numa_cpy_wd_z(struct w_dat_t *wd, int z_id, const void *s, file_t *f);

// init OMP for numa configuration
void numa_init_omp(int cfg_n_procs, int cfg_n_nodes);

// check omp thread proc match numa_map configuration
void omp_proc_bind_numa_check(void);
