// numa infos for windows

#include <windows.h>
#include "l_util.h"          // msg_error
#include "mem_alloc.h"
#include "numa.h"

// --------------------------------------
// get some processors/numa configuration
// note: processor group not managed, will return only processors in current group (max 64)

struct numa_inf_t numa = { 0 };

void init_numa_info(void)
{
  DWORD sz = 0;
  uint64_t p_msk = 0;              // physical processors mask
  int i, j, lc = 0;                // not physical processors count (ht)
  unsigned char node_plist[MAX_NUMA_NODES][MAX_NUMA_PROCS];

  if (!GetLogicalProcessorInformation(NULL, &sz))
  {
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
    {
      SYSTEM_LOGICAL_PROCESSOR_INFORMATION *pi, *pi_buff;
      pi_buff = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc_check(sz);
      if (GetLogicalProcessorInformation(pi_buff, &sz))
      {
        char *pi_end = (char *)pi_buff + sz;

        // get one physical processor if HT enabled, define p_msk
        for (pi = pi_buff; (char *)pi < pi_end; pi++)
        {
          if (pi->Relationship == RelationProcessorCore)
          {
            // get one processor in mask
            uint64_t m = pi->ProcessorMask;
            uint64_t m_p = 1;                    // physical mask
            if (!m)                              // something is wrong
              break;
            while (!(m & 1))
              { m >>= 1; m_p <<= 1; }

            // count remaining as HT processor
            m >>= 1;                             // pass first found
            while (m)
              { lc += m & 1; m >>= 1; }

            if (p_msk & m_p)                     // already defined ?
              break;
            p_msk |= m_p;                        // update global mask for physicals
            numa.n_procs++;                      // cores count
          }
        }
        if ((!p_msk) || ((char *)pi != pi_end))  // a break occured
          msg_error("init_numa_info failed (1)");

        // get processors nodes
        for (pi = pi_buff; (char *)pi < pi_end; pi++)
        {
          if (pi->Relationship == RelationNumaNode)
          {
            uint64_t m = pi->ProcessorMask;
            uint64_t m_p = 1;
            int p_id = 0;                        // proc id
            int n_id = pi->NumaNode.NodeNumber;  // node id
            if (n_id >= MAX_NUMA_NODES)
              break;
            if (n_id >= numa.n_nodes)
              numa.n_nodes = n_id + 1;
            while (m)
            {
              if (m & 1)
              {
                numa.proc_node[p_id] = n_id;
                if (p_msk & m_p)                 // ignore if HT proc
                  node_plist[n_id][numa.node_nprocs[n_id]++] = p_id;
              }
              m >>= 1;
              m_p <<= 1;
              p_id++;
            }
          }
        }
      }
      free_check(pi_buff);
    }
  }

  // main thread
  numa.mt_node = numa.proc_node[numa_get_thread_proc()];
  numa.mt_procs = numa.node_nprocs[numa.mt_node];

  if (!numa.mt_procs)                            // something is wrong
    msg_error("init_numa_info failed (2)");

  // create sorted procs list for user, with procs for main thread at begin
  memcpy(numa.proc_list, node_plist[numa.mt_node], numa.mt_procs);
  memset(numa.proc_node, numa.mt_node, numa.mt_procs);
  j = numa.mt_procs;
  for (i=0; i<numa.n_nodes; i++)
  {
    if (i != numa.mt_node)
    {
      int np = numa.node_nprocs[i];
      if (!np)                                     // something is wrong
        msg_error("node %d contain 0 processors.", i);
      memcpy(numa.proc_list + j, node_plist[i], np);
      memset(numa.proc_node + j, i, np);
      j += np;
    }
  }
  CHECK(j == numa.n_procs);

  // user infos
  msg_info("numa node(s): %d, mp node: %d, num logical/physical procs.: %d/%d (HT %s)\n", 
     numa.n_nodes, numa.mt_node, numa.n_procs+lc, numa.n_procs, lc ? "on" : "off");
}

// set proc for current thread
bool numa_set_thread_proc(int proc_id)
{
  int proc = (int)SetThreadIdealProcessor(GetCurrentThread(), proc_id);
  if (proc < 0)                                  // return -1 if fail
    msg_error("numa_set_thread_proc failed");
  return proc;                                   // return previous processor id
}

// return proc for current thread
int numa_get_thread_proc(void)
{
  int proc = (int)SetThreadIdealProcessor(GetCurrentThread(), MAXIMUM_PROCESSORS);
  if (proc < 0)                                  // return -1 if fail
    msg_error("numa_get_thread_proc failed");
  return (int)proc;
}

// display mem available in nodes (dev usage)
void numa_disp_mem(void)
{
  int n;
  for (n=0; n<numa.n_nodes; n++)
  {
    ULONGLONG sz;
    if (GetNumaAvailableMemoryNode((UCHAR)n, &sz))
      msg_info(" - memory in node %d: %.2f Gb\n", n, (double)sz/(1024.0*1024*1024));
  }
}

// ------------------------------------
// memory alloctions into nodes

#if 0
// debug usage: temporarily replace virtual alloc to checked malloc to check for memory leaks
#define VirtualAllocExNuma(p, addr, sz, flags, p_flags, node) malloc_check(sz)
#define VirtualFree(p, addr, flags) (free_check(p), 1)
#endif

// alloc memory in node
void *numa_alloc(size_t sz, int node)
{
  void *p = VirtualAllocExNuma(GetCurrentProcess(), NULL, sz, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, node);
  if (!p)
    msg_error("numa_alloc failed");
  return p;
}

// free memory
void numa_free(void *p)
{
  if (p)
  {
    BOOL res = VirtualFree(p, 0, MEM_RELEASE);
    if (!res)
      msg_info("numa_free failed.\n");
  }
}
