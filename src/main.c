#include "l_util.h"
#include "model.h"

#ifdef CHECK_EXIT
#include "mem_alloc.h"
#include "omp_numa.h"
#endif

int main(int argc, char *argv[])
{
  if (APP_ERROR())                 // catch error return point
    return -1;

#if 1
  if (argc != 2)
  {
    msg_info("Usage: llama_st <run_config.json>\n");
    msg_info("Example: llama_st run_json/run_llama2.json\n");
    return -1;
  }
  build_model(argv[1]);
#else
  // dev mode
  build_model("run_json/run_tinyllama.json");
  //build_model("run_json/run_llama1.json");
  //build_model("run_json/run_llama2.json");
  //build_model("run_json/run_codellama.json");
  //build_model("run_json/run_mistral.json");
  //build_model("run_json/run_mathstral.json");
  //build_model("run_json/run_zephyr.json");
  //build_model("run_json/run_mixtral.json");
  //build_model("run_json/run_vigogne2.json");
  //build_model("run_json/run_llama3.json");
  //build_model("run_json/run_llama3.1.json");
  //build_model("run_json/run_qwen2.5.json");
#endif

  // run generate or chat
  if (model.config.run_mode == run_mode_generate)
    generate();
  else 
  if (model.config.run_mode == run_mode_chat)
    chat();
  else
    msg_info("undefined run mode: %d\n", model.config.run_mode);

  // free memory
  free_model();

#ifdef CHECK_EXIT
  // some exit checks
  omp_proc_bind_numa_check();   // check no change occured
  dbg_print_alloc();            // check free
  wait_return_exit();           // press return to exit
#endif

  return 0;
}
