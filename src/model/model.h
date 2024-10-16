// user application header
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"

// model identifier
enum e_model_id
{
  model_id_tinyllama = 0,    // "tinyllama",
  model_id_llama1,           // "llama1",
  model_id_llama2,           // "llama2",
  model_id_code_llama,       // "codellama",
  model_id_llama3,           // "llama3",
  model_id_llama31,          // "llama31",
  model_id_mistral,          // "mistral",
  model_id_mathstral,        // "mathstral",
  model_id_zephyr,           // "zephyr",
  model_id_mixtral,          // "mixtral",
  model_id_vigogne2,         // "vigogne2",
  model_id_qwen2,            // "qwen2",
  model_id_count,            // models count
};

extern const char *model_id_names[model_id_count];

// application run mode
enum e_run_mode
{
  run_mode_generate = 0,
  run_mode_chat,
};

// chat mode config
struct chat_cfg_t
{
  bool chat_use_colors;             // use colors for user/assistant text

  // forward tokens display options
  int fwd_disp_mode;                // 0: display nothing, 1: tokens list

  // method used to generate the chat prompt format
  int chat_prompt_mode;

  // prompt names displayed for assistant and user
  char *chat_assistant_name;
  char *chat_user_name;

  // mode 0
  char *cm0_sys_prompt;
  char *cm0_user_prompt;

  // mode 1
  char *cm1_sys_template;
  char *cm1_user_first_template;
  char *cm1_user_template;
  char *cm1_end_template;
  char *cm1_sys_prompt;
  char *cm1_user_prompt;

  // mode 2
  char *cm2_sys_template;
  char *cm2_user_template;
  char *cm2_user_name_sw;           // swith user/assistant string in generate mode
  char *cm2_sys_prompt;
  char *cm2_user_prompt;
};

// run configuration defined in json
struct run_conf_t
{
  // model identifier
  char *model_ident;               // define model type for model specificities

  // model load
  struct
  {
    int model_num_safetensors;     // count of .safetensors files in model
    char *model_path;              // path to model, ex "C:/llama2/llama2-7b-chat-hf"
    char *tokenizer_name;          // tokenizer file name (ex tokenizer.json)
  } load;

  // set or override rope freq
  float rope_set;                  // set/change rope inv freq value, ignored if 0

  // sampler config defined in sampler struct

  // load parameters
  bool cvt_sf16;                   // convert model to sfloat16 at load
  bool cvt_f12;                    // convert model to float12 at load
  bool cvt_f8;                     // convert model to float8 at load

  // hardware parameters
  int num_procs;                   // num procs used for threads
  int numa_nodes;                  // num numa nodes to init
  int simd_mode;                   // -1: best auto, 0:off(fpu) 1:sse 2:avx

  // checks
  bool test_nan_logits;            // test for NAN at sampling in all logits results

  // run mode
  enum e_run_mode run_mode;        // 0: generate, 1:chat
  int gen_run_steps;               // number of steps to run. 0 = max (model max_seq_len)
  char *token_eos_str;             // end of string token (assistant reply end)
  char *token_eot_str;             // end of text token (dialog/generate end)

  // token display option
  bool tok_disp_raw;
  bool tok_disp_split;             // separate each token with ','
  bool tok_disp_prob;              // display sampling information
 
  // generate mode config
  char *gen_mode_prompt;           // init prompt for generate run_mode

  // chat mode config
  struct chat_cfg_t chat;

  // defined using strings
  enum e_model_id e_model_id;
  int token_eos;                   // eos token
  int token_eot;                   // eot token
};

struct model_t
{
  struct run_conf_t config;
  struct tokenizer_t tokenizer;
  struct transformer_t transformer;
  struct sampler_t sampler;
};

extern struct model_t model;

void build_model(const char *conf_file_name);

void free_model(void);

// chat loop
void chat(void);

// generation loop
void generate(void);
