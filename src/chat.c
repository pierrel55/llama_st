// simple console chat used to test models.
#include <stdio.h>        // _snprintf
#include "l_util.h"
#include "term_utf8.h"
#include "model.h"

// colors id
enum e_col_id
{
  col_sys = 0,           // system messages
  col_usr,               // user input (keyboard)
  col_llm,               // llm generation
};

// print in sys color
static void print_sys(const char *str)
{
  text_color(col_sys);
  msg_info(str);
}

// menu command
enum e_cmd
{
  cmd_none = 0,          // no menu selected
  cmd_resume,            // resume after menu exit
  cmd_regen,             // forget + regen last llm reply
  cmd_forget,            // forget last user input + llm reply
  cmd_rst_kp,            // reset chat, keep current sys prompt
  cmd_rst_np,            // reset chat, enter new sys prompt
  cmd_exit               // exit program
};

static enum e_cmd user_menu(bool usr_inp, bool en_regen, bool en_forget);

// keybord input some text, manage commands
static enum e_cmd kbd_input_prompt(char *buff, int buff_sizeof, bool usr_inp, bool en_regen, bool en_forget)
{
  while (1)              // repeat while text empty or exit command entered
  {
    int l_input;
    text_color(col_usr);
    l_input = kbd_input_utf8(buff, buff_sizeof);
    if (l_input < 0)             // string too long or cannot utf8 encode input
      print_sys("\ninput error. please retry.\n");
    else
    if (l_input > 0)             // not empty
    {
      // test for command
      if ((l_input == 2) && (buff[0] == '#') && (buff[1] == 'm'))  // #m => enter menu
        return user_menu(usr_inp, en_regen, en_forget);
      return cmd_none;
    }
  }
}

// extended token print depend of conf options, common for chat and generate mode
void tokenizer_decode_print_ex(int token_id, float prob)
{
  struct run_conf_t *conf = &model.config;
  tokenizer_decode_print(token_id, conf->tok_disp_raw);
  if (conf->tok_disp_prob && (prob >= 0.0f))
    msg_info("[%.2f],", prob);
  else
  if (conf->tok_disp_prob || conf->tok_disp_split)
    msg_info(",");
}

// forward tokenizer token list to transformer, def logits on last token if def_logits true
static void forward_user_tokens(bool display, bool def_logits)
{
  struct tokenizer_t *tokenizer = &model.tokenizer; 
  int i, n_token = tokenizer->mt_list.n_list;
  int def_logits_i = def_logits ? n_token - 1 : -1;

  if (display)
    text_color(col_usr);
  for (i=0; i<n_token; i++)
  {
    int tok_id = tokenizer->mt_list.mt[i].tok_id;
    if (display)
      tokenizer_decode_print_ex(tok_id, -1.0f);
    forward(tok_id, false, i == def_logits_i);
  }
  if (display)
    print_sys("\n");
}

// chat mode 2: test if switch name token list match last tokens list + new sampled token
#define MAX_SW_NAME 64            // max switch name len, must be power of 2 value

// define end string using new token, return true if match sw_name
static bool cmp_sw_name(char *e_str, int *e_wr, int new_token, const char *sw_name, int sw_name_len)
{
  // update end string using ring buffer
  const char *s = tokenizer_decode(new_token);
  int i = *e_wr;
  while (*s)
  {
    e_str[i] = *s++;
    i = (i + 1) & (MAX_SW_NAME - 1);
  }
  *e_wr = i;

  // compare if result match sw_name
  i = (i - sw_name_len) & (MAX_SW_NAME - 1);
  while (*sw_name)
  {
    if (*sw_name++ != e_str[i])
      return false;
    i = (i + 1) & (MAX_SW_NAME - 1);
  }
  return true;
}

// ------------------------------------------------------------
// chat

static char str_kbd[2048];        // keyboard input
static char str_fwd[4096];        // forward string

// chat loop
void chat(void)
{
  struct chat_cfg_t *chat = &model.config.chat;
  const struct run_conf_t *conf = &model.config;
  struct tokenizer_t *tokenizer = &model.tokenizer;
  struct transformer_t *transformer = &model.transformer;
  struct sampler_t *sampler = &model.sampler;

  int pr_mode = chat->chat_prompt_mode;
  const char *sys_prompt, *user_prompt;
  const char *sys_template, *user_template0, *user_template, *end_template;
  bool def_sys_prompt, use_user_template0;
  enum e_cmd cmd;
  int pos_input, pos_reply;

  // mode 2 specific, contain end of generated text
  char m2_end_str[MAX_SW_NAME] = { 0 };
  int m2_end_wr = 0;
  int end_template_len = 0;
 
  // init console to enable colors
  if (chat->chat_use_colors)
    term_init();

  text_color(col_sys);
  msg_info("------------------------------------\n"
           "Start chat mode %d\n", pr_mode);
  msg_info("- Enter '#m' on a new line to enter menu.\n");
  msg_info("- Press 'esc' key to break generation.\n");

  // -----------------------------------
  // init templates and prompts for mode
  if (pr_mode == 0)
  {
    // use model ident to define templates
    switch (conf->e_model_id)
    {
      case model_id_tinyllama:
        // https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
        sys_template   = "<|system|>\n%s</s>\n";
        user_template0 = NULL;
        user_template  = "<|user|>\n%s</s>\n<|assistant|>\n";
        end_template   = "</s>\n";        // end of assistant reply
      break;
      case model_id_llama2:
      case model_id_code_llama:           // same format as llama2 chat
        // https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        sys_template   = "<s>[INST] <<SYS>>\n%s\n<</SYS>>\n\n";
        user_template0 = "%s [/INST] ";   // llama2 use a first user template
        user_template  = "<s>[INST] %s [/INST] ";
        end_template   = "</s>";
      break;
      case model_id_llama3:
      case model_id_llama31:              // same format as llama3
        // https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        // https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1
        sys_template   = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>";
        user_template0 = NULL;
        user_template  = "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        end_template   = "<|eot_id|>\n";
      break;
      case model_id_mistral:
      case model_id_mathstral:
        // todo: find clear documentation.
        // https://docs.mistral.ai/guides/prompting_capabilities/
        // https://community.aws/content/2dFNOnLVQRhyrOrMsloofnW0ckZ/how-to-prompt-mistral-ai-models-and-why
        // https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/discussions/49
        sys_template   = NULL;             // no sys template ?
        user_template0 = NULL;
        user_template  = "<s>[INST] %s [/INST]";
        end_template   = "</s>\n";
      break;
      case model_id_zephyr:
        // https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
        sys_template   = "<|system|>\n%s</s>\n";
        user_template0 = NULL;
        user_template  = "<|user|>\n%s</s>\n<|assistant|>\n";
        end_template   = "</s>\n";        // end of assistant reply
      break;
      case model_id_mixtral:
        // https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1#instruction-format
        // <s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
        if (chat->cm0_sys_prompt[0])
          msg_info("sys prompt ignored for instruct mode\n");
        sys_template   = NULL;                     // no sys template
        user_template0 = "<s> [INST] %s [/INST]";  // 1st instruction
        user_template  = "[INST] %s [/INST] ";     // Follow-up instruction
        end_template   = "</s>";
      break;
      case model_id_vigogne2:
        // https://huggingface.co/bofenghuang/vigogne-2-7b-chat
        // https://huggingface.co/bofenghuang/vigogne-2-70b-chat
#if 0
        // 7B, see https://huggingface.co/bofenghuang/vigogne-2-7b-chat
        sys_template   = "<s><|system|>: %s\n";
        user_template0 = NULL;
        user_template  = "<|user|>: %s\n<|assistant|>:";
        end_template   = "</s>\n";        // end of assistant reply
#else
        // format for 70B, see https://huggingface.co/bofenghuang/vigogne-2-70b-chat
        sys_template   = "<s>[INST] <<SYS>>\n%s\n<</SYS>>\n\n";
        user_template0 = "%s [/INST] ";    // first user template
        user_template  = "[INST] %s [/INST]";
        end_template   = "</s>\n";
#endif
      break;
      case model_id_qwen2:                 // qwen2.5
        sys_template   = "<|im_start|>system\n%s<|im_end|>\n";
        user_template0 = NULL;
        user_template  = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
        end_template   = "<|im_end|>\n";
      break;
      default:
        msg_error("chat_prompt_mode = 0 not supported for model %s.", model_id_names[conf->e_model_id]);
    }
    sys_prompt     = chat->cm0_sys_prompt;
    user_prompt    = chat->cm0_user_prompt;
  }
  else
  if (pr_mode == 1)
  {
    sys_template   = chat->cm1_sys_template;
    user_template0 = chat->cm1_user_first_template;
    user_template  = chat->cm1_user_template;
    sys_prompt     = chat->cm1_sys_prompt;
    user_prompt    = chat->cm1_user_prompt;
    end_template   = chat->cm1_end_template;
  }
  else
  if (pr_mode == 2)
  {
    sys_template   = chat->cm2_sys_template;
    user_template0 = NULL;
    user_template  = chat->cm2_user_template;
    sys_prompt     = chat->cm2_sys_prompt;
    user_prompt    = chat->cm2_user_prompt;
    end_template   = chat->cm2_user_name_sw;

    end_template_len = (int)strlen(end_template);
    if ((end_template_len < 5) || (end_template_len > MAX_SW_NAME))
      msg_error("cm2_user_name_sw must contain %d to %d characters", 5, MAX_SW_NAME);
  }
  else
    msg_error("invalid chat prompt mode configuration (%d)", pr_mode);

#if 0
  // display initial prompt
  if (!chat->fwd_disp_mode)
  {
    if (sys_prompt && sys_prompt[0])
      msg_info("System prompt: %s\n", sys_prompt);
    if (user_prompt && user_prompt[0])
      msg_info("User prompt  : %s\n", user_prompt);
  }
#endif

dialog_rst_np:                             // reset + def new prompt if sys_prompt = NULL
  transformer->state.cache.n_tokens = 0;
  def_sys_prompt = sys_template && sys_template[0];

dialog_rst_kp:                             // reset + keep curent prompt
  use_user_template0 = user_template0 && user_template0[0];
  pos_input = 0;                           // reset user input position
  pos_reply = 0;                           // reset llm reply position

  // dialog loop
  while (1)
  {
    // ------------------------------
    // system prompt

    while (def_sys_prompt)
    {
      int str_fwd_len = 0;                 // len of string to tokenize/forward
      bool kbd_in = false;
      if (!sys_prompt || !sys_prompt[0])   // input system prompt if not defined or empty
      {
        kbd_in = true;                     // flag keyboard input (no echo required)
        print_sys("Enter system prompt:");
        cmd = kbd_input_prompt(str_kbd, sizeof(str_kbd), false, false, false);
        // interpret menu command (only cmd_exit allowed)
        if (cmd == cmd_none)
          sys_prompt = str_kbd;
        else
        if (cmd == cmd_exit)
          goto exit_chat;
        else
          continue;
      }
      // format using template
      str_fwd_len = _snprintf(str_fwd, sizeof(str_fwd), sys_template, sys_prompt);
      sys_prompt = NULL;
      // check length
      if ((str_fwd_len < 0) || (str_fwd_len == sizeof(str_fwd)))
      {
        print_sys("\nprompt is too long. please re-enter.\n");
        continue;
      }
      def_sys_prompt = false;
      // encode and forward
      print_sys("forward system prompt..\n");
      tokenizer_encode(str_fwd);
      forward_user_tokens(!kbd_in || chat->fwd_disp_mode, false);
      // define num tokens to keep in sys prompt if context cache compacted
      transformer->state.cache.n_tokens_sys = transformer->state.cache.n_tokens;
      break;
    }

    // ------------------------------
    // user prompt

    while (1)
    {
      int str_fwd_len = 0;                       // len of string to tokenize/forward
      bool kbd_in = false;
      // user template
      const char *u_tpl = use_user_template0 ? user_template0 : user_template;

      print_sys(chat->chat_user_name);
      if (!user_prompt || !user_prompt[0])       // input user prompt if not defined or empty
      {
        kbd_in = true;
        cmd = kbd_input_prompt(str_kbd, sizeof(str_kbd), true, pos_reply != 0, pos_input != 0);
        if (cmd == cmd_none)
          user_prompt = str_kbd;
        else
        {
          // -----------------------
          // interpret menu commands
          if (cmd == cmd_exit)
            goto exit_chat;
          if (cmd == cmd_rst_np)                 // reset + enter new sys prompt
            goto dialog_rst_np;
          if (cmd == cmd_rst_kp)                 // reset + keep current sys prompt
          {
            transformer->state.cache.n_tokens = transformer->state.cache.n_tokens_sys;
            goto dialog_rst_kp;
          }
          if ((cmd == cmd_regen) && pos_reply)   // forget and regen last reply
          {
            // regen logits of last injected token
            pos_reply--;
            transformer->state.cache.n_tokens = pos_reply;
            forward(transformer->state.cache.tokens[pos_reply].token_id, false, true);
            break;                               // generate
          }
          if ((cmd == cmd_forget) && pos_input)  // forget last user input and llm reply
          {
            pos_reply = 0;
            transformer->state.cache.n_tokens = pos_input;
          }
          continue;                              // re-enter user prompt
        }
      }
      str_fwd_len = _snprintf(str_fwd, sizeof(str_fwd), u_tpl, user_prompt);
      user_prompt = NULL;
      if ((str_fwd_len < 0) || (str_fwd_len == sizeof(str_fwd)))
      {
        print_sys("prompt is too long. please re-enter.\n");
        continue;
      }
      use_user_template0 = false;
      pos_input = transformer->state.cache.n_tokens;
      // encode and forward
      tokenizer_encode(str_fwd);
      forward_user_tokens(!kbd_in || chat->fwd_disp_mode, true);
      break;
    }
   
    // ------------------------------------------
    // get LLM reply

#ifdef PACK_KV_CACHE
    {
      // reserve size in kv cache for reply. (to be tested)
      int nd = reserve_kv_cache(500);
      if (nd)                                    // num of deleted tokens
      {
#if 1
        // inform user
        text_color(col_sys);
        msg_info(">info: cache compacted, %d forgotten tokens.\n", nd);
#endif
        // adjust local pos
        if (pos_input) pos_input -= nd;
        if (pos_reply) pos_reply -= nd;
        CHECK((pos_input >= 0) && (pos_reply >= 0));
      }
    }
#else
    // if context full, give option to save dialog text and init new dialog
    if (transformer->state.cache.n_tokens == transformer->config.seq_len)
    {
      while (1)
      {
        enum e_cmd cmd;
        print_sys("Context cache full.., select menu option.\n");
        cmd = user_menu(true, false, false);
        if (cmd == cmd_exit)
          goto exit_chat;
        if (cmd == cmd_rst_np)                   // reset + enter new sys prompt
          goto dialog_rst_np;
        if (cmd == cmd_rst_kp)                   // reset + keep current sys prompt
        {
          transformer->state.cache.n_tokens = transformer->state.cache.n_tokens_sys;
          goto dialog_rst_kp;
        }
      }
    }
#endif

    pos_reply = transformer->state.cache.n_tokens;
    // msg_info("{pos_reply:%d}", pos_reply);    // debug usage
    print_sys(chat->chat_assistant_name);
    while (1)
    {
      // get token from logits
      struct prob_index_t *pi = sampler_sample();

      // user can press escape key to shorten answer if too long (ex: LLM generate 1000 pi digits..)
      if (read_key() == 27)
      {
        print_sys("{esc stop}");
        pi->index = model.config.token_eos;      // force eos
      }

      // stop if eos/eot generated
      if (   (pi->index == model.config.token_eot)
          || (pi->index == model.config.token_eos))
      {
        // forward end template
        tokenizer_encode(end_template);
        forward_user_tokens(chat->fwd_disp_mode, false);
        break;
      }

      // print generated token
      text_color(col_llm);
      tokenizer_decode_print_ex(pi->index, pi->prob);

      // specific mode 2, detect end template generated by LLM (cm2_user_name_sw)
      if ((pr_mode == 2) && cmp_sw_name(m2_end_str, &m2_end_wr, pi->index, end_template, end_template_len))
      {
        forward(pi->index, true, false);   // forward end template last token
        break;
      }
 
      // get next token logits
      forward(pi->index, true, true);
    }
    print_sys("\n");
  }

exit_chat:
  text_color(-1);   // restore default console color
  term_cb_clear();  // free if allocated datas
}

// --------------------------
// menu

// copy dialog text to clipboard
static void cb_copy_dialog_text(bool raw_mode)
{
  struct transformer_t *t = &model.transformer;
  int i, rp = -1; 

  term_cb_clear();
  for (i=0; i<t->state.cache.n_tokens; i++)
  {
    struct ctoken_t *ct = &t->state.cache.tokens[i];
    int token_id = ct->token_id;

    if (ct->sampled != rp)                       // role changed
    {
      // define some strings to ident injected/sampled tokens in text + position in cache.
      char role_str[256];
      sprintf(role_str, "\n{{%d:%s}}\n", i, ct->sampled ? "LLM" : "USER");
      term_cb_add_utf8(role_str);
      rp = ct->sampled;
    }

    if (raw_mode)
    {
      struct tokenizer_t *tk = &model.tokenizer;
      const char *s = tokenizer_get_token_str(token_id);
      term_cb_add_utf8(s);
      if (!strcmp(s, "<0x0A>"))
        term_cb_add_utf8("\n");                  // emit lf for visibility
      term_cb_add_utf8(",");
    }
    else
      term_cb_add_utf8(tokenizer_decode(token_id));
    
  }
  term_cb_copy();
}

// menus: acceded using '#m' keyboard input on a new user input line.
// return:
// cmd_resume: resume keyboard input
// cmd_regen : regen last llm reply
// cmd_exit  : exit program
static enum e_cmd run_menu(bool usr_inp, bool en_regen, bool en_forget)
{
  struct run_conf_t *conf = &model.config;
  while (1)
  {
    int k = read_key();
    switch (k)
    {
      case '1':
        if (!en_regen) break;
        msg_info(">1 : regen reply.\n");
        return cmd_regen;
      case '2':
        if (!en_forget) break;
        msg_info(">2 : forget last user input and llm reply.\n");
        return cmd_forget;
      case '3':
        conf->chat.fwd_disp_mode = !conf->chat.fwd_disp_mode;
        msg_info(">3 : fwd_disp_mode = %d\n", conf->chat.fwd_disp_mode);
      break;
      case '4':
        conf->tok_disp_raw = !conf->tok_disp_raw; 
        conf->tok_disp_split = conf->tok_disp_raw;
        msg_info(">4 : tok_disp_raw = %d\n", conf->tok_disp_raw);
      break;
      case '5':
        if (!usr_inp) break;
        cb_copy_dialog_text(false);
        msg_info(">5 : copy dialog text to clipboard.\n");
      break;
      case '6':
        if (!usr_inp) break;
        cb_copy_dialog_text(true);
        msg_info(">6 : copy raw dialog text to clipboard.\n");
      break;
      case '7':
        if (!usr_inp) break;
        msg_info(">7 : reset chat (keep sys prompt).\n");
        return cmd_rst_kp;
      case '8':
        if (!usr_inp) break;
        msg_info(">8 : reset chat (define new sys prompt).\n");
        return cmd_rst_np;
      case 'q':
        msg_info(">q : exit program.\n");
        return cmd_exit;
      case 27:
        return cmd_resume;
    }
    sleep_ms(50);
  }
}

static enum e_cmd user_menu(bool usr_inp, bool en_regen, bool en_forget)
{
  enum e_cmd cmd;
  struct transformer_t *t = &model.transformer;

  text_color(col_sys);
  msg_info("------------------------------------\n");
  msg_info("MENU: ctx size %d/%d (nd:%d)\n", t->state.cache.n_tokens, t->config.seq_len, t->state.cache.n_tokens_del);
  if (en_regen)  // cannot be done if no previous reply
  msg_info(" 1   - forget and regenerate last llm reply.\n");
  if (en_forget)
  msg_info(" 2   - forget last user input and llm reply.\n");

  msg_info(" 3   - show/hide user forward tokens.\n");
  msg_info(" 4   - enable/disable tokens raw display mode.\n");
  
  if (usr_inp)   // cannot be done if entering new system prompt
  {
  msg_info(" 5   - save dialog text to clipboard.\n");
  msg_info(" 6   - save raw dialog text to clipboard.\n");    // separate tokens + include special tokens
  msg_info(" 7   - reset chat (keep current sys prompt).\n");
  msg_info(" 8   - reset chat (enter new sys prompt).\n");
  }
  msg_info(" q   - exit program.\n");
  msg_info(" esc - exit this menu.\n");
  msg_info("Select option key:\n");
  cmd = run_menu(usr_inp, en_regen, en_forget);
  msg_info("------------------------------------\n");
  return cmd;
}
