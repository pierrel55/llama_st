#include "l_util.h"
#include "model.h"
#include "term_utf8.h"
#include "time_ev.h"

// re-use function defined in chat.c
void tokenizer_decode_print_ex(int token_id, float prob);

// generation loop
void generate(void)
{
  int i, t0, t1, n_gen, run_steps;
  struct run_conf_t *conf = &model.config;
  const struct mt_list_t *mt_list = &model.tokenizer.mt_list;

  msg_info("Generate: max %d tokens..\n", conf->gen_run_steps);
  msg_info("- Press 'esc' key to break generation.\n");
  T_RESET();                   // dev mode, eval code time

  // time stats
  t0 = time_in_ms();

  // forward init prompt
  tokenizer_encode(conf->gen_mode_prompt);
  for (i=0; i<mt_list->n_list; i++)
  {
    int token = mt_list->mt[i].tok_id;
    forward(token, false, i == (mt_list->n_list-1));
    tokenizer_decode_print_ex(token, -1.0f);
  }

  // generate
  for (run_steps = 0; run_steps != conf->gen_run_steps; run_steps++)
  {
    // get token from logits into samp
    struct prob_index_t *pi = sampler_sample();

    // stop gen is s key pressed
    if (read_key() == 27)
    {
      msg_info("{esc stop}");
      break;
    }

    // data-dependent terminating condition
    if (   (pi->index == model.config.token_eos)
        || (pi->index == model.config.token_eot))
      break;

    // print generated token
    tokenizer_decode_print_ex(pi->index, pi->prob);

    // update logits
    forward(pi->index, true, true);
  }

  T_PRINT();

  // time elapsed
  t1 = time_in_ms();
  n_gen = model.transformer.state.cache.n_tokens;
  msg_info("\ntotal time: %.2fs for %d tokens, tok/s: %.2f\n", (t1-t0) / 1000.0, n_gen, n_gen*1000.0 / (t1-t0));
}
