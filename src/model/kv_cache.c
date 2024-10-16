// kv cache: is part of transformer.c but moved in separate file for clarity.
// this code use a method to ensure kv cache is never full. (experimental.. is that good ?)
// the method consists in 'forgetting' the oldest tokens.
// in chat mode:
//  - series of one user entry + one llm reply are deleted after system prompt
// in generate mode:
//  - only delete some first tokens.
// the hole produced after systemp promt is removed and rope for kv datas following hole is updated as if tokens follow prompt without hole.
// todo: difficult to test effect as require context almost full to operate.

#ifdef PACK_KV_CACHE

#include "l_util.h"
#include "model.h"

// 'forget' some tokens in kv cache to reduce context size.
static void reduce_kv_cache(int min_tokens_delete)
{
  struct transformer_t *t = &model.transformer;
  const struct transformer_config_t *p = &t->config;
  struct transformer_runstate_t *s = &t->state;

  int n_ctx = s->cache.n_tokens;       // current tokens count in context
  int min_del = n_ctx/20;              // min tokens to delete (5% of context)
  int i0, i, n_del;

  if (model.config.run_mode == 0)
  {
    // generate mode, very unlikely to happen (no eot produced before context full)
    // done to ensure no cache overflow.
    i0 = 0;
    i = min_del;
  }
  else                                 // chat mode
  {
    if (min_tokens_delete < min_del)
      min_tokens_delete = min_del;

    i0 = t->state.cache.n_tokens_sys;  // keep sys prompt if defined
    for (i=i0; i<n_ctx; )
    {
      // pass one user entry
      for (; i<n_ctx; i++)
        if (t->state.cache.tokens[i].sampled)
          break;

      // pass one llm reply
      for (; i<n_ctx; i++)
        if (!t->state.cache.tokens[i].sampled)
          break;

      // test if enough deleted
      if ((i - i0) >= min_tokens_delete)
        break;
    }
  }

  // count of deleted tokens in cache
  n_del = i - i0;

  // update user info
  t->state.cache.n_tokens_del += n_del;

#if 0
  // debug: display deleted token list
  {
    int j;
    msg_info("\n------\n kv delete %d tokens:\n", n_del);
    for (j=i0; j<i; j++)
    {
      tokenizer_decode_print(s->cache.tokens[j].token_id, true);
      msg_info(",");
    }
    msg_info("\n------\n");
  }
#endif

  // define < 0 rope rotation = - num of deleted tokens
  set_RoPE_pos(s->rope_sin_cos, -n_del, s->rope_freq, p->head_size/2);

  // compact and update kv cache rope
  s->cache.n_tokens = i0;
  s->cache.n_tokens_samp = 0;

  for (; i<n_ctx; i++)
  {
    int l, pos = s->cache.n_tokens++;

    // remove kv cache hole
    for (l=0; l<p->n_layers; l++)
    {
      size_t i_ofs = ((size_t)l * n_ctx + i  ) * p->kv_dim;
      size_t p_ofs = ((size_t)l * n_ctx + pos) * p->kv_dim;
      RoPE(&s->k_cache[i_ofs], &s->v_cache[i_ofs], s->rope_sin_cos, p->head_size, p->kv_dim, p->kv_dim);
      memcpy(&s->k_cache[p_ofs], &s->k_cache[i_ofs], p->kv_dim * sizeof(float));
      memcpy(&s->v_cache[p_ofs], &s->v_cache[i_ofs], p->kv_dim * sizeof(float));
    }

    // compact token list
    s->cache.tokens[pos] = s->cache.tokens[i];
    if (s->cache.tokens[pos].sampled)
      s->cache.n_tokens_samp++;
    else
      s->cache.n_tokens_samp = 0;
  }
}

// reserve tokens in kv cache for llm generation.
// return cound of deleted tokens
int reserve_kv_cache(int min_token_reserve)
{
  int token_prev = model.transformer.state.cache.n_tokens;
  int token_left = model.transformer.config.seq_len - token_prev;
  if (token_left < min_token_reserve)
  {
    reduce_kv_cache(min_token_reserve - token_left);
    return token_prev - model.transformer.state.cache.n_tokens;  // return num of deleted tokens
  }
  return 0;
}

#endif // PACK_KV_CACHE