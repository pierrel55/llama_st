#include <stdlib.h>
#include "l_util.h"
#include "mem_alloc.h"
#include "model.h"
#include "utf8.h"

#define EN_TOK(i) (sampler->tk_select[(i) >> 5] & (1 << ((i) & 31)))

// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
static unsigned int random_u32(uint64_t *state)
{
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// random float32 in [0,1)
static float random_f32(uint64_t *state)
{
  return (random_u32(state) >> 8) / 16777216.0f;
}

// qsort sort sample
static int compare(const void *a, const void *b)
{
  struct prob_index_t *a_ = (struct prob_index_t *)a;
  struct prob_index_t *b_ = (struct prob_index_t *)b;
  if (a_->prob > b_->prob) return -1;
  if (a_->prob < b_->prob) return 1;
  return 0;
}

// return the index that has the highest probability
static struct prob_index_t *sample_argmax(void)
{
  int vocab_size = model.transformer.config.vocab_size;
  float *logits = model.transformer.state.logits;
  struct prob_index_t *pi = model.sampler.probindex;

  int i, max_i = 0;
  float max_p = logits[0];
  for (i=1; i<vocab_size; i++) 
    if (logits[i] > max_p) 
    {
      max_i = i;
      max_p = logits[i];
    }
  pi->index = max_i;
  pi->prob = 1.0f;
  return pi;
}

// sampler_sample the token given the logits and some hyperparameters
struct prob_index_t *sampler_sample(void)
{
  int vocab_size = model.transformer.config.vocab_size;
  float *logits = model.transformer.state.logits;
  struct sampler_t *sampler = &model.sampler;
  struct sampler_conf_t *cfg = &sampler->conf;
  struct prob_index_t *probindex = sampler->probindex;
  bool topp_eos = cfg->topp_eos;
  float cutoff, prob_sum, eos_prob, r;
  int i, n;

  // nan check
  if (model.config.test_nan_logits && !check_no_nan_f32(logits, vocab_size))
    msg_info("<logits contain NAN>");

  // apply temperature
  if (cfg->temperature <= 0.01f)
    return sample_argmax();

  if ((cfg->temperature <= 0.99f) || (cfg->temperature >= 1.01f))
  {
    float k = 1.0f / cfg->temperature;
    for (i=0; i<vocab_size; i++) 
      logits[i] *= k;
  }

  // apply eos_amp 
  if (cfg->eos_amp > 0.01f)
  {
    int nt_samp = model.transformer.state.cache.n_tokens_samp;
    if (nt_samp > cfg->eos_amp_n)
    {
      float w = (float)(nt_samp - cfg->eos_amp_n) / cfg->eos_amp_n;
      float ki = 1.0f + w * cfg->eos_amp;
      float kd = 1.0f - w * cfg->eos_amp;
      int eos = model.config.token_eos;
      int eot = model.config.token_eot;
      if (logits[eos] > 0) logits[eos] *= ki; else logits[eos] *= kd;
      if (logits[eot] > 0) logits[eot] *= ki; else logits[eot] *= kd;
      topp_eos = true;                 // force topp_eos
    }
  }

  // apply repeat penalty (custom algorithm)
  if (cfg->repeat_penalty > 0.01f) 
  {
    int n_max = model.transformer.state.cache.n_tokens_samp;  // do not apply on injected tokens
    int n = cfg->repeat_penalty_n < n_max ? cfg->repeat_penalty_n : n_max;
    const struct ctoken_t *ct = model.transformer.state.cache.tokens + model.transformer.state.cache.n_tokens - n;
    float ki = 1.0f + cfg->repeat_penalty;
    float kd = 1.0f - cfg->repeat_penalty;
    for (i=0; i<n; i++)
    {
      int l = utf8_char_len(tokenizer_decode(ct[i].token_id));
      if (l >= 4)                      // apply on long piece only
      {
        if (logits[i] >= 0)
          logits[i] *= kd;             // decrease value
        else
          logits[i] *= ki;
      }
    }
  }

  // apply softmax to the logits to get the probabilities for next token
  softmax(logits, vocab_size);

  // quicksort indices in descending order of probabilities
  // values smaller than (1 - cfg->topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  cutoff = (1.0f - cfg->topp) / (vocab_size - 1);
  n = 0;
  for (i=0; i<vocab_size; i++)
  {
    float prob = logits[i];
    if (prob >= cutoff)
    {
      if (sampler->tk_select && !EN_TOK(i))
        continue;
      probindex[n].index = i;
      probindex[n].prob = prob;
      n++;
    }
  }
  qsort(probindex, n, sizeof(struct prob_index_t), compare);

  // apply topk
  if (cfg->topk && (n > cfg->topk))
    n = cfg->topk;

  // truncate the list where cumulative probability exceeds topp
  prob_sum = 0;
  eos_prob = 0;
  for (i=0; i<n; i++) 
  {
    int index = probindex[i].index;
    float prob = probindex[i].prob;
    prob_sum += prob;

    if (    (index == model.config.token_eos)
         || (index == model.config.token_eot))
    {
      eos_prob = prob;
      if (topp_eos)
        break;
    }
    else
    // if eos prob in top list and prob < topp_minp then end list.
    if (eos_prob && (prob < cfg->topp_minp))
      break;

    if (prob_sum >= cfg->topp)
      break;                           // we've exceeded topp by including last_idx
  }
  n = i;                               // n = n_topp - 1
 
  // random sample from the truncated list
  r = random_f32(&sampler->rng_state) * prob_sum;
  prob_sum = 0.0f;
  for (i=0; i<n; i++)
  {
    prob_sum += probindex[i].prob;
    if (prob_sum > r)
      break;
  }
  return &probindex[i];
}

// -------------------------------------------
// init sampler

// test if token contain non-allowed utf8 chars
static bool tk_reject(const char *s, int *r_codes, int n_codes)
{
  while (*s)
  {
    int code;
    int l = utf8_char_decode(s, &code);
    if (l > 1)
    {
      int j;
      for (j=0; j<n_codes; j++)
        if (r_codes[j] == code)
          break;
      if (j == n_codes)
        return true;
    }
    s += l;
  }
  return false;
}

void build_sampler(void)
{
  struct sampler_t *sampler = &model.sampler;
  struct sampler_conf_t *cfg = &model.sampler.conf;

  #define P_ADJ(typ, var, dis, min, max) if (cfg->var != dis) adjust_range_##typ(&cfg->var, #var, min , max)

  // adjust user defined parameters in allowed ranges
  P_ADJ(f32, temperature      , 1.0f, 0.0f , 2.0f );   // 0.0 to 2.0: 0:greedy decoding 2.0:maximum creativity
  P_ADJ(f32, topp             , 0.5f, 0.01f, 0.99f);   // 0.01 to 0.99: max probability sum of top tokens
  P_ADJ(int, topk             , 0   , 5    , 200  );   // (integer) size of top tokens list 5..200
  P_ADJ(f32, topp_minp        , 0.0f, 0.0f , 1.0f );   // 0.0 to 1.0: (experimental) !=0: min token probability required to continue generate if EOS contained in topp list
  P_ADJ(f32, repeat_penalty   , 0.0f, 0.0f , 2.0f );   // 0.0..2.0 repeat penalty (0.0 = disable)
  P_ADJ(int, repeat_penalty_n , 0   ,   10 , 1000 );   // (integer) count of last generated tokens used to apply repeat penalty (0 = disable, min 10)
  P_ADJ(f32, eos_amp          , 0.0f, 0.0f , 2.0f );   // 0.0 to 2.0 eos probability increase coefficient when more than eos_amp_n tokens generated. (0 = disable)
  P_ADJ(int, eos_amp_n        , 0   ,   10 , 1000 );   // (integer) count of token generated before starting eos_amp influence (0 = disable, min 10)

  // alloc working ddtas
  sampler->probindex = malloc_check(model.transformer.config.vocab_size * sizeof(struct prob_index_t));

  // init rng
  sampler->rng_state = cfg->rand_seed;

  // display infos
  msg_info("sampler config:\n");
  msg_info("  temperature      : %.2f\n", cfg->temperature);
  msg_info("  topp             : %.2f\n", cfg->topp);
  msg_info("  topk             : %d\n"  , cfg->topk);
  msg_info("  topp_minp        : %.2f\n", cfg->topp_minp);
  msg_info("  topp_eos         : %s\n"  , cfg->topp_eos ? "true" : "false");
  msg_info("  repeat_penalty   : %.2f\n", cfg->repeat_penalty);
  msg_info("  repeat_penalty_n : %d\n"  , cfg->repeat_penalty_n);
  msg_info("  eos_amp          : %.2f\n", cfg->eos_amp);
  msg_info("  eos_amp_n        : %d\n"  , cfg->eos_amp_n);
  msg_info("  rand seed        : %d\n"  , cfg->rand_seed);

  // disable invalid configuration
  if (cfg->repeat_penalty_n == 0) cfg->repeat_penalty = 0.0f;
  if (cfg->eos_amp_n        == 0) cfg->eos_amp        = 0.0f;

  // restrict chars, used to restrict tokens that can be sampled.
  // this was defined for very small twen models (0.5B) that could generate Chenese tokens mixed with non Chenese text.
  if (cfg->ch_restrict)
  {
    int vocab_max = model.transformer.config.vocab_size;
    int *msk, n_msk, i;
    int r_codes[256];
    const char *s = cfg->ch_restrict;
    int n_codes = utf8_get_char_count(cfg->ch_restrict);
    if (!n_codes || (n_codes > 256))
      msg_error("ch_restrict string containd invalid utf8 encoding or more than 256 characters");

    // alloc binary array
    n_msk = (vocab_max + 31) / 32;
    msk = calloc_check(n_msk * sizeof(int));
    sampler->tk_select = msk;

    // define allowed tokens
    for (i=0; i<vocab_max; i++)
    {
      const char *tk = tokenizer_get_token_str(i);
      bool reject = tk_reject(tk, r_codes, n_codes);
      if (!reject)
        msk[i >> 5] |= (1 << (i & 31));
    }
    sampler->tk_select = msk;
#if 0
    // debug: print rejected tokens
    for (i=0; i<vocab_max; i++)
      if (!EN_TOK(i))
      {
        tokenizer_decode_print(i, true);
        msg_info("\n");
      }
#endif
  }
}

void free_sampler(void)
{
  free_check(model.sampler.probindex);
  free_check(model.sampler.tk_select);
}