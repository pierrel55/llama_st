// ----------------------------------------------------------------------------
// The struct sampler_t, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

// struct used when sorting probabilities during top-p sampling
struct prob_index_t
{
  float prob;
  int index;
};

// sampler config
struct sampler_conf_t
{
  float temperature;               // 0.0 to 2.0: 0:greedy decoding 2.0:maximum creativity
  float topp;                      // 0.01 to 0.99: max probability sum of top tokens
  int topk;                        // (integer) limit size of top tokens list 5..200 (0 = disable)
  float topp_minp;                 // 0.0 to 1.0: (experimental) !=0: min token probability required to continue generate if EOS contained in topp list
  bool topp_eos;                   // true: limit topp list size to token with probability >= EOS
  float repeat_penalty;            // 0.0..2.0 repeat penalty (0.0 = disable)
  int repeat_penalty_n;            // (integer) count of last generated tokens used to apply repeat penalty (0 = disable, min 10)
  float eos_amp;                   // 0.0 to 2.0 amplify eos probability when more than eos_inc_n tokens generated. (0 = disable)
  int eos_amp_n;                   // (integer) count of tokens generated before starting eos_amp influence (0 = disable, min 10)
  int rand_seed;                   // (integer) random seed
  char *ch_restrict;               // if string defined, define ascii + allowed chars list in sampled tokens.
};

struct sampler_t
{
  struct sampler_conf_t conf;      // config

  uint64_t rng_state;
  struct prob_index_t *probindex;  // buffer used in top-p sampling
  int *tk_select;                  // binary array for restricted tokens, (NULL if unused)
};

void build_sampler(void);
void free_sampler(void);

// sample from transformer logits
struct prob_index_t *sampler_sample(void);
