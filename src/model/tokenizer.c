#include <stdio.h>
#include <stdlib.h>
#include "l_util.h"
#include "utf8.h"
#include "term_utf8.h"
#include "mem_alloc.h"
#include "model.h"
#include "load_tokenizer.h"

// ------------------------------------
// encode text

// define merge score between two tokens
static void set_m_score(struct m_tok_t *mt, int token_id)
{
  const struct merge_id_t *m_id = bpe_find_merge(mt->tok_id, token_id);
  if (!m_id)
    mt->score = -1;
  else
  {
    mt->score = m_id->merge_id;                  // score
    mt->tok_id_m = m_id->tok_id_m;               // merge token mt->tok_id + token_id
  }
}

// add token in mt_list and define score of previous token in list with added token
static void mt_add_token(int token_id, bool def_score)
{
  struct tokenizer_t *t = &model.tokenizer;
  struct m_tok_t *mt;
  struct mt_list_t *mt_list = &t->mt_list;
  if (mt_list->n_list == mt_list->n_alloc)
  {
    mt_list->n_alloc += 1024;
    mt_list->mt = (struct m_tok_t *)realloc_check(mt_list->mt, mt_list->n_alloc * sizeof(struct m_tok_t));
  }
  mt = &mt_list->mt[mt_list->n_list++];

  // set score of previous with new
  if (def_score && (mt_list->n_list > 1))
    set_m_score(mt - 1, token_id);
  
  // init new in list
  mt->score = -1;
  mt->tok_id = token_id;
  mt->tok_id_m = -1;
}

#if 0
// debug usage, print token list
static void PRINT_TL(void)
{
  const struct tokenizer_t *t = &model.tokenizer;
  const struct mt_list_t *mt_list = &t->mt_list;
  int i;
  for (i=0; i<t->mt_list.n_list; i++)
  {
    tokenizer_decode_print(t->mt_list.mt[i].tok_id, true);
    msg_info("[%d],", t->mt_list.mt[i].tok_id);
  }
  msg_info("\n");
}
#else
#define PRINT_TL()
#endif

// encode one token per utf8 char
static void char_encode_mt_list(const char *text)
{
  struct tokenizer_t *t = &model.tokenizer;

mt_encode_continue:
  while (*text)
  {
    char utf8_char[8];
    int tok_id, l_utf8;

    // convert special tokens string to single token.
    if (*text == '<')
    {
      int id;
      for (id=t->id_special_base; id<=t->id_special_last; id++)
      {
        const char *s = tokenizer_get_token_str(id);
        const char *c = text;
        for (;*s && (*s == *c); s++, c++);
        if (!*s)
        {
          mt_add_token(id, false);
          text = c;
          goto mt_encode_continue;           // exit double loop
        }
      }
    }

    l_utf8 = utf8_char_len(text);            // length of utf8 char (max is 4)
    if (!l_utf8)                             // check encoding of string
      msg_error("tokenizer_encode: invalid utf8 char encoding");

    memcpy(utf8_char, text, l_utf8);         // need 0 ended string
    utf8_char[l_utf8] = 0;
    text += l_utf8;

    tok_id = tokenizer_find_token_id(utf8_char);
    if (tok_id >= 0)
      mt_add_token(tok_id, true);
    else
    if (t->token_id_0xff)                    // byte fallback used
    {
      int i;
      for (i=0; i<l_utf8; i++)
        mt_add_token(t->token_id_0x0 + (unsigned char)utf8_char[i], false);
    }
    else
    {
      int utf8_code;
      utf8_char_decode(utf8_char, &utf8_code);
      msg_error("tokenizer_encode: no token for utf8 code %d", utf8_code);
    }
  }
}

// encode text, define tokeniser token list
void tokenizer_encode(const char *text)
{
  struct tokenizer_t *t = &model.tokenizer;
  t->mt_list.n_list = 0;                         // reset list size

  CHECK(text && text[0]);
  if (!text || !text[0])                         // should not occur
    return;

  // encode all utf8 char as token id and define initial merge scores
  char_encode_mt_list(text);
  PRINT_TL();

  // --------------------------------------------
  // merge token pairs
  while (1)                                      // repeat until no merge found
  {
    int score = t->merge.list_size;              // max score value (min score is best merge)
    int i, i_best = -1, i_max = t->mt_list.n_list - 1;

    for (i=0; (i<i_max) && score; i++)
    {
      struct m_tok_t *mt = &t->mt_list.mt[i];
      if ((mt->score >= 0) && (mt->score < score))
      {
        score = mt->score;
        i_best = i;
      }
    }

    PRINT_TL();
    if (i_best < 0)                              // no merge found
      break;

    // concat list
    for (i=i_best+1; i<i_max; i++)
      t->mt_list.mt[i] = t->mt_list.mt[i+1];
    t->mt_list.n_list--;

    // replace token
    t->mt_list.mt[i_best].tok_id = t->mt_list.mt[i_best].tok_id_m;
    PRINT_TL();

    // update score with next
    set_m_score(&t->mt_list.mt[i_best], t->mt_list.mt[i_best+1].tok_id);

    // update score of previous
    if (i_best > 0)
      set_m_score(&t->mt_list.mt[i_best-1], t->mt_list.mt[i_best].tok_id);
  }
}

// return token string.
// input bin buff can be small (2 chars) as is used only to receive one 8 bits binary value
const char *tokenizer_decode(int token_id)
{
  struct tokenizer_t *t = &model.tokenizer;

  // < 0 token, bug somewhere if occur
  CHECK(token_id >= 0);

  // not printable special tokens
  if ((token_id < 0) || ((token_id >= t->id_special_base) && (token_id <= t->id_special_last)))
    return "";

  if (t->token_id_0xff)       // byte fallback used
  {
    if ((token_id <= t->token_id_0x0 + 255) && (token_id >= t->token_id_0x0))
    {
      char c = token_id - t->token_id_0x0;
      if ((c == '\r') || (c == '\n'))
        return "\n";
      if (c == '\t')
        return "\t";
      // todo: interpret other codes ?
      return "";
    }
  }

  // normal token
  return tokenizer_get_token_str(token_id);
}

// decode and print token
void tokenizer_decode_print(int token_id, bool disp_raw)
{
  const struct tokenizer_t *t = &model.tokenizer; 
  CHECK((unsigned int)token_id < (unsigned int)t->tok_index_list_size);

  // raw display mode (display all, <unk> <s> </s> etc.., test usage)
  if (disp_raw)
  {
    const char *s = tokenizer_get_token_str(token_id); 
    print_utf8_raw(s);
    if (!strcmp(s, "<0x0A>"))
      print_utf8("\n");                          // emit lf for visibility
  }
  else
  {
    const char *s = tokenizer_decode(token_id);
    if (!t->mode_ll3)
    {
      // following BOS (1) token_id, sentencepiece decoder strips any leading whitespace (see PR #89)
      static int prev_token = 0;                 // previous printed token
      if ((prev_token == t->token_id_bos_ws) && (s[0] == ' '))
        s++;
      prev_token = token_id;
    }
    if (s[0])                                    // printable
      print_utf8(s);                             // print in utf8
  }
}

// load and init tokenizer from .json file
void build_tokenizer(void)
{
  char tmp[256];
  const char *file_name = model.config.load.tokenizer_name;

  // ll3 mode: no strips leading whitespace (PR #89), 
  model.tokenizer.mode_ll3 = (    (model.config.e_model_id == model_id_llama3) 
                               || (model.config.e_model_id == model_id_llama31)
                               || (model.config.e_model_id == model_id_qwen2));

  // if file_name empty, define default name as model_path + tokenizer.json
  if (!file_name[0])
  {
    int l = _snprintf(tmp, sizeof(tmp), "%s/%s", model.config.load.model_path, "tokenizer.json");
    if ((l < 0) || (l == sizeof(tmp)))
      msg_error("tokenizer path too long or invalid format");
    file_name = tmp;
  }

  // load datas
  msg_info("load tokenizer: %s\n", file_name);
  load_tokenizer(file_name);
}

// free allocated mem
void free_tokenizer(void)
{
  struct tokenizer_t *t = &model.tokenizer;
  free_check(t->dic_tokens.buff);
  free_check(t->tok_index);
  free_check(t->merge.id_list);
  free_check(t->mt_list.mt);
}

#if 0
// -------------------------------
// test code

static void dump_token_list(int mode)
{
  const struct mt_list_t *mt_list = &model.tokenizer.mt_list;
  int i;
  for (i=0; i<mt_list->n_list; i++)
  {
    int tok_id = mt_list->mt[i].tok_id;
    if (mode == 0)
    {
      msg_info("%d (%d): \'", i, tok_id);
      tokenizer_decode_print(tok_id, true);
      msg_info("\'\n");
    }
    else
    if (mode == 1)
      tokenizer_decode_print(tok_id, true);
    else
      tokenizer_decode_print(tok_id, false);
  }
}

// load utf8 text from file
static char *load_text_utf8(const char *file_name)
{
  char *txt;
  file_t f = { 0 };
  f_open(&f, file_name, "rb");
  txt = malloc_check(f.size+1);
  f_read(txt, f.size, &f);
  f_close(&f);
  txt[f.size] = 0;
  return txt;
}

int main(void)
{
  char *text, *text_load = NULL;
  if (APP_ERROR())
    return -1;

#if 0
  text = "we couldn't find any more pairs to merge, so we're done";
  //text = "input a string, return utf8 encoded buffer size in bytes, -1 if error.";
  //text = "at twenty five and twelve the fifteen teenager and we go home";
#else
  text_load = load_text_utf8("gen_7b_utf8.txt");    // 279
  //text_load = load_text_utf8("txt_utf8.txt");        // 55
  //text_load = load_text_utf8("tok_bpe_utf8.txt");    // 2178
  text = text_load + 3;  // pass UTF8 header (239, 187, 191)
  utf8_cvt_crlf_to_cr(text);            // convert lf
  // print_utf8(text_load);          // display source texte
#endif

  model.config.load.tokenizer_name = "E:/llama2/llama2-7b-chat-hf/tokenizer.json";
  build_tokenizer();
  tokenizer_encode(text, true, true);

  dump_token_list(0);
  msg_info("\n-----\n");
  dump_token_list(1);
  msg_info("\n-----\n");
  dump_token_list(2);
  msg_info("\n-----\n");
  msg_info("\nn_tokens = %d\n", model.tokenizer.mt_list.n_list);

  free_tokenizer();
  free_check(text_load);
  dbg_print_alloc();

  wait_return_exit();
}
#endif
