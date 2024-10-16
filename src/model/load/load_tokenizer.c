// load tokenizer.json format

#include <stdio.h>
#include <stdlib.h>
#include "l_util.h"
#include "mem_alloc.h"
#include "utf8.h"
#include "json.h"
#include "model.h"
#include "load_tokenizer.h"

// max token string length including 0 end
// llama2: 64 is enough
// llama3: some tokens are very long (ex: token 118594)
#define MAX_TOKEN_LENGTH 260

// utf8 sentencepiece default whitespace (llama < 3)
struct
{
  char u_len;              // utf8 len
  char u[4];               // utf8 char
} sp_ws = { 3, { 0xe2, 0x96, 0x81, 0 } };

// ---------------------------------------------
// string list allocated in contiguous mem array

// add string of length len in dictionary, return store offset in dictionary buffer.
static int str_dic_add(struct str_dic_t *dic, const char *str, int len, int alloc_inc)
{
  int ofs = dic->wr_ofs;
  if ((dic->wr_ofs + len) >= dic->sz_alloc)
  {
    dic->sz_alloc += (len + 1 + alloc_inc);
    dic->buff = realloc_check(dic->buff, dic->sz_alloc);
  }
  memcpy(dic->buff + dic->wr_ofs, str, len);
  *(dic->buff + dic->wr_ofs + len) = 0;
  dic->wr_ofs += (len + 1);
  dic->n_strings++;
  return ofs;
}

// ------------------------------------
// tokens

// sort / bsearch tokens
static int cmp_tokens(const void *a, const void *b)
{
  const struct tok_index_t *ta = (const struct tok_index_t *)a;
  const struct tok_index_t *tb = (const struct tok_index_t *)b;
  return strcmp(ta->str, tb->str);
}

// sort token list for bsearch
static void sort_token_list(struct tokenizer_t *t)
{
  qsort(t->tok_index, t->tok_index_list_size, sizeof(struct tok_index_t), cmp_tokens);
}

// find a token id from utf8 string, return -1 if not found
int tokenizer_find_token_id(const char *str)
{
  const struct tokenizer_t *t = &model.tokenizer;
  struct tok_index_t tok = { str, 0 };
  struct tok_index_t *res = bsearch(&tok, t->tok_index, t->tok_index_list_size, sizeof(struct tok_index_t), cmp_tokens);
  return res ? res->tok_id : -1;
}

// find a special token from string and check is special token
int tokenizer_find_sp_token_id(const char *str)
{
  struct tokenizer_t *t = &model.tokenizer;
  int id = tokenizer_find_token_id(str);
  if ((id < t->id_special_base) || (id > t->id_special_last))
    msg_error("failed to get special token '%s'", str);
  return id;
}

// find a token, produce error if not found
static int find_token_check(const char *str)
{
  int id = tokenizer_find_token_id(str);
  if (id < 0)
    msg_error("failed to get token '%s'", str);
  return id;
}

// return token string from token index
const char *tokenizer_get_token_str(int token_id)
{
  const struct tokenizer_t *t = &model.tokenizer;
  if ((unsigned int)token_id >= (unsigned int)t->tok_index_list_size)
    return "<unk>";                              // should never occur
  return t->tok_index[t->tok_index[token_id].id_to_sort].str;
}

// ------------------------------------
// merges

// sort / bsearch merges
static int cmp_merges(const void *a, const void *b)
{
  const struct merge_id_t *ma = (const struct merge_id_t *)a;
  const struct merge_id_t *mb = (const struct merge_id_t *)b;
  if (ma->tok_id_l > mb->tok_id_l)
    return 1;
  if (ma->tok_id_l < mb->tok_id_l)
    return -1;
  if (ma->tok_id_r > mb->tok_id_r)
    return 1;
  if (ma->tok_id_r < mb->tok_id_r)
    return -1;
  return 0;
}

// sort merge strings for bsearch
static void sort_merge_list(struct tokenizer_t *t)
{
  qsort(t->merge.id_list, t->merge.list_size, sizeof(struct merge_id_t), cmp_merges);
}

// find merge datas, return NULL if not found
const struct merge_id_t *bpe_find_merge(int l_id, int r_id)
{
  const struct tokenizer_t *t = &model.tokenizer;
  struct merge_id_t merge = { l_id, r_id, 0 };
  return bsearch(&merge, t->merge.id_list, t->merge.list_size, sizeof(struct merge_id_t), cmp_merges);
}

// ------------------------------------
// load

// copy utf8 string and replace special chars
static int utf8_replace_char(char *s)
{
  char *d0 = s;
  char *d = s;
  while (*s)
  {
    int l = utf8_char_len(s);
    if (!l)
      msg_error("token invalid utf8 encoding");
    if (l == 1)                                 // for speed
      *d++ = *s++;
    else
    switch (model.config.e_model_id)
    {
      #define CMP_CH2(a, b) ((unsigned char)s[0] == a) && ((unsigned char)s[1] == b) && (l == 2)
      case model_id_llama3:
      case model_id_llama31:
      case model_id_qwen2:
        if      (CMP_CH2(0xc4, 0xa0)) { *d++ =  ' '; s+=2; }
        else if (CMP_CH2(0xc4, 0x8a)) { *d++ = '\n'; s+=2; }
        else { memcpy(d, s, l); d+=l; s+=l; }
      break;
      default:  // sentencepiece whitespace
        if ((l == sp_ws.u_len) && !memcmp(s, sp_ws.u, l)) { *d++ = ' '; s+=l; }
        else { memcpy(d, s, l); d+=l; s+=l; }
    }
  }
  *d = 0;
  return (int)(d - d0);
}

// init a merge string
static void def_merge_string(const char *str, struct tokenizer_t *t)
{
  struct merge_id_t *m_id;

  // remove space in merge string and check resulting token exist in token list
  char m_str[MAX_TOKEN_LENGTH];  // merge string without space
  char l_str[MAX_TOKEN_LENGTH];  // left string
  char r_str[MAX_TOKEN_LENGTH];  // right string
  char *ms_max = m_str + sizeof(m_str) - 2;
  char *ms = m_str;
  char *ls = l_str;
  char *rs = r_str;
  const char *s1 = str;
  const char *sp = NULL;

  // get left and right part of merge string
  while (*s1 && (ms < ms_max))
  {
    if (*s1 == ' ')
    {
      if (sp)                          // multi spaces found
        msg_error("merge string: multi spaces");
      sp = s1;
    }
    else
    {
      *ms++ = *s1;
      if (!sp)
        *ls++ = *s1;
      else
        *rs++ = *s1;
    }
    s1++;
  }
  
  // checks, str too long or space not found or at invalid position
  if (*s1 || !sp || (ls == l_str) || (rs == r_str))
    msg_error("merge string: space not found or invalid position");  // or MAX_TOKEN_LENGTH too small

  // terminate strings
  *ms = 0;
  *ls = 0;
  *rs = 0;

  // replace special encoded chars
  utf8_replace_char(m_str);
  utf8_replace_char(l_str);
  utf8_replace_char(r_str);

  // save
  if (t->merge.list_size == t->merge.n_alloc)
  {
    t->merge.n_alloc += 5000;
    t->merge.id_list = (struct merge_id_t *)realloc_check(t->merge.id_list, t->merge.n_alloc * sizeof(struct merge_id_t));
  }
  m_id = &t->merge.id_list[t->merge.list_size];
  m_id->merge_id = t->merge.list_size;

  m_id->tok_id_m = tokenizer_find_token_id(m_str);
  if (m_id->tok_id_m < 0)             // cannot find token for merged string
    msg_error("merge string: no token match");

  // define left and right tokens
  m_id->tok_id_r = tokenizer_find_token_id(r_str);
  m_id->tok_id_l = tokenizer_find_token_id(l_str);
  if ((m_id->tok_id_l < 0) || (m_id->tok_id_r < 0))
    msg_error("merge string: sub token not found");
  t->merge.list_size++;
}

// add a token in tok_index list
static int token_list_add(struct tokenizer_t *t, int tok_id, const char *str, int l_str, int nt_alloc)
{
  int dic_ofs;

  // check
  if ((tok_id < 0) || (l_str <= 0) || (!str[0]))
    msg_error("invalid token index %d or empty token '%s'", tok_id, str);
  
  // check len
  if (l_str >= MAX_TOKEN_LENGTH)
    msg_error("token %d length > max length (%d, max = %d)", tok_id, l_str, MAX_TOKEN_LENGTH-1);

  // reserve memory for tok_index list
  if (tok_id >= nt_alloc)
  {
    int i;
    int nt_alloc_prev = nt_alloc;
    nt_alloc = tok_id + 32000 + 8;
    t->tok_index = (struct tok_index_t *)realloc_check(t->tok_index, nt_alloc * sizeof(struct tok_index_t));
    for (i=nt_alloc_prev; i<nt_alloc; i++)
      t->tok_index[i].tok_id = -1;               // flag no data
  }
  
  // need to check if token already saved (same token can be defined in added_tokens list and model.vocab list)
  dic_ofs = t->tok_index[tok_id].tok_id;
  if (dic_ofs >= 0)
  {
    char *dic_str = &t->dic_tokens.buff[dic_ofs];
    if (strcmp(dic_str, str))
      msg_error("token %d have different string definition '%s' : '%s'", tok_id, dic_str, str);
  }
  else
  {
    // store token string in dic
    dic_ofs = str_dic_add(&t->dic_tokens, str, l_str, 65536);

    // note: temporary use tok_id to store offset in dic.
    // cannot define string ptr now because realloc usage.
    t->tok_index[tok_id].tok_id = dic_ofs;
  }

  return nt_alloc;
}

// load tokenizer datas from .json
void load_tokenizer(const char *file_name)
{
  struct tokenizer_t *t = &model.tokenizer;
  struct js_read_inf_t inf = { 0 };
  struct h_json_t *h = js_load_file(file_name, false);
  bool chk_bpe = false;
  int nt_alloc = 0;                              // allocated elements in tok_index list

  js_seek_origin(h);

  while (js_read_param(h, &inf))
  {
    // js_print_param(h, &inf);  // debug, display json key + value
    
    // ----------------------------------
    // added tokens list (special tokens)
    if (js_cmp_key_list(h, "added_tokens"))      // is [ properties list ]
    {
      do                                         // iterate array loop
      {
        int id = -1;
        char content[MAX_TOKEN_LENGTH];
        content[0] = 0;
        do                                       // iterate array element properties
        {
          // ex: (0) :added_tokens:id = 128000
          // ex: (1) :added_tokens:content = "<|begin_of_text|>"
          if (js_cmp_key_list(h, "added_tokens.id"))
            id = js_get_num_value_i32(h);
          else
          if (js_cmp_key_list(h, "added_tokens.content"))
            js_copy_key_value_str(h, content, sizeof(content));
        }
        while (!inf.is_lev_end && js_read_param(h, &inf));

        if ((id < 0) || !content[0])
          msg_error("added_tokens: missing id or content value");

        // def base index for special tokens
        if (!t->id_special_count)
          t->id_special_base = id;

        // check ordered id
        if (id != (t->id_special_base + t->id_special_count))
          msg_error("added_tokens: not ordered index");

        t->id_special_count++;

        // add token in list
        // msg_info("special token %d: %s\n", id, content);
        nt_alloc = token_list_add(t, id, content, (int)strlen(content), nt_alloc);
      }
      while (!inf.is_arr_end && js_read_param(h, &inf));

      // def last inclusive index
      t->id_special_last = t->id_special_base + t->id_special_count - 1;
    }
    else
    // --------------------------------
    // parameters checks
    if (js_cmp_key_list(h, "model.type"))        // model.type = BPE
    {
      js_check_key_value_str(h, "BPE");
      chk_bpe = true;
    }
    else
    // --------------------------------
    // sentencepiece whitespace code (should be "\xE2\x96\x81" if not using llama3)
    if (js_cmp_key_list(h, "decoder.decoders.pattern.String"))
    {
      js_copy_key_value_str(h, sp_ws.u, sizeof(sp_ws.u));
      sp_ws.u_len = utf8_char_len(sp_ws.u);
      if (!sp_ws.u_len)
        msg_error("sentencepiece whitespace code is not utf8");
    }
    else
    // --------------------------------
    // load tokens
    if (js_cmp_key_list(h, "model.vocab"))       // model.vocab. token, id_token
    {
      int i = 0;
      // load tokens
      do
      {
        int l_key;
        char key_ident[MAX_TOKEN_LENGTH];

        // get token index
        int tok_id = js_get_num_value_i32(h);

        // check ordered load index
        if (tok_id != i)
          msg_error("json invalid token index %d, expect %d", tok_id, i);
        i++;

        // get token string to key_ident
        js_copy_key_ident_str(h, key_ident, sizeof(key_ident), 2);

        // replace sentencepiece space code to real space
        l_key = utf8_replace_char(key_ident);

        // add token in list
        nt_alloc = token_list_add(t, tok_id, key_ident, l_key, nt_alloc);
      }
      while (!inf.is_lev_end && js_read_param(h, &inf));  // read next

      // get total token count
      for (i=0; i<nt_alloc; i++)
        if (t->tok_index[i].tok_id < 0)
          break;

      // check no hole in list
      if (i != t->dic_tokens.n_strings)
        msg_error("token index hole..");   // bug somewhere or index hole in .json between model.vocab and added_tokens

      // realloc to exact size to free unused memory
      t->tok_index = (struct tok_index_t *)realloc_check(t->tok_index, i * sizeof(struct tok_index_t));
      t->tok_index_list_size = i;

      // convert string offset in dic to string pointers, required because use of qsort and bsearch
      for (i=0; i<t->tok_index_list_size; i++)
      {
        int dic_ofs = t->tok_index[i].tok_id;
        t->tok_index[i].str = &t->dic_tokens.buff[dic_ofs];  // set pointer to string
        t->tok_index[i].tok_id = i;                          // set id before sort
        t->tok_index[i].id_to_sort = 0;
      }
      
      // sort token list for bsearch
      sort_token_list(t);

      // set id_to_sort, used to get token string from token index in sorted table
      for (i=0; i<t->tok_index_list_size; i++)
      {
        CHECK(t->tok_index[t->tok_index[i].tok_id].id_to_sort == 0);
        t->tok_index[t->tok_index[i].tok_id].id_to_sort = i;
      }
    }
    else
    // --------------------------------
    // load merge strings
    if (js_cmp_key_list(h, "model.merges")) // is [ string list ]
    {
      // :model:merges[0] = "?ûü t"  ()
      do
      {
        char key_value[MAX_TOKEN_LENGTH];
        js_copy_key_value_str(h, key_value, sizeof(key_value));
        def_merge_string(key_value, t);
      }
      while (!inf.is_arr_end && js_read_param(h, &inf));  // read next

      // sort merge strings for bsearch
      sort_merge_list(t);
    }
  }

  js_close(h);

  // check all is loaded
  if (!chk_bpe)
    msg_info("tokenizer load: warning: BPE key not found");

  if (!t->merge.list_size)
    msg_error("tokenizer load: merge strings not found");

  if (!t->mode_ll3)    // byte fallback tokens definition expected + strip leading whitespace following bos
  {
    t->token_id_bos_ws = find_token_check("<s>");
    t->token_id_0x0  = find_token_check("<0x00>");
    t->token_id_0xff = find_token_check("<0xFF>");
    if ((t->token_id_0xff - t->token_id_0x0) != 0xff)
      msg_error("byte fallback token index error");
  }
}

#if 0
// ------------------------------------
// test code
int main(void)
{
  int i;
  const char *dic_str;
  struct tokenizer_t *t = &model.tokenizer;

  if (APP_ERROR())
    return -1;

#if 1
  model.config.load.tokenizer_name = "D:/llama2_st/70b-chat-hf/tokenizer.json";
  model.config.e_model_id = model_id_llama2;
#endif
#if 1
  model.config.load.tokenizer_name = "D:/llama3_st/8b-instruct/tokenizer.json";
  model.config.e_model_id = model_id_llama3;
#endif

  build_tokenizer();
  
  for (i=0; i<t->dic_tokens.n_strings; i++)
  {
    const char *str = tokenizer_get_token_str(i);
    int tok_id = tokenizer_find_token_id(str);
    CHECK(tok_id == i);
    CHECK(utf8_get_char_count(str) > 0);
  }

  dic_str = t->dic_tokens.buff;
  for (i=0; i<t->dic_tokens.n_strings; i++)
  {
    int tok_id = tokenizer_find_token_id(dic_str);
    const char *str = tokenizer_get_token_str(tok_id);
    CHECK(str == dic_str);
    dic_str += strlen(dic_str) + 1;
  }

  free_tokenizer();
  dbg_print_alloc();
  wait_return_exit();
}
#endif
