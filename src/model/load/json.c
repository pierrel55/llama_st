// minimal json parser limited to requested features to import safetensors and tokenizer datas.
#include <stdlib.h>
#include <stdarg.h>
#include "l_util.h"
#include "mem_alloc.h"
#include "json.h"

#define JS_MAX_KEY_LEVEL 8         // max key level

// string pointer with len
typedef struct
{
  const char *s;                   // warning, not 0 terminated
  int len;                         // len
  bool coated;
} lstr_t;

// parse buffer
struct s_t
{
  char *origin;                    // alloc origin
  const char *p;                   // current parse pointer
  bool ext_mode;                   // allow comments and string concat option
  int pad;                         // padding
};

// handle
struct h_json_t
{
  struct s_t s;                    // parse buffer
  lstr_t key[JS_MAX_KEY_LEVEL];    // key names
  bool is_arr[JS_MAX_KEY_LEVEL];   // into [ ] array flag
  int ne_lst[JS_MAX_KEY_LEVEL];    // key ctr in { level
  int ne_arr[JS_MAX_KEY_LEVEL];    // key ctr in [ level
  int key_lvl;                     // '{' level
  int key_len;
  lstr_t value;                    // key value
};

// ------------------------------------
// error

// json parse error message, show line and stop position
static void json_err(struct s_t *s, const char *err)
{
  // rewind and format message to display (remove multi spaces and CR/LF)
  char msg[50];
  const char *r = --s->p;
  char *w = &msg[sizeof(msg)-1];
  int ln = 1;
  *w = 0;
  for (;(w > msg) && (r >= s->origin); r--)
  {
    if      ((unsigned char)*r > ' ') *(--w) = *r;
    else if ((unsigned char)*w > ' ') *(--w) = ' ';
  }
  // find line number
  for (r = s->origin; r < s->p; r++)
    if (*r == '\n') ln++;
  msg_error("json: interpret failed (%s) line %d: ..%s", err, ln, w);
}

// ------------------------------------
// string parse utils

// pass spaces, cr, lf, tab, comments
static void pass_spc(struct s_t *s)
{
  while (*s->p)
  {
    if ((unsigned char)*s->p <= ' ')
    {
      s->p++;
      continue;
    }

    if (!s->ext_mode)         // strict json, no comments allowed
      break;

    if (*s->p != '/')         // cannot be a comment start
      break;

    if (s->p[1] == '/')       // '//' comment
    {
      while ((unsigned char)*s->p >= ' ')
        s->p++;
      continue;
    }

    if (s->p[1] == '*')       // /* */ comment begin
    {
      // find '*/' comment end
      while (1)
      {
        if (!*s->p)           // reach eof
          json_err(s, "unterminated /* */ comment");
        if ((*s->p == '*') && (s->p[1] == '/'))
          break;
        s->p++;
      }
      s->p += 2;
      continue;
    }
    break;                    // '/?' ?
  }
}

// test char and move to next if match
static bool test_char_pass(struct s_t *s, char c)
{
  if (*s->p != c)
    return false;
  s->p++;
  pass_spc(s);
  return true;
}

// ------------------------------------

// define handle and init mem base
// if ext_mode define, allow comments and string concat option.
struct h_json_t *js_get_handle(char *mem, bool ext_mode)
{
  struct h_json_t *h = (struct h_json_t *)calloc_check(sizeof(struct h_json_t));
  h->s.origin = mem;
  h->s.ext_mode = ext_mode;
  return h;
}

// open file, alloc buffer and pass first '{', return file buffer
struct h_json_t *js_load_file(const char *file_name, bool ext_mode)
{
  char *mem;
  file_t file = { 0 };
  f_open(&file, file_name, "rb");
  mem = (char *)malloc_check(file.size+1);
  f_read(mem, file.size, &file);
  mem[file.size] = 0;
  f_close(&file);
  return js_get_handle(mem, ext_mode);
}

// free mem
void js_close(struct h_json_t *h)
{
  free_check(h->s.origin);
  free_check(h);
}

// reset state, seek to start of file
void js_seek_origin(struct h_json_t *h)
{
  unsigned char *s = h->s.origin; 
  h->s.p = h->s.origin;                // reset to origin
  // test if utf-8 signature EF/BB/BF is found and pass it
  if ((s[0] == 0xEF) && (s[1] == 0xBB) && (s[2] == 0xBF))
    h->s.p += 3;
  pass_spc(&h->s);
  if (*h->s.p != '{')                  // ensure will enter level 0
    json_err(&h->s, "{ expected");
  h->key_lvl = -1;
  h->key_len = 0;
  h->value.s = NULL;
  h->value.len = 0;
}

// get identifier or value string
static void get_js_str(lstr_t *lstr, struct s_t *s)
{
  if (*s->p == '\"')                   // coated string
  {
    lstr->coated = true;
    lstr->s = ++s->p;                  // pass first "
    while (1)
    {
      if (!*s->p)
        json_err(s, "\" expected");    // unterminated string
      if (*s->p == '\"')               // found string end
        break;
      if ((*s->p == '\\') && s->p[1])  // pass backslash + next char
        s->p += 2;
      else
        s->p++;
    }
    lstr->len = (int)(s->p++ - lstr->s);
  }
  else                                 // non coated string
  {
    char c;
    lstr->coated = false;
    lstr->s = s->p;
    while ((c = *s->p) && (     ((c >= '0') && (c <= '9')) || (c == '.') || (c == '-')
                             || ((c >= 'a') && (c <= 'z')) ))
      s->p++;
    lstr->len = (int)(s->p - lstr->s);
  }
  pass_spc(s);
}

// read a key in buffer, return false if eof reached
int js_read_param(struct h_json_t *h, struct js_read_inf_t *inf)
{
  struct s_t *s = &h->s; 
  inf->is_lev_end = false;
  inf->is_arr_end = false;

  if (!*s->p)
    return false;                      // eof

  // { "ident":
  if (test_char_pass(s, '{'))          // test for '{'
  {
    if (h->key_lvl == JS_MAX_KEY_LEVEL)
      json_err(s, "too many '{'");
    h->key_lvl++;
    h->ne_lst[h->key_lvl] = 0;
  }

  if (!h->is_arr[h->key_lvl])          // is values list
  {
    lstr_t *key;
    h->ne_lst[h->key_lvl]++;

    // get key name
    key = &h->key[h->key_lvl];

    get_js_str(key, s);
    if (!key->len || !key->coated)
      json_err(s, "key name expected"); // key name empty or not coated

    // pass ':'
    if (!test_char_pass(s, ':'))
      json_err(s, ": expected");

    // array
    if (test_char_pass(s, '['))        // is array
    {
      h->is_arr[h->key_lvl] = true;
      h->ne_arr[h->key_lvl] = 0;
    }

    // sub object '{'
    if (*s->p == '{')
      return js_read_param(h, inf);    // recurse
  }

  get_js_str(&h->value, s);            // get value

#if 0
  // empty key value allowed only in extended mode
  if (!h->value.len && !(h->s.ext_mode && h->value.coated))
#else
  // empty not coated (qwen use empty strings in tokenizer.json)
  if (!h->value.len && !h->value.coated)
#endif
    json_err(s, "empty key value");

  // extended mode concat string option ".." + ".."
  if ((*s->p == '+') && h->value.coated && s->ext_mode)
  {
    const char *s0 = h->value.s;       // save origin
    int l = h->value.len;              // 1st part len
    while (test_char_pass(s, '+') && (*s->p == '\"'))
    {
      get_js_str(&h->value, s);
      l += h->value.len;
    }
    h->value.s = s0;
    h->value.len = l;                  // is concatained len
  }
  
  // update index
  if (h->is_arr[h->key_lvl])
    h->ne_arr[h->key_lvl]++;

  // set user inf
  inf->lst_id = h->ne_lst[h->key_lvl] - 1;
  inf->arr_id = h->ne_arr[h->key_lvl] - 1;
  
  // move to next element, set information
  h->key_len = h->key_lvl;             // returned value len
  while (*s->p && (h->key_lvl >= 0))
  {
    if (test_char_pass(s, ','))
      return true;

    if (test_char_pass(s, '}'))
    {
      h->key_lvl--;
      inf->is_lev_end = true;
    }
    else
    if (test_char_pass(s, ']'))
    {
      if (!h->is_arr[h->key_lvl])      // ] ]
        json_err(s, "unexpected ']'");
      h->is_arr[h->key_lvl] = false;
      h->ne_arr[h->key_lvl] = 0;
      inf->is_arr_end = true;
    }
    else
      json_err(s, "unexpected char");  // unexpected char
  }

  // check parse states
  if (*s->p || (h->key_lvl != -1))
    json_err(s,  "missing '}'");
  inf->file_eof = true;
  return true;
}

// ----------------------------------
// read key ident and value

// copy lstr_t to str, check size and convert backslash sequences.
static int ls_cpy(char *d, int d_size, const lstr_t *ls)
{
  char *d0 = d;
  char *d_end = d + d_size - 1;
  const char *s = ls->s;
  const char *s_end = ls->s + ls->len;
  while (s < s_end)
  {
    if (d == d_end)
      msg_error("json: ls_cpy: d_size too small");
    if (*s == '\\')
    {
            if (s[1] == 'r')  *d = '\r';
      else  if (s[1] == 'n')  *d = '\n';
      else  if (s[1] == 't')  *d = '\t';  // tab
      else  if (s[1] == '\"') *d = '\"';
      else  if (s[1] == '\\') *d = '\\';
      else  if (s[1] == 'f')  *d = '\f';  // form feed
      else  if (s[1] == 'b')  *d = '\b';  // back space
      else  if (s[1] == 'u')  // ex "\u0001"  (\u followed by four-hex-digits)
      {
        char *e;
        int n = strtol(s+2, &e, 16);
        if (((e - ls->s) != 6) || (n > 255))
          msg_error("json: ls_cpy: \\u invalid hex number");
        *d = (unsigned char)n;
        s += 4;
      }
      else
        msg_error("json: ls_cpy: unsupported backslash sequence: '\\%c'", s[1]);
      d++;
      s += 2;
    }
    else
      *d++ = *s++;
  }
  *d = 0;
  return (int)(d - d0);
}

// ------------------
// key name

// get key identifier string at level id
char *js_copy_key_ident_str(const struct h_json_t *h, char *d, int d_size, int id)
{
  if ((unsigned int)id > (unsigned int)(h->key_len + 1))
    msg_error("json: js_copy_key_ident_str: invalid id");
  ls_cpy(d, d_size, &h->key[id]);
  return d;
}

// compare key[level id] with key (basic, require not fragmented without backslash in string)
bool js_cmp_key_ident(struct h_json_t *h, const char *key, int id)
{
  const lstr_t *ls = &h->key[id];
  if ((unsigned int)id > (unsigned int)(h->key_len + 1))
    msg_error("json: js_cmp_key_ident: invalid id");
  return (!memcmp(key, ls->s, ls->len) && !key[ls->len]);
}

// get current key ident temporary string, (print message usage only)
char *js_get_key_ident_str_tmp(const struct h_json_t *h, int id)
{
  static char tmp[256];
  return js_copy_key_ident_str(h, tmp, sizeof(tmp), id);
}

// ------------------
// key value

// get current key value string
char *js_copy_key_value_str(const struct h_json_t *h, char *d, int d_size)
{
  if (h->s.ext_mode && h->value.coated)   // may be '+' fragmented, need to re-parse
  {
    int ls = 0;
    int ld = 0;
    struct s_t s = h->s;
    *d = 0;                               // define 0 end if 0 sized strings ("")
    s.p = h->value.s - 1;                 // seek to first "
    while (ls != h->value.len)
    {
      lstr_t lstr;
      get_js_str(&lstr, &s);
      ld += ls_cpy(d + ld, d_size - ld, &lstr);
      ls += lstr.len;
      test_char_pass(&s, '+');
    }
  }
  else
    ls_cpy(d, d_size, &h->value);
  return d;
}

// get current key value as allocated 0 terminated string
char *js_get_key_value_str_alloc(const struct h_json_t *h)
{
  char *d = (char *)malloc_check(h->value.len + 1);
  return js_copy_key_value_str(h, d, h->value.len + 1);
}

// get current key value temporary string, (print message usage only)
char *js_get_key_value_str_tmp(const struct h_json_t *h)
{
  static char tmp[256];
  return js_copy_key_value_str(h, tmp, sizeof(tmp));
}

// ----------------------------------
// convert string values to numeric

bool js_get_num_value_bool(const struct h_json_t *h)
{
  const lstr_t *ls = &h->value;
  if (!ls->coated)                 // reserved for identifiers
  {
    if ((ls->len == 4) && !memcmp(ls->s,  "true", ls->len))
      return true;
    if ((ls->len == 5) && !memcmp(ls->s, "false", ls->len))
      return false;

    // in extended mode also allow 0/1 usage instead of false/true
    if (h->s.ext_mode)
    {
      int b = js_get_num_value_i32(h);
      if (b == 1)
        return true;
      if (b == 0)
        return false;
    }
  }
  msg_error("json: js_get_num_value_bool");
}

int64_t js_get_num_value_i64(const struct h_json_t *h)
{
  const lstr_t *ls = &h->value;
  if (!ls->coated)
  {
    char *e;
    int64_t n = _strtoi64(ls->s, &e, 10);
    if ((e - ls->s) == ls->len)  // check full string used
      return n;
  }
  msg_error("json: js_get_num_value_i64");
}

int js_get_num_value_i32(const struct h_json_t *h)
{
  const lstr_t *ls = &h->value;
  if (!ls->coated)
  {
    char *e;
    int64_t n = _strtoi64(ls->s, &e, 10);
    if (n != (int)n)
      msg_error("json: js_get_num_value_i32 (int32 ovf)");
    if ((e - ls->s) == ls->len)
       return (int)n;
  }
  msg_error("json: js_get_num_value_i32");
}

float js_get_num_value_f32(const struct h_json_t *h)
{
  const lstr_t *ls = &h->value;
  if (!ls->coated)
  {
    char *e;
    float n = (float)strtod(ls->s, &e);
    if ((e - ls->s) == ls->len)
      return n;
  }
  msg_error("json: js_get_num_value_f32");
}

// ----------------------------------
// compare key names

// compare if key list match.
// list is encoded using '.' separator, ex: a.b.c.d
bool js_cmp_key_list(const struct h_json_t *h, const char *key_list)
{
  int i;
  for (i=0; i<=h->key_len; i++)
  {
    const lstr_t *ls = &h->key[i];
    const char *s = ls->s;
    const char *s_max = ls->s + ls->len;
    while ((s < s_max) && (*s == *key_list)) { s++; key_list++; }
    if (!*key_list)
      return s == s_max;
    if (*key_list != '.')
      return false;
    key_list++;
  }
  return true;
}

// find key value list in file, check uniqueness (avoid usage on big files, is slow)
bool js_find_key_list(struct h_json_t *h, const char *key_list)
{
  struct js_read_inf_t inf = { 0 };
  lstr_t key_save = { 0 };
  bool key_found = false;
  js_seek_origin(h);

  while (js_read_param(h, &inf))
  {
    if (js_cmp_key_list(h, key_list))
    {
      if (key_found)
        msg_error("json key_list \"%s\" duplicated", key_list);
      key_save = h->value;
      key_found = true;
    }
  }
  h->value = key_save;             // save found to handle
  return key_found;
}

// find key value in file, produce error if not found
void js_find_key_list_check(struct h_json_t *h, const char *key)
{
  if (!js_find_key_list(h, key))
    msg_error("json key \"%s\" not found", key);
}

// find and read an array of integers, return array len, -1 if not found
int js_find_read_int_array_key_list(struct h_json_t *h, const char *key_list, int *arr, int arr_max)
{
  struct js_read_inf_t inf = { 0 };
  js_seek_origin(h);
  while (js_read_param(h, &inf))
  {
    if (js_cmp_key_list(h, key_list))
    {
      int i = 0;
      do
      {
        if (inf.arr_id < 0)
          msg_error("json \"%s\" is not an array", key_list);
        if (i >= arr_max)
          msg_error("json \"%s\" dest array too small", key_list);
        arr[i++] = js_get_num_value_i32(h);
      } while (!inf.is_arr_end && js_read_param(h, &inf));
      return i;
    }
  }
  return -1;
}

#if 0
// reserved for future use
enum e_js_num_type
{
  js_num_bool = 0,
  js_num_i32,
  js_num_i64,
  js_num_f32,
};

static const char js_num_type_sz[4] = { sizeof(bool), sizeof(int), sizeof(int64_t), sizeof(float) };

// get a variable type numeric value
void js_get_num_value(const struct h_json_t *h, void *p_value, enum e_js_num_type num_type)
{
  if      (num_type == js_num_bool) *(bool    *)p_value = js_get_num_value_bool(h);
  else if (num_type == js_num_i32)  *(int     *)p_value = js_get_num_value_i32(h);
  else if (num_type == js_num_i64)  *(int64_t *)p_value = js_get_num_value_i64(h);
  else if (num_type == js_num_f32)  *(float   *)p_value = js_get_num_value_f32(h);
  else
    msg_error("js_get_num_value: undefined type %d", num_type);
}

// find and read an array of numeric values, return array len, -1 if not found
int js_find_read_array_key_list(struct h_json_t *h, const char *key_list, void *arr, int arr_max, enum e_js_num_type num_type)
{
  struct js_read_inf_t inf = { 0 };
  js_seek_origin(h);

  while (js_read_param(h, &inf))
  {
    if (js_cmp_key_list(h, key_list))
    {
      int i = 0;
      char *p = (char *)arr;
      do
      {
        if (inf.arr_id < 0)
          msg_error("json \"%s\" is not an array", key_list);
        if (i >= arr_max)
          msg_error("json \"%s\" dest array too small", key_list);
        js_get_num_value(h, p, num_type);
        p += js_num_type_sz[num_type];
        i++;
      } while (!inf.is_arr_end && js_read_param(h, &inf));
      return i;
    }
  }
  return -1;
}
#endif

// -------------------------------------------

// compare key value string. (basic, require not fragmented without backslash in string)
bool js_cmp_key_value_str(struct h_json_t *h, const char *str)
{
  const lstr_t *ls = &h->value;
  return (!memcmp(str, ls->s, ls->len) && !str[ls->len]);
}

// check key value match with the one expected
void js_check_key_value_str(struct h_json_t *h, const char *str)
{
  if (!js_cmp_key_value_str(h, str))             // missmatch
  {
    char tmp[1024];
    ls_cpy(tmp, sizeof(tmp), &h->value);
    msg_error("js_check_key_value_str: expected key value is %s, found %s", str, tmp);
  }
}

// -------------------------------------------
// print/debug

// print param
void js_print_param(const struct h_json_t *h, const struct js_read_inf_t *inf)
{
  int i;
  char tmp[1024];
  if (inf)
    msg_info("(%d) ", inf->lst_id);              // key index
  for (i=0; i<=h->key_len; i++)
    msg_info(":%s", (ls_cpy(tmp, sizeof(tmp), &h->key[i]), tmp));
  if (inf && (inf->arr_id >= 0))                 // if array, print index
    msg_info("[%d]", inf->arr_id);
  msg_info(h->value.coated ? " = \"%s\"" : " = %s", (ls_cpy(tmp, sizeof(tmp), &h->value), tmp));
  if (inf)
  {
    msg_info("  (");
    if (inf->is_lev_end) msg_info("le");
    if (inf->is_arr_end) msg_info(" ae");
    if (inf->file_eof) msg_info(" eof");
    msg_info(")");
  }
  msg_info("\n");
}

// dump file content (identify keys values)
void js_dump_file(const char *file_name, bool ext_mode)
{
  struct js_read_inf_t inf = { 0 };
  struct h_json_t *h = js_load_file(file_name, ext_mode);
  js_seek_origin(h);

  // print file content
  while (js_read_param(h, &inf))
    js_print_param(h, &inf);

  js_close(h);
}

#if 0
// -------------------------------------------
// test code

int main(void)
{
  if (APP_ERROR())                 // catch error return point
    return -1;

  //js_dump_file("_config.json");
  //js_dump_file("_model.safetensors.index.json");
  //js_dump_file("_hdr.json");
  js_dump_file("_tokenizer_trunc.json");
  //js_dump_file("_tokenizer.json");

  dbg_print_alloc();
  wait_return_exit();
}
#endif
