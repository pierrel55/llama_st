#include <stdbool.h>
#include <string.h>
#include "utf8.h"

// ------------------------------------
// UTF8

// encode one char to utf8, return length, 0 if error
int utf8_char_encode(char *s, int code)
{
  // 7 bits  0bbb.bbbb
  if (code < (1 << 7))
  {
    s[0] = code;
    return 1;
  }
  // 5 + 6 bits  110b.bbbb 10bb.bbbb
  if (code < (1 << (5 + 6)))
  {
    s[0] = 0xc0 | (code >> 6);
    s[1] = 0x80 | (code & 0x3f);
    return 2;
  }
  // 4 + 6 + 6 bits  1110.bbbb 10bb.bbbb 10bb.bbbb
  if (code < (1 << (4 + 6 + 6)))
  {
    s[0] = 0xe0 | (code >> 12);
    s[1] = 0x80 | ((code >> 6) & 0x3f);
    s[2] = 0x80 | (code & 0x3f);
    return 3;
  }
  // 3 + 6 + 6 + 6 bits  1111.0bbb 10bb.bbbb 10bb.bbbb 10bb.bbbb
  if (code < (1 << (3 + 6 + 6 + 6)))
  {
    s[0] = 0xf0 | (code >> 18);
    s[1] = 0x80 | ((code >> 12) & 0x3f);
    s[2] = 0x80 | ((code >> 6) & 0x3f);
    s[3] = 0x80 | (code & 0x3f);
    return 4;
  }
  // code too big to encode
  return 0;
}

// return encoded value and length
int utf8_char_decode(const char *s, int *code)
{
  // 7 bits  0bbb.bbbb
  if (!(s[0] & 0x80))
  {
    *code = s[0];
    return 1;
  }
  // 5 + 6 bits  110b.bbbb 10bb.bbbb
  if ((s[0] & 0xe0) == 0xc0)
  {
    *code = ((s[0] & 0x1f) << 6) | (s[1] & 0x3f);
    if ((s[1] & 0xc0) == 0x80)   
      return 2;
  }
  // 4 + 6 + 6 bits  1110.bbbb 10bb.bbbb 10bb.bbbb
  if ((s[0] & 0xf0) == 0xe0)
  {
    *code = ((s[0] & 0xf) << 12) | ((s[1] & 0x3f) << 6) | (s[2] & 0x3f);
    if (((s[1] & 0xc0) == 0x80) && ((s[2] & 0xc0) == 0x80))
      return 3;
  }
  // 3 + 6 + 6 + 6 bits  1111.0bbb 10bb.bbbb 10bb.bbbb 10bb.bbbb
  if ((s[0] & 0xf8) == 0xf0)
  {
    *code = ((s[0] & 0x7) << 18) | ((s[1] & 0x3f) << 12) | ((s[2] & 0x3f) << 6) | (s[3] & 0x3f);
    if (((s[1] & 0xc0) == 0x80) && ((s[2] & 0xc0) == 0x80) && ((s[3] & 0xc0) == 0x80))
      return 4;
  }
  // invalid encoding
  *code = 0;
  return 0;
}

// return encoded length
int utf8_char_len(const char *s)
{
  if (!(s[0] & 0x80))
    return 1;
  if ((s[0] & 0xe0) == 0xc0)
  {
    if ((s[1] & 0xc0) == 0x80)
      return 2;
  }
  else
  if ((s[0] & 0xf0) == 0xe0)
  {
    if (((s[1] & 0xc0) == 0x80) && ((s[2] & 0xc0) == 0x80))
      return 3;
  }
  else
  if ((s[0] & 0xf8) == 0xf0)
  {
    if (((s[1] & 0xc0) == 0x80) && ((s[2] & 0xc0) == 0x80) && ((s[3] & 0xc0) == 0x80))
      return 4;
  }
  // invalid encoding
  return 0;
}

// return count of utf8 char coded in string, return 0 if coding error found
int utf8_get_char_count(const char *s)
{
  int char_count = 0;
  while (*s)
  {
    int l = utf8_char_len(s);
    if (!l) 
      return 0;                  // encoding error
    s += l;
    char_count++;
  }
  return char_count;
}

// convert text cr + lf or single lf to single cr
bool utf8_cvt_crlf_to_cr(char *s)
{
  char *d = s;
  while (*s)
  {
    int l = utf8_char_len(s);
    if (!l)
      break;                     // encoding error

    if (*s == 0x0d)              // CR + LF to CR
    {
      *d++ = 0x0d;
      s += (s[1] == 0x0a) ? 2 : 1;
    }
    else
    if (*s == 0x0a)              // LF to CR
    {
      *d++ = 0x0d;
      s++;
    }
    else
    while (l--)
      *d++ = *s++;
  }
  *d = 0;
  return !*s;
}

#if 0
// -----------------------------------------------
// test code
// https://fr.wikipedia.org/wiki/UTF-8
#include <stdio.h>

typedef struct
{
  int code;
  char *s;
  int len;
} ut_t;

// wiki page examples
const ut_t u_list[] = {
  { 159      , "\xC2\x9F", 2 },
  { 160      , "\xC2\xA0", 2 },
  { 191      , "\xC2\xBF", 2 },
  { 192      , "\xC3\x80", 2 },
  { 233      , "\xC3\xA9", 2 },
  { 2047     , "\xDF\xBF", 2 },
  { 2048     , "\xE0\xA0\x80", 3 },
  { 8364     , "\xE2\x82\xAC", 3 },
  { 55295 	 , "\xED\x9F\xBF", 3 },
  { 57344    , "\xEE\x80\x80", 3 },
  { 63743    , "\xEF\xA3\xBF", 3 },
  { 63744    , "\xEF\xA4\x80", 3 },
  { 64975    , "\xEF\xB7\x8F", 3 },
  { 64976    , "\xEF\xB7\x90", 3 },
  { 65007    , "\xEF\xB7\xAF", 3 },
  { 65008    , "\xEF\xB7\xB0", 3 },
  { 65533    , "\xEF\xBF\xBD", 3 },
  { 65534    , "\xEF\xBF\xBE", 3 },
  { 65535    , "\xEF\xBF\xBF", 3 },
  { 65536    , "\xF0\x90\x80\x80", 4 },
  { 119070   , "\xF0\x9D\x84\x9E", 4 },
  { 131069   , "\xF0\x9F\xBF\xBD", 4 },
  { 131070   , "\xF0\x9F\xBF\xBE", 4 },
  { 131071   , "\xF0\x9F\xBF\xBF", 4 },
  { 131072   , "\xF0\xA0\x80\x80", 4 },
  { 196605   , "\xF0\xAF\xBF\xBD", 4 },
  { 196606   , "\xF0\xAF\xBF\xBE", 4 },
  { 196607   , "\xF0\xAF\xBF\xBF", 4 },
  { 196608   , "\xF0\xB0\x80\x80", 4 },
  { 262141   , "\xF0\xBF\xBF\xBD", 4 },
  { 262142   , "\xF0\xBF\xBF\xBE", 4 },
  { 262143   , "\xF0\xBF\xBF\xBF", 4 },
  { 917504   , "\xF3\xA0\x80\x80", 4 },
  { 983037   , "\xF3\xAF\xBF\xBD", 4 },
  { 983038   , "\xF3\xAF\xBF\xBE", 4 },
  { 983039   , "\xF3\xAF\xBF\xBF", 4 },
  { 983040   , "\xF3\xB0\x80\x80", 4 },
  { 1048573  , "\xF3\xBF\xBF\xBD", 4 },
  { 1048574  , "\xF3\xBF\xBF\xBE", 4 },
  { 1048575  , "\xF3\xBF\xBF\xBF", 4 },
  { 1048576  , "\xF4\x80\x80\x80", 4 },
  { 1114109  , "\xF4\x8F\xBF\xBD", 4 },
  { 1114110  , "\xF4\x8F\xBF\xBE", 4 },
  { 1114111  , "\xF4\x8F\xBF\xBF", 4 },
  { 0, NULL, 0 } };

int main(void)
{
  int i;
  for (i=0; u_list[i].s; i++)
  {
    const ut_t *u = &u_list[i];
    int l, code;
    l = utf8_char_len(u->s);
    if (l != u->len)
      break;
    l = utf8_char_decode(u->s, &code);
    if (l != u->len)
      break;
    if (code != u->code)
      break;
  }
  if (u_list[i].s)
    printf("decode failed.\n");

  // check encode
  for (i=0; i < (1 << (3 + 6 + 6 + 6)); i++)
  {
    char s[8];
    int code;
    int le = utf8_char_encode(s, i);
    int ld = utf8_char_decode(s, &code);
    if (!le || (ld != le) || (code != i))
    {
      printf("encode failed.\n");
      break;
    }
  }
}
#endif