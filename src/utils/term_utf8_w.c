// UTF8 terminal for windows
#include <windows.h>
#include <wchar.h>
#include <conio.h>
#include "l_util.h"
#include "utf8.h"
#include "term_utf8.h"

// --------------------------
// terminal color functions

static HANDLE hConsole = INVALID_HANDLE_VALUE;   // console handle
static unsigned int last_color_set = 7;          // last set color index
static bool en_setcol = false;                   // can set color (init remap success)

// https://learn.microsoft.com/fr-fr/windows/win32/gdi/colorref
// format is 0x00bbggrr
#define _RGB(r, g, b) ((b << 16) | (g << 8) | r)

// define 4 user colors (max is 15, 0 is default background color)
static COLORREF user_col[4] =
{
  _RGB(220, 220, 220),
  _RGB(180, 255, 180),
  _RGB(180, 180, 255),
  _RGB(250, 250, 250)
};

// define a RGB color using "r.g.b" string, ex: "180.255.180"
int term_get_color(const char *col_str)
{
  const char *c = col_str;
  char *e;
  int r, g, b;
  if (*c)        
  { 
    r = strtol(c, &e, 0);
    if ((e > c) && (*e == '.'))
    {
      c = e+1; g = strtol(c, &e, 0);
      if ((e > c) && (*e == '.'))
      {
        c = e+1; b = strtol(c, &e, 0);
        if ((e > c) && !*e)
          return _RGB(r, g, b);
      }
    }
  }
  msg_error("invalid rgb color string '%s'\n", col_str);
}

// define user_col[col_id] RGB color, must be done before call to term_init()
void term_def_color(int col_id, int color)
{
  user_col[col_id & 3] = color;
}

// init user color table
void term_init(void)
{
  SetConsoleTitle(TEXT("llama st"));
  hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hConsole != INVALID_HANDLE_VALUE)
  {
    CONSOLE_SCREEN_BUFFER_INFOEX scr_buff;
    scr_buff.cbSize = sizeof(scr_buff);
    if (GetConsoleScreenBufferInfoEx(hConsole, &scr_buff))
    {
      int i;
      for (i=0; i<4; i++)
        scr_buff.ColorTable[i+1] = user_col[i];  // change default colors 1..4

      if (SetConsoleScreenBufferInfoEx(hConsole, &scr_buff))
      {
        en_setcol = true;
        return;
      }
    }
  }
  msg_info("windows console init failed.");      // then colors selection disabled
}

// select a text color [0..3]
void text_color(int col_id)
{
  col_id = col_id < 0 ? 7 : 1 + col_id;          // default console color = 7
  if ((col_id != last_color_set) && en_setcol)
  {
    SetConsoleTextAttribute(hConsole, col_id);
    last_color_set = col_id;
  }
}

// ensure cursor position at new line
void cursor_nl(void)
{
  CONSOLE_SCREEN_BUFFER_INFO bi;
  if (    (hConsole != INVALID_HANDLE_VALUE) 
       && GetConsoleScreenBufferInfo(hConsole, &bi))
  {
    if (bi.dwCursorPosition.X == 0)
      return;
    _putwch('\n');
  }
}

// wait for ms (debug usage)
void term_wait_ms(int ms)
{
  Sleep(ms);
}

// read a key without wait
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/getch-getwch?view=msvc-170
int read_key(void)
{
  if (_kbhit())
  {
    int k = _getch();
    if (!k || (k == 0xe0))
      k = _getch();         // 59..67 => F1..F8
    return k;
  }
  return 0;
}

// ------------------------------------

// print UTF8 string and manage cr as lf (printed as cr + lf)
bool print_utf8(const char *s)
{
  static int code = 0;
  while (*s)
  {
    int prev_code = code;
    int l = utf8_char_decode(s, &code);
    s += l;
    if (!l || (l==4))
      return false;              // bad s encoding or code > 16 bits

    // want to convert cr and cr+lf to single lf
    if ((code == '\n') && (prev_code == '\r'))
      continue;
    if (code == '\r')            // replace cr with lf
      _putwch('\n');
    else
      _putwch(code);             // display wide char
  }
  return true;
}

// print UTF8 string and display control chars code (debug/check usage)
void print_utf8_raw(const char *s)
{
  while (*s)
  {
    int code;
    int l = utf8_char_decode(s, &code);
    if (!l || (l==4))
    {
      msg_info("{utf8 %d}", l);  // inform if cannot encode
      return;
    }
    if (code < ' ')
    {
      msg_info("{%d}", code);
      if (code == 10)
        _putwch('\n');           // also emit lf for visibility
    }
    else
      _putwch(code);             // display wide char
    s += l;
  }
}

#define MAX_INP_LEN 2048

// keyboard input a string, return utf8 encoded buffer size
int kbd_input_utf8(char *s, int s_sizeof)
{
  char *s0 = s;
  wchar_t wstr[MAX_INP_LEN];     // key input wide buffer (16-bit wide encoded as UTF-16LE)
  size_t l_wstr;
  if (_cgetws_s(wstr, MAX_INP_LEN-1, &l_wstr) == 0)
  {
    char *s = s0;
    char *s_max = s0 + s_sizeof - 6;          // reserve for max utf8 char length + cr + 0 ending
    int i;
    for (i=0; (i<l_wstr) && (s < s_max); i++)
    {
      int l = utf8_char_encode(s, (unsigned int)wstr[i]);
      if (!l)                                 // cannot encode ?
        break;
      s += l;
    }
    if (i == l_wstr)                          // not truncated, no encode error
    {
      // *s++ = '\r';                         // add cr
      // *s++ = '\n';                         // add lf
      *s = 0;
      return (int)(s - s0);
    }
  }
  *s0 = 0;
  return -1;                                  // error occured
}

// sleep for ms time
void sleep_ms(int ms)
{
  Sleep(ms);
}

// --------------------------------------------------
// clipboard (chat menu), copy utf8 text to clipboard

#include "mem_alloc.h"

static struct
{
  wchar_t *text;
  int size;
  int size_alloc;
} clp_brd = { 0 };

// clear clipboard datas
void term_cb_clear(void)
{
  free_check(clp_brd.text);
  clp_brd.text = NULL;
  clp_brd.size = 0;
  clp_brd.size_alloc = 0;
}

// add utf8 string in clipboard
void term_cb_add_utf8(const char *s)
{
  int i, l = utf8_get_char_count(s);
  if ((clp_brd.size + l) >= clp_brd.size_alloc)
  {
    clp_brd.size_alloc = clp_brd.size + l + 32768;
    clp_brd.text = realloc_check(clp_brd.text, clp_brd.size_alloc * sizeof(wchar_t));
  }
  for (i=0; i<l; i++)
  {
    int code, l_utf8 = utf8_char_decode(s, &code);
    clp_brd.text[clp_brd.size++] = code;
    s += l_utf8;
  }
  clp_brd.text[clp_brd.size] = 0;
}

// copy text to clipboard
void term_cb_copy(void)
{
  int len = clp_brd.size;
  if (len && OpenClipboard(NULL))
  {
    HGLOBAL hglbCopy;
    EmptyClipboard();
    hglbCopy = GlobalAlloc(GMEM_MOVEABLE, (len+1) * sizeof(wchar_t)); 
    if (hglbCopy) 
    { 
      LPTSTR lptstrCopy = (LPTSTR)GlobalLock(hglbCopy);
      if (lptstrCopy)
      {
        memcpy(lptstrCopy, clp_brd.text, (len+1) * sizeof(wchar_t));
        GlobalUnlock(hglbCopy);
        SetClipboardData(CF_UNICODETEXT, hglbCopy);
      }
      GlobalFree(hglbCopy);
    }
    CloseClipboard();
  }
}

#if 0
// ------------------------------------
// test input string + echo
int main(void)
{
  term_init();
  print_utf8("system color.\n");
  text_color(0);
  print_utf8("Dialog started.. (use # + <return> on new line to exit.)\n");

  while (1)
  {
    char u8[1024];
    int u8_len;

    text_color(1);
    print_utf8("user:");
    u8_len = kbd_input_utf8(u8, sizeof(u8));
    if (!u8_len)
    {
      print_utf8("\ninput error.\n");
      continue;
    }

    if (u8_len > 1)
    {
      text_color(2);
      print_utf8("echo:");
      if (!print_utf8(u8))
      {
        print_utf8("\nprint utf8 error.\n");
        continue;
      }

      if ((u8[0] == '#') && (u8[1] == '\r'))
        break;
    }
  }
  text_color(-1);
  print_utf8("system color.\n");
  return 0;
}
#endif

#if 0
int main(void)
{
  hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hConsole != INVALID_HANDLE_VALUE)
  {
    int i;
    for (i=0; i<32; i++)
    {
      SetConsoleTextAttribute(hConsole, i);
      msg_info("color id %d\n", i);
    }
  }
  return 0;
}
#endif