// UTF8 terminal

// define a RGB color using "r.g.b" string, ex: "180.255.180"
int term_get_color(const char *col_str);

// define user_col[col_id] RGB color, must be done before call to term_init()
void term_def_color(int col_id, int color);

// set print color
void text_color(int col_id);

// single init, required if text_color() used
void term_init(void);

// wait for ms (debug usage)
void term_wait_ms(int ms);

// print UTF8 string and manage cr alone as cr + lf
bool print_utf8(const char *s);

// print UTF8 string and display control chars code (debug/check usage)
void print_utf8_raw(const char *s);

// ensure cursor position at new line
void cursor_nl(void);
void cursor_nl_set(void);

// keyboard input a string, return utf8 encoded buffer size
int kbd_input_utf8(char *s, int s_sizeof);

// read a key without wait
int read_key(void);

// sleep for ms time
void sleep_ms(int ms);

// clipboard (chat menu), copy utf8 text to clipboard
void term_cb_clear(void);
void term_cb_add_utf8(const char *utf8);
void term_cb_copy(void);
