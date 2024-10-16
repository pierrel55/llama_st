// ------------------------------------
// UTF8

// encode one char to utf8, return length, 0 if error
int utf8_char_encode(char *s, int code);

// return char encoded code value and length
int utf8_char_decode(const char *s, int *code);

// return char encoded length and test if coding is valid
int utf8_char_len(const char *s);

// return count of utf8 char coded in string, return 0 if coding error found
int utf8_get_char_count(const char *s);

// text convert cr + lf or lf alone to cr
bool utf8_cvt_crlf_to_cr(char *s);
