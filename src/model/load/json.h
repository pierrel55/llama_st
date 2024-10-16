// json file read

// user info 
struct js_read_inf_t
{
  int lst_id;                      // into { key id
  int arr_id;                      // into [ value id, -1 if not array
  bool is_lev_end;                 // last read is end of list into { } bloc
  bool is_arr_end;                 // last read is end of an array into [ ] bloc
  bool file_eof;                   // last read is end of file
};

// define handle and init mem base
struct h_json_t *js_get_handle(char *mem, bool ext_mode);

// file read
struct h_json_t *js_load_file(const char *file_name, bool ext_mode);

// free mem
void js_close(struct h_json_t *h);

// reset state, seek to start of file
void js_seek_origin(struct h_json_t *h);

// param read
int js_read_param(struct h_json_t *h, struct js_read_inf_t *inf);

// key identifiers
char *js_copy_key_ident_str(const struct h_json_t *h, char *d, int d_size, int id);
bool js_cmp_key_ident(struct h_json_t *h, const char *key, int id);
char *js_get_key_ident_str_tmp(const struct h_json_t *h, int id);

// key value
char *js_copy_key_value_str(const struct h_json_t *h, char *d, int d_size);
char *js_get_key_value_str_alloc(const struct h_json_t *h);
char *js_get_key_value_str_tmp(const struct h_json_t *h);

// get key value as numeric
bool js_get_num_value_bool(const struct h_json_t *h);
int64_t js_get_num_value_i64(const struct h_json_t *h);
int js_get_num_value_i32(const struct h_json_t *h);
float js_get_num_value_f32(const struct h_json_t *h);

// find key in file
bool js_cmp_key_list(const struct h_json_t *h, const char *key_list);
bool js_find_key_list(struct h_json_t *h, const char *key_list);
void js_find_key_list_check(struct h_json_t *h, const char *key_list);

// find and read an array of integers, return array len, -1 if not found
int js_find_read_int_array_key_list(struct h_json_t *h, const char *key_list, int *arr, int arr_max);

// compare key value
bool js_cmp_key_value_str(struct h_json_t *h, const char *str);
void js_check_key_value_str(struct h_json_t *h, const char *str);

// ----------
// print util

void js_print_param(const struct h_json_t *h, const struct js_read_inf_t *inf);
void js_dump_file(const char *file_name, bool ext_mode);
