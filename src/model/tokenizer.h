// strings dictionnary
struct str_dic_t
{
  char *buff;                          // all strings 0 ended concatenated
  int sz_alloc;                        // buff alloc size in bytes
  int wr_ofs;                          // write offset in buff
  int n_strings;                       // count of strings in buff
};

// token
struct tok_index_t
{
  const char *str;                     // to string in tokenizer_t dic_tokens
  int tok_id;                          // origin tokenizer index (non sorted list)
  int id_to_sort;                      // origin index to sorted position (tok_id to str convert)
};

// merge tokens id
struct merge_id_t
{
  int tok_id_l;                        // left token id
  int tok_id_r;                        // right token id
  int tok_id_m;                        // merged token id
  int merge_id;                        // id in merge list
};

// token list element used by tokenizer encode
struct m_tok_t
{
  int score;                           // merge score with righ token
  int tok_id;                          // token id
  int tok_id_m;                        // token id if merge with righ token
};

// merge tokens id list
struct mt_list_t
{
  struct m_tok_t *mt;
  int n_list;
  int n_alloc;
};

// BPE tokenizer datas
struct tokenizer_t
{
  bool mode_ll3;                       // llama3 tokenizer mode

  struct str_dic_t dic_tokens;         // token string list
  struct tok_index_t *tok_index;       // token id list
  int tok_index_list_size;
  
  // merge list
  struct
  {
    struct merge_id_t *id_list;        // merge tokens id
    int list_size;
    int n_alloc;
  } merge;

  int id_special_base;                 // start index of special tokens not in model.vocab list
  int id_special_last;                 // last special tokens index
  int id_special_count;                // count of special tokens

  // byte fallback + strips leading whitespace specific (see PR #89), active if not mode_ll3
  int token_id_bos_ws;                 
  int token_id_0x0;                    // <0x00> byte fallback
  int token_id_0xff;                   // <0xff> byte fallback

  // tokenizer_encode token list result
  struct mt_list_t mt_list;
};

// find a token id from utf8 string, return -1 if not found
int tokenizer_find_token_id(const char *str);

// find a special token from string and check is special token
int tokenizer_find_sp_token_id(const char *str);

// return token string from token index
const char *tokenizer_get_token_str(int token_id);

// encode text, define mt_list token list
void tokenizer_encode(const char *text);

// return decoded token string.
const char *tokenizer_decode(int token_id);

// decode and print
void tokenizer_decode_print(int token_id, bool disp_raw);

// load and init tokenizer from .json file
void build_tokenizer(void);

// free allocated mem
void free_tokenizer(void);
