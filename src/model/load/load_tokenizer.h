// find merge datas, return NULL if not found (code in load_tokenizer.c)
const struct merge_id_t *bpe_find_merge(int l_id, int r_id);

// load (private to tokenizer.c, defined in load_tokenizer.c)
void load_tokenizer(const char *file_name);
