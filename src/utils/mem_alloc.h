// ------------------------------------
// memory allocation with check

void *malloc_check(size_t size);
void *calloc_check(size_t size);
void *realloc_check(void *ptr, size_t size);
void free_check(void *ptr);

// print currently allocated size
void dbg_print_alloc(void);

// alloc string
char *str_alloc(const char *str, int len);

#define VAR_ALLOC(var, typ, ne) typ *var = (typ *)malloc_check((ne)*sizeof(typ))
