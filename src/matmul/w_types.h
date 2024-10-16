// weights data types
enum e_w_type
{
  w_type_f32 = 0,
  w_type_f16,
  w_type_bf16,
  w_type_sf16,
  w_type_f12,
  w_type_f8,
  w_type_COUNT,
};

// types sizeof
static const unsigned int w_type_sizeof[w_type_COUNT] = { 4, 2, 2, 2, 2, 1 };

// names of types (in matmul.c)
extern const char *w_type_name[w_type_COUNT];

// C types
typedef unsigned short f16_t;
typedef unsigned short bf16_t;
typedef unsigned short sf16_t;
typedef unsigned short f12_t;
typedef unsigned char f8_t;
