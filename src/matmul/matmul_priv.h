// common private header for matmul/conversion code
#define N_64K (1 << 16)

#define ABS_F16(x) ((x) & 0x7FFF)      // abs for F16/BF16/SF16, clear sign bit

// --------------------------------------
// data conversion to float 32 functions

extern const cvt_f16_to_f32_t cvt_f16_to_f32_procs[simd_n];
extern const cvt_bf16_to_f32_t cvt_bf16_to_f32_procs[simd_n];
extern const cvt_sf16_to_f32_t cvt_sf16_to_f32_procs[simd_n];

// --------------------------------------
// vector to matrix multiply functions

extern const matmul_f32_f32_t matmul_f32_f32_procs[simd_n];
extern const matmul_f32_f16_t matmul_f32_f16_procs[simd_n];
extern const matmul_f32_bf16_t matmul_f32_bf16_procs[simd_n];
extern const matmul_f32_sf16_t matmul_f32_sf16_procs[simd_n];
extern const matmul_f32_f12_t matmul_f32_f12_procs[simd_n];
extern const matmul_f32_f8_t matmul_f32_f8_procs[simd_n];

// --------------------------------------
// SF16 conversions, code in matmul_sf16.c

void init_conv_sf16(void);
void cvt_f16_to_sf16(sf16_t *sf16, const f16_t *f16, size_t ne);

// --------------------------------------
// F16 conversions, code in matmul_f16.c

void init_sw_f16c(void);
void free_sw_f16c(void);
void cvt_f32_to_f16(f16_t *f16, const float *f32, size_t ne);

// --------------------------------------
// F12 conversions, code in matmul_f12.c

void init_conv_f12(void);
void cvt_f16_to_f12(f12_t *f12, const f16_t *f16, size_t ne);
void cvt_bf16_to_f12(f12_t *f12, const f16_t *f16, size_t ne);

// --------------------------------------
// F8 conversions, code in matmul_f8.c

void init_conv_f8(void);
void cvt_f16_to_f8(f8_t *f8, const f16_t *f16, size_t ne);
void cvt_bf16_to_f8(f8_t *f8, const bf16_t *bf16, size_t ne);