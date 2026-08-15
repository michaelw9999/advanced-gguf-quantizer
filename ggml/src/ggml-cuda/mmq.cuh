#pragma once

#include "common.cuh"
#include "vecdotq.cuh"
#include "mma.cuh"

#include <climits>
#include <cstdint>

using namespace ggml_cuda_mma;

#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
#define MMQ_ITER_K             256
#define MMQ_ITER_K_FP4         512
#define MMQ_NWARPS               8

#ifndef GGML_CUDA_NVFP4_FP8_EXACT_SUB16
#define GGML_CUDA_NVFP4_FP8_EXACT_SUB16 1
#endif

#ifndef GGML_CUDA_MXFP6_MMQ_KGROUP
#define GGML_CUDA_MXFP6_MMQ_KGROUP 4
#endif

#ifndef GGML_CUDA_MXFP6_FP8_MMQ_KGROUP
#define GGML_CUDA_MXFP6_FP8_MMQ_KGROUP 3
#endif

typedef void (*load_tiles_mmq_t)(const char * __restrict__ x, int * x_tile, const int kbx0, const int i_max, const int stride);
typedef void (*vec_dot_mmq_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00);
typedef void (*mmq_write_back_t)(const float * __restrict__ sum, const int32_t * __restrict__ get_rows_to_sorted,
    float * __restrict__ dst, const int stride, const int i_max, const int j_max);

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4*QK8_1]; // 128 values quantized to 8 bit each
};

// this struct is used for fp4 data types (currently only used for Blackwell)
// mxfp4 has block size 32, each int32 of d4 contains 2 e8m0 scales in the lower 16 bits
// nvfp4 has block size 16, each int32 of d4 contains 4 ue4m3 scales
struct block_fp4_mmq {
    uint32_t d4[4];       // 8 E8M0 scales (1 per 32 values), 2 packed per uint32: d4[0]={s0,s1}, d4[1]={s2,s3}, etc.
    int8_t   qs[4 * 32];  // 256 FP4 values packed as 4-bit pairs (2 per byte)
};
struct block_mxfp8_mmq {
    uint32_t d4[4];                // 4 E8M0 scales per 128 activation values
    uint8_t  qs[4 * QK8_1];       // 128 E4M3 values
};
struct block_nvfp4_mmq {
    uint32_t sc4_u32[4];  // 16 UE4M3 scales, one per 16-value subblock.
    uint32_t qs_u32[32];  // 256 E2M1 values packed as 4-bit pairs.
};
struct tile_mxfp6_e2m3_mmq {
    uint32_t qs_u32[QK_MXFP6_E2M3_FRAGS][QK_MXFP6_E2M3_PTX_BYTES/4]; // Direct FP6 PTX byte containers for B operand loads.
    uint8_t  e[QK_MXFP6_E2M3_FRAGS];                             // One UE8M0 scale per K32 activation fragment.
    uint8_t  pad[3];                                        // Keep MMQ activation blocks int-strided.
};

static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_fp4_mmq)  == sizeof(block_q8_1_mmq),    "Unexpected block_fp4_mmq size");
static_assert(sizeof(block_mxfp8_mmq) == sizeof(block_q8_1_mmq),   "Unexpected block_mxfp8_mmq size");
static_assert(sizeof(block_nvfp4_mmq) == sizeof(block_q8_1_mmq),   "Unexpected block_nvfp4_mmq size");
static_assert(QK_MXFP6_E2M3_PTX_BYTES % 4 == 0,                   "MXFP6_E2M3 PTX bytes must be u32-loadable");
static_assert(sizeof(tile_mxfp6_e2m3_mmq) == QK_MXFP6_E2M3_FRAGS*QK_MXFP6_E2M3_PTX_BYTES + QK_MXFP6_E2M3_FRAGS + 3, "Unexpected tile_mxfp6_e2m3_mmq size");
static_assert(sizeof(tile_mxfp6_e2m3_mmq) == 36,                        "Unexpected tile_mxfp6_e2m3_mmq size");
static_assert(sizeof(tile_mxfp6_e2m3_mmq) % sizeof(int) == 0,          "tile_mxfp6_e2m3_mmq must be int-strided");


static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ggml_type type_x) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q5_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q5_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q8_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_MXFP4:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_MXFP8:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_MXFP6_E2M3:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_NVFP4:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q2_K:
            return MMQ_Q8_1_DS_LAYOUT_D2S6;
        case GGML_TYPE_Q3_K:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_IQ1_S:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

static int get_mmq_x_max_host(const int cc) {
    return (turing_mma_available(cc) || amd_wmma_available(cc)) ? 128 :
        GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA ?
#ifdef GGML_CUDA_FORCE_MMQ
            128                     : 64;
#else
            MMQ_DP4A_MAX_BATCH_SIZE : 64;
#endif // GGML_CUDA_FORCE_MMQ
}

static constexpr __device__ int get_mmq_x_max_device() {
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    return 128;
#else // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

#if defined(GGML_USE_HIP)
    return 64;
#else // defined(GGML_USE_HIP)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#ifdef GGML_CUDA_FORCE_MMQ
    return 128;
#else // GGML_CUDA_FORCE_MMQ
    return MMQ_DP4A_MAX_BATCH_SIZE;
#endif // GGML_CUDA_FORCE_MMQ
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 64;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA

#endif // defined(GGML_USE_HIP)
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

static int get_mmq_y_host(const int cc) {
    return GGML_CUDA_CC_IS_AMD(cc) ? (GGML_CUDA_CC_IS_RDNA1(cc) ? 64 : 128) :
        ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ? 128 : 64);
}

static constexpr __device__ int get_iter_k([[maybe_unused]] const ggml_type type) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    return type == GGML_TYPE_MXFP4 ? MMQ_ITER_K_FP4 : MMQ_ITER_K;

#endif // defined(BLACKWELL_MMA_AVAILABLE)
    return MMQ_ITER_K;
}

static constexpr __device__ int get_mmq_y_device() {
#if defined(GGML_USE_HIP)
#if defined(RDNA1)
    return 64;
#else
    return 128;
#endif // defined RDNA1
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 128;
#else
    return 64;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // defined(GGML_USE_HIP)
}

// Decouple shared memory tile sizes from WARP_SIZE to allow for different warp sizes.
// The K dimension of the tiles has either,
// 1*MMQ_TILE_NE_K==32 (always for TILE_Y_K) or 2*MMQ_TILE_NE_K==64 (typically for TILE_X_K),
// 32 bit elements for the quantized data (does not include scales).
// In other words, the size of the quantized data in the K dimension is a multiple of MMQ_TILE_NE_K.
// The final tile size in K direction is padded to avoid shared memory bank conflicts,
// in terms of 32 bit elements that means K % 2 == 1 for dp4a or K % 8 == 4 for mma.
#define MMQ_TILE_NE_K 32

#define MMQ_DP4A_TXS_Q4_0    tile_x_sizes{mmq_y*MMQ_TILE_NE_K   + mmq_y, mmq_y*MMQ_TILE_NE_K/QI4_0   + mmq_y/QI4_0,     0}
#define MMQ_DP4A_TXS_Q4_1    tile_x_sizes{mmq_y*MMQ_TILE_NE_K   + mmq_y, mmq_y*MMQ_TILE_NE_K/QI4_1   + mmq_y/QI4_1,     0}
#define MMQ_DP4A_TXS_Q8_0    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K*2/QI8_0 + mmq_y/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K*4/QI8_0 + mmq_y/(QI8_0/4), 0}
#define MMQ_DP4A_TXS_Q8_1    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K*2/QI8_1 + mmq_y/(QI8_1/2), 0}
#define MMQ_DP4A_TXS_Q2_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K         + mmq_y,           0}
#define MMQ_DP4A_TXS_Q3_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y,                                         mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q4_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K   + mmq_y, mmq_y*MMQ_TILE_NE_K/QI4_K,                     mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q5_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K/QI5_K   + mmq_y/QI5_K,     mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q6_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K/QI6_K   + mmq_y/QI6_K,     mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int mmq_y) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q4_0:    return MMQ_DP4A_TXS_Q4_0;
        case GGML_TYPE_Q4_1:    return MMQ_DP4A_TXS_Q4_1;
        case GGML_TYPE_Q5_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q5_1:    return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_Q8_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_MXFP4:   return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_MXFP8:   return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_MXFP6_E2M3:   return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_NVFP4:   return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_Q2_K:    return MMQ_DP4A_TXS_Q2_K;
        case GGML_TYPE_Q3_K:    return MMQ_DP4A_TXS_Q3_K;
        case GGML_TYPE_Q4_K:    return MMQ_DP4A_TXS_Q4_K;
        case GGML_TYPE_Q5_K:    return MMQ_DP4A_TXS_Q5_K;
        case GGML_TYPE_Q6_K:    return MMQ_DP4A_TXS_Q6_K;
        case GGML_TYPE_IQ2_XXS: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ2_XS:  return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ2_S:   return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ3_XXS: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ3_S:   return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ1_S:   return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_XS:  return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_NL:  return MMQ_DP4A_TXS_Q8_0;
        default:                return tile_x_sizes{0, 0, 0};
    }
}

#define MMQ_MMA_TILE_X_K_Q8_0  (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0                   + 4)
#define MMQ_MMA_TILE_X_K_FP4   (2*MMQ_TILE_NE_K + 8                                       + 4) // MXFP4 and NVFP4 Blackwell
#define MMQ_MMA_TILE_X_K_NVFP4 (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2                         + 4) // NVFP4 Generic
#define MMQ_MMA_TILE_X_K_Q8_1  (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0                   + 4)
#define MMQ_MMA_TILE_X_K_Q2_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K                           + 4)
#define MMQ_MMA_TILE_X_K_Q3_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2                         + 4)
#define MMQ_MMA_TILE_X_K_Q6_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K   + MMQ_TILE_NE_K/8 + 7)

static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q8_1 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q2_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q3_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q6_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_FP4  % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_FP4 == MMQ_MMA_TILE_X_K_Q8_1, "Wrong tile size for MXFP4");
static_assert(MMQ_MMA_TILE_X_K_NVFP4 % 8 == 4, "Wrong padding.");


static constexpr __host__ __device__ int mmq_get_mma_tile_x_k(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q4_0:    return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q4_1:    return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q5_0:    return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q5_1:    return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q8_0:    return MMQ_MMA_TILE_X_K_Q8_0;
        // tile sizes are the same for Q8_1 and FP4 for blackwell
        case GGML_TYPE_MXFP4:   return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_MXFP8:   return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_MXFP6_E2M3:   return MMQ_MMA_TILE_X_K_Q8_0;
#if defined(BLACKWELL_MMA_AVAILABLE)
        case GGML_TYPE_NVFP4:   return MMQ_MMA_TILE_X_K_FP4;
#else
        case GGML_TYPE_NVFP4:   return MMQ_MMA_TILE_X_K_NVFP4;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
        case GGML_TYPE_Q2_K:    return MMQ_MMA_TILE_X_K_Q2_K;
        case GGML_TYPE_Q3_K:    return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_Q4_K:    return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q5_K:    return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q6_K:    return MMQ_MMA_TILE_X_K_Q6_K;
        case GGML_TYPE_IQ2_XXS: return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ2_XS:  return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ2_S:   return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ3_XXS: return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ3_S:   return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ1_S:   return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_XS:  return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_NL:  return MMQ_MMA_TILE_X_K_Q8_0;
        default:                return 0;
    }
}

// block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)
#define MMQ_TILE_Y_K     (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)
#define MMQ_TILE_Y_FP4_K MMQ_TILE_Y_K

static int mmq_get_granularity_host(const int mmq_x, const int cc) {
    if (blackwell_mma_available(cc)) {
        return 32;
    } else if (amd_mfma_available(cc) || amd_wmma_available(cc)) {
        return mmq_x >= 128 ? 32 : 16;
    } else if (turing_mma_available(cc) && mmq_x >= 48) {
        return 16;
    } else {
        return 8;
    }
}

#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 128 ? 32 : 16;
}
#elif defined(TURING_MMA_AVAILABLE)
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}
#else
static constexpr __device__ int mmq_get_granularity_device(const int /*mmq_x*/) {
    return 8;
}
#endif // AMD_MFMA_AVAILABLE

#if defined(GGML_USE_HIP)
static int mmq_get_nwarps_host(const int cc, const int warp_size) {
    return amd_mfma_available(cc) ? 8 : 256/warp_size;
}
#else
static int mmq_get_nwarps_host(const int /*cc*/, const int warp_size) {
    return 256/warp_size;
}
#endif // (GGML_USE_HIP)

static constexpr __device__ int mmq_get_nwarps_device() {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    return 8;
#else
    return 256/ggml_cuda_get_physical_warp_size();
#endif // AMD_MFMA_AVAILABLE
}

// ------------------------------------------------------------

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q1_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int blocks_per_iter = MMQ_ITER_K / QK1_0;
    constexpr int threads_per_row = blocks_per_iter * QI1_0;
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int scale_entries_per_block = QK1_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_block;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI1_0;
    const int kqsx = txi % QI1_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q1_0 * bxi = (const block_q1_0 *) x + kbx0 + i*stride + kbx;
        const int qs_offset = 4*kqsx;
        const int qs0 = bxi->qs[qs_offset + 0] | (bxi->qs[qs_offset + 1] << 8) |
                        (bxi->qs[qs_offset + 2] << 16) | (bxi->qs[qs_offset + 3] << 24);

        int unpacked_bytes[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int shift = j * 4;
            const int bits4 = (qs0 >> shift) & 0x0F;
            const int b0 = (bits4 & 0x01) ? 1 : -1;
            const int b1 = (bits4 & 0x02) ? 1 : -1;
            const int b2 = (bits4 & 0x04) ? 1 : -1;
            const int b3 = (bits4 & 0x08) ? 1 : -1;
            unpacked_bytes[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
        }

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + dst_offset + j] = unpacked_bytes[j];
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j] = unpacked_bytes[j];
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q1_0 * bxi = (const block_q1_0 *) x + kbx0 + i*stride + scale_block;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + ksx] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + ksx] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_0);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_0;
    const int kqsx = txi % QI4_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + 0]     = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0           + kbxd] = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_0*VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_0_Q8_1_MMQ];

                constexpr int max_cpy = ggml_cuda_get_max_cpy_bytes();
                constexpr int mcpy_int = max_cpy / sizeof(int);
                static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

                int tmp0[4], tmp1[4];

                #pragma unroll
                for (int l0 = 0; l0 < 4 / mcpy_int; ++l0) {
                    ggml_cuda_memcpy_1<max_cpy>(tmp0 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + l0 * mcpy_int]  );
                    ggml_cuda_memcpy_1<max_cpy>(tmp1 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + QI4_0 + l0 * mcpy_int]);
                }

                u[0]=tmp0[0]; u[2]=tmp0[1]; u[4]=tmp0[2]; u[6]=tmp0[3];
                u[1]=tmp1[0]; u[3]=tmp1[1]; u[5]=tmp1[2]; u[7]=tmp1[3];

                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                    (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_0], u,
                     x_df[i*(MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + k0/(QR4_0*QI4_0)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_1);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_1;
    const int kqsx = txi % QI4_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbx;
        const int qs0 = get_int_b4(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI4_1) + kqsx + 0]     = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI4_1) + kqsx + QI4_1] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_1;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*MMQ_MMA_TILE_X_K_Q8_1           + kbxd] = bxi->dm;
#else
        x_dm[i*(MMQ_TILE_NE_K/QI4_1) + i/QI4_1 + kbxd] = bxi->dm;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_1*VDR_Q4_1_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_1_Q8_1_MMQ];

                constexpr int max_cpy = ggml_cuda_get_max_cpy_bytes();
                constexpr int mcpy_int = max_cpy / sizeof(int);
                static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

                int tmp0[4], tmp1[4];

                #pragma unroll
                for (int l0 = 0; l0 < 4 / mcpy_int; ++l0) {
                    ggml_cuda_memcpy_1<max_cpy>(tmp0 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + l0 * mcpy_int]  );
                    ggml_cuda_memcpy_1<max_cpy>(tmp1 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + QI4_1 + l0 * mcpy_int]);
                }

                u[0]=tmp0[0]; u[2]=tmp0[1]; u[4]=tmp0[2]; u[6]=tmp0[3];
                u[1]=tmp1[0]; u[3]=tmp1[1]; u[5]=tmp1[2]; u[7]=tmp1[3];

                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                    (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_1], u,
                     x_dm[i*(MMQ_TILE_NE_K/QI4_1) + i/QI4_1 + k0/(QR4_1*QI4_1)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_0);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI5_0;
    const int kqsx = txi % QI5_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0           + kbxd] = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI5_0) + i/QI5_0 + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_1);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI5_1;
    const int kqsx = txi % QI5_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_b4(bxi->qs, kqsx);
        const int qh = get_int_b4(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_1;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*MMQ_MMA_TILE_X_K_Q8_1           + kbxd] = bxi->dm;
#else
        x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + kbxd] = bxi->dm;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    // MMQ_ITER_K / (4 * QR8_0) == 64 required. but NV has only 32 threads per warp
    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI8_0;
    const int kqsx = txi % QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / QI8_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0                 + kbxd] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_mxfp4(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_MXFP4, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR_MXFP4);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI_MXFP4;
    const int kqsx = txi % QI_MXFP4;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i*stride + kbx;

        const int aux_q4 = get_int_b1(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        const int k0 = kbx * (2 * QI_MXFP4) + kqsx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + k0 + 0]        = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + k0 + QI_MXFP4] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0]        = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI_MXFP4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI_MXFP4;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_1                 + kbxd] = ggml_cuda_e8m0_to_fp32(bxi->e)*0.5f;
#else
        x_df[i*(MMQ_TILE_NE_K/QI_MXFP4) + i/QI_MXFP4 + kbxd] = ggml_cuda_e8m0_to_fp32(bxi->e)*0.5f;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_mxfp6(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_MXFP6_E2M3, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    // One MXFP6_E2M3 byte-container block is one K32 fragment. A generic MMQ K256
    // iteration therefore spans eight consecutive blocks, arranged like Q8_0:
    // four K32 blocks in the first half of tile_x and four in the second half.
    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI_MXFP6_E2M3;
    const int kqsx = txi % QI_MXFP6_E2M3;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const compact_mxfp6_k32 * bxi = (const compact_mxfp6_k32 *) x + kbx0 + i*stride + kbx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 0             + txi] = mxfp6_e2m3_pack4_i8x8(ggml_cuda_mxfp6_e2m3_get4_u8containers(bxi[0], 0, kqsx));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K + txi] = mxfp6_e2m3_pack4_i8x8(ggml_cuda_mxfp6_e2m3_get4_u8containers(bxi[MMQ_TILE_NE_K/QI_MXFP6_E2M3], 0, kqsx));
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 0             + txi] = mxfp6_e2m3_pack4_i8x8(ggml_cuda_mxfp6_e2m3_get4_u8containers(bxi[0], 0, kqsx));
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = mxfp6_e2m3_pack4_i8x8(ggml_cuda_mxfp6_e2m3_get4_u8containers(bxi[MMQ_TILE_NE_K/QI_MXFP6_E2M3], 0, kqsx));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / QI_MXFP6_E2M3;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const compact_mxfp6_k32 * bxi = (const compact_mxfp6_k32 *) x + kbx0 + i*stride;
        const float scale = ggml_cuda_e8m0_to_fp32(bxi[kbxd].e[0]) * (1.0f / 8.0f);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0                 + kbxd] = scale;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI_MXFP6_E2M3) + i/(QI_MXFP6_E2M3/2) + kbxd] = scale;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

static __device__ __forceinline__ int mxfp8_pack4_i8(const uint32_t packed4) {
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint8_t code = (packed4 >> (8 * i)) & 0xff;
        const int q = max(-127, min(127, (int) roundf(ggml_cuda_e4m3_to_fp32(code) * 0.25f)));
        out |= (uint32_t) (uint8_t) q << (8 * i);
    }
    return (int) out;
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_mxfp8(
        const char * __restrict__ x, int * __restrict__ x_tile,
        const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int * x_qs = x_tile;
    float * x_df = (float *) (x_qs + 2 * MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_MXFP8, mmq_y);
    int * x_qs = x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif

    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = threadIdx.x % threads_per_row;
    const int frag = txi / 8;
    const int word = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows * nwarps) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if constexpr (need_check) {
            i = min(i, i_max);
        }
        const block_mxfp8 * bxi = (const block_mxfp8 *) x + kbx0 + i * stride;
        const int q0 = mxfp8_pack4_i8(get_int_b4(bxi->qs[frag + 0], word));
        const int q1 = mxfp8_pack4_i8(get_int_b4(bxi->qs[frag + 4], word));
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 0 + txi] = q0;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K + txi] = q1;
#else
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + 0 + txi] = q0;
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = q1;
#endif
    }

    const int kbxd = threadIdx.x % 8;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * (warp_size / 8)) {
        int i = i0 + threadIdx.y * (warp_size / 8) + threadIdx.x / 8;
        if constexpr (need_check) {
            i = min(i, i_max);
        }
        const block_mxfp8 * bxi = (const block_mxfp8 *) x + kbx0 + i * stride;
        const float scale = ggml_cuda_e8m0_to_fp32(bxi->e[kbxd]) * 4.0f;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = scale;
#else
        x_df[i * (2 * MMQ_TILE_NE_K / QI8_0) + i / (QI8_0 / 2) + kbxd] = scale;
#endif
    }
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_mxfp8_native(
        const char * __restrict__ x, int * __restrict__ x_tile,
        const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int sram_stride = MMQ_MMA_TILE_X_K_Q8_1;

    int *      x_qs = x_tile;
    uint32_t * x_sc = (uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);

    const mxfp8_tile * __restrict__ tiles = (const mxfp8_tile *) x;
    const int lane = int(threadIdx.x);

#pragma unroll
    for (int i0 = int(threadIdx.y) * MXFP8_MMA_TILE_ROWS; i0 < mmq_y; i0 += nwarps * MXFP8_MMA_TILE_ROWS) {
        const int64_t tile = kbx0 + int64_t(i0 / MXFP8_MMA_TILE_ROWS) * stride;
        const int row_lo = i0 + (lane >> 2);
        const int row_hi = row_lo + 8;
        const int word   = lane & 3;

#pragma unroll
        for (int frag = 0; frag < QK_MXFP8_FRAGS; ++frag) {
            const uint4 packed = tiles[tile].qs[frag][lane];
            x_qs[row_lo * sram_stride + frag * 8 + word + 0] = int(packed.x);
            x_qs[row_hi * sram_stride + frag * 8 + word + 0] = int(packed.y);
            x_qs[row_lo * sram_stride + frag * 8 + word + 4] = int(packed.z);
            x_qs[row_hi * sram_stride + frag * 8 + word + 4] = int(packed.w);
            if (lane < MXFP8_MMA_TILE_ROWS) {
                x_sc[(i0 + lane) * sram_stride + frag] = tiles[tile].sc[frag][lane];
            }
        }
    }
    GGML_UNUSED(i_max);
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_mxfp4_fp4(const char * __restrict__ x,
                                                            int * __restrict__ x_tile,
                                                            const int kbx0,
                                                            const int i_max,
                                                            const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    int *      x_qs = (int *) x_tile;
    uint32_t * x_sc = (uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);

    const int txi = threadIdx.x;

    constexpr int iter_k = get_iter_k(GGML_TYPE_MXFP4);

    constexpr int threads_per_row = iter_k / QK_MXFP4;  // each thread processes 1 block
    constexpr int rows_per_warp   = warp_size / threads_per_row;
    const int     kbx             = txi % threads_per_row;
    const int     row_in_warp     = txi / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (need_check) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i * stride + kbx;

        // quantize_mxfp4_mmq permutes nibbles to match the quantized format
        const int k0 = kbx * 4;
        memcpy(x_qs + i * MMQ_MMA_TILE_X_K_FP4 + k0, bxi->qs, 16);

        // Load E8M0 scales: pack 2 consecutive scales into one uint32
        if (kbx % 2 == 0) {
            uint32_t e = bxi->e;
            e |= ((bxi + 1)->e << 8);
            x_sc[i * MMQ_MMA_TILE_X_K_FP4 + kbx / 2] = e;
        }
    }
}

#if defined(BLACKWELL_MMA_AVAILABLE)
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_mxfp8_mxfp8_mma(
        const int * __restrict__ x, const int * __restrict__ y,
        float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int stride = MMQ_MMA_TILE_X_K_Q8_1;
    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / tile_C::I;
    constexpr int nfrags = MMQ_TILE_NE_K / tile_A::J;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);
    const int * x_qs = x;
    const uint32_t * x_sc = (const uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);
    const int * y_qs = y + 4;
    const uint32_t * y_sc = (const uint32_t *) y;
    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    tile_A A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            load_ldmatrix(A[n][frag], x_qs + (i0 + n * tile_A::I) * stride + k0, stride);
            scaleA[n][frag] = x_sc[(i0 + n * tile_A::I + tidx_A) * stride + k0 / tile_A::J];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
        tile_B B[nfrags];
        uint32_t scaleB[nfrags];
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            load_generic(B[frag], y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            const int sum_offset = (j0 / tile_C::J + n) * tile_C::ne;
            tile_C C;
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] = sum[sum_offset + l];
            }
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                mma_block_scaled_fp8_e4m3(C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);
            }
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[sum_offset + l] = C.x[l];
            }
        }
    }
}

// Shared MMA kernel for MXFP4 and NVFP4 on Blackwell.
// Both quantizations encode values as e2m1 (FP4) and produce one uint32 scale per
// m16n8k64 MMA call; only the PTX kind (scale_vec::2X ue8m0 vs scale_vec::4X ue4m3)
// and the per-type stride constant differ.
template <int mmq_x, int mmq_y, ggml_type type>
static __device__ __forceinline__ void vec_dot_fp4_fp4_mma(const int * __restrict__ x,
                                                           const int * __restrict__ y,
                                                           float * __restrict__ sum,
                                                           const int k00) {
    static_assert(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4,
                  "vec_dot_fp4_fp4_mma: type must be MXFP4 or NVFP4");

    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int stride        = MMQ_MMA_TILE_X_K_FP4;
    constexpr int granularity   = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx           = rows_per_warp / tile_C::I;
    constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int *      x_qs = (const int *) x;
    const uint32_t * x_sc = (const uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);
    const int *      y_qs = (const int *) y + 4;
    const uint32_t * y_sc = (const uint32_t *) y;

    // 2 threads per quad supply the packed scale register to the block_scale MMA,
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    const int i0     = (threadIdx.y / ntx) * rows_per_warp;

    tile_A   A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            load_ldmatrix(A[n][frag], x_qs + (i0 + n * tile_A::I) * stride + k0, stride);
            scaleA[n][frag] = x_sc[(i0 + n * tile_A::I + tidx_A) * stride + k0 / tile_A::J];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
        tile_B   B[nfrags];
        uint32_t scaleB[nfrags];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            load_generic(B[frag], y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                tile_C C = {};
                mma_block_scaled_fp4<type>(C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l];
                }
            }
        }
    }
}

template <bool need_check>
static __device__ __forceinline__ void load_nvfp4_tileA(
                                        tile<16, 8, int> & t,
                                        uint32_t & scale,
                                        const block_nvfp4_blackwell * __restrict__ x,
                                        const int nvfp4_blocks_per_row,
                                        const int row_base,
                                        const int frag_abs,
                                        const int row_max) {

    const int lane = int(threadIdx.x) & 31;
    int row_lo_abs = row_base + (lane >> 2);
    int row_hi_abs = row_lo_abs + 8;
    if constexpr (need_check) {
        row_lo_abs = min(row_lo_abs, row_max);
        row_hi_abs = min(row_hi_abs, row_max);
    }

    const int block_rel = frag_abs / 4;
    const int frag_idx = frag_abs % 4;
    const int word_idx = lane & 3;
    int * tx = (int *) t.x;

    if constexpr (!need_check) {
        const block_nvfp4_blackwell_frag & frag = x[(row_base / 16) * nvfp4_blocks_per_row + block_rel].tiles[frag_idx];
        const uint4 packed = reinterpret_cast<const uint4 *>(frag.regs)[lane];
        tx[0] = (int) packed.x;
        tx[1] = (int) packed.y;
        tx[2] = (int) packed.z;
        tx[3] = (int) packed.w;
        scale = frag.scales_u32[lane];
        return;
    }

    const block_nvfp4_blackwell & block_lo = x[(row_lo_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const block_nvfp4_blackwell & block_hi = x[(row_hi_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const int row_lo = (row_lo_abs % 16);
    const int row_hi = (row_hi_abs % 16);

    tx[0] = (int) ggml_cuda_nvfp4_tile_q_word(block_lo, row_lo, frag_idx, word_idx + 0);
    tx[1] = (int) ggml_cuda_nvfp4_tile_q_word(block_hi, row_hi, frag_idx, word_idx + 0);
    tx[2] = (int) ggml_cuda_nvfp4_tile_q_word(block_lo, row_lo, frag_idx, word_idx + 4);
    tx[3] = (int) ggml_cuda_nvfp4_tile_q_word(block_hi, row_hi, frag_idx, word_idx + 4);
    scale = (lane & 1) == 0
        ? ggml_cuda_nvfp4_tile_scale_word(block_lo, row_lo, frag_idx)
        : ggml_cuda_nvfp4_tile_scale_word(block_hi, row_hi, frag_idx);
}

static __device__ __forceinline__ void load_nvfp4_tileB(
                                        tile<8, 8, int> & t,
                                        uint32_t & scale,
                                        const block_nvfp4_mmq & y_block,
                                        const int frag_idx) {

    const int lane = int(threadIdx.x) & 31;
    const int group = lane & 3;
    const uint32_t * __restrict__ y_qs = y_block.qs_u32 + frag_idx * 8;
    int * const tx = (int *) t.x;

    tx[0] = (int) y_qs[group + 0];
    tx[1] = (int) y_qs[group + 4];

    uint32_t scale_word = 0;
    if (group == 0) {
        scale_word = y_block.sc4_u32[frag_idx];
    }
    scale = __shfl_sync(0xFFFFFFFFu, scale_word, lane & ~3);
}

template <int mmq_x>
static __device__ __forceinline__ void load_nvfp4_tileB(
                                        tile<8, 8, int> & t,
                                        uint32_t & scale,
                                        const block_nvfp4_mmq * __restrict__ y_blocks_j,
                                        const int frag_idx) {
    const int col = (int(threadIdx.x) & 31) >> 2;
    load_nvfp4_tileB(t, scale, y_blocks_j[col], frag_idx);
}

#if defined(BLACKWELL_MMA_AVAILABLE)
static __device__ __forceinline__ uint32_t nvfp4_pack4_mxf8f6f4_e2m1(const uint32_t packed8, const int off4) {
    return
        ((uint32_t) (((packed8 >> (4 * (off4 + 0))) & 0x0Fu) <<  2)      ) |
        ((uint32_t) (((packed8 >> (4 * (off4 + 1))) & 0x0Fu) <<  2) <<  8) |
        ((uint32_t) (((packed8 >> (4 * (off4 + 2))) & 0x0Fu) <<  2) << 16) |
        ((uint32_t) (((packed8 >> (4 * (off4 + 3))) & 0x0Fu) <<  2) << 24);
}

static __device__ __forceinline__ uint8_t nvfp4_ue4m3_scale_to_e8m0_x2(const uint8_t scale) {
    if (scale == 0 || scale == 0x7F || scale == 0xFF) {
        return 0;
    }

    const int exp = (scale >> 3) & 0x0F;
    const int man = scale & 0x07;
    int unbiased = -127;
    if (exp == 0) {
        unbiased = man <= 1 ? -9 : man <= 2 ? -8 : man <= 5 ? -7 : -6;
    } else {
        unbiased = exp - 7 + (man >= 4);
    }
    return (uint8_t) max(0, min(254, unbiased + 127));
}

static __device__ __forceinline__ float nvfp4_fp8_scale_correction(const uint8_t scale) {
    const uint8_t approx = nvfp4_ue4m3_scale_to_e8m0_x2(scale);
    const float exact_raw = 2.0f * ggml_cuda_ue4m3_to_fp32(scale);
    const float approx_raw = ggml_cuda_e8m0_to_fp32(approx);
    return approx_raw > 0.0f ? exact_raw / approx_raw : 0.0f;
}

template <bool need_check>
static __device__ __forceinline__ void fixup_nvfp4_fp8_tileC(
                                        tile<16, 8, float> & t,
                                        const block_nvfp4_blackwell * __restrict__ x,
                                        const int nvfp4_blocks_per_row,
                                        const int row_base,
                                        const int frag_abs32,
                                        const int row_max,
                                        const int sub16) {
    const int block_rel   = frag_abs32 / 8;
    const int frag32_rel  = frag_abs32 & 7;
    const int nv_frag_idx = frag32_rel >> 1;
    const int half32      = frag32_rel & 1;
    const int scale_shift = 8 * (half32 * 2 + sub16);

#pragma unroll
    for (int l = 0; l < tile<16, 8, float>::ne; ++l) {
        const int row_abs = row_base + tile<16, 8, float>::get_i(l);
        if constexpr (need_check) {
            if (row_abs > row_max) {
                t.x[l] = 0.0f;
                continue;
            }
        }

        const block_nvfp4_blackwell & block = x[(row_abs / 16) * nvfp4_blocks_per_row + block_rel];
        const uint32_t scale_word = ggml_cuda_nvfp4_tile_scale_word(block, row_abs & 15, nv_frag_idx);
        const uint8_t scale_code = (scale_word >> scale_shift) & 0xFFu;
        t.x[l] *= nvfp4_fp8_scale_correction(scale_code);
    }
}

static __device__ __forceinline__ uint32_t nvfp4_load_mxf8f6f4_word(
                                        const block_nvfp4_blackwell & block,
                                        const int row,
                                        const int nv_frag_idx,
                                        const int half32,
                                        const int sub16,
                                        const int out_word) {
    const int out_sub16 = out_word >> 2;
    if (out_sub16 != sub16) {
        return 0;
    }

    const int rel_word = out_word & 3;
    const int src_pack_idx = half32 * 4 + sub16 * 2 + (rel_word >> 1);
    const int off4 = (rel_word & 1) * 4;
    const uint32_t packed8 = ggml_cuda_nvfp4_tile_q_word(block, row, nv_frag_idx, src_pack_idx);
    return nvfp4_pack4_mxf8f6f4_e2m1(packed8, off4);
}

template <bool need_check>
static __device__ __forceinline__ void load_nvfp4_fp8_tileA_sub16(
                                        tile<16, 8, int> & t,
                                        uint32_t & scale,
                                        const block_nvfp4_blackwell * __restrict__ x,
                                        const int nvfp4_blocks_per_row,
                                        const int row_base,
                                        const int frag_abs32,
                                        const int row_max,
                                        const int sub16) {
    const int lane = int(threadIdx.x) & 31;
    int row_lo_abs = row_base + (lane >> 2);
    int row_hi_abs = row_lo_abs + 8;
    const bool row_lo_valid = !need_check || row_lo_abs <= row_max;
    const bool row_hi_valid = !need_check || row_hi_abs <= row_max;
    if constexpr (need_check) {
        row_lo_abs = min(row_lo_abs, row_max);
        row_hi_abs = min(row_hi_abs, row_max);
    }

    const int block_rel   = frag_abs32 / 8;
    const int frag32_rel  = frag_abs32 & 7;
    const int nv_frag_idx = frag32_rel >> 1;
    const int half32      = frag32_rel & 1;
    const int word_idx    = lane & 3;
    int * tx = (int *) t.x;

    const block_nvfp4_blackwell & block_lo = x[(row_lo_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const block_nvfp4_blackwell & block_hi = x[(row_hi_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const int row_lo = row_lo_abs & 15;
    const int row_hi = row_hi_abs & 15;

    tx[0] = row_lo_valid ? (int) nvfp4_load_mxf8f6f4_word(block_lo, row_lo, nv_frag_idx, half32, sub16, word_idx + 0) : 0;
    tx[1] = row_hi_valid ? (int) nvfp4_load_mxf8f6f4_word(block_hi, row_hi, nv_frag_idx, half32, sub16, word_idx + 0) : 0;
    tx[2] = row_lo_valid ? (int) nvfp4_load_mxf8f6f4_word(block_lo, row_lo, nv_frag_idx, half32, sub16, word_idx + 4) : 0;
    tx[3] = row_hi_valid ? (int) nvfp4_load_mxf8f6f4_word(block_hi, row_hi, nv_frag_idx, half32, sub16, word_idx + 4) : 0;

    const int tid = lane & 3;
    uint8_t sc = 0x7F;
    if (tid == 0 && row_lo_valid) {
        const uint32_t scale_word = ggml_cuda_nvfp4_tile_scale_word(block_lo, row_lo, nv_frag_idx);
        sc = nvfp4_ue4m3_scale_to_e8m0_x2((scale_word >> (8 * (half32 * 2 + sub16))) & 0xFFu);
    } else if (tid == 1 && row_hi_valid) {
        const uint32_t scale_word = ggml_cuda_nvfp4_tile_scale_word(block_hi, row_hi, nv_frag_idx);
        sc = nvfp4_ue4m3_scale_to_e8m0_x2((scale_word >> (8 * (half32 * 2 + sub16))) & 0xFFu);
    }
    scale = uint32_t(sc);
}

template <bool need_check>
static __device__ __forceinline__ void load_nvfp4_fp8_tileA_k32(
                                        tile<16, 8, int> & t,
                                        uint32_t & scale,
                                        const block_nvfp4_blackwell * __restrict__ x,
                                        const int nvfp4_blocks_per_row,
                                        const int row_base,
                                        const int frag_abs32,
                                        const int row_max) {
    const int lane = int(threadIdx.x) & 31;
    int row_lo_abs = row_base + (lane >> 2);
    int row_hi_abs = row_lo_abs + 8;
    const bool row_lo_valid = !need_check || row_lo_abs <= row_max;
    const bool row_hi_valid = !need_check || row_hi_abs <= row_max;
    if constexpr (need_check) {
        row_lo_abs = min(row_lo_abs, row_max);
        row_hi_abs = min(row_hi_abs, row_max);
    }

    const int block_rel   = frag_abs32 / 8;
    const int frag32_rel  = frag_abs32 & 7;
    const int nv_frag_idx = frag32_rel >> 1;
    const int half32      = frag32_rel & 1;
    const int word_idx    = lane & 3;
    int * tx = (int *) t.x;

    const block_nvfp4_blackwell & block_lo = x[(row_lo_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const block_nvfp4_blackwell & block_hi = x[(row_hi_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const int row_lo = row_lo_abs & 15;
    const int row_hi = row_hi_abs & 15;

    const int sub_lo = (word_idx + 0) >> 2;
    const int sub_hi = (word_idx + 4) >> 2;
    tx[0] = row_lo_valid ? (int) nvfp4_load_mxf8f6f4_word(block_lo, row_lo, nv_frag_idx, half32, sub_lo, word_idx + 0) : 0;
    tx[1] = row_hi_valid ? (int) nvfp4_load_mxf8f6f4_word(block_hi, row_hi, nv_frag_idx, half32, sub_lo, word_idx + 0) : 0;
    tx[2] = row_lo_valid ? (int) nvfp4_load_mxf8f6f4_word(block_lo, row_lo, nv_frag_idx, half32, sub_hi, word_idx + 4) : 0;
    tx[3] = row_hi_valid ? (int) nvfp4_load_mxf8f6f4_word(block_hi, row_hi, nv_frag_idx, half32, sub_hi, word_idx + 4) : 0;

    const int tid = lane & 3;
    uint8_t sc = 0x7F;
    if (tid == 0 && row_lo_valid) {
        const uint32_t scale_word = ggml_cuda_nvfp4_tile_scale_word(block_lo, row_lo, nv_frag_idx);
        const uint8_t sc0 = nvfp4_ue4m3_scale_to_e8m0_x2((scale_word >> (8 * (half32 * 2 + 0))) & 0xFFu);
        const uint8_t sc1 = nvfp4_ue4m3_scale_to_e8m0_x2((scale_word >> (8 * (half32 * 2 + 1))) & 0xFFu);
        sc = max(sc0, sc1);
    } else if (tid == 1 && row_hi_valid) {
        const uint32_t scale_word = ggml_cuda_nvfp4_tile_scale_word(block_hi, row_hi, nv_frag_idx);
        const uint8_t sc0 = nvfp4_ue4m3_scale_to_e8m0_x2((scale_word >> (8 * (half32 * 2 + 0))) & 0xFFu);
        const uint8_t sc1 = nvfp4_ue4m3_scale_to_e8m0_x2((scale_word >> (8 * (half32 * 2 + 1))) & 0xFFu);
        sc = max(sc0, sc1);
    }
    scale = uint32_t(sc);
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)

template <bool need_check>
static __device__ __forceinline__ void load_mxfp6_e2m3_tileA(
                                        tile<16, 8, int> & t,
                                        uint32_t & scale,
                                        const tile_mxfp6_e2m3_blackwell * __restrict__ x,
                                        const int blocks_per_row,
                                        const int row_base,
                                        const int frag_abs,
                                        const int row_max) {
    const int lane = int(threadIdx.x) & 31;
    int row_lo_abs = row_base + (lane >> 2);
    int row_hi_abs = row_lo_abs + 8;
    const bool row_lo_valid = !need_check || row_lo_abs <= row_max;
    const bool row_hi_valid = !need_check || row_hi_abs <= row_max;
    if constexpr (need_check) {
        row_lo_abs = min(row_lo_abs, row_max);
        row_hi_abs = min(row_hi_abs, row_max);
    }

    const int block_rel = frag_abs / QK_MXFP6_E2M3_FRAGS;
    const int frag_idx  = frag_abs % QK_MXFP6_E2M3_FRAGS;
    int * tx = (int *) t.x;

    if constexpr (!need_check) {
        const tile_mxfp6_e2m3_blackwell_frag & frag = x[(row_base / 16) * blocks_per_row + block_rel].tiles[frag_idx];
        const uint32_t w0 = frag.regs[lane][0];
        const uint32_t w1 = frag.regs[lane][1];
        const uint32_t w2 = frag.regs[lane][2];
        tx[0] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers(w0);
        tx[1] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers((w0 >> 24) | (w1 << 8));
        tx[2] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers((w1 >> 16) | (w2 << 16));
        tx[3] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers(w2 >> 8);
        scale = frag.scales[lane];
        return;
    }

    if (!row_lo_valid) {
        tx[0] = 0;
        tx[1] = 0;
        tx[2] = 0;
        tx[3] = 0;
        scale = 0x7Fu;
        return;
    }

    const tile_mxfp6_e2m3_blackwell & block_lo = x[(row_lo_abs / 16) * blocks_per_row + block_rel];
    const tile_mxfp6_e2m3_blackwell & block_hi = x[(row_hi_abs / 16) * blocks_per_row + block_rel];
    const int row_lo = row_lo_abs & 15;
    const int row_hi = row_hi_abs & 15;
    const int word_idx = lane & 3;

    tx[0] = row_lo_valid ? (int) ggml_cuda_mxfp6_e2m3_bw_tile_q_word(block_lo, row_lo, frag_idx, word_idx + 0) : 0;
    tx[1] = row_hi_valid ? (int) ggml_cuda_mxfp6_e2m3_bw_tile_q_word(block_hi, row_hi, frag_idx, word_idx + 0) : 0;
    tx[2] = row_lo_valid ? (int) ggml_cuda_mxfp6_e2m3_bw_tile_q_word(block_lo, row_lo, frag_idx, word_idx + 4) : 0;
    tx[3] = row_hi_valid ? (int) ggml_cuda_mxfp6_e2m3_bw_tile_q_word(block_hi, row_hi, frag_idx, word_idx + 4) : 0;

    // Candidate scale_vec::1X selector mapping, mirrored by the MXFP6_E2M3 microkernel:
    // tidA 0 supplies rows 0..7 and tidA 1 supplies rows 8..15 within each quad.
    const int tid = lane & 3;
    uint8_t sc = 0x7F;
    if (tid == 0 && row_lo_valid) {
        sc = ggml_cuda_mxfp6_e2m3_bw_tile_scale(block_lo, row_lo, frag_idx);
    } else if (tid == 1 && row_hi_valid) {
        sc = ggml_cuda_mxfp6_e2m3_bw_tile_scale(block_hi, row_hi, frag_idx);
    }
    scale = uint32_t(sc);
}

static __device__ __forceinline__ void load_mxfp6_e2m3_tileA_full(
                                        tile<16, 8, int> & t,
                                        uint32_t & scale,
                                        const tile_mxfp6_e2m3_blackwell * __restrict__ x,
                                        const int blocks_per_row,
                                        const int row_base,
                                        const int frag_idx) {
    const int lane = int(threadIdx.x) & 31;
    const tile_mxfp6_e2m3_blackwell_frag & frag =
        x[(row_base >> 4) * blocks_per_row].tiles[frag_idx];
    const uint32_t w0 = frag.regs[lane][0];
    const uint32_t w1 = frag.regs[lane][1];
    const uint32_t w2 = frag.regs[lane][2];
    int * tx = (int *) t.x;

    tx[0] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers(w0);
    tx[1] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers((w0 >> 24) | (w1 << 8));
    tx[2] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers((w1 >> 16) | (w2 << 16));
    tx[3] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers(w2 >> 8);
    scale = frag.scales[lane];
}

template <bool need_check>
static __device__ __forceinline__ void load_mxfp6_e2m3_tileA_maybe_checked(
                                        tile<16, 8, int> & t,
                                        uint32_t & scale,
                                        const tile_mxfp6_e2m3_blackwell * __restrict__ x,
                                        const int blocks_per_row,
                                        const int row_base,
                                        const int frag_abs,
                                        const int row_max) {
    if constexpr (need_check) {
        if (row_base + 15 <= row_max) {
            load_mxfp6_e2m3_tileA<false>(t, scale, x, blocks_per_row, row_base, frag_abs, row_max);
        } else {
            load_mxfp6_e2m3_tileA<true>(t, scale, x, blocks_per_row, row_base, frag_abs, row_max);
        }
    } else {
        load_mxfp6_e2m3_tileA<false>(t, scale, x, blocks_per_row, row_base, frag_abs, row_max);
    }
}

template <int mmq_x>
static __device__ __forceinline__ void load_mxfp6_e2m3_tileB(
                                        tile<8, 8, int> & t,
                                        uint32_t & scale,
                                        const tile_mxfp6_e2m3_mmq * __restrict__ y_blocks_j,
                                        const int frag_idx) {
    GGML_UNUSED(mmq_x);

    const int lane = int(threadIdx.x) & 31;
    const int col  = lane >> 2;
    const int tid  = lane & 3;
    const tile_mxfp6_e2m3_mmq & block = y_blocks_j[col];
    int * tx = (int *) t.x;

    tx[0] = (int) block.qs_u32[frag_idx][tid + 0];
    tx[1] = (int) block.qs_u32[frag_idx][tid + 4];

    uint32_t sc = 0;
    if (tid == 0) {
        sc = block.e[frag_idx];
    }
    scale = __shfl_sync(0xFFFFFFFFu, sc, lane & ~3);
}

template <int mmq_x>
static __device__ __forceinline__ void load_fp8_tileB(
                                        tile<8, 8, int> & t,
                                        uint32_t & scale,
                                        const block_fp8 * __restrict__ y_blocks_j,
                                        const int ncols_y,
                                        const int frag_abs) {
    GGML_UNUSED(mmq_x);

    const int lane = int(threadIdx.x) & 31;
    const int col  = lane >> 2;
    const int tid  = lane & 3;
    const int block_rel = frag_abs / QK_FP8_FRAGS;
    const int frag_idx  = frag_abs % QK_FP8_FRAGS;
    const block_fp8 & block = y_blocks_j[col + block_rel*ncols_y];
    int * tx = (int *) t.x;

    tx[0] = (int) ggml_cuda_fp8_get4_u8containers(block, frag_idx, tid + 0);
    tx[1] = (int) ggml_cuda_fp8_get4_u8containers(block, frag_idx, tid + 4);

    uint32_t sc = 0;
    if (tid == 0) {
        sc = block.e[frag_idx];
    }
    scale = __shfl_sync(0xFFFFFFFFu, sc, lane & ~3);
}

static __device__ __forceinline__ uint8_t q8_1_mmq_fp8_scale_code(const float d) {
    if (!(d > 0.0f) || !isfinite(d)) {
        return 127;
    }
    int code = (int) ceilf(log2f((127.0f * d) / 448.0f)) + 127;
    return (uint8_t) max(0, min(254, code));
}

static __device__ __forceinline__ uint32_t q8_1_mmq_word_to_fp8x4(
        const block_q8_1_mmq & bq8, const int frag, const int word, const float mul) {
    const int base = 32*frag + 4*word;
    const uint32_t lo = uint32_t(__nv_cvt_float2_to_fp8x2(
        make_float2(float(bq8.qs[base + 0]) * mul, float(bq8.qs[base + 1]) * mul), __NV_SATFINITE, __NV_E4M3));
    const uint32_t hi = uint32_t(__nv_cvt_float2_to_fp8x2(
        make_float2(float(bq8.qs[base + 2]) * mul, float(bq8.qs[base + 3]) * mul), __NV_SATFINITE, __NV_E4M3));
    return lo | (hi << 16);
}

template <int mmq_x>
static __device__ __forceinline__ void load_q8_1_mmq_as_fp8_tileB(
                                        tile<8, 8, int> & t,
                                        uint32_t & scale,
                                        const block_q8_1_mmq * __restrict__ y_blocks_j,
                                        const int ncols_y,
                                        const int frag_abs) {
    GGML_UNUSED(mmq_x);

    const int lane = int(threadIdx.x) & 31;
    const int col  = lane >> 2;
    const int tid  = lane & 3;
    const int block_rel = frag_abs >> 2;
    const int frag_idx  = frag_abs & 3;
    const block_q8_1_mmq & block = y_blocks_j[col + block_rel*ncols_y];
    const uint8_t scale_code = q8_1_mmq_fp8_scale_code(block.d4[frag_idx]);
    const float scale_f = ggml_cuda_e8m0_to_fp32(scale_code);
    const float mul = scale_f > 0.0f ? block.d4[frag_idx] / scale_f : 0.0f;
    int * tx = (int *) t.x;

    tx[0] = int(q8_1_mmq_word_to_fp8x4(block, frag_idx, tid + 0, mul));
    tx[1] = int(q8_1_mmq_word_to_fp8x4(block, frag_idx, tid + 4, mul));
    scale = __shfl_sync(0xFFFFFFFFu, uint32_t(scale_code), lane & ~3);
}

template <int n>
static __device__ __forceinline__ void mxfp6_accumulate_scaled(float * __restrict__ sum, const float * __restrict__ x, const float scale) {
    if (scale == 1.0f) {
#pragma unroll
        for (int l = 0; l < n; ++l) {
            sum[l] += x[l];
        }
    } else {
#pragma unroll
        for (int l = 0; l < n; ++l) {
            sum[l] += scale * x[l];
        }
    }
}

template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void vec_dot_mxfp6_e2m3_mxfp6_e2m3_mma(
        const tile_mxfp6_e2m3_blackwell * __restrict__ x_blocks, const int stride_row_x,
        const tile_mxfp6_e2m3_mmq * __restrict__ y, float * __restrict__ sum,
        const int k00, const int i_max, const float tensor_scale) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;
    constexpr int nwarps          = mmq_get_nwarps_device();
    constexpr int granularity     = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp   = 2 * granularity;
    constexpr int ntx             = rows_per_warp / tile_C::I;
    constexpr int rows_per_slab   = nwarps * tile_C::I;
    constexpr int groups_per_slab = mmq_x / tile_C::J;

    const int ty = threadIdx.y;
    const int ty_ntx_mod = ty % ntx;
    const int ty_ntx_div = ty / ntx;
    const tile_mxfp6_e2m3_mmq * __restrict__ y_blocks = y + ty_ntx_mod * tile_C::J;

#pragma unroll
    for (int frag = 0; frag < QK_MXFP6_E2M3_FRAGS; ++frag) {
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
            const tile_mxfp6_e2m3_mmq * __restrict__ y_blocks_j = y_blocks + j0;
#pragma unroll
            for (int slab_row0 = 0; slab_row0 < mmq_y; slab_row0 += rows_per_slab) {
                const int i0 = slab_row0 + ty_ntx_div * rows_per_warp;
                const int sum_j = slab_row0 / rows_per_slab * groups_per_slab + j0 / tile_C::J;
                tile_B B;
                uint32_t scaleB;
                load_mxfp6_e2m3_tileB<mmq_x>(B, scaleB, y_blocks_j, frag);
#pragma unroll
                for (int n = 0; n < ntx; ++n) {
                    tile_A A;
                    uint32_t scaleA;
                    tile_C C = {};
                    const int row_base = i0 + n * tile_A::I;
                    if constexpr (need_check) {
                        load_mxfp6_e2m3_tileA_maybe_checked<true>(A, scaleA, x_blocks, stride_row_x,
                            row_base, k00 * QK_MXFP6_E2M3_FRAGS + frag, i_max);
                    } else {
                        GGML_UNUSED(i_max);
                        load_mxfp6_e2m3_tileA_full(A, scaleA, x_blocks, stride_row_x, row_base,
                            k00 * QK_MXFP6_E2M3_FRAGS + frag);
                    }
                    float * __restrict__ sum_n = sum + (sum_j + n) * tile_C::ne;
                    mma_block_scaled_mxfp6_e2m3(C, A, B, scaleA, scaleB);
                    mxfp6_accumulate_scaled<tile_C::ne>(sum_n, C.x, tensor_scale);
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void vec_dot_mxfp6_e2m3_fp8_e4m3_mma_frag(
        const tile_mxfp6_e2m3_blackwell * __restrict__ x_blocks, const int stride_row_x,
        const block_fp8 * __restrict__ y, const int ncols_y, float * __restrict__ sum,
        const int frag_abs, const int i_max, const float tensor_scale) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;
    constexpr int nwarps          = mmq_get_nwarps_device();
    constexpr int granularity     = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp   = 2 * granularity;
    constexpr int ntx             = rows_per_warp / tile_C::I;
    constexpr int rows_per_slab   = nwarps * tile_C::I;
    constexpr int groups_per_slab = mmq_x / tile_C::J;

    const int ty = threadIdx.y;
    const int ty_ntx_mod = ty % ntx;
    const int ty_ntx_div = ty / ntx;
    const block_fp8 * __restrict__ y_blocks = y + ty_ntx_mod * tile_C::J;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
        const block_fp8 * __restrict__ y_blocks_j = y_blocks + j0;
        tile_B B;
        uint32_t scaleB;
        load_fp8_tileB<mmq_x>(B, scaleB, y_blocks_j, ncols_y, frag_abs);

#pragma unroll
        for (int slab_row0 = 0; slab_row0 < mmq_y; slab_row0 += rows_per_slab) {
            const int i0 = slab_row0 + ty_ntx_div * rows_per_warp;
            const int sum_j = slab_row0 / rows_per_slab * groups_per_slab + j0 / tile_C::J;

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_A A;
                uint32_t scaleA;
                const int row_base = i0 + n * tile_A::I;
                load_mxfp6_e2m3_tileA_maybe_checked<need_check>(A, scaleA, x_blocks, stride_row_x,
                    row_base, frag_abs, i_max);

                tile_C C = {};
                float * __restrict__ sum_n = sum + (sum_j + n) * tile_C::ne;
                mma_block_scaled_mxfp6_e2m3_fp8_e4m3(C, A, B, scaleA, scaleB);
                    mxfp6_accumulate_scaled<tile_C::ne>(sum_n, C.x, tensor_scale);
            }
        }
    }
}

template <int mmq_x, int mmq_y, bool need_check, int k_group>
static __device__ __forceinline__ void vec_dot_mxfp6_e2m3_fp8_e4m3_mma_kgroup_scoped(
        const tile_mxfp6_e2m3_blackwell * __restrict__ x_blocks, const int stride_row_x,
        const block_fp8 * __restrict__ y, const int ncols_y,
        float * __restrict__ sum, const int frag_start, const int i_max, const float tensor_scale) {
    static_assert(k_group >= 2 && k_group <= 6, "FP8 activation grouped path is intended for 2-6 K32 blocks");
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;
    constexpr int nwarps          = mmq_get_nwarps_device();
    constexpr int granularity     = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp   = 2 * granularity;
    constexpr int ntx             = rows_per_warp / tile_C::I;
    constexpr int rows_per_slab   = nwarps * tile_C::I;
    constexpr int groups_per_slab = mmq_x / tile_C::J;

    const int ty = threadIdx.y;
    const int ty_ntx_mod = ty % ntx;
    const int ty_ntx_div = ty / ntx;
    const block_fp8 * __restrict__ y_blocks = y + ty_ntx_mod * tile_C::J;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int slab_row0 = 0; slab_row0 < mmq_y; slab_row0 += rows_per_slab) {
            const int i0 = slab_row0 + ty_ntx_div * rows_per_warp;
            const int sum_j = slab_row0 / rows_per_slab * groups_per_slab + j0 / tile_C::J;
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C = {};
                const int row_base = i0 + n * tile_A::I;

#pragma unroll
                for (int g = 0; g < k_group; ++g) {
                    tile_A A;
                    tile_B B;
                    uint32_t scaleA;
                    uint32_t scaleB;
                    const int frag_abs = frag_start + g;
                    load_fp8_tileB<mmq_x>(B, scaleB, y_blocks + j0, ncols_y, frag_abs);
                    load_mxfp6_e2m3_tileA_maybe_checked<need_check>(A, scaleA, x_blocks + g, stride_row_x,
                        row_base, 0, i_max);
                    mma_block_scaled_mxfp6_e2m3_fp8_e4m3(C, A, B, scaleA, scaleB);
                }

                float * __restrict__ sum_n = sum + (sum_j + n) * tile_C::ne;
                mxfp6_accumulate_scaled<tile_C::ne>(sum_n, C.x, tensor_scale);
            }
        }
    }
}

template <int mmq_x, int mmq_y, bool need_check, int k_group>
static __device__ __forceinline__ void vec_dot_mxfp6_e2m3_mxfp6_e2m3_mma_kgroup_streamed(
        const tile_mxfp6_e2m3_blackwell * __restrict__ x_blocks, const int stride_row_x,
        const tile_mxfp6_e2m3_mmq * __restrict__ y, const int ncols_y,
        float * __restrict__ sum, const int i_max, const float tensor_scale) {
    static_assert(QK_MXFP6_E2M3_FRAGS == 1, "MXFP6 grouped path assumes one K32 fragment per block");
    static_assert(k_group >= 2 && k_group <= 6, "MXFP6 grouped path is intended for 2-6 K32 blocks");
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;
    constexpr int nwarps          = mmq_get_nwarps_device();
    constexpr int granularity     = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp   = 2 * granularity;
    constexpr int ntx             = rows_per_warp / tile_C::I;
    constexpr int rows_per_slab   = nwarps * tile_C::I;
    constexpr int groups_per_slab = mmq_x / tile_C::J;

    const int ty = threadIdx.y;
    const int ty_ntx_mod = ty % ntx;
    const int ty_ntx_div = ty / ntx;
    const tile_mxfp6_e2m3_mmq * __restrict__ y_blocks = y + ty_ntx_mod * tile_C::J;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int slab_row0 = 0; slab_row0 < mmq_y; slab_row0 += rows_per_slab) {
            const int i0 = slab_row0 + ty_ntx_div * rows_per_warp;
            const int sum_j = slab_row0 / rows_per_slab * groups_per_slab + j0 / tile_C::J;
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C = {};
                const int row_base = i0 + n * tile_A::I;
#pragma unroll
                for (int g = 0; g < k_group; ++g) {
                    tile_A A;
                    tile_B B;
                    uint32_t scaleA;
                    uint32_t scaleB;
                    load_mxfp6_e2m3_tileB<mmq_x>(B, scaleB, y_blocks + j0 + ncols_y * g, 0);
                    if constexpr (need_check) {
                        load_mxfp6_e2m3_tileA_maybe_checked<true>(A, scaleA, x_blocks + g, stride_row_x,
                            row_base, 0, i_max);
                    } else {
                        GGML_UNUSED(i_max);
                        load_mxfp6_e2m3_tileA_full(A, scaleA, x_blocks + g, stride_row_x, row_base, 0);
                    }
                    mma_block_scaled_mxfp6_e2m3(C, A, B, scaleA, scaleB);
                }
                float * __restrict__ sum_n = sum + (sum_j + n) * tile_C::ne;
                mxfp6_accumulate_scaled<tile_C::ne>(sum_n, C.x, tensor_scale);
            }
        }
    }
}

template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void vec_dot_mxfp6_e2m3_mxfp6_e2m3_mma_k3_scoped(
        const tile_mxfp6_e2m3_blackwell * __restrict__ x_blocks, const int stride_row_x,
        const tile_mxfp6_e2m3_mmq * __restrict__ y, const int ncols_y,
        float * __restrict__ sum, const int i_max, const float tensor_scale) {
    static_assert(QK_MXFP6_E2M3_FRAGS == 1, "MXFP6 K3 path assumes one K32 fragment per block");
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;
    constexpr int nwarps          = mmq_get_nwarps_device();
    constexpr int granularity     = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp   = 2 * granularity;
    constexpr int ntx             = rows_per_warp / tile_C::I;
    constexpr int rows_per_slab   = nwarps * tile_C::I;
    constexpr int groups_per_slab = mmq_x / tile_C::J;

    const int ty = threadIdx.y;
    const int ty_ntx_mod = ty % ntx;
    const int ty_ntx_div = ty / ntx;
    const tile_mxfp6_e2m3_mmq * __restrict__ y_blocks = y + ty_ntx_mod * tile_C::J;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int slab_row0 = 0; slab_row0 < mmq_y; slab_row0 += rows_per_slab) {
            const int i0 = slab_row0 + ty_ntx_div * rows_per_warp;
            const int sum_j = slab_row0 / rows_per_slab * groups_per_slab + j0 / tile_C::J;
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C = {};
                const int row_base = i0 + n * tile_A::I;

#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    tile_A A;
                    tile_B B;
                    uint32_t scaleA;
                    uint32_t scaleB;
                    load_mxfp6_e2m3_tileB<mmq_x>(B, scaleB, y_blocks + j0 + ncols_y * g, 0);
                    if constexpr (need_check) {
                        load_mxfp6_e2m3_tileA_maybe_checked<true>(A, scaleA, x_blocks + g, stride_row_x,
                            row_base, 0, i_max);
                    } else {
                        GGML_UNUSED(i_max);
                        load_mxfp6_e2m3_tileA_full(A, scaleA, x_blocks + g, stride_row_x, row_base, 0);
                    }
                    mma_block_scaled_mxfp6_e2m3(C, A, B, scaleA, scaleB);
                }

                float * __restrict__ sum_n = sum + (sum_j + n) * tile_C::ne;
                    mxfp6_accumulate_scaled<tile_C::ne>(sum_n, C.x, tensor_scale);
            }
        }
    }
}

template <int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void vec_dot_nvfp4_nvfp4_mma(
        const block_nvfp4_blackwell * __restrict__ x_blocks, const int stride_row_x,
        const block_nvfp4_mmq * __restrict__ y, float * __restrict__ sum,
        const int k00, const int i_max, const float tensor_scale) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;
    constexpr int nwarps          = mmq_get_nwarps_device();
    constexpr int granularity     = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp   = 2 * granularity;
    constexpr int ntx             = rows_per_warp / tile_C::I;
    constexpr int rows_per_slab   = nwarps * tile_C::I;
    constexpr int groups_per_slab = mmq_x / tile_C::J;

    const int ty = threadIdx.y;
    const int ty_ntx_mod = ty % ntx;
    const int ty_ntx_div = ty / ntx;
    const block_nvfp4_mmq * __restrict__ y_blocks = y + ty_ntx_mod * tile_C::J;

#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
        const int frag0 = k01 / 8;
        const int frag1 = frag0 + 1;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
            const block_nvfp4_mmq * __restrict__ y_blocks_j = y_blocks + j0;
            tile_B   B[2];
            uint32_t scaleB[2];
            load_nvfp4_tileB<mmq_x>(B[0], scaleB[0], y_blocks_j, frag0);
            load_nvfp4_tileB<mmq_x>(B[1], scaleB[1], y_blocks_j, frag1);

#pragma unroll
            for (int slab_row0 = 0; slab_row0 < mmq_y; slab_row0 += rows_per_slab) {
                tile_A   A[ntx][2];
                uint32_t scaleA[ntx][2];
                const int i0 = slab_row0 + ty_ntx_div * rows_per_warp;
                const int sum_j = slab_row0 / rows_per_slab * groups_per_slab + j0 / tile_C::J;

#pragma unroll
                for (int n = 0; n < ntx; ++n) {
                    const int row_base = i0 + n * tile_A::I;
                    load_nvfp4_tileA<need_check>(A[n][0], scaleA[n][0], x_blocks, stride_row_x, row_base, k00 / 8 + frag0, i_max);
                    load_nvfp4_tileA<need_check>(A[n][1], scaleA[n][1], x_blocks, stride_row_x, row_base, k00 / 8 + frag1, i_max);
                }

#pragma unroll
                for (int n = 0; n < ntx; ++n) {
                    tile_C C[2] = {};
                    float * __restrict__ sum_n = sum + (sum_j + n) * tile_C::ne;
                    mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C[0], A[n][0], B[0], scaleA[n][0], scaleB[0]);
                    mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C[1], A[n][1], B[1], scaleA[n][1], scaleB[1]);
#pragma unroll
                    for (int l = 0; l < tile_C::ne; ++l) {
                        sum_n[l] += tensor_scale * (C[0].x[l] + C[1].x[l]);
                    }
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot_nvfp4_q8_1_dp4a(
    const int * __restrict__ x,
    const int * __restrict__ y,
    float * __restrict__ sum,
    const int k00) {

    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int k_half = (k00 % MMQ_ITER_K) / MMQ_TILE_NE_K;

    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        const int * __restrict__ yb = y + j * MMQ_TILE_Y_K;
        const int * __restrict__ y_qs_int = yb + 4;

        for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
            const int i = i0 + threadIdx.x;

            const int * __restrict__ x_qs_row = x + i * (4*MMQ_TILE_NE_K);
            const int * __restrict__ x_sc_row = x + mmq_y * (4*MMQ_TILE_NE_K) + i * 16;
            const int * __restrict__ y_qs_lane = y_qs_int;
            const uint8_t * __restrict__ x_sc = reinterpret_cast<const uint8_t *>(x_sc_row);
            const half2 * __restrict__ y_ds_lane = reinterpret_cast<const half2 *>(yb);

            float acc = 0.0f;
#pragma unroll
            for (int lane_rel = 0; lane_rel < 2; ++lane_rel) {
                const int lane = 2 * k_half + lane_rel;
                const uint8_t * __restrict__ sc_lane = x_sc + lane * (QK_NVFP4 / QK_NVFP4_SUB);
                const int * __restrict__ u_lane = y_qs_lane + lane_rel * 16;
                const uint32_t * __restrict__ qs_lane = reinterpret_cast<const uint32_t *>(x_qs_row) + lane * (QK_NVFP4 / 8);
                const float y_d0 = __low2float(y_ds_lane[lane_rel * 2 + 0]);
                const float y_d1 = __low2float(y_ds_lane[lane_rel * 2 + 1]);

#pragma unroll
                for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
                    const int u_base = (sub >> 1) * 8 + (sub & 1) * 4;
                    const uint32_t packed0 = qs_lane[sub * 2 + 0];
                    const uint32_t packed1 = qs_lane[sub * 2 + 1];

                    const int x0 = (int) nvfp4_pack4_i8_packed8(packed0, 0);
                    const int x1 = (int) nvfp4_pack4_i8_packed8(packed0, 4);
                    const int x2 = (int) nvfp4_pack4_i8_packed8(packed1, 0);
                    const int x3 = (int) nvfp4_pack4_i8_packed8(packed1, 4);

                    int sumi = ggml_cuda_dp4a(x0, u_lane[u_base + 0], 0);
                    sumi = ggml_cuda_dp4a(x1, u_lane[u_base + 1], sumi);
                    sumi = ggml_cuda_dp4a(x2, u_lane[u_base + 2], sumi);
                    sumi = ggml_cuda_dp4a(x3, u_lane[u_base + 3], sumi);

                    const float sx = ggml_cuda_ue4m3_to_fp32(sc_lane[sub]);
                    const float sy = sub < 2 ? y_d0 : y_d1;
                    acc += (sx * sy) * float(sumi);
                }
            }

            sum[j0 / nwarps * mmq_y / warp_size + i0 / warp_size] += acc;
        }
    }
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0 % MMQ_TILE_NE_K],
                     x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            float dB;
            const int j = j0 + tile_C::get_j(0);
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            } else {
                dB = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    const float dA = x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + k0/QI8_0];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA*dB;
                }
            }
        }
    }
#else
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
    float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/QI8_0] = x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + k0/QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] =             y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
                }
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_0], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}


template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + k0/QI8_1], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    float2 dmA = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + k0/QI8_1]);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.x*dsB.x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.y*dsB.y;
                }
            }
        }
    }
#else
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

// Used for NVFP4, Q3_K, IQ2_S, and IQ2_XS
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q8_0_16_q8_1_impl<QI8_0>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0],
                    &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + k0/(QI8_0/2)],
                    y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

// Used for Q3_K, IQ2_S, and IQ2_XS:
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q3_K + k0, MMQ_MMA_TILE_X_K_Q3_K);
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * x_df[i*MMQ_MMA_TILE_X_K_Q3_K + k0/4] * dB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/8], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q3_K + k0, MMQ_MMA_TILE_X_K_Q3_K);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*MMQ_MMA_TILE_X_K_Q3_K + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR2_K);
    constexpr int nrows = ggml_cuda_get_physical_warp_size() / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2*l)) & 0x03030303;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q2_K + k] = x_qs_k;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x*(sc_m & 0x0F), bxi_dmf.y*(sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + kqsx] = x_dm_ik;
#else
        x_dm[i*(MMQ_TILE_NE_K + 1)   + kqsx] = x_dm_ik;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    float2 y_df[mmq_x/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        y_df[j0/nwarps] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
    }

#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K/2; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                constexpr int ns = 2;
                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }
        }
    }

    // Some compilers fail to unroll the loop over k01 if there is a conditional statement for ns in the inner loop.
    // As a workaround 2 separate loops are used instead.
#pragma unroll
    for (int k01 = MMQ_TILE_NE_K/2; k01 < MMQ_TILE_NE_K; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                constexpr int ns = 1;
                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }
        }
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q2_K + k0, MMQ_MMA_TILE_X_K_Q2_K);
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = (k01 < MMQ_TILE_NE_K/2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K]).x : __half22float2(y_ds[j*MMQ_TILE_Y_K]).y;
            const float sB = (k01 >= MMQ_TILE_NE_K * 3/4) ? 0
                                              : (((k01/4)%2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).y
                                                             : __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).x);

            tile_C Cm;
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
#pragma unroll
                for (int l = 0; l < tile_A::ne; ++l) {
                    A1.x[l] = 0x01010101;
                }
                mma(Cm, A1, B);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd;
                mma(Cd, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const float2 dm = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + k0/4]);
                    float tmp = Cd.x[l]*dm.x;
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm.x[l]*dm.y;
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*dB;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= dm.y*sB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];
    float  mA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/QI8_1], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q2_K + k0, MMQ_MMA_TILE_X_K_Q2_K);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1/2) {
                const int k0 = k00 + k01;

                const float2 dm = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q2_K + k0/(QI8_1/2)]);

                dA[n][l][k01/(QI8_1/2)] = dm.x;
                mA[n][l][k01/(QI8_1/2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
        float2 dB[tile_C::ne/2];

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int j = j0 + tile_C::get_j(l);

            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B B[2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

            tile_C Cm[2];
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                mma(Cm[0], A1, B[0]);
                mma(Cm[1], A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd[2];

                mma(Cd[0], A[n][k01/4 + 0], B[0]);
                mma(Cd[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    float tmp = Cd[0].x[l]*dA[n][l/2][k01/4 + 0] + Cd[1].x[l]*dA[n][l/2][k01/4 + 1];
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm[0].x[l]*mA[n][l/2][k01/4 + 0] + Cm[1].x[l]*mA[n][l/2][k01/4 + 1];
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*(k01 < MMQ_TILE_NE_K/2 ? dB[l%2].x : dB[l%2].y);
                }
            }
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K * 3/4; k01 += QI8_1) {
            float2 sB[tile_C::ne/2];

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 0]*sB[l%2].x;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 1]*sB[l%2].y;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR3_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + k] = x_qs_k;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/4;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int ksc = threadIdx.x % 4;

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        const int8_t * sc8 = (const int8_t *) &sc;
        const float d = bxi->d;

#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_df[i*MMQ_MMA_TILE_X_K_Q3_K + sizeof(int)*ksc + l] = d*sc8[l];
        }
#else
        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = sc;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if !(defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE))
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        x_df[i] = bxi->d;
    }
#endif // !(defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)) || defined(AMD_WMMA_AVAILABLE)
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const int8_t * scales = ((const int8_t *) (x_sc + i*(MMQ_TILE_NE_K/8) + i/8)) + k0/4;

                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q3_K_q8_1_impl_mmq(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], scales,
                    x_df[i], y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        // Need if on AMD instead of % because warp_size == 64
        // This causes double work and throughput loss (MI300X)
        // H100 loses about 100 t/s with 'if' condition over '%'
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/2;
        if (i < mmq_y) {
#else
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % mmq_y;
        {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            if (need_check) {
                i = min(i, i_max);
            }

            const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

    #pragma unroll
            for (int l = 0; l < sizeof(int); ++l) {
                x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
#else
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

        x_dm[i] = bxi->dm;
    }
    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (MMQ_TILE_NE_K/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = scales8;
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = (const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/32] + 2*(k01/16);

                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[i*(MMQ_TILE_NE_K + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;
        const int ky = QR5_K*txi;

        const int ql = get_int_b4(bxi->qs, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, txi % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (txi / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (txi / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + QI5_K/4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kq0] = ql0 | qh0;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + kq1] = ql1 | qh1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq0] = ql0 | qh0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq1] = ql1 | qh1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
#if defined(AMD_MFMA_AVAILABLE)
        // Need if on AMD instead of % because warp_size == 64
        // This causes double work and throughput loss (MI300X)
        // H100 loses about 100 t/s with 'if' condition over '%'
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/2;
        if (i < mmq_y) {
#else
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % mmq_y;
        {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            if (need_check) {
                i = min(i, i_max);
            }

            const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
            for (int l = 0; l < int(sizeof(int)); ++l) {
                x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
#else
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

        x_dm[i] = bxi->dm;
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (MMQ_TILE_NE_K/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = scales8;
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR5_K*VDR_Q5_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = ((const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k00/32]) + 2*(k01/16);

                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q5_K_q8_1_impl_mmq(
                    &x_qs[i*(QR5_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/QI6_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR6_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*txi - txi % (QI6_K/2) + 0;
        const int kq1 = 2*txi - txi % (QI6_K/2) + QI6_K/2;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q6_K]           = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + threadIdx.x%4] = get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8));
#else
        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + threadIdx.x%(MMQ_TILE_NE_K/8)] = get_int_b2(bxi->scales, threadIdx.x%(QI6_K/8));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR6_K*VDR_Q6_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const int8_t * sc = ((const int8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/16]);

                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q6_K_q8_1_impl_mmq(
                    &x_qs[i*(QR6_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc,
                    x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q6_K + k0, MMQ_MMA_TILE_X_K_Q6_K);
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const int8_t * sc = (const int8_t *) (x_sc + i*MMQ_MMA_TILE_X_K_Q6_K + k00/16);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * sc[k01/4] * x_df[i*MMQ_MMA_TILE_X_K_Q6_K] * dB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A   A[ntx][8];
    int    scA[ntx][tile_C::ne/2][8];
    float   dA[ntx][tile_C::ne/2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/4 + 0], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (k0 + 0),         MMQ_MMA_TILE_X_K_Q6_K);
            load_ldmatrix(A[n][k01/4 + 1], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q6_K + (k0 + tile_A::J), MMQ_MMA_TILE_X_K_Q6_K);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

                const int      sc_packed = x_sc[i*MMQ_MMA_TILE_X_K_Q6_K + k0/16];
                const int8_t * sc        = (const int8_t *) &sc_packed;

#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

            dA[n][l] = x_df[i*MMQ_MMA_TILE_X_K_Q6_K];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
        float tmp[ntx][tile_C::ne] = {{0.0f}};

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + 0         + k01, MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + tile_B::J + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_nl(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_NL, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_NL);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_NL;
    const int kqsx = txi % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbx;

        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = kbx * (2 * QI4_NL) + kqsx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0]      = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + QI4_NL] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0]      = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI4_NL] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_NL;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0             + kbxd] = __half2float(bxi->d);
#else
        x_df[i*(MMQ_TILE_NE_K/QI4_NL) + i/QI4_NL + kbxd] = __half2float(bxi->d);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_xxs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XXS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_xxs * bxi = (const block_iq2_xxs *) x + kbx0 + i*stride;

        const int q2 = get_int_b2(bxi->qs, 2*kqsx+0);
        const uint8_t * aux8 = (const uint8_t *) &q2;
        const uint32_t aux32 = get_int_b2(bxi->qs, 2*kqsx+1);

#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[l]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 1)] = grid1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0   + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_xs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_xs * bxi = (const block_iq2_xs *) x + kbx0 + i*stride;

        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;

    #pragma unroll
        for (int l = 0; l < QR2_XS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l] >> 9);

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K                   + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K                   + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_s(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_S, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_s * bxi = (const block_iq2_s *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->qs, QK_K/32 + kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR2_S; ++l) {
            const int * grid_pos = (const int *)(iq2s_grid + (qs[l] | ((qh << (8-2*l)) & 0x300)));

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K                   + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K                   + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_xxs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_XXS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_xxs * bxi = (const block_iq3_xxs *) x + kbx0 + i*stride;

        const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * q3 = (const uint8_t *) &q3_packed;
        const uint32_t aux32 = get_int_b2(bxi->qs, QK_K/16 + kqsx);

#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[2*l+0]], iq3xxs_grid[q3[2*l+1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7*l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = aux32 >> 28;
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0     + kqsx] = (ls*d + d/2)/2;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4   + kqsx] = (ls*d + d/2)/2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_s(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_s * bxi = (const block_iq3_s *) x + kbx0 + i*stride;

        const int2      qs_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->signs, kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR3_S; ++l) {
            const int2 grid_pos = make_int2(
                iq3s_grid[qs[2*l+0] | ((qh << (8 - 2*l)) & 0x100)],
                iq3s_grid[qs[2*l+1] | ((qh << (7 - 2*l)) & 0x100)]);

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + (2*l+1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = 1 + 2*((bxi->scales[kqsx/2] >> (((2*kqsx) << 1) & 0x04)) & 0x0F);
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0     + kqsx] = ls*d;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4   + kqsx] = ls*d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq1_s(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR1_S);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq1_s * bxi = (const block_iq1_s *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

    #pragma unroll
        for (int l = 0; l < QR1_S/2; ++l) {
            const int grid = iq1s_grid_gpu[qs[l] | (((qh >> (3*l)) & 0x07) << 8)];

            const int grid0 = (grid >> 0) & 0x0F0F0F0F;
            const int grid1 = (grid >> 4) & 0x0F0F0F0F;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 8*kqsx + (2*l+1)] = grid1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+1)] = grid1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const float  d1q   = __half2float(bxi->d) * (((qh >> 11) & 0x0E) + 1);
        const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_ds[i*MMQ_MMA_TILE_X_K_Q8_1     + kqsx] = make_half2(d1q, d1q*delta);
#else
        x_ds[i*(MMQ_TILE_NE_K/4) + i/4   + kqsx] = make_half2(d1q, d1q*delta);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_xs(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_XS);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const int aux_q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = 8 * (kqsx / 4) + kqsx % 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 4] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0] = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int rows_per_warp = warp_size / 8;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / (MMQ_TILE_NE_K/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const float d = __half2float(bxi->d);

        const int ls = ((bxi->scales_l[(threadIdx.x % 8)/2] >> (4*(threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2*(threadIdx.x % 8))) & 0x03) << 4);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0   + threadIdx.x % 8] = d * (ls - 32);
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + threadIdx.x % 8] = d * (ls - 32);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template<int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_dp4a(
        const float * __restrict__ sum, const int32_t * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[ids_dst[j]*stride + i] = sum[(j0/nwarps) * (mmq_y/warp_size) + i0/warp_size];
        }
    }
}

template<ggml_type type, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
        const float * __restrict__ sum, const int32_t * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int tileC_IJ = mmq_get_granularity_device(0);
    typedef tile<tileC_IJ, tileC_IJ, int, DATA_LAYOUT_J_MAJOR> tile_C;
    constexpr int rows_per_warp = granularity;
#else
    typedef tile<16, 8, int> tile_C;
    constexpr int rows_per_warp = 2 * granularity;
#endif // defined(AMD_MFMA_AVAILABLE)
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    static_assert(nwarps*tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");
#else
    GGML_UNUSED(nwarps);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (need_check && i > i_max) {
                    continue;
                }

                dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------

template <int mmq_x, int mmq_y, bool need_check, ggml_type type>
struct mmq_type_traits;

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q1_0> {
    static constexpr int              vdr          = VDR_Q1_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q1_0<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q4_0> {
    static constexpr int              vdr          = VDR_Q4_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_0<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q4_1> {
    static constexpr int              vdr          = VDR_Q4_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_1<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_1_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q5_0> {
    static constexpr int              vdr          = VDR_Q5_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_0<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q5_1> {
    static constexpr int              vdr          = VDR_Q5_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_1<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_1_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q8_0> {
    static constexpr int              vdr          = VDR_Q8_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q8_0<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_MXFP4> {
    static constexpr int              vdr          = VDR_MXFP4_Q8_1_MMQ;
#ifdef BLACKWELL_MMA_AVAILABLE
    static constexpr load_tiles_mmq_t load_tiles  = load_tiles_mxfp4_fp4<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_fp4_fp4_mma<mmq_x, mmq_y, GGML_TYPE_MXFP4>;
#else
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_mxfp4<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
#endif // BLACKWELL_MMA_AVAILABLE
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_MXFP8> {
    static constexpr int vdr = VDR_MXFP8_Q8_1_MMQ;
#ifdef BLACKWELL_MMA_AVAILABLE
    static constexpr load_tiles_mmq_t load_tiles = load_tiles_mxfp8_native<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t vec_dot_mma = vec_dot_mxfp8_mxfp8_mma<mmq_x, mmq_y>;
#else
    static constexpr load_tiles_mmq_t load_tiles = load_tiles_mxfp8<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t vec_dot_mma =
        vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
#endif
    static constexpr vec_dot_mmq_t vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_MXFP6_E2M3> {
    static constexpr int              vdr          = VDR_MXFP6_E2M3_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_mxfp6<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_NVFP4> {
#ifdef BLACKWELL_MMA_AVAILABLE
    static constexpr int              vdr          = VDR_NVFP4_NVFP4_MMQ;
#else
    static constexpr int              vdr          = VDR_NVFP4_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_nvfp4<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
#endif // BLACKWELL_MMA_AVAILABLE
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q2_K> {
    static constexpr int              vdr          = VDR_Q2_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q2_K<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q2_K_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q2_K_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q3_K> {
    static constexpr int              vdr          = VDR_Q3_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q3_K<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q3_K_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q4_K> {
    static constexpr int              vdr          = VDR_Q4_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_K<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_K_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q5_K> {
    static constexpr int              vdr          = VDR_Q5_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_K<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_K_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q6_K> {
    static constexpr int              vdr          = VDR_Q6_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q6_K<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q6_K_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q6_K_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ2_XXS> {
    static constexpr int              vdr          = VDR_IQ2_XXS_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_xxs<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ2_XS> {
    static constexpr int              vdr          = VDR_IQ2_XS_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_xs<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ2_S> {
    static constexpr int              vdr          = VDR_IQ2_S_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_s<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ3_XXS> {
    static constexpr int              vdr          = VDR_IQ3_XXS_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_xxs<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ3_S> {
    static constexpr int              vdr          = VDR_IQ3_S_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_s<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ1_S> {
    static constexpr int              vdr          = VDR_IQ1_S_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq1_s<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_1_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ4_NL> {
    static constexpr int              vdr          = VDR_IQ4_NL_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_nl<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ4_XS> {
    static constexpr int              vdr          = VDR_IQ4_XS_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_xs<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check, bool fixup>
static __device__ __forceinline__ void process_nvfp4_tiles(const char * __restrict__ x,
        const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const uint32_t scale_channel) {
    // Much simpler than mmq_process_tile because the repacked block
    // contains the tensor scale with the weights, so  the vecdot directly consumes
    // from src0 directly from GMEM. This lets us skip smem B staging and do 1 vecdot per kblock.
    // This gives a big speedup while increasing correctness.
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int blocks_per_tile = QK_K / QK_NVFP4; // 4;

    const block_nvfp4_mmq * __restrict__ y_nv = reinterpret_cast<const block_nvfp4_mmq *>(y);
    const block_nvfp4_blackwell_tensor * __restrict__ x_tensor = (const block_nvfp4_blackwell_tensor *) x;
    float weight_scale = x_tensor->weight_scales ? x_tensor->weight_scales[scale_channel] : x_tensor->weight_scale;
    float input_scale  = x_tensor->input_scales  ? x_tensor->input_scales[scale_channel]  : x_tensor->input_scale;
    weight_scale = weight_scale > 0.0f && isfinite(weight_scale) ? weight_scale : 1.0f;
    input_scale  = input_scale  > 0.0f && isfinite(input_scale)  ? input_scale  : 1.0f;
    const float tensor_scale = weight_scale * input_scale;

    const int k_block_start = kb0_start / 4;
    const int k_block_stop  = (kb0_stop + blocks_per_tile - 1) / blocks_per_tile;

    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};
    const block_nvfp4_blackwell * __restrict__ x_blocks = x_tensor->tiles + offset_x;
    for (int k_block = k_block_start; k_block < k_block_stop; ++k_block) {
        vec_dot_nvfp4_nvfp4_mma<mmq_x, mmq_y, need_check>(
            x_blocks + k_block, stride_row_x, y_nv + ncols_y * k_block,
            sum, 0, tile_x_max_i, tensor_scale);
    }

    constexpr mmq_write_back_t write_back = mmq_write_back_mma<GGML_TYPE_NVFP4, mmq_x, mmq_y, need_check>;

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        write_back(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

#if defined(BLACKWELL_MMA_AVAILABLE)
template <int mmq_x, int mmq_y, bool need_check, bool fixup>
static __device__ __forceinline__ void process_mxfp6_tiles(const char * __restrict__ x,
        const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const uint32_t scale_channel) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = mmq_get_nwarps_device();

    const tile_mxfp6_e2m3_mmq * __restrict__ y_mx = reinterpret_cast<const tile_mxfp6_e2m3_mmq *>(y);
    const tile_mxfp6_e2m3_blackwell_tensor * __restrict__ x_tensor = (const tile_mxfp6_e2m3_blackwell_tensor *) x;
    const float tensor_scale = x_tensor->weight_scales ? x_tensor->weight_scales[scale_channel] : x_tensor->weight_scale;
    const tile_mxfp6_e2m3_blackwell * __restrict__ x_blocks = x_tensor->tiles + offset_x;
    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    if constexpr (need_check) {
        if (tile_x_max_i >= mmq_y - 1) {
            constexpr int k_group = GGML_CUDA_MXFP6_MMQ_KGROUP;
            int k_block = kb0_start;
            for (; k_block + k_group - 1 < kb0_stop; k_block += k_group) {
                if constexpr (k_group == 3) {
                    vec_dot_mxfp6_e2m3_mxfp6_e2m3_mma_k3_scoped<mmq_x, mmq_y, false>(
                        x_blocks + k_block, stride_row_x, y_mx + ncols_y * k_block,
                        ncols_y, sum, tile_x_max_i, tensor_scale);
                } else {
                    vec_dot_mxfp6_e2m3_mxfp6_e2m3_mma_kgroup_streamed<mmq_x, mmq_y, false, k_group>(
                        x_blocks + k_block, stride_row_x, y_mx + ncols_y * k_block,
                        ncols_y, sum, tile_x_max_i, tensor_scale);
                }
            }
            for (; k_block < kb0_stop; ++k_block) {
                vec_dot_mxfp6_e2m3_mxfp6_e2m3_mma<mmq_x, mmq_y, false>(
                    x_blocks + k_block, stride_row_x, y_mx + ncols_y * k_block,
                    sum, 0, tile_x_max_i, tensor_scale);
            }

            constexpr mmq_write_back_t write_back_checked = mmq_write_back_mma<GGML_TYPE_MXFP6_E2M3, mmq_x, mmq_y, true>;

            if (fixup) {
                write_back_checked(sum, ids_dst, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
            } else {
                write_back_checked(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
            }
            return;
        }
    }

    constexpr int k_group = GGML_CUDA_MXFP6_MMQ_KGROUP;
    int k_block = kb0_start;
    for (; k_block + k_group - 1 < kb0_stop; k_block += k_group) {
        if constexpr (k_group == 3) {
            vec_dot_mxfp6_e2m3_mxfp6_e2m3_mma_k3_scoped<mmq_x, mmq_y, need_check>(
                x_blocks + k_block, stride_row_x, y_mx + ncols_y * k_block,
                ncols_y, sum, tile_x_max_i, tensor_scale);
        } else {
            vec_dot_mxfp6_e2m3_mxfp6_e2m3_mma_kgroup_streamed<mmq_x, mmq_y, need_check, k_group>(
                x_blocks + k_block, stride_row_x, y_mx + ncols_y * k_block,
                ncols_y, sum, tile_x_max_i, tensor_scale);
        }
    }
    for (; k_block < kb0_stop; ++k_block) {
        vec_dot_mxfp6_e2m3_mxfp6_e2m3_mma<mmq_x, mmq_y, need_check>(
            x_blocks + k_block, stride_row_x, y_mx + ncols_y * k_block,
            sum, 0, tile_x_max_i, tensor_scale);
    }

    constexpr mmq_write_back_t write_back = mmq_write_back_mma<GGML_TYPE_MXFP6_E2M3, mmq_x, mmq_y, need_check>;

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        write_back(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

template <int mmq_x, int mmq_y, bool need_check, bool fixup>
static __device__ __forceinline__ void process_mxfp6_tiles_fp8(const char * __restrict__ x,
        const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const uint32_t scale_channel) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = mmq_get_nwarps_device();

    const block_fp8 * __restrict__ y_fp8 = reinterpret_cast<const block_fp8 *>(y);
    const tile_mxfp6_e2m3_blackwell_tensor * __restrict__ x_tensor = (const tile_mxfp6_e2m3_blackwell_tensor *) x;
    const float tensor_scale = x_tensor->weight_scales ? x_tensor->weight_scales[scale_channel] : x_tensor->weight_scale;
    const tile_mxfp6_e2m3_blackwell * __restrict__ x_blocks = x_tensor->tiles + offset_x;
    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    if constexpr (need_check) {
        if (tile_x_max_i >= mmq_y - 1) {
            constexpr int k_group = GGML_CUDA_MXFP6_FP8_MMQ_KGROUP;
            int k_frag = kb0_start;
            for (; k_frag + k_group - 1 < kb0_stop; k_frag += k_group) {
                vec_dot_mxfp6_e2m3_fp8_e4m3_mma_kgroup_scoped<mmq_x, mmq_y, false, k_group>(
                    x_blocks + k_frag, stride_row_x, y_fp8, ncols_y, sum, k_frag, tile_x_max_i, tensor_scale);
            }
            for (; k_frag < kb0_stop; ++k_frag) {
                vec_dot_mxfp6_e2m3_fp8_e4m3_mma_frag<mmq_x, mmq_y, false>(
                    x_blocks, stride_row_x, y_fp8, ncols_y, sum, k_frag, tile_x_max_i, tensor_scale);
            }

            constexpr mmq_write_back_t write_back_checked = mmq_write_back_mma<GGML_TYPE_MXFP6_E2M3, mmq_x, mmq_y, true>;

            if (fixup) {
                write_back_checked(sum, ids_dst, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
            } else {
                write_back_checked(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
            }
            return;
        }
    }

    constexpr int k_group = GGML_CUDA_MXFP6_FP8_MMQ_KGROUP;
    int k_frag = kb0_start;
    for (; k_frag + k_group - 1 < kb0_stop; k_frag += k_group) {
        vec_dot_mxfp6_e2m3_fp8_e4m3_mma_kgroup_scoped<mmq_x, mmq_y, need_check, k_group>(
            x_blocks + k_frag, stride_row_x, y_fp8, ncols_y, sum, k_frag, tile_x_max_i, tensor_scale);
    }
    for (; k_frag < kb0_stop; ++k_frag) {
        vec_dot_mxfp6_e2m3_fp8_e4m3_mma_frag<mmq_x, mmq_y, need_check>(
            x_blocks, stride_row_x, y_fp8, ncols_y, sum, k_frag, tile_x_max_i, tensor_scale);
    }

    constexpr mmq_write_back_t write_back = mmq_write_back_mma<GGML_TYPE_MXFP6_E2M3, mmq_x, mmq_y, need_check>;

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        write_back(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

#else
template <int mmq_x, int mmq_y, bool need_check, bool fixup>
static __device__ __forceinline__ void process_mxfp6_tiles(const char * __restrict__ x,
        const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const uint32_t scale_channel) {
    GGML_UNUSED_VARS(x, offset_x, y, ids_dst, dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
        tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, scale_channel);
    NO_DEVICE_CODE;
}

#endif // defined(BLACKWELL_MMA_AVAILABLE)

template <ggml_type type, int mmq_x, bool need_check, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int              nwarps     = mmq_get_nwarps_device();
    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              mmq_y      = get_mmq_y_device();
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, mmq_y, need_check, type>::load_tiles;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_mma;
    constexpr mmq_write_back_t write_back = mmq_write_back_mma<type, mmq_x, mmq_y, need_check>;
#else
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_dp4a;
    constexpr mmq_write_back_t write_back = mmq_write_back_dp4a<mmq_x, mmq_y, need_check>;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

#if defined(BLACKWELL_MMA_AVAILABLE)
    // FP4 tile stores 8 blocks
    constexpr int ne_block = type == GGML_TYPE_MXFP4 ? QK_K : 4 * QK8_1;
#else
    constexpr int ne_block = 4 * QK8_1;
#endif  // defined(BLACKWELL_MMA_AVAILABLE)

    constexpr int ITER_K          = get_iter_k(type);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        write_back(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}


// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <ggml_type type, int mmq_x, bool need_check, bool native_mxfp6 = false, bool mxfp6_fp8_acts = false>
#if defined(GGML_USE_HIP)
#if defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(CDNA) || defined(GCN)
    __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 2)
#endif // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(CDNA) || defined(GCN)
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#if defined(BLACKWELL_MMA_AVAILABLE)
    __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 1)
#else
    __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 1)
#endif // defined(BLACKWELL_MMA_AVAILABLE)
#else
    __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 2)
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // defined(GGML_USE_HIP)
static __global__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx) {

    // Skip unused template specializations for faster compilation:
    if (mmq_x > get_mmq_x_max_device() || mmq_x % mmq_get_granularity_device(mmq_x) != 0) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr int qk    = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y = get_mmq_y_device();
    int y_block_ints = int(sizeof(block_q8_1_mmq)/sizeof(int));
#if defined(BLACKWELL_MMA_AVAILABLE)
    if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
        y_block_ints = native_mxfp6 ?
            (mxfp6_fp8_acts ? int(sizeof(block_fp8)/sizeof(int)) : int(sizeof(tile_mxfp6_e2m3_mmq)/sizeof(int))) :
            int(sizeof(block_q8_1_mmq)/sizeof(int));
    }
    if constexpr (type == GGML_TYPE_NVFP4) {
        y_block_ints = int(sizeof(block_nvfp4_mmq)/sizeof(int));
    }
#endif // defined(BLACKWELL_MMA_AVAILABLE)

    const uint32_t nty = (nrows_x + mmq_y - 1) / mmq_y; // Number of tiles y

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

        if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    // On non-CDNA AMD or old CUDA the performance with stream-k was worse, use conventional tiling instead:
#if (defined(GGML_USE_HIP) && !defined(CDNA)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
    {
        const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
        const int wt = tmp2.x;
        const int zt = tmp2.y;
        const int jt = blockIdx.y;
        const int it = blockIdx.x;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y   = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst;

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;

            if (jt*mmq_x >= col_diff) {
                return;
            }

            // __syncthreads(); // There is no previous tile that could cause a race condition.
#pragma unroll
            for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*mmq_x + j];
            }
            __syncthreads();
        }

        offset_y   += (col_low + jt*mmq_x)*y_block_ints;
        offset_dst += it*mmq_y;

        const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
        const int tile_y_max_j = col_diff - jt*mmq_x - 1;

        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;

        constexpr bool fixup = false;
#if defined(BLACKWELL_MMA_AVAILABLE)
        if constexpr (type == GGML_TYPE_NVFP4) {
            const int sample_x = fastdiv(wt, sample_ratio);
            const int channel_x = fastdiv(zt, channel_ratio);
            const int offset_x_fp4 = sample_x*stride_sample_x + channel_x*stride_channel_x + (it*mmq_y / 16)*stride_row_x;
            const int scale_channel = ids_dst ? zt : channel_x;
            process_nvfp4_tiles<mmq_x, mmq_y, need_check, fixup>
                (x, offset_x_fp4, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y,
                 stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z, scale_channel);
        } else if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
            if (native_mxfp6) {
                const int sample_x = fastdiv(wt, sample_ratio);
                const int channel_x = fastdiv(zt, channel_ratio);
                const int offset_x_mx = sample_x*stride_sample_x + channel_x*stride_channel_x + (it*mmq_y / 16)*stride_row_x;
                const int scale_channel = ids_dst ? zt : channel_x;
                if constexpr (mxfp6_fp8_acts) {
                    process_mxfp6_tiles_fp8<mmq_x, mmq_y, need_check, fixup>
                        (x, offset_x_mx, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y,
                         stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z, scale_channel);
                } else {
                    process_mxfp6_tiles<mmq_x, mmq_y, need_check, fixup>
                        (x, offset_x_mx, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y,
                         stride_col_dst, tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z, scale_channel);
                }
            } else {
                mul_mat_q_process_tile<type, mmq_x, need_check, fixup>
                    (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
                     tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
            }
        } else
#endif // defined(BLACKWELL_MMA_AVAILABLE)
        mul_mat_q_process_tile<type, mmq_x, need_check, fixup>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
        return;
    }
#endif // (defined(GGML_USE_HIP) && !defined(CDNA4) && !defined(CDNA3)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA

    constexpr int ITER_K          = get_iter_k(type);
    constexpr int blocks_per_iter = ITER_K / qk;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        int tmp = fastdiv(kbc, blocks_per_ne00);
        uint2 tmp2 = fast_div_modulo(tmp, ntx);
        const int jt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nchannels_y);
        const int zt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nsamples_y);
        const int wt = tmp2.y;
        const int it = tmp2.x;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y   = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst;

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;

            if (jt*mmq_x >= col_diff) {
                kbc += blocks_per_ne00.z;
                kbc -= fastmodulo(kbc, blocks_per_ne00);

                kb0_start = 0;
                kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));

                continue;
            }

            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*mmq_x + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * mmq_x) * y_block_ints;
        offset_dst += it*mmq_y;

        const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
        const int tile_y_max_j = col_diff - jt*mmq_x - 1;

        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
#if defined(BLACKWELL_MMA_AVAILABLE)
        if constexpr (type == GGML_TYPE_NVFP4) {
            const int sample_x = fastdiv(wt, sample_ratio);
            const int channel_x = fastdiv(zt, channel_ratio);
            const int offset_x_fp4 = sample_x*stride_sample_x + channel_x*stride_channel_x + (it*mmq_y / 16)*stride_row_x;
            const int scale_channel = ids_dst ? zt : channel_x;
            process_nvfp4_tiles<mmq_x, mmq_y, need_check, fixup>
                (x, offset_x_fp4, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y,
                 stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, scale_channel);
        } else if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
            if (native_mxfp6) {
                const int sample_x = fastdiv(wt, sample_ratio);
                const int channel_x = fastdiv(zt, channel_ratio);
                const int offset_x_mx = sample_x*stride_sample_x + channel_x*stride_channel_x + (it*mmq_y / 16)*stride_row_x;
                const int scale_channel = ids_dst ? zt : channel_x;
                if constexpr (mxfp6_fp8_acts) {
                    process_mxfp6_tiles_fp8<mmq_x, mmq_y, need_check, fixup>
                        (x, offset_x_mx, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y,
                         stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, scale_channel);
                } else {
                    process_mxfp6_tiles<mmq_x, mmq_y, need_check, fixup>
                        (x, offset_x_mx, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y,
                         stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, scale_channel);
                }
            } else {
                mul_mat_q_process_tile<type, mmq_x, need_check, fixup>
                    (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
                     tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
            }
        } else
#endif // defined(BLACKWELL_MMA_AVAILABLE)
        mul_mat_q_process_tile<type, mmq_x, need_check, fixup>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = fastdiv(kbc, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;

    // Defaults for regular matrix multiplication:
    int col_low    = 0;
    int col_high   = ncols_dst;
    int col_diff   = ncols_dst;
    int offset_y   = wt*stride_sample_y   + zt*stride_channel_y;
    int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst;

    if (ids_dst) {
        col_low  = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y   = 0;
        offset_dst = 0;

        if (jt*mmq_x >= col_diff) {
            return;
        }

        // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
            const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

            if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) {
                break;
            }

            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y += (col_low + jt * mmq_x) * y_block_ints;
    offset_dst += it*mmq_y;

    const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
    const int tile_y_max_j = col_diff - jt*mmq_x - 1;

    const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
#if defined(BLACKWELL_MMA_AVAILABLE)
    if constexpr (type == GGML_TYPE_NVFP4) {
        const int sample_x = fastdiv(wt, sample_ratio);
        const int channel_x = fastdiv(zt, channel_ratio);
        const int offset_x_fp4 = sample_x*stride_sample_x + channel_x*stride_channel_x + (it*mmq_y / 16)*stride_row_x;
        const int scale_channel = ids_dst ? zt : channel_x;
        process_nvfp4_tiles<mmq_x, mmq_y, need_check, fixup>
            (x, offset_x_fp4, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y,
             stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, scale_channel);
    } else if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
        if (native_mxfp6) {
            const int sample_x = fastdiv(wt, sample_ratio);
            const int channel_x = fastdiv(zt, channel_ratio);
            const int offset_x_mx = sample_x*stride_sample_x + channel_x*stride_channel_x + (it*mmq_y / 16)*stride_row_x;
            const int scale_channel = ids_dst ? zt : channel_x;
            if constexpr (mxfp6_fp8_acts) {
                process_mxfp6_tiles_fp8<mmq_x, mmq_y, need_check, fixup>
                    (x, offset_x_mx, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y,
                     stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, scale_channel);
            } else {
                process_mxfp6_tiles<mmq_x, mmq_y, need_check, fixup>
                    (x, offset_x_mx, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y,
                     stride_col_dst, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, scale_channel);
            }
        } else {
            mul_mat_q_process_tile<type, mmq_x, need_check, fixup>
                (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
                 tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
        }
    } else
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    mul_mat_q_process_tile<type, mmq_x, need_check, fixup>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}

template <ggml_type type, int mmq_x, bool need_check>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device()/2, 1)
static __global__ void mul_mat_q_stream_k_fixup(
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
        float * __restrict__ tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
        const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst, const uint3 nsamples_y,
        const int stride_sample_dst, const uint3 ntx) {
    constexpr int mmq_y           = get_mmq_y_device();
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int ITER_K          = get_iter_k(type);
    constexpr int blocks_per_iter = ITER_K / qk;

    constexpr int nwarps = mmq_get_nwarps_device()/2;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    float sum[mmq_x / nwarps] = {0.0f};
    const int i = blockIdx.y*warp_size + threadIdx.x;

    const int nty = (nrows_x + mmq_y - 1) / mmq_y;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc0      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc0_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        int kbc = int64_t(bidx)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;


#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps] += tmp_last_tile[bidx*(mmq_x*mmq_y) + j*mmq_y + i];
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    int tmp = fastdiv(kbc0, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;

    if (!ids_dst) {
        const int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst + it*mmq_y;
        dst += offset_dst;

        const int i_max = nrows_x   - it*mmq_y - 1;
        const int j_max = ncols_dst - jt*mmq_x - 1;
        if (need_check && i > i_max) {
            return;
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j > j_max) {
                return;
            }

            dst[j*stride_col_dst + i] += sum[j0/nwarps];
        }
        return;
    }

    __shared__ int ids_dst_shared[mmq_x];
    const int col_low  = expert_bounds[zt + 0];
    const int col_high = expert_bounds[zt + 1];
    const int col_diff = col_high - col_low;

    for (int j = threadIdx.y*warp_size + threadIdx.x; j < mmq_x; j += nwarps*warp_size) {
        ids_dst_shared[j] = j < col_diff - jt*mmq_x ? ids_dst[col_low + jt*mmq_x + j] : 0;
    }
    __syncthreads();

    const int offset_dst = it*mmq_y;
    dst += offset_dst;

    const int i_max = nrows_x  - it*mmq_y - 1;
    const int j_max = col_diff - jt*mmq_x - 1;
    if (need_check && i > i_max) {
        return;
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

        dst[ids_dst_shared[j]*stride_col_dst + i] += sum[j0/nwarps];
    }
}

struct mmq_args {
    const char * x; ggml_type type_x; const int * y; const int32_t * ids_dst; const int32_t * expert_bounds; float * dst;
    int64_t ncols_x; int64_t nrows_x; int64_t ncols_dst; int64_t stride_row_x; int64_t ncols_y; int64_t nrows_dst;
    int64_t nchannels_x; int64_t nchannels_y; int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
    int64_t nsamples_x; int64_t nsamples_y; int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
    bool use_stream_k; bool native_mxfp6; bool mxfp6_fp8_acts; int64_t ncols_max;
};

template<ggml_type type>
static size_t mmq_get_nbytes_shared(
        const int mmq_x, const int mmq_y, const int cc, const int warp_size, const int nwarps, const bool native_mxfp6 = false) {
    if constexpr (type == GGML_TYPE_NVFP4) {
        if (blackwell_mma_available(cc)) {
            GGML_UNUSED_VARS(mmq_y, cc, warp_size, nwarps);
            return mmq_x*sizeof(int);
        }
    }
    if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
        if (native_mxfp6 && blackwell_mma_available(cc)) {
            GGML_UNUSED_VARS(mmq_y, cc, warp_size, nwarps);
            return mmq_x*sizeof(int);
        }
    }
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);
    const int mmq_tile_x_k = mmq_get_mma_tile_x_k(type);
    const size_t nbs_ids = mmq_x*sizeof(int);
    const size_t nbs_x = (turing_mma_available(cc) || amd_mfma_available(cc) || amd_wmma_available(cc)) ? mmq_y*mmq_tile_x_k*sizeof(int) : txs.qs*sizeof(int) + txs.dm*sizeof(half2) + txs.sc*sizeof(int);
    const size_t nbs_y = mmq_x * (sizeof(block_q8_1_mmq));
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, nwarps*warp_size*sizeof(int));
}

template <ggml_type type, int mmq_x>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const int nwarps = mmq_get_nwarps_host(cc, warp_size);
    const int mmq_y = get_mmq_y_host(cc);

    const dim3 block_dims(warp_size, nwarps, 1);

    const int nbytes_shared = mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size, nwarps, args.native_mxfp6);

    if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
        if (args.native_mxfp6) {
            CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x, false, true, false>), nbytes_shared);
            CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x,  true, true, false>), nbytes_shared);
            CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x, false, true, true>), nbytes_shared);
            CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x,  true, true, true>), nbytes_shared);
        } else {
            CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x, false, false>), nbytes_shared);
            CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x,  true, false>), nbytes_shared);
        }
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x, false, false>), nbytes_shared);
        CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x,  true, false>), nbytes_shared);
    }

    const int nty  = (args.nrows_x   + mmq_y - 1) / mmq_y;
    const int ntx  = (args.ncols_max + mmq_x - 1) / mmq_x;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 block_nums_xy_tiling(nty, ntx, ntzw);

    GGML_ASSERT(args.nchannels_y % args.nchannels_x == 0);
    GGML_ASSERT(args.nsamples_y  % args.nsamples_x  == 0);
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio  = args.nsamples_y  / args.nsamples_x;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
    const uint3 ntx_fd             = init_fastdiv_values(ntx);
    const uint3 nchannels_y_fd     = init_fastdiv_values(args.nchannels_y);
    const uint3 nsamples_y_fd      = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd   = init_fastdiv_values(channel_ratio);
    const uint3 sample_ratio_fd    = init_fastdiv_values(sample_ratio);

    if (!args.use_stream_k) {
        if (args.nrows_x % mmq_y == 0) {
            constexpr bool need_check = false;
            if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
                if (args.native_mxfp6) {
                    if (args.mxfp6_fp8_acts) {
                        mul_mat_q<type, mmq_x, need_check, true, true><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                            (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                             ntx_fd);
                    } else {
                        mul_mat_q<type, mmq_x, need_check, true, false><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                            (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                             ntx_fd);
                    }
                } else {
                    mul_mat_q<type, mmq_x, need_check, false><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                         ntx_fd);
                }
            } else {
                mul_mat_q<type, mmq_x, need_check, false><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                    (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                     blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                     channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                     sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                     ntx_fd);
            }
        } else {
            constexpr bool need_check = true;
            if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
                if (args.native_mxfp6) {
                    if (args.mxfp6_fp8_acts) {
                        mul_mat_q<type, mmq_x, need_check, true, true><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                            (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                             ntx_fd);
                    } else {
                        mul_mat_q<type, mmq_x, need_check, true, false><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                            (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                             ntx_fd);
                    }
                } else {
                    mul_mat_q<type, mmq_x, need_check, false><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                         ntx_fd);
                }
            } else {
                mul_mat_q<type, mmq_x, need_check, false><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                    (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                     blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                     channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                     sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                     ntx_fd);
            }
        }
        return;
    }

    // For the stream-k kernel it is possible to run it with tiling by setting the number of CUDA blocks equal to the number of tiles.
    // This is worthwhile if the efficiency of tiling is high and skipping the fixup kernel is more important.
    const int ntiles_dst = ntx * nty * ntzw;
    const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm*tiles_nwaves);
    const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);

    GGML_ASSERT(ntiles_dst * blocks_per_ne00_fd.z < (1 << 30)); // Assert that variable kbc will not overflow.

    const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0;

    ggml_cuda_pool & pool = ctx.pool(id);
    ggml_cuda_pool_alloc<float> tmp_fixup(pool);
    if (fixup_needed) {
        tmp_fixup.alloc(block_nums_stream_k.x * mmq_x*mmq_y);
    }

    const dim3 block_nums_fixup(block_nums_stream_k.x, mmq_y/warp_size, 1);
    const dim3 block_dims_fixup(block_dims.x, block_dims.y/2, block_dims.z);

    if (args.nrows_x % mmq_y == 0) {
        constexpr bool need_check = false;
        if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
            if (args.native_mxfp6) {
                if (args.mxfp6_fp8_acts) {
                    mul_mat_q<type, mmq_x, need_check, true, true><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
                        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                         ntx_fd);
                } else {
                    mul_mat_q<type, mmq_x, need_check, true, false><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
                        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                         ntx_fd);
                }
            } else {
                mul_mat_q<type, mmq_x, need_check, false><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
                    (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                     blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                     channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                     sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                     ntx_fd);
            }
        } else {
            mul_mat_q<type, mmq_x, need_check, false><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
                (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                 blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                 channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                 sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                 ntx_fd);
        }

        if (!fixup_needed) {
            return;
        }

        CUDA_CHECK(cudaGetLastError());
        mul_mat_q_stream_k_fixup<type, mmq_x, need_check><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
            (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
             args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
             ntx_fd);
    } else {
        constexpr bool need_check = true;
        if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
            if (args.native_mxfp6) {
                if (args.mxfp6_fp8_acts) {
                    mul_mat_q<type, mmq_x, need_check, true, true><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
                        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                         ntx_fd);
                } else {
                    mul_mat_q<type, mmq_x, need_check, true, false><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
                        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                         ntx_fd);
                }
            } else {
                mul_mat_q<type, mmq_x, need_check, false><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
                    (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                     blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                     channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                     sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                     ntx_fd);
            }
        } else {
            mul_mat_q<type, mmq_x, need_check, false><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
                (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                 blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                 channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                 sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                 ntx_fd);
        }

        if (!fixup_needed) {
            return;
        }

        CUDA_CHECK(cudaGetLastError());
        mul_mat_q_stream_k_fixup<type, mmq_x, need_check><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
            (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
             args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
             ntx_fd);
    }
}

template <ggml_type type>
void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int    id     = ggml_cuda_get_device();
    const int    cc     = ggml_cuda_info().devices[id].cc;
    const size_t smpbo  = ggml_cuda_info().devices[id].smpbo;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const int nwarps    = mmq_get_nwarps_host(cc, warp_size);

    int mmq_x_max = get_mmq_x_max_host(cc);
#if defined(BLACKWELL_MMA_AVAILABLE)
    if constexpr (type == GGML_TYPE_NVFP4) {
        if (args.ids_dst != nullptr && mmq_x_max > 64) {
            mmq_x_max = 64;
        }
    }
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    const int mmq_y = get_mmq_y_host(cc);

    int mmq_x_best  = 0;
    int ntiles_x_best = INT_MAX;

    for (int mmq_x = 8; mmq_x <= mmq_x_max && ntiles_x_best > 1; mmq_x += 8) {
        const int granularity = mmq_get_granularity_host(mmq_x, cc);

        if (mmq_x % granularity != 0 ||
                mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size, nwarps, args.native_mxfp6) > smpbo) {
            continue;
        }

        const int ntiles_x = (args.ncols_max + mmq_x - 1) / mmq_x;

        if (ntiles_x < ntiles_x_best) {
            mmq_x_best = mmq_x;
            ntiles_x_best = ntiles_x;
        }
    }

    switch (mmq_x_best) {
        case   8:
            launch_mul_mat_q<type,   8>(ctx, args, stream);
            break;
        case  16:
            launch_mul_mat_q<type,  16>(ctx, args, stream);
            break;
        case  24:
            launch_mul_mat_q<type,  24>(ctx, args, stream);
            break;
        case  32:
            launch_mul_mat_q<type,  32>(ctx, args, stream);
            break;
        case  40:
            launch_mul_mat_q<type,  40>(ctx, args, stream);
            break;
        case  48:
            launch_mul_mat_q<type,  48>(ctx, args, stream);
            break;
        case  56:
            launch_mul_mat_q<type,  56>(ctx, args, stream);
            break;
        case  64:
            launch_mul_mat_q<type,  64>(ctx, args, stream);
            break;
        case  72:
            launch_mul_mat_q<type,  72>(ctx, args, stream);
            break;
        case  80:
            launch_mul_mat_q<type,  80>(ctx, args, stream);
            break;
        case  88:
            launch_mul_mat_q<type,  88>(ctx, args, stream);
            break;
        case  96:
            launch_mul_mat_q<type,  96>(ctx, args, stream);
            break;
        case 104:
            launch_mul_mat_q<type, 104>(ctx, args, stream);
            break;
        case 112:
            launch_mul_mat_q<type, 112>(ctx, args, stream);
            break;
        case 120:
            launch_mul_mat_q<type, 120>(ctx, args, stream);
            break;
        case 128:
            launch_mul_mat_q<type, 128>(ctx, args, stream);
            break;
        default:
            fprintf(stderr, "mmq_x_best=%d\n", mmq_x_best);
            GGML_ABORT("fatal error");
            break;
    }
}

#define DECL_MMQ_CASE(type)                                                        \
    template void mul_mat_q_case<type>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \

extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q8_0);
extern DECL_MMQ_CASE(GGML_TYPE_MXFP4);
extern DECL_MMQ_CASE(GGML_TYPE_MXFP8);
extern DECL_MMQ_CASE(GGML_TYPE_MXFP6_E2M3);
extern DECL_MMQ_CASE(GGML_TYPE_NVFP4);
extern DECL_MMQ_CASE(GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q6_K);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ1_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_NL);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);

// -------------------------------------------------------------------------------------------------------------------------

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts);
