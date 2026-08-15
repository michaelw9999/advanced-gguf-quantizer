#include "mmvq.cuh"
#include "mma.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdlib>
#include <cstdint>
#include <cstring>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static inline bool ggml_cuda_nvfp4_enable_q8_to_fp8_mmvq() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    if (mode != nullptr && (strstr(mode, "fp8-direct") != nullptr || strstr(mode, "direct-fp8") != nullptr)) {
        return false;
    }
    return mode != nullptr &&
        (strcmp(mode, "q8-to-fp8") == 0 || strcmp(mode, "fp8-cols") == 0 || strcmp(mode, "fp8") == 0 ||
         strcmp(mode, "auto") == 0 || strcmp(mode, "1") == 0 || strncmp(mode, "fp8-", 4) == 0);
}

static inline bool ggml_cuda_nvfp4_mmvq_approx_scale() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    return mode != nullptr && strstr(mode, "approx") != nullptr;
}

static inline bool ggml_cuda_nvfp4_mmvq_native_acts() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    return mode != nullptr && strstr(mode, "native") != nullptr;
}

static inline bool ggml_cuda_nvfp4_mmvq_native_fused() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    return mode != nullptr && strstr(mode, "native") != nullptr && strstr(mode, "fused") != nullptr;
}

static inline int ggml_cuda_nvfp4_mmvq_native_split_k() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    if (mode != nullptr && strstr(mode, "split32") != nullptr) {
        return 32;
    }
    if (mode != nullptr && strstr(mode, "split16") != nullptr) {
        return 16;
    }
    if (mode != nullptr && strstr(mode, "split8") != nullptr) {
        return 8;
    }
    if (mode != nullptr && strstr(mode, "split4") != nullptr) {
        return 4;
    }
    if (mode != nullptr && strstr(mode, "split2") != nullptr) {
        return 2;
    }
    if (mode != nullptr && (strstr(mode, "split1") != nullptr || strstr(mode, "nosplit") != nullptr)) {
        return 1;
    }
    return 16;
}

static inline bool ggml_cuda_nvfp4_mmvq_direct_fp8_acts() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    return mode != nullptr && (strstr(mode, "fp8-direct") != nullptr || strstr(mode, "direct-fp8") != nullptr);
}

static inline bool ggml_cuda_nvfp4_mmvq_direct_fp8_fulln_only() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    return mode != nullptr && (strstr(mode, "fulln") != nullptr || strstr(mode, "full-n") != nullptr);
}

static inline bool ggml_cuda_nvfp4_mmvq_direct_fp8_fast_quant() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    return mode != nullptr && strstr(mode, "fastq") != nullptr;
}

static inline int ggml_cuda_nvfp4_mmvq_direct_fp8_split_k() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    if (mode != nullptr && strstr(mode, "split8") != nullptr) {
        return 8;
    }
    if (mode != nullptr && strstr(mode, "split4") != nullptr) {
        return 4;
    }
    if (mode != nullptr && strstr(mode, "split2") != nullptr) {
        return 2;
    }
    return 0;
}

static inline bool ggml_cuda_nvfp4_mmvq_force_small_k() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    return mode != nullptr && strstr(mode, "q8-small-k") != nullptr;
}

static inline int ggml_cuda_nvfp4_mmvq_q8_vdr() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    if (mode != nullptr && strstr(mode, "q8-vdr1") != nullptr) {
        return 1;
    }
    if (mode != nullptr && strstr(mode, "q8-vdr2") != nullptr) {
        return 2;
    }
    if (mode != nullptr && strstr(mode, "q8-vdr4") != nullptr) {
        return 4;
    }
    if (mode != nullptr && strstr(mode, "q8-vdr8") != nullptr) {
        return 8;
    }
    return 0;
}

static inline int ggml_cuda_nvfp4_mmvq_warps() {
    const char * mode = getenv("GGML_CUDA_NVFP4_MMVQ");
    if (mode != nullptr && strstr(mode, "w1") != nullptr) {
        return 1;
    }
    if (mode != nullptr && strstr(mode, "w2") != nullptr) {
        return 2;
    }
    if (mode != nullptr && strstr(mode, "w4") != nullptr) {
        return 4;
    }
    if (mode != nullptr && strstr(mode, "w8") != nullptr) {
        return 8;
    }
    return 16;
}

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_MXFP8:   return vec_dot_mxfp8_q8_1;
        case GGML_TYPE_MXFP6_E2M3:   return nullptr;
#if defined(BLACKWELL_MMA_AVAILABLE)
        case GGML_TYPE_NVFP4:   return nullptr; // changed to 5arg not matching vec_dot_q_cuda_t for now;
#else
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_MXFP8:   return VDR_MXFP8_Q8_1_MMVQ;
        case GGML_TYPE_MXFP6_E2M3:   return VDR_MXFP6_E2M3_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_MXFP6_E2M3:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_MXFP6_E2M3:   return 7;
        case GGML_TYPE_NVFP4:   return 7;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_MXFP6_E2M3:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
#if defined(BLACKWELL_MMA_AVAILABLE)
            if (type == GGML_TYPE_NVFP4) {
                return 12;
            }
#endif // defined(BLACKWELL_MMA_AVAILABLE)
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11, int64_t ne01) {
    if (type == GGML_TYPE_MXFP8 && blackwell_mma_available(cc)) {
        const bool use_exact_mmvq = ne11 <= MMVQ_MAX_BATCH_SIZE && ne01 <= MMVQ_MXFP8_MAX_ROWS;
        return use_exact_mmvq || ggml_cuda_should_use_mxfp8_mma_1col(type, cc, ne11, ne01);
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mxfp8_mma_1col(enum ggml_type type, int cc, int64_t ne11, int64_t ne01) {
    return type == GGML_TYPE_MXFP8 && blackwell_mma_available(cc) && ne11 == 1 &&
           ne01 > MMVQ_MXFP8_MAX_ROWS && ne01 % 16 == 0;
}

static __device__ __forceinline__ float ggml_cuda_mmvq_apply_glu(
        const float result, const float gate_value, const ggml_glu_op glu_op) {
    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:
            return result * ggml_cuda_op_silu_single(gate_value);
        case GGML_GLU_OP_GEGLU:
            return result * ggml_cuda_op_gelu_single(gate_value);
        case GGML_GLU_OP_SWIGLU_OAI:
            return ggml_cuda_op_swiglu_oai_single(gate_value, result);
        default:
            return result * gate_value;
    }
}

static __device__ __forceinline__ void load_mxfp8_tileA_1col_mmvq(
        ggml_cuda_mma::tile<16, 8, int> & A, uint32_t & scaleA,
        const mxfp8_tile * __restrict__ tile,
        int row_lo, int row_hi, int frag, int word) {
    const uint4 packed = tile->qs[frag][ggml_cuda_mxfp8_frag_lane(row_lo, word)];
    int * ax = reinterpret_cast<int *>(A.x);
    ax[0] = packed.x;
    ax[1] = packed.y;
    ax[2] = packed.z;
    ax[3] = packed.w;
    scaleA = uint32_t(tile->sc[frag][row_lo]) | (uint32_t(tile->sc[frag][row_hi]) << 16);
}

static __device__ __forceinline__ void load_mxfp8_tileB_1col_mmvq(
        ggml_cuda_mma::tile<8, 8, int> & B, uint32_t & scaleB,
        const block_mxfp8_mmq * __restrict__ y, const int frag) {
    const int lane = int(threadIdx.x) & 31;
    int * bx = reinterpret_cast<int *>(B.x);
    uint32_t scale = 0x7Fu;

    if (lane < 4) {
        const block_mxfp8_mmq & block = y[frag >> 2];
        const int frag_in_block = frag & 3;
        const uint32_t * q = reinterpret_cast<const uint32_t *>(block.qs + frag_in_block * QK_MXFP8_SUB);
        bx[0] = int(q[lane]);
        bx[1] = int(q[lane + 4]);
        if (lane == 0) {
            const uint32_t d4 = *reinterpret_cast<const uint32_t *>(block.d4);
            scale = (d4 >> (8 * frag_in_block)) & 0xFFu;
        }
    } else {
        bx[0] = 0;
        bx[1] = 0;
    }

    scaleB = __shfl_sync(0xFFFFFFFFu, scale, 0);
}

#define GGML_CUDA_MXFP8_1COL_MIN_BLOCKS(W) ((W) <= 16 ? 2 : 1)

template <int warps_per_tile, bool has_gate, bool has_x_bias, bool has_gate_bias>
static __global__ __launch_bounds__(32 * warps_per_tile, GGML_CUDA_MXFP8_1COL_MIN_BLOCKS(warps_per_tile))
void mul_mat_vec_mxfp8_mma_1col(
        const mxfp8_tile * __restrict__ vx, const mxfp8_tile * __restrict__ vgate,
        const block_mxfp8_mmq * __restrict__ y,
        const float * __restrict__ x_bias, const float * __restrict__ gate_bias,
        const ggml_glu_op glu_op, float * __restrict__ dst, const int nfrags32,
        const int tiles_per_row, const int tile_stride_channel, const int tile_stride_sample,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int row_base    = 16 * int(blockIdx.x);
    const int tile_row    = int(blockIdx.x);
    const int channel_dst = int(blockIdx.y);
    const int sample_dst  = int(blockIdx.z);
    const int channel_x   = channel_ratio == 1 ? channel_dst : channel_dst / channel_ratio;
    const int sample_x    = sample_ratio == 1 ? sample_dst : sample_dst / sample_ratio;
    const int tile_offset =
        sample_x * tile_stride_sample + channel_x * tile_stride_channel + tile_row * tiles_per_row;

    const mxfp8_tile * x = vx + tile_offset;
    const mxfp8_tile * gate = has_gate ? vgate + tile_offset : nullptr;
    const block_mxfp8_mmq * y_cur =
        y + sample_dst * stride_sample_y + channel_dst * stride_channel_y;

    const int lane    = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int row_lo  = lane >> 2;
    const int row_hi  = row_lo + 8;
    const int word    = lane & 3;
    tile_C C = {};
    tile_C gate_C = {};

    for (int frag = 2 * warp_id; frag < nfrags32; frag += 2 * warps_per_tile) {
        tile_B B0, B1;
        uint32_t scaleB0, scaleB1;
        load_mxfp8_tileB_1col_mmvq(B0, scaleB0, y_cur, frag + 0);
        load_mxfp8_tileB_1col_mmvq(B1, scaleB1, y_cur, frag + 1);

        const mxfp8_tile * tile_cur = x + (frag >> 3);
        const int frag_in_tile = frag & 7;

        tile_A A0, A1;
        uint32_t scaleA0, scaleA1;
        load_mxfp8_tileA_1col_mmvq(A0, scaleA0, tile_cur, row_lo, row_hi, frag_in_tile + 0, word);
        load_mxfp8_tileA_1col_mmvq(A1, scaleA1, tile_cur, row_lo, row_hi, frag_in_tile + 1, word);

        mma_block_scaled_mxfp8(C, A0, B0, scaleA0, scaleB0);
        mma_block_scaled_mxfp8(C, A1, B1, scaleA1, scaleB1);

        if constexpr (has_gate) {
            const mxfp8_tile * gate_tile = gate + (frag >> 3);
            load_mxfp8_tileA_1col_mmvq(A0, scaleA0, gate_tile, row_lo, row_hi, frag_in_tile + 0, word);
            load_mxfp8_tileA_1col_mmvq(A1, scaleA1, gate_tile, row_lo, row_hi, frag_in_tile + 1, word);
            mma_block_scaled_mxfp8(gate_C, A0, B0, scaleA0, scaleB0);
            mma_block_scaled_mxfp8(gate_C, A1, B1, scaleA1, scaleB1);
        }
    }

    __shared__ float partial[warps_per_tile][16];
    __shared__ float partial_gate[has_gate ? warps_per_tile : 1][16];
    if ((lane & 3) == 0) {
        const int row_part = lane >> 2;
        partial[warp_id][row_part + 0] = C.x[0];
        partial[warp_id][row_part + 8] = C.x[2];
        if constexpr (has_gate) {
            partial_gate[warp_id][row_part + 0] = gate_C.x[0];
            partial_gate[warp_id][row_part + 8] = gate_C.x[2];
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane < 16) {
        float result = 0.0f;
        float gate_value = 0.0f;
#pragma unroll
        for (int warp = 0; warp < warps_per_tile; ++warp) {
            result += partial[warp][lane];
            if constexpr (has_gate) {
                gate_value += partial_gate[warp][lane];
            }
        }

        const int dst_idx =
            sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row_base + lane;
        if constexpr (has_x_bias) {
            result += x_bias[dst_idx];
        }
        if constexpr (has_gate) {
            if constexpr (has_gate_bias) {
                gate_value += gate_bias[dst_idx];
            }
            result = ggml_cuda_mmvq_apply_glu(result, gate_value, glu_op);
        }
        dst[dst_idx] = result;
    }
}

template <int warps_per_tile>
static void launch_mul_mat_vec_mxfp8_mma_1col(
        const mxfp8_tile * src0, const ggml_cuda_mm_fusion_args_device & fusion,
        const block_mxfp8_mmq * src1, float * dst, const int nfrags32,
        const int nrows, const int nchannels_dst, const int nsamples_dst,
        const int tiles_per_row, const int tile_stride_channel, const int tile_stride_sample,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio, cudaStream_t stream) {
    const dim3 block_nums(nrows / 16, nchannels_dst, nsamples_dst);
    const dim3 block_dims(32, warps_per_tile, 1);

    if (fusion.gate != nullptr) {
        if (fusion.x_bias != nullptr && fusion.gate_bias != nullptr) {
            mul_mat_vec_mxfp8_mma_1col<warps_per_tile, true, true, true><<<block_nums, block_dims, 0, stream>>>(
                src0, (const mxfp8_tile *) fusion.gate, src1,
                (const float *) fusion.x_bias, (const float *) fusion.gate_bias, fusion.glu_op, dst, nfrags32,
                tiles_per_row, tile_stride_channel, tile_stride_sample, stride_channel_y, stride_sample_y,
                stride_channel_dst, stride_sample_dst, channel_ratio, sample_ratio);
        } else if (fusion.x_bias != nullptr) {
            mul_mat_vec_mxfp8_mma_1col<warps_per_tile, true, true, false><<<block_nums, block_dims, 0, stream>>>(
                src0, (const mxfp8_tile *) fusion.gate, src1,
                (const float *) fusion.x_bias, nullptr, fusion.glu_op, dst, nfrags32,
                tiles_per_row, tile_stride_channel, tile_stride_sample, stride_channel_y, stride_sample_y,
                stride_channel_dst, stride_sample_dst, channel_ratio, sample_ratio);
        } else if (fusion.gate_bias != nullptr) {
            mul_mat_vec_mxfp8_mma_1col<warps_per_tile, true, false, true><<<block_nums, block_dims, 0, stream>>>(
                src0, (const mxfp8_tile *) fusion.gate, src1,
                nullptr, (const float *) fusion.gate_bias, fusion.glu_op, dst, nfrags32,
                tiles_per_row, tile_stride_channel, tile_stride_sample, stride_channel_y, stride_sample_y,
                stride_channel_dst, stride_sample_dst, channel_ratio, sample_ratio);
        } else {
            mul_mat_vec_mxfp8_mma_1col<warps_per_tile, true, false, false><<<block_nums, block_dims, 0, stream>>>(
                src0, (const mxfp8_tile *) fusion.gate, src1,
                nullptr, nullptr, fusion.glu_op, dst, nfrags32,
                tiles_per_row, tile_stride_channel, tile_stride_sample, stride_channel_y, stride_sample_y,
                stride_channel_dst, stride_sample_dst, channel_ratio, sample_ratio);
        }
    } else {
        GGML_ASSERT(fusion.x_bias == nullptr && fusion.gate_bias == nullptr);
        mul_mat_vec_mxfp8_mma_1col<warps_per_tile, false, false, false><<<block_nums, block_dims, 0, stream>>>(
            src0, nullptr, src1, nullptr, nullptr, fusion.glu_op, dst, nfrags32,
            tiles_per_row, tile_stride_channel, tile_stride_sample, stride_channel_y, stride_sample_y,
            stride_channel_dst, stride_sample_dst, channel_ratio, sample_ratio);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool fp8_acts = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t physical_rows_x, const uint32_t ids_stride, const bool native_mxfp6, const bool native_fp8) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

#if defined(BLACKWELL_MMA_AVAILABLE)
    constexpr int qk  = type == GGML_TYPE_NVFP4 ? QK_NVFP4 : ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = type == GGML_TYPE_NVFP4 ? QI_NVFP4 : ggml_cuda_type_traits<type>::qi;
#else
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_k = qk / QK8_1;

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block*blockIdx.x;
    int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }
    // Keep the no-fusion instantiation small; dense TG1 uses this generic path.
    [[maybe_unused]] float x_biases[has_fusion ? ncols_dst : 1]    = { 0.0f };
    [[maybe_unused]] float gate_biases[has_fusion ? ncols_dst : 1] = { 0.0f };
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    // Avoid gate scratch in the no-fusion instantiation; it costs registers/shared memory.
    float tmp_gate[has_fusion ? ncols_dst : 1][has_fusion ? rows_per_cuda_block : 1] = {{0.0f}};

    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
	    const block_q8_1  * y_q8  = nullptr;
	    const block_fp8 * y_fp8 = nullptr;
	    if constexpr (fp8_acts) {
	        y_fp8 = ((const block_fp8 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
	    } else {
	        y_q8 = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
	    }
		    const int kbx_begin = tid / (qi/vdr);
		    const int kbx_step  = blocks_per_iter;

    for (int kbx = kbx_begin; kbx < blocks_per_row_x; kbx += kbx_step) {
        const int kby = kbx * blocks_per_k;
        const int kqs_base = vdr * (tid % (qi/vdr));

	#pragma unroll
		        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
	            for (int i = 0; i < rows_per_cuda_block; ++i) {
	                if (rows_per_cuda_block != 1 && uint32_t(row0 + i) >= nrows_x) {
	                    continue;
	                }
	                const block_q8_1  * y_ptr_q8  = nullptr;
	                const block_fp8 * y_ptr_fp8 = nullptr;
	                if constexpr (fp8_acts) {
	                    y_ptr_fp8 = &y_fp8[j*stride_col_y];
	                } else {
	                    y_ptr_q8 = &y_q8[j*stride_col_y + kby];
	                }
	                int kbx_q = kbx_offset + i*stride_row_x + kbx;
#if defined(BLACKWELL_MMA_AVAILABLE)
	                if constexpr (type == GGML_TYPE_NVFP4) {
	                    const int row = row0 + i;
	                    const uint32_t block_rel =
	                        sample_x*stride_sample_x + channel_x*stride_channel_x + (row / 16)*stride_row_x + (kbx >> 2);
	                    kbx_q = int(((uint32_t) (row & 15) << 28) | ((uint32_t) (kbx & 3) << 24) | block_rel);
	                } else if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
	                    if (native_mxfp6) {
	                        const int row = row0 + i;
	                        const uint32_t block_rel =
	                            sample_x*stride_sample_x + channel_x*stride_channel_x + (row / 16)*stride_row_x + kbx;
	                        kbx_q = int(((uint32_t) (row & 15) << 28) | block_rel);
	                    }
	                }
#endif // defined(BLACKWELL_MMA_AVAILABLE)
                {
#if defined(BLACKWELL_MMA_AVAILABLE)
	                if constexpr (type == GGML_TYPE_NVFP4) {
	                    tmp[j][i] += vec_dot_nvfp4_q8_1_bw(vx, y_ptr_q8, kbx_q, kqs_base, channel_x);
	                } else if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
	                    if constexpr (fp8_acts) {
	                        if (native_mxfp6) {
	                            tmp[j][i] += vec_dot_mxfp6_e2m3_fp8_mmvq_bw(vx, y_ptr_fp8 + (kbx >> 1), kbx_q, kqs_base, channel_x, kbx & 1);
	                        }
	                    } else if (native_mxfp6) {
	                        tmp[j][i] += vec_dot_mxfp6_e2m3_q8_1_bw(vx, y_ptr_q8, kbx_q, kqs_base, channel_x);
	                    }
	                } else
#endif // defined(BLACKWELL_MMA_AVAILABLE)
	                {
	                    tmp[j][i] += vec_dot_q_cuda(vx, y_ptr_q8, kbx_q, kqs_base);
	                }
	                if constexpr (has_fusion) {
	                    if (use_gate) {
#if defined(BLACKWELL_MMA_AVAILABLE)
	                        if constexpr (type == GGML_TYPE_NVFP4) {
	                            tmp_gate[j][i] += vec_dot_nvfp4_q8_1_bw(vgate, y_ptr_q8, kbx_q, kqs_base, channel_x);
	                        } else if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
	                            if constexpr (fp8_acts) {
	                                if (native_mxfp6) {
	                                    tmp_gate[j][i] += vec_dot_mxfp6_e2m3_fp8_mmvq_bw(vgate, y_ptr_fp8 + (kbx >> 1), kbx_q, kqs_base, channel_x, kbx & 1);
	                                }
	                            } else if (native_mxfp6) {
	                                tmp_gate[j][i] += vec_dot_mxfp6_e2m3_q8_1_bw(vgate, y_ptr_q8, kbx_q, kqs_base, channel_x);
	                            }
	                        } else
#endif // defined(BLACKWELL_MMA_AVAILABLE)
	                        {
	                            tmp_gate[j][i] += vec_dot_q_cuda(vgate, y_ptr_q8, kbx_q, kqs_base);
	                        }
	                        }
	                    }
		                }
		            }
	        }
	    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][has_fusion ? ncols_dst : 1][has_fusion ? rows_per_cuda_block : 1][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            float result = tmp[j][threadIdx.x];
            if constexpr (has_fusion) {
                if (use_bias) {
                    result += x_biases[j];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_biases[j];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI: {
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        }
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[j*stride_col_dst + threadIdx.x] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias, tmp_gate);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool fp8_acts = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const uint32_t physical_rows_x,
        const bool native_mxfp6, const bool native_fp8) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

	    const block_q8_1  * y_q8  = nullptr;
	    const block_fp8 * y_fp8 = nullptr;
	    if constexpr (fp8_acts) {
	        y_fp8 = ((const block_fp8 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
	    } else {
	        y_q8 = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
	    }
	    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;
    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
	        for (int i = 0; i < c_rows_per_block; ++i) {
	            if (c_rows_per_block != 1 && uint32_t(row0 + i) >= nrows_x) {
	                continue;
	            }
	            int kbx_q = kbx_offset + i*stride_row_x + kbx;
#if defined(BLACKWELL_MMA_AVAILABLE)
            if constexpr (type == GGML_TYPE_NVFP4) {
                const int row = row0 + i;
                const uint32_t block_rel = channel_x*stride_channel_x + (row / 16)*stride_row_x + (kbx >> 2);
                kbx_q = int(((uint32_t) (row & 15) << 28) | ((uint32_t) (kbx & 3) << 24) | block_rel);
	                tmp[i] += vec_dot_nvfp4_q8_1_bw(vx, &y_q8[kby], kbx_q, kqs, channel_x);
	            } else if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
                if (fp8_acts) {
                    if (native_mxfp6) {
                        const int row = row0 + i;
                        const uint32_t block_rel = channel_x*stride_channel_x + (row / 16)*stride_row_x + kbx;
                        kbx_q = int(((uint32_t) (row & 15) << 28) | block_rel);
                        tmp[i] += vec_dot_mxfp6_e2m3_fp8_mmvq_bw(vx, y_fp8 + (kbx >> 1), kbx_q, kqs, channel_x, kbx & 1);
                    }
                } else if (native_mxfp6) {
                    const int row = row0 + i;
                    const uint32_t block_rel = channel_x*stride_channel_x + (row / 16)*stride_row_x + kbx;
                    kbx_q = int(((uint32_t) (row & 15) << 28) | block_rel);
                    tmp[i] += vec_dot_mxfp6_e2m3_q8_1_bw(vx, &y_q8[kby], kbx_q, kqs, channel_x);
	                }
	            } else
#endif // defined(BLACKWELL_MMA_AVAILABLE)
	            {
	                tmp[i] += vec_dot_q_cuda(vx, &y_q8[kby], kbx_q, kqs);
	            }
	        }
	    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
	        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
	        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
	        const uint32_t physical_rows_x, const uint32_t ids_stride, cudaStream_t stream, bool native_mxfp6 = false, bool fp8_acts = false,
            bool native_fp8 = false) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            if (fp8_acts) {
                ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, true>, launch_params,
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, physical_rows_x, ids_stride, native_mxfp6, native_fp8);
            } else {
                ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, false>, launch_params,
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, physical_rows_x, ids_stride, native_mxfp6, native_fp8);
            }
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    if (fp8_acts) {
        ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, physical_rows_x, ids_stride, native_mxfp6, native_fp8);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, physical_rows_x, ids_stride, native_mxfp6, native_fp8);
    }
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
	        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
	        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
	        const uint32_t ncols_dst, const uint32_t ids_stride,
	        const int warp_size, const int nchannels_dst, const uint32_t physical_rows_x, cudaStream_t stream, bool native_mxfp6 = false, bool fp8_acts = false,
            bool native_fp8 = false) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    if (fp8_acts) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, dst,
            ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, physical_rows_x, native_mxfp6, native_fp8);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, dst,
            ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, physical_rows_x, native_mxfp6, native_fp8);
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
	        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
	        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
	        const int ids_stride, cudaStream_t stream, bool native_mxfp6 = false, bool fp8_acts = false,
            bool native_fp8 = false, const int physical_rows_x = 0) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_ids = ids != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
#if defined(BLACKWELL_MMA_AVAILABLE)
        constexpr int qk                    = type == GGML_TYPE_NVFP4 ? QK_NVFP4 : ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = type == GGML_TYPE_NVFP4 ? QI_NVFP4 : ggml_cuda_type_traits<type>::qi;
#else
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
#if defined(BLACKWELL_MMA_AVAILABLE)
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (c_ncols_dst == 1 && ggml_cuda_nvfp4_mmvq_force_small_k()) {
                    use = true;
                }
            }
            if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
                use = false;
            }
#endif // defined(BLACKWELL_MMA_AVAILABLE)
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
	        mul_mat_vec_q_moe_launch<type>(
	            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
	            stride_row_x, stride_col_y, stride_col_dst,
	            stride_channel_x, stride_channel_y, stride_channel_dst,
	            ncols_dst, ids_stride, warp_size, nchannels_dst, physical_rows_x, stream, native_mxfp6, fp8_acts, native_fp8);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            bool use_small_k = should_use_small_k(c_ncols_dst);

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
	                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
	                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, physical_rows_x, ids_stride,
	                    stream, native_mxfp6, fp8_acts, native_fp8);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
	                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
	                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, physical_rows_x, ids_stride,
	                    stream, native_mxfp6, fp8_acts, native_fp8);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
	                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
	                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
	                 dims.first, dims.second, 0, physical_rows_x, ids_stride, stream, native_mxfp6, fp8_acts, native_fp8);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
	                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
	                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
	                 dims.first, dims.second, 0, physical_rows_x, ids_stride, stream, native_mxfp6, fp8_acts, native_fp8);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
	                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
	                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
	                 dims.first, dims.second, 0, physical_rows_x, ids_stride, stream, native_mxfp6, fp8_acts, native_fp8);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
	                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
	                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
	                 dims.first, dims.second, 0, physical_rows_x, ids_stride, stream, native_mxfp6, fp8_acts, native_fp8);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
	                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
	                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
	                 dims.first, dims.second, 0, physical_rows_x, ids_stride, stream, native_mxfp6, fp8_acts, native_fp8);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
	                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
	                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
	                 dims.first, dims.second, 0, physical_rows_x, ids_stride, stream, native_mxfp6, fp8_acts, native_fp8);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
	                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
	                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
	                 dims.first, dims.second, 0, physical_rows_x, ids_stride, stream, native_mxfp6, fp8_acts, native_fp8);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(has_fusion);
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
	        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
	        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
	        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
	        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
	        const int ids_stride, cudaStream_t stream, bool native_mxfp6 = false, bool fp8_acts = false,
            bool native_fp8 = false, const int physical_rows_x = 0) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP8:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP8>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
	        case GGML_TYPE_MXFP6_E2M3:
	            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP6_E2M3>
	                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
	                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
	                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                     ids_stride, stream, native_mxfp6, fp8_acts, native_fp8, physical_rows_x);
	            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
	    }
	}

template <ggml_type type, int warps_per_row>
static __device__ __forceinline__ float dot_row_fp8_acts_1col(
        const void * __restrict__ vx, const block_fp8 * __restrict__ y,
        const int x_offset, const int row, const int channel_x, const int lane, const int warp_id, const int blocks_per_row_x) {
    constexpr int warp_size = 32;
    float sum = 0.0f;
    if constexpr (type == GGML_TYPE_MXFP6_E2M3) {
        const tile_mxfp6_e2m3_blackwell_tensor * tensor = (const tile_mxfp6_e2m3_blackwell_tensor *) vx;
        const tile_mxfp6_e2m3_blackwell * x = tensor->tiles + x_offset;
        const int row_in_tile = row & 15;
        const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
        for (int q = lane + warp_id * warp_size; q < blocks_per_row_x * QI_MXFP6_E2M3; q += warp_size * warps_per_row) {
            const int kbx   = q / QI_MXFP6_E2M3;
            const int qword = q - kbx * QI_MXFP6_E2M3;
            const int frag  = qword >> 3;
            const int word  = qword & 7;
            const int y_block = kbx >> 1;
            const int y_frag  = kbx & 1;
            const uint32_t x6_4 = ggml_cuda_mxfp6_e2m3_bw_tile_q_word(x[kbx], row_in_tile, frag, word);
            const uint32_t y8_4 = ggml_cuda_fp8_get4_u8containers(y[y_block], y_frag, word);
            const float d = tensor_scale * ggml_cuda_e8m0_to_fp32(ggml_cuda_mxfp6_e2m3_bw_tile_scale(x[kbx], row_in_tile, frag)) *
                ggml_cuda_e8m0_to_fp32(y[y_block].e[y_frag]);
            const int x4 = mxfp6_e2m3_pack4_i8x8(x6_4);
            float sumf = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sumf += float((int8_t) ((x4 >> (8 * i)) & 0xFF)) * (1.0f / 8.0f) *
                    ggml_cuda_e4m3_to_fp32((y8_4 >> (8 * i)) & 0xFF);
            }
            sum += d * sumf;
        }
    } else if constexpr (type == GGML_TYPE_NVFP4) {
        const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
        const block_nvfp4_blackwell * x = tensor->tiles + x_offset;
        const int row_in_tile = row & 15;
        const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
        constexpr int qwords_per_nvfp4_tile = QK_K / 4;
        for (int q = lane + warp_id * warp_size; q < blocks_per_row_x * qwords_per_nvfp4_tile; q += warp_size * warps_per_row) {
            const int kbx       = q / qwords_per_nvfp4_tile;
            const int qword_abs = q - kbx * qwords_per_nvfp4_tile;
            const int elem_abs  = qword_abs * 4;

            const int nv_frag_idx = elem_abs / QK_NVFP4;
            const int elem_frag   = elem_abs - nv_frag_idx * QK_NVFP4;
            const int half32      = elem_frag >> 5;
            const int word        = (elem_frag & 31) >> 2;
            const int sub16       = word >> 2;
            const int rel_word    = word & 3;
            const int pack_idx    = half32 * 4 + sub16 * 2 + (rel_word >> 1);
            const int off4        = (rel_word & 1) * 4;
            const int y_elem      = kbx * QK_K + elem_abs;
            const int y_block     = y_elem / QK_FP8;
            const int y_rem       = y_elem - y_block * QK_FP8;
            const int y_frag      = y_rem / QK_FP8_SUB;
            const int y_word      = (y_rem - y_frag * QK_FP8_SUB) >> 2;
            const block_nvfp4_blackwell & block = x[kbx];
            const uint32_t x8 = ggml_cuda_nvfp4_tile_q_word(block, row_in_tile, nv_frag_idx, pack_idx);
            const uint32_t y4 = ggml_cuda_fp8_get4_u8containers(y[y_block], y_frag, y_word);
            float sumf = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint8_t xcode = (x8 >> (4 * (off4 + i))) & 0x0F;
                const uint8_t ycode = (y4 >> (8 * i)) & 0xFF;
                sumf += float(kvalues_mxfp4[xcode]) * ggml_cuda_e4m3_to_fp32(ycode);
            }

            const uint32_t scale_word = ggml_cuda_nvfp4_tile_scale_word(block, row_in_tile, nv_frag_idx);
            const uint8_t scale_code = (scale_word >> (8 * (half32 * 2 + sub16))) & 0xFF;
            const float d = tensor_scale * ggml_cuda_nvfp4_scale_to_fp32_half(scale_code) *
                ggml_cuda_e8m0_to_fp32(y[y_block].e[y_frag]);
            sum += d * sumf;
        }
    }
    return sum;
}

template <ggml_type type, int rows_per_block, int warps_per_row>
static __global__ void mul_mat_vec_fp8_acts_1col(
        const void * __restrict__ vx, const void * __restrict__ vgate, const block_fp8 * __restrict__ y,
        const float * __restrict__ x_bias, const float * __restrict__ gate_bias,
        float * __restrict__ dst, const ggml_glu_op glu_op,
        const int nrows_x, const int blocks_per_row_x,
        const int stride_row_x, const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row_in_block = threadIdx.z;
    const int row = blockIdx.x * rows_per_block + row_in_block;
    const bool valid_row = row < nrows_x;
    __shared__ float partial[rows_per_block][warps_per_row];
    __shared__ float partial_gate[rows_per_block][warps_per_row];

    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;
    int x_offset = sample_x*stride_sample_x + channel_x*stride_channel_x;
    if constexpr (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP6_E2M3) {
        x_offset += (row / 16) * stride_row_x;
    } else {
        x_offset += row * stride_row_x;
    }
    const block_fp8 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;

    float sum = valid_row ? dot_row_fp8_acts_1col<type, warps_per_row>(vx, y_cur, x_offset, row, channel_x, lane, warp_id, blocks_per_row_x) : 0.0f;
    float sum_gate = 0.0f;
    if (vgate != nullptr) {
        sum_gate = valid_row ? dot_row_fp8_acts_1col<type, warps_per_row>(vgate, y_cur, x_offset, row, channel_x, lane, warp_id, blocks_per_row_x) : 0.0f;
    }

    sum = warp_reduce_sum<warp_size>(sum);
    if (vgate != nullptr) {
        sum_gate = warp_reduce_sum<warp_size>(sum_gate);
    }
    if (lane == 0) {
        partial[row_in_block][warp_id] = sum;
        if (vgate != nullptr) {
            partial_gate[row_in_block][warp_id] = sum_gate;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane < warps_per_row ? partial[row_in_block][lane] : 0.0f;
        if (vgate != nullptr) {
            sum_gate = lane < warps_per_row ? partial_gate[row_in_block][lane] : 0.0f;
        }
        sum = warp_reduce_sum<warp_size>(sum);
        if (vgate != nullptr) {
            sum_gate = warp_reduce_sum<warp_size>(sum_gate);
        }
    }

    if (valid_row && warp_id == 0 && lane == 0) {
        if (x_bias != nullptr) {
            sum += x_bias[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row];
        }
        if (vgate != nullptr) {
            if (gate_bias != nullptr) {
                sum_gate += gate_bias[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row];
            }
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    sum *= ggml_cuda_op_silu_single(sum_gate);
                    break;
                case GGML_GLU_OP_GEGLU:
                    sum *= ggml_cuda_op_gelu_single(sum_gate);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    sum = ggml_cuda_op_swiglu_oai_single(sum_gate, sum);
                    break;
                default:
                    sum *= sum_gate;
                    break;
            }
        }
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = sum;
    }
}

#if defined(BLACKWELL_MMA_AVAILABLE)
static __device__ __forceinline__ void load_mxfp6_e2m3_tileA_416(
        ggml_cuda_mma::tile<16, 8, int> & A, uint32_t & scaleA,
        const tile_mxfp6_e2m3_blackwell_frag & frag) {
    const int lane = threadIdx.x & 31;
    const uint32_t w0 = frag.regs[lane][0];
    const uint32_t w1 = frag.regs[lane][1];
    const uint32_t w2 = frag.regs[lane][2];
    int * ax = (int *) A.x;
    ax[0] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers(w0);
    ax[1] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers((w0 >> 24) | (w1 << 8));
    ax[2] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers((w1 >> 16) | (w2 << 16));
    ax[3] = (int) ggml_cuda_mxfp6_e2m3_unpack4_u8containers(w2 >> 8);
    scaleA = frag.scales[lane];
}

static __device__ __forceinline__ uint32_t nvfp4_mmvq_pack4_mxf8f6f4_e2m1(const uint32_t packed8, const int off4) {
    return
        ((uint32_t) (((packed8 >> (4 * (off4 + 0))) & 0x0Fu) <<  2)      ) |
        ((uint32_t) (((packed8 >> (4 * (off4 + 1))) & 0x0Fu) <<  2) <<  8) |
        ((uint32_t) (((packed8 >> (4 * (off4 + 2))) & 0x0Fu) <<  2) << 16) |
        ((uint32_t) (((packed8 >> (4 * (off4 + 3))) & 0x0Fu) <<  2) << 24);
}

static __device__ __forceinline__ uint8_t nvfp4_mmvq_ue4m3_scale_to_e8m0_x2(const uint8_t scale) {
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

static __device__ __forceinline__ float nvfp4_mmvq_fp8_scale_correction(const uint8_t scale) {
    const uint8_t approx = nvfp4_mmvq_ue4m3_scale_to_e8m0_x2(scale);
    const float exact_raw = 2.0f * ggml_cuda_ue4m3_to_fp32(scale);
    const float approx_raw = ggml_cuda_e8m0_to_fp32(approx);
    return approx_raw > 0.0f ? exact_raw / approx_raw : 0.0f;
}

static __device__ __forceinline__ uint32_t nvfp4_mmvq_load_mxf8f6f4_word(
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
    return nvfp4_mmvq_pack4_mxf8f6f4_e2m1(packed8, off4);
}

static __device__ __forceinline__ void load_nvfp4_tileA_native_mmvq(
        ggml_cuda_mma::tile<16, 8, int> & A, uint32_t & scaleA,
        const block_nvfp4_blackwell * __restrict__ x,
        const int nvfp4_blocks_per_row,
        const int row_base,
        const int frag_abs) {
    const int lane = int(threadIdx.x) & 31;
    const int block_rel = frag_abs / 4;
    const int frag_idx  = frag_abs & 3;
    const block_nvfp4_blackwell_frag & frag = x[(row_base / 16) * nvfp4_blocks_per_row + block_rel].tiles[frag_idx];
    const uint4 packed = reinterpret_cast<const uint4 *>(frag.regs)[lane];
    int * ax = (int *) A.x;
    ax[0] = (int) packed.x;
    ax[1] = (int) packed.y;
    ax[2] = (int) packed.z;
    ax[3] = (int) packed.w;
    scaleA = frag.scales_u32[lane];
}

static __device__ __forceinline__ void load_nvfp4_tileB_native_1col_mmvq(
        ggml_cuda_mma::tile<8, 8, int> & B, uint32_t & scaleB,
        const block_nvfp4_mmq & y_block,
        const int frag_idx) {
    const int lane = int(threadIdx.x) & 31;
    const int group = lane & 3;
    int * bx = (int *) B.x;
    if (lane < 4) {
        const uint32_t * __restrict__ y_qs = y_block.qs_u32 + frag_idx * 8;
        bx[0] = (int) y_qs[group + 0];
        bx[1] = (int) y_qs[group + 4];
    } else {
        bx[0] = 0;
        bx[1] = 0;
    }
    uint32_t scale_word = 0;
    if (lane == 0) {
        scale_word = y_block.sc4_u32[frag_idx];
    }
    scaleB = __shfl_sync(0xFFFFFFFFu, scale_word, 0);
}

static __device__ __forceinline__ uint8_t nvfp4_mmvq_scale_code_from_amax_fast(const float amax) {
    const uint8_t code = ggml_cuda_fp32_to_ue4m3(amax * (1.0f / 6.0f));
    return code == 0x7Fu ? 0x7Eu : code;
}

static __device__ __forceinline__ uint32_t nvfp4_mmvq_quantize_fp4x8(
        const float * __restrict__ y,
        const int64_t base_idx,
        const int elem_base,
        const int ne10,
        const float inv_input_scale,
        const float inv_subblock_scale,
        float & amax) {
    uint32_t packed = 0;
    amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int elem = elem_base + k;
        const float raw = elem < ne10 ? y[base_idx + elem] : 0.0f;
        const float v = isfinite(raw) ? raw * inv_input_scale : 0.0f;
        amax = fmaxf(amax, fabsf(v));
        packed |= uint32_t(ggml_cuda_float_to_fp4_e2m1(v, inv_subblock_scale)) << (4 * k);
    }
    return packed;
}

static __device__ __forceinline__ void nvfp4_mmvq_load_vals8(
        const float * __restrict__ y,
        const int64_t base_idx,
        const int elem_base,
        const int ne10,
        const float inv_input_scale,
        float vals[8],
        float & amax) {
    amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int elem = elem_base + k;
        const float raw = elem < ne10 ? y[base_idx + elem] : 0.0f;
        const float v = isfinite(raw) ? raw * inv_input_scale : 0.0f;
        vals[k] = v;
        amax = fmaxf(amax, fabsf(v));
    }
}

static __device__ __forceinline__ uint32_t nvfp4_mmvq_pack_vals8(const float vals[8], const float inv_scale) {
    uint32_t packed = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        packed |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k], inv_scale)) << (4 * k);
    }
    return packed;
}

static __device__ __forceinline__ float nvfp4_mmvq_half_err8(
        const float vals[8],
        const float test_scale,
        const float test_inv_scale) {
    float err = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint8_t q = ggml_cuda_float_to_fp4_e2m1(vals[k], test_inv_scale);
        const float diff = fabsf(vals[k]) - fabsf(float(kvalues_mxfp4[q & 0x7])) * test_scale;
        err += diff * diff;
    }
    return err;
}

static __device__ __forceinline__ uint8_t nvfp4_mmvq_select_scale_code_exact(
        const float vals[8],
        const float amax,
        float & subblock_scale) {
    static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2 };
    const int first_code = int(ggml_cuda_fp32_to_ue4m3(amax * (1.0f / 6.0f)));
    const uint32_t pair_mask = (int(threadIdx.x) & 2) == 0 ? 0x00000003u : 0x0000000Cu;
    float best_err = 1.0e30f;
    uint8_t best_code = 0;
    subblock_scale = 0.0f;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        const int test_code = first_code + test_offsets[i];
        if (test_code < 0 || test_code > 0x7E) {
            continue;
        }
        const float test_scale = ggml_cuda_nvfp4_scale_to_fp32_half(uint8_t(test_code));
        if (!(test_scale > 0.0f) || !isfinite(test_scale)) {
            continue;
        }
        const float test_inv_scale = 0.5f / test_scale;
        const float half_err = nvfp4_mmvq_half_err8(vals, test_scale, test_inv_scale);
        const float cur_err = half_err + __shfl_xor_sync(pair_mask, half_err, 1);
        if (cur_err < best_err) {
            best_err = cur_err;
            best_code = uint8_t(test_code);
            subblock_scale = test_scale;
        }
    }
    return best_code;
}

template <bool exact_scale>
static __device__ __forceinline__ void load_nvfp4_tileB_fused_1col_mmvq(
        ggml_cuda_mma::tile<8, 8, int> & B, uint32_t & scaleB,
        const float * __restrict__ y,
        const int ne10,
        const int stride_channel_y,
        const int stride_sample_y,
        const int channel_dst,
        const int sample_dst,
        const int frag_abs,
        const float inv_input_scale) {
    const int lane = int(threadIdx.x) & 31;
    const int group = lane & 3;
    const int frag_idx = frag_abs & 3;
    const int qword0 = frag_idx * 8 + group + 0;
    const int qword1 = frag_idx * 8 + group + 4;
    const int sub0 = qword0 >> 1;
    const int sub1 = qword1 >> 1;
    const int half0 = qword0 & 1;
    const int half1 = qword1 & 1;
    const int elem_base0 = (frag_abs >> 2) * QK_K + sub0 * QK_NVFP4_SUB + half0 * 8;
    const int elem_base1 = (frag_abs >> 2) * QK_K + sub1 * QK_NVFP4_SUB + half1 * 8;
    const int64_t base_idx = int64_t(sample_dst) * stride_sample_y + int64_t(channel_dst) * stride_channel_y;

    float amax0 = 0.0f;
    float amax1 = 0.0f;
    uint32_t packed0 = 0;
    uint32_t packed1 = 0;
    uint8_t code0 = 0;
    uint8_t code1 = 0;
    float sub_scale0 = 0.0f;
    float sub_scale1 = 0.0f;
    if (lane < 4) {
        float vals0[8];
        float vals1[8];
        float half_amax0 = 0.0f;
        float half_amax1 = 0.0f;
        if constexpr (exact_scale) {
            nvfp4_mmvq_load_vals8(y, base_idx, elem_base0, ne10, inv_input_scale, vals0, half_amax0);
            nvfp4_mmvq_load_vals8(y, base_idx, elem_base1, ne10, inv_input_scale, vals1, half_amax1);
            amax0 = fmaxf(half_amax0, __shfl_xor_sync(0x0000000Fu, half_amax0, 1));
            amax1 = fmaxf(half_amax1, __shfl_xor_sync(0x0000000Fu, half_amax1, 1));
            code0 = nvfp4_mmvq_select_scale_code_exact(vals0, amax0, sub_scale0);
            code1 = nvfp4_mmvq_select_scale_code_exact(vals1, amax1, sub_scale1);
            const float inv_scale0 = sub_scale0 > 0.0f ? 0.5f / sub_scale0 : 0.0f;
            const float inv_scale1 = sub_scale1 > 0.0f ? 0.5f / sub_scale1 : 0.0f;
            packed0 = nvfp4_mmvq_pack_vals8(vals0, inv_scale0);
            packed1 = nvfp4_mmvq_pack_vals8(vals1, inv_scale1);
        } else {
            packed0 = nvfp4_mmvq_quantize_fp4x8(y, base_idx, elem_base0, ne10, inv_input_scale, 1.0f, half_amax0);
            packed1 = nvfp4_mmvq_quantize_fp4x8(y, base_idx, elem_base1, ne10, inv_input_scale, 1.0f, half_amax1);
            amax0 = fmaxf(half_amax0, __shfl_xor_sync(0x0000000Fu, half_amax0, 1));
            amax1 = fmaxf(half_amax1, __shfl_xor_sync(0x0000000Fu, half_amax1, 1));
            code0 = nvfp4_mmvq_scale_code_from_amax_fast(amax0);
            code1 = nvfp4_mmvq_scale_code_from_amax_fast(amax1);
            sub_scale0 = ggml_cuda_nvfp4_scale_to_fp32_half(code0);
            sub_scale1 = ggml_cuda_nvfp4_scale_to_fp32_half(code1);
            const float inv_scale0 = sub_scale0 > 0.0f ? 0.5f / sub_scale0 : 0.0f;
            const float inv_scale1 = sub_scale1 > 0.0f ? 0.5f / sub_scale1 : 0.0f;
            packed0 = nvfp4_mmvq_quantize_fp4x8(y, base_idx, elem_base0, ne10, inv_input_scale, inv_scale0, half_amax0);
            packed1 = nvfp4_mmvq_quantize_fp4x8(y, base_idx, elem_base1, ne10, inv_input_scale, inv_scale1, half_amax1);
        }
    }

    int * bx = (int *) B.x;
    if (lane < 4) {
        bx[0] = int(packed0);
        bx[1] = int(packed1);
    } else {
        bx[0] = 0;
        bx[1] = 0;
    }

    const uint32_t sc0 = __shfl_sync(0xFFFFFFFFu, uint32_t(code0), 0);
    const uint32_t sc1 = __shfl_sync(0xFFFFFFFFu, uint32_t(code0), 2);
    const uint32_t sc2 = __shfl_sync(0xFFFFFFFFu, uint32_t(code1), 0);
    const uint32_t sc3 = __shfl_sync(0xFFFFFFFFu, uint32_t(code1), 2);
    scaleB = sc0 | (sc1 << 8) | (sc2 << 16) | (sc3 << 24);
}

static __device__ __forceinline__ float vec_dot_nvfp4_q8_1_bw_vdr1_mmvq(
        const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_1,
        const int32_t kbx,
        const int32_t iqs,
        const uint32_t channel_x) {
    const uint32_t packed_kbx = (uint32_t) kbx;
    const int row_in_tile = packed_kbx >> 28;
    const int frag = (packed_kbx >> 24) & 0x0F;
    const int block_rel = packed_kbx & 0x00FFFFFF;
    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vbq;
    const block_nvfp4_blackwell_frag & frag_tile = tensor->tiles[block_rel].tiles[frag];
    const int lane_base = (row_in_tile & 7) * 4;
    const int reg_base  = row_in_tile >> 3;
    const uint32_t scale_word = frag_tile.scales_u32[lane_base + reg_base];
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;

    const int sub = iqs >> 1;
    const int half = iqs & 1;
    const int lane = lane_base + 2*(sub & 1);
    const int reg  = reg_base + 2*(sub >> 1);
    const uint32_t q_lo = frag_tile.regs[lane + 0][reg];
    const uint32_t q_hi = frag_tile.regs[lane + 1][reg];
    const block_q8_1 * bq8 = bq8_1 + (sub >> 1);
    const int32_t i8 = ((sub & 1) << 2);

    int sumi = 0;
    if (half == 0) {
        const int x0 = get_int_from_table_16_contiguous4(q_lo & 0xFFFFu, kvalues_mxfp4);
        const int x2 = get_int_from_table_16_contiguous4(q_hi & 0xFFFFu, kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(x0, get_int_b4(bq8->qs, i8 + 0), sumi);
        sumi = ggml_cuda_dp4a(x2, get_int_b4(bq8->qs, i8 + 2), sumi);
    } else {
        const int x1 = get_int_from_table_16_contiguous4(q_lo >> 16, kvalues_mxfp4);
        const int x3 = get_int_from_table_16_contiguous4(q_hi >> 16, kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(x1, get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(x3, get_int_b4(bq8->qs, i8 + 3), sumi);
    }

    const float d = tensor_scale *
        ggml_cuda_nvfp4_scale_to_fp32_half((scale_word >> (8*sub)) & 0xFF) *
        __low2float(bq8->ds);
    return d * float(sumi);
}

template <int vdr>
static __device__ __forceinline__ float vec_dot_nvfp4_q8_1_bw_vdr_mmvq(
        const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_1,
        const int32_t kbx,
        const int32_t iqs,
        const uint32_t channel_x) {
    static_assert(vdr == 2 || vdr == 4 || vdr == 8, "only VDR2/VDR4/VDR8 are handled here");
    const uint32_t packed_kbx = (uint32_t) kbx;
    const int row_in_tile = packed_kbx >> 28;
    const int frag = (packed_kbx >> 24) & 0x0F;
    const int block_rel = packed_kbx & 0x00FFFFFF;
    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vbq;
    const block_nvfp4_blackwell_frag & frag_tile = tensor->tiles[block_rel].tiles[frag];
    const int lane_base = (row_in_tile & 7) * 4;
    const int reg_base  = row_in_tile >> 3;
    const uint32_t scale_word = frag_tile.scales_u32[lane_base + reg_base];
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < vdr/2; ++i) {
        const int32_t iqs0 = iqs + 2*i;
        const int32_t is = iqs0 >> 1;
        const int32_t sub = is & 3;
        const int lane = lane_base + 2*(sub & 1);
        const int reg  = reg_base + 2*(sub >> 1);
        const uint32_t q_lo = frag_tile.regs[lane + 0][reg];
        const uint32_t q_hi = frag_tile.regs[lane + 1][reg];
        const int x0 = get_int_from_table_16_contiguous4(q_lo & 0xFFFFu, kvalues_mxfp4);
        const int x1 = get_int_from_table_16_contiguous4(q_lo >> 16, kvalues_mxfp4);
        const int x2 = get_int_from_table_16_contiguous4(q_hi & 0xFFFFu, kvalues_mxfp4);
        const int x3 = get_int_from_table_16_contiguous4(q_hi >> 16, kvalues_mxfp4);
        const block_q8_1 * bq8 = bq8_1 + (is >> 1);
        const int32_t i8 = ((is & 1) << 2);

        int sumi = ggml_cuda_dp4a(x0, get_int_b4(bq8->qs, i8 + 0), 0);
        sumi = ggml_cuda_dp4a(x2, get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi = ggml_cuda_dp4a(x1, get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(x3, get_int_b4(bq8->qs, i8 + 3), sumi);

        const float d = tensor_scale *
            ggml_cuda_nvfp4_scale_to_fp32_half((scale_word >> (8*sub)) & 0xFF) *
            __low2float(bq8->ds);
        sum += d * float(sumi);
    }

    return sum;
}

template <int nwarps, int vdr>
static __global__ __launch_bounds__(32*nwarps, 1) void mul_mat_vec_nvfp4_q8_vdr_1col(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int ncols_x,
        const int stride_row_x, const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    constexpr int warp_size = 32;
    constexpr int qi = QI_NVFP4;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = warp_id * warp_size + lane;
    const int row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;
    const int blocks_per_row_x = ncols_x / QK_NVFP4;
    const int kbx_begin = tid / (qi / vdr);
    const int kqs = tid % (qi / vdr);
    const block_q8_1 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;

    float sum = 0.0f;
    for (int kbx = kbx_begin; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const uint32_t block_rel = sample_x*stride_sample_x + channel_x*stride_channel_x +
            (row / 16)*stride_row_x + (kbx >> 2);
        const int kbx_q = int(((uint32_t) (row & 15) << 28) | ((uint32_t) (kbx & 3) << 24) | block_rel);
        if constexpr (vdr == 1) {
            sum += vec_dot_nvfp4_q8_1_bw_vdr1_mmvq(vx, y_cur + 2*kbx, kbx_q, kqs, channel_x);
        } else {
            sum += vec_dot_nvfp4_q8_1_bw_vdr_mmvq<vdr>(vx, y_cur + 2*kbx, kbx_q, kqs, channel_x);
        }
    }

    sum = warp_reduce_sum<warp_size>(sum);
    __shared__ float partial[nwarps];
    if (lane == 0) {
        partial[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane < nwarps ? partial[lane] : 0.0f;
        sum = warp_reduce_sum<warp_size>(sum);
        if (lane == 0) {
            dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = sum;
        }
    }
}

static __device__ __forceinline__ void load_nvfp4_fp8_tileA_sub16_mmvq(
        ggml_cuda_mma::tile<16, 8, int> & A, uint32_t & scaleA,
        const block_nvfp4_blackwell * __restrict__ x,
        const int nvfp4_blocks_per_row,
        const int row_base,
        const int frag_abs32,
        const int sub16) {
    const int lane = int(threadIdx.x) & 31;
    const int row_lo_abs = row_base + (lane >> 2);
    const int row_hi_abs = row_lo_abs + 8;

    const int block_rel   = frag_abs32 / 8;
    const int frag32_rel  = frag_abs32 & 7;
    const int nv_frag_idx = frag32_rel >> 1;
    const int half32      = frag32_rel & 1;
    const int word_idx    = lane & 3;
    int * ax = (int *) A.x;

    const block_nvfp4_blackwell & block_lo = x[(row_lo_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const block_nvfp4_blackwell & block_hi = x[(row_hi_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const int row_lo = row_lo_abs & 15;
    const int row_hi = row_hi_abs & 15;

    ax[0] = (int) nvfp4_mmvq_load_mxf8f6f4_word(block_lo, row_lo, nv_frag_idx, half32, sub16, word_idx + 0);
    ax[1] = (int) nvfp4_mmvq_load_mxf8f6f4_word(block_hi, row_hi, nv_frag_idx, half32, sub16, word_idx + 0);
    ax[2] = (int) nvfp4_mmvq_load_mxf8f6f4_word(block_lo, row_lo, nv_frag_idx, half32, sub16, word_idx + 4);
    ax[3] = (int) nvfp4_mmvq_load_mxf8f6f4_word(block_hi, row_hi, nv_frag_idx, half32, sub16, word_idx + 4);

    const int tid = lane & 3;
    uint8_t sc = 0x7F;
    if (tid == 0) {
        const uint32_t scale_word = ggml_cuda_nvfp4_tile_scale_word(block_lo, row_lo, nv_frag_idx);
        sc = nvfp4_mmvq_ue4m3_scale_to_e8m0_x2((scale_word >> (8 * (half32 * 2 + sub16))) & 0xFFu);
    } else if (tid == 1) {
        const uint32_t scale_word = ggml_cuda_nvfp4_tile_scale_word(block_hi, row_hi, nv_frag_idx);
        sc = nvfp4_mmvq_ue4m3_scale_to_e8m0_x2((scale_word >> (8 * (half32 * 2 + sub16))) & 0xFFu);
    }
    scaleA = uint32_t(sc);
}

static __device__ __forceinline__ void fixup_nvfp4_fp8_tileC_mmvq(
        ggml_cuda_mma::tile<16, 8, float> & C,
        const block_nvfp4_blackwell * __restrict__ x,
        const int nvfp4_blocks_per_row,
        const int row_base,
        const int frag_abs32,
        const int sub16) {
    const int block_rel   = frag_abs32 / 8;
    const int frag32_rel  = frag_abs32 & 7;
    const int nv_frag_idx = frag32_rel >> 1;
    const int half32      = frag32_rel & 1;
    const int scale_shift = 8 * (half32 * 2 + sub16);

    static_assert(ggml_cuda_mma::tile<16, 8, float>::ne == 4, "unexpected 16x8 C tile layout");
    const int row_lo_abs = row_base + (int(threadIdx.x) >> 2);
    const int row_hi_abs = row_lo_abs + 8;
    const block_nvfp4_blackwell & block_lo = x[(row_lo_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const block_nvfp4_blackwell & block_hi = x[(row_hi_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const uint32_t scale_word_lo = ggml_cuda_nvfp4_tile_scale_word(block_lo, row_lo_abs & 15, nv_frag_idx);
    const uint32_t scale_word_hi = ggml_cuda_nvfp4_tile_scale_word(block_hi, row_hi_abs & 15, nv_frag_idx);
    const float corr_lo = nvfp4_mmvq_fp8_scale_correction((scale_word_lo >> scale_shift) & 0xFFu);
    const float corr_hi = nvfp4_mmvq_fp8_scale_correction((scale_word_hi >> scale_shift) & 0xFFu);
    C.x[0] *= corr_lo;
    C.x[1] *= corr_lo;
    C.x[2] *= corr_hi;
    C.x[3] *= corr_hi;
}

static __device__ __forceinline__ uint32_t q8_1_word_to_fp8x4_mmvq(
        const block_q8_1 & bq8, const int word, const float mul) {
    const int base = 4 * word;
    const float2 lo = make_float2(float(bq8.qs[base + 0]) * mul, float(bq8.qs[base + 1]) * mul);
    const float2 hi = make_float2(float(bq8.qs[base + 2]) * mul, float(bq8.qs[base + 3]) * mul);
    const uint32_t p0 = uint32_t(__nv_cvt_float2_to_fp8x2(lo, __NV_SATFINITE, __NV_E4M3));
    const uint32_t p1 = uint32_t(__nv_cvt_float2_to_fp8x2(hi, __NV_SATFINITE, __NV_E4M3));
    return p0 | (p1 << 16);
}

static __device__ __forceinline__ void load_q8_1_as_fp8_tileB_cols_mmvq(
        ggml_cuda_mma::tile<8, 8, int> & B, uint32_t & scaleB,
        const block_q8_1 * __restrict__ q8_blocks,
        const int stride_col_y,
        const int ncols_dst) {
    const int lane = int(threadIdx.x) & 31;
    const int col  = lane >> 2;
    const int tid  = lane & 3;
    int * bx = (int *) B.x;
    if (col < ncols_dst) {
        const block_q8_1 & bq8 = q8_blocks[col*stride_col_y];
        const float d = __low2float(bq8.ds);
        bx[0] = int(q8_1_word_to_fp8x4_mmvq(bq8, tid + 0, d));
        bx[1] = int(q8_1_word_to_fp8x4_mmvq(bq8, tid + 4, d));
    } else {
        bx[0] = 0;
        bx[1] = 0;
    }
    scaleB = 0x7Fu;
}

static __device__ __forceinline__ void load_q8_1_as_fp8_tileB_1col_mmvq(
        ggml_cuda_mma::tile<8, 8, int> & B, uint32_t & scaleB,
        const block_q8_1 * __restrict__ q8_block) {
    const int lane = int(threadIdx.x) & 31;
    const int tid  = lane & 3;
    int * bx = (int *) B.x;
    if (lane < 4) {
        const float d = __low2float(q8_block->ds);
        bx[0] = int(q8_1_word_to_fp8x4_mmvq(*q8_block, tid + 0, d));
        bx[1] = int(q8_1_word_to_fp8x4_mmvq(*q8_block, tid + 4, d));
    } else {
        bx[0] = 0;
        bx[1] = 0;
    }
    scaleB = 0x7Fu;
}

static __device__ __forceinline__ void load_fp8_tileB_cols_mmvq(
        ggml_cuda_mma::tile<8, 8, int> & B, uint32_t & scaleB,
        const block_fp8 * __restrict__ fp8_blocks,
        const int stride_col_y,
        const int ncols_dst,
        const int frag_abs32) {
    const int lane = int(threadIdx.x) & 31;
    const int col  = lane >> 2;
    const int tid  = lane & 3;
    const int block_rel = frag_abs32 / QK_FP8_FRAGS;
    const int frag_idx  = frag_abs32 % QK_FP8_FRAGS;
    int * bx = (int *) B.x;
    uint32_t sc = 0x7Fu;
    if (col < ncols_dst) {
        const block_fp8 & bfp8 = fp8_blocks[col*stride_col_y + block_rel];
        bx[0] = int(ggml_cuda_fp8_get4_u8containers(bfp8, frag_idx, tid + 0));
        bx[1] = int(ggml_cuda_fp8_get4_u8containers(bfp8, frag_idx, tid + 4));
        if (tid == 0) {
            sc = bfp8.e[frag_idx];
        }
    } else {
        bx[0] = 0;
        bx[1] = 0;
    }
    scaleB = __shfl_sync(0xFFFFFFFFu, sc, lane & ~3);
}

static __device__ __forceinline__ void load_fp8_tileB_1col_mmvq(
        ggml_cuda_mma::tile<8, 8, int> & B, uint32_t & scaleB,
        const block_fp8 * __restrict__ fp8_blocks,
        const int frag_abs32) {
    const int lane = int(threadIdx.x) & 31;
    const int tid  = lane & 3;
    const int block_rel = frag_abs32 / QK_FP8_FRAGS;
    const int frag_idx  = frag_abs32 % QK_FP8_FRAGS;
    int * bx = (int *) B.x;
    uint32_t sc = 0x7Fu;
    if (lane < 4) {
        const block_fp8 & bfp8 = fp8_blocks[block_rel];
        bx[0] = int(ggml_cuda_fp8_get4_u8containers(bfp8, frag_idx, tid + 0));
        bx[1] = int(ggml_cuda_fp8_get4_u8containers(bfp8, frag_idx, tid + 4));
        if (lane == 0) {
            sc = bfp8.e[frag_idx];
        }
    } else {
        bx[0] = 0;
        bx[1] = 0;
    }
    scaleB = __shfl_sync(0xFFFFFFFFu, sc, 0);
}

template <int warps_per_tile>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_nvfp4_native_1col(
        const void * __restrict__ vx, const block_nvfp4_mmq * __restrict__ y,
        float * __restrict__ dst, const int nvfp4_blocks_per_row,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles + sample_x*stride_sample_x + channel_x*stride_channel_x;
    const block_nvfp4_mmq * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const float weight_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    const float input_scale  = tensor->input_scales  ? tensor->input_scales[channel_x]  : tensor->input_scale;
    const float tensor_scale = weight_scale * (input_scale != 0.0f ? input_scale : 1.0f);
    const int row_base = tile_row * tile_C::I;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx = warp_id; kbx < nvfp4_blocks_per_row; kbx += warps_per_tile) {
#pragma unroll
        for (int frag = 0; frag < 4; ++frag) {
            tile_A A;
            tile_B B;
            tile_C C_sub = {};
            uint32_t scaleA;
            uint32_t scaleB;
            load_nvfp4_tileA_native_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, kbx * 4 + frag);
            load_nvfp4_tileB_native_1col_mmvq(B, scaleB, y_cur[kbx], frag);
            mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C_sub, A, B, scaleA, scaleB);
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] += C_sub.x[l];
            }
        }
    }

    __shared__ float partial[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial[row_part + 0][warp_id] = C.x[0];
        partial[row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial[row_in_tile][w];
        }
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row_base + row_in_tile] =
            tensor_scale * sum;
    }
}

template <int warps_per_tile, int rows_per_block>
static __global__ __launch_bounds__(32*warps_per_tile*rows_per_block, 1) void mul_mat_vec_nvfp4_native_1col_rows(
        const void * __restrict__ vx, const block_nvfp4_mmq * __restrict__ y,
        float * __restrict__ dst, const int nvfp4_blocks_per_row,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio,
        const int ne01) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x * rows_per_block + threadIdx.z;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles + sample_x*stride_sample_x + channel_x*stride_channel_x;
    const block_nvfp4_mmq * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const float weight_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    const float input_scale  = tensor->input_scales  ? tensor->input_scales[channel_x]  : tensor->input_scale;
    const float tensor_scale = weight_scale * (input_scale != 0.0f ? input_scale : 1.0f);
    const int row_base = tile_row * tile_C::I;
    const bool valid_row = row_base < ne01;

    const int lane = int(threadIdx.x) & 31;
    const int warp_id = threadIdx.y;
    __shared__ int b_shared[warps_per_tile][32][2];
    __shared__ uint32_t scaleB_shared[warps_per_tile];
    tile_C C = {};
    for (int kbx = warp_id; kbx < nvfp4_blocks_per_row; kbx += warps_per_tile) {
#pragma unroll
        for (int frag = 0; frag < 4; ++frag) {
            if (threadIdx.z == 0) {
                tile_B B_pre;
                uint32_t scaleB_pre;
                load_nvfp4_tileB_native_1col_mmvq(B_pre, scaleB_pre, y_cur[kbx], frag);
                const int * bx_pre = (const int *) B_pre.x;
                b_shared[warp_id][lane][0] = bx_pre[0];
                b_shared[warp_id][lane][1] = bx_pre[1];
                if (lane == 0) {
                    scaleB_shared[warp_id] = scaleB_pre;
                }
            }
            __syncthreads();

            if (valid_row) {
                tile_A A;
                tile_B B;
                tile_C C_sub = {};
                uint32_t scaleA;
                int * bx = (int *) B.x;
                bx[0] = b_shared[warp_id][lane][0];
                bx[1] = b_shared[warp_id][lane][1];
                load_nvfp4_tileA_native_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, kbx * 4 + frag);
                mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C_sub, A, B, scaleA, scaleB_shared[warp_id]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    C.x[l] += C_sub.x[l];
                }
            }
            __syncthreads();
        }
    }

    __shared__ float partial_warp[rows_per_block][16][warps_per_tile];
    if (valid_row && (lane & 3) == 0) {
        const int row_part = lane >> 2;
        partial_warp[threadIdx.z][row_part + 0][warp_id] = C.x[0];
        partial_warp[threadIdx.z][row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (valid_row && warp_id == 0 && lane < 16) {
        const int row_in_tile = lane;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial_warp[threadIdx.z][row_in_tile][w];
        }
        const int row = row_base + row_in_tile;
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = tensor_scale * sum;
    }
}

template <int warps_per_tile, int split_k, bool exact_scale>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_nvfp4_native_fused_1col_splitk(
        const void * __restrict__ vx, const float * __restrict__ y,
        float * __restrict__ partial, const int nvfp4_blocks_per_row,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int channel_ratio, const int sample_ratio,
        const int ne10, const int ne01, const int nchannels_dst) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int split_id = blockIdx.z % split_k;
    const int sample_dst = blockIdx.z / split_k;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles + sample_x*stride_sample_x + channel_x*stride_channel_x;
    const float weight_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    const float input_scale  = tensor->input_scales  ? tensor->input_scales[channel_x]  : tensor->input_scale;
    const float input_scale_safe = input_scale != 0.0f && isfinite(input_scale) ? input_scale : 1.0f;
    const float inv_input_scale = 1.0f / input_scale_safe;
    const float tensor_scale = weight_scale * input_scale_safe;
    const int row_base = tile_row * tile_C::I;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx = split_id*warps_per_tile + warp_id; kbx < nvfp4_blocks_per_row; kbx += warps_per_tile*split_k) {
#pragma unroll
        for (int frag = 0; frag < 4; ++frag) {
            tile_A A;
            tile_B B;
            tile_C C_sub = {};
            uint32_t scaleA;
            uint32_t scaleB;
            const int frag_abs = kbx * 4 + frag;
            load_nvfp4_tileA_native_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, frag_abs);
            load_nvfp4_tileB_fused_1col_mmvq<exact_scale>(B, scaleB, y, ne10, stride_channel_y, stride_sample_y,
                    channel_dst, sample_dst, frag_abs, inv_input_scale);
            mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C_sub, A, B, scaleA, scaleB);
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] += C_sub.x[l];
            }
        }
    }

    __shared__ float partial_warp[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial_warp[row_part + 0][warp_id] = C.x[0];
        partial_warp[row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial_warp[row_in_tile][w];
        }
        const int row = row_base + row_in_tile;
        partial[((sample_dst*nchannels_dst + channel_dst)*split_k + split_id)*ne01 + row] = tensor_scale * sum;
    }
}

template <int warps_per_tile, int split_k, int rows_per_block, bool exact_scale>
static __global__ __launch_bounds__(32*warps_per_tile*rows_per_block, 1) void mul_mat_vec_nvfp4_native_fused_1col_splitk_rows(
        const void * __restrict__ vx, const float * __restrict__ y,
        float * __restrict__ partial, const int nvfp4_blocks_per_row,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int channel_ratio, const int sample_ratio,
        const int ne10, const int ne01, const int nchannels_dst) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x * rows_per_block + threadIdx.z;
    const int channel_dst = blockIdx.y;
    const int split_id = blockIdx.z % split_k;
    const int sample_dst = blockIdx.z / split_k;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles + sample_x*stride_sample_x + channel_x*stride_channel_x;
    const float weight_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    const float input_scale  = tensor->input_scales  ? tensor->input_scales[channel_x]  : tensor->input_scale;
    const float input_scale_safe = input_scale != 0.0f && isfinite(input_scale) ? input_scale : 1.0f;
    const float inv_input_scale = 1.0f / input_scale_safe;
    const float tensor_scale = weight_scale * input_scale_safe;
    const int row_base = tile_row * tile_C::I;
    const bool valid_row = row_base < ne01;

    const int lane = int(threadIdx.x) & 31;
    const int warp_id = threadIdx.y;
    __shared__ int b_shared[warps_per_tile][32][2];
    __shared__ uint32_t scaleB_shared[warps_per_tile];
    tile_C C = {};
    for (int kbx = split_id*warps_per_tile + warp_id; kbx < nvfp4_blocks_per_row; kbx += warps_per_tile*split_k) {
#pragma unroll
        for (int frag = 0; frag < 4; ++frag) {
            const int frag_abs = kbx * 4 + frag;
            if (threadIdx.z == 0) {
                tile_B B_pre;
                uint32_t scaleB_pre;
                load_nvfp4_tileB_fused_1col_mmvq<exact_scale>(B_pre, scaleB_pre, y, ne10, stride_channel_y, stride_sample_y,
                        channel_dst, sample_dst, frag_abs, inv_input_scale);
                const int * bx_pre = (const int *) B_pre.x;
                b_shared[warp_id][lane][0] = bx_pre[0];
                b_shared[warp_id][lane][1] = bx_pre[1];
                if (lane == 0) {
                    scaleB_shared[warp_id] = scaleB_pre;
                }
            }
            __syncthreads();

            if (valid_row) {
                tile_A A;
                tile_B B;
                tile_C C_sub = {};
                uint32_t scaleA;
                int * bx = (int *) B.x;
                bx[0] = b_shared[warp_id][lane][0];
                bx[1] = b_shared[warp_id][lane][1];
                load_nvfp4_tileA_native_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, frag_abs);
                mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C_sub, A, B, scaleA, scaleB_shared[warp_id]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    C.x[l] += C_sub.x[l];
                }
            }
            __syncthreads();
        }
    }

    __shared__ float partial_warp[rows_per_block][16][warps_per_tile];
    if (valid_row && (lane & 3) == 0) {
        const int row_part = lane >> 2;
        partial_warp[threadIdx.z][row_part + 0][warp_id] = C.x[0];
        partial_warp[threadIdx.z][row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (valid_row && warp_id == 0 && lane < 16) {
        const int row_in_tile = lane;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial_warp[threadIdx.z][row_in_tile][w];
        }
        const int row = row_base + row_in_tile;
        partial[((sample_dst*nchannels_dst + channel_dst)*split_k + split_id)*ne01 + row] = tensor_scale * sum;
    }
}

template <int warps_per_tile, int rows_per_block, bool exact_scale>
static __global__ __launch_bounds__(32*warps_per_tile*rows_per_block, 1) void mul_mat_vec_nvfp4_native_fused_1col_rows(
        const void * __restrict__ vx, const float * __restrict__ y,
        float * __restrict__ dst, const int nvfp4_blocks_per_row,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio,
        const int ne10, const int ne01) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x * rows_per_block + threadIdx.z;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles + sample_x*stride_sample_x + channel_x*stride_channel_x;
    const float weight_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    const float input_scale  = tensor->input_scales  ? tensor->input_scales[channel_x]  : tensor->input_scale;
    const float input_scale_safe = input_scale != 0.0f && isfinite(input_scale) ? input_scale : 1.0f;
    const float inv_input_scale = 1.0f / input_scale_safe;
    const float tensor_scale = weight_scale * input_scale_safe;
    const int row_base = tile_row * tile_C::I;
    const bool valid_row = row_base < ne01;

    const int lane = int(threadIdx.x) & 31;
    const int warp_id = threadIdx.y;
    __shared__ int b_shared[warps_per_tile][32][2];
    __shared__ uint32_t scaleB_shared[warps_per_tile];
    tile_C C = {};
    for (int kbx = warp_id; kbx < nvfp4_blocks_per_row; kbx += warps_per_tile) {
#pragma unroll
        for (int frag = 0; frag < 4; ++frag) {
            const int frag_abs = kbx * 4 + frag;
            if (threadIdx.z == 0) {
                tile_B B_pre;
                uint32_t scaleB_pre;
                load_nvfp4_tileB_fused_1col_mmvq<exact_scale>(B_pre, scaleB_pre, y, ne10, stride_channel_y, stride_sample_y,
                        channel_dst, sample_dst, frag_abs, inv_input_scale);
                const int * bx_pre = (const int *) B_pre.x;
                b_shared[warp_id][lane][0] = bx_pre[0];
                b_shared[warp_id][lane][1] = bx_pre[1];
                if (lane == 0) {
                    scaleB_shared[warp_id] = scaleB_pre;
                }
            }
            __syncthreads();

            if (valid_row) {
                tile_A A;
                tile_B B;
                tile_C C_sub = {};
                uint32_t scaleA;
                int * bx = (int *) B.x;
                bx[0] = b_shared[warp_id][lane][0];
                bx[1] = b_shared[warp_id][lane][1];
                load_nvfp4_tileA_native_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, frag_abs);
                mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C_sub, A, B, scaleA, scaleB_shared[warp_id]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    C.x[l] += C_sub.x[l];
                }
            }
            __syncthreads();
        }
    }

    __shared__ float partial_warp[rows_per_block][16][warps_per_tile];
    if (valid_row && (lane & 3) == 0) {
        const int row_part = lane >> 2;
        partial_warp[threadIdx.z][row_part + 0][warp_id] = C.x[0];
        partial_warp[threadIdx.z][row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (valid_row && warp_id == 0 && lane < 16) {
        const int row_in_tile = lane;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial_warp[threadIdx.z][row_in_tile][w];
        }
        const int row = row_base + row_in_tile;
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = tensor_scale * sum;
    }
}

template <int warps_per_tile, bool exact_scale>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_nvfp4_q8_to_fp8_1col(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int nvfp4_blocks_per_row, const int blocks_per_row_q8,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles + sample_x*stride_sample_x + channel_x*stride_channel_x;
    const block_q8_1 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    const int row_base = tile_row * tile_C::I;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx32 = warp_id; kbx32 < blocks_per_row_q8; kbx32 += warps_per_tile) {
        tile_B B_pre;
        uint32_t scaleB_pre;
        load_q8_1_as_fp8_tileB_1col_mmvq(B_pre, scaleB_pre, y_cur + kbx32);

#pragma unroll
        for (int sub16 = 0; sub16 < 2; ++sub16) {
            tile_A A;
            tile_C C_sub = {};
            uint32_t scaleA;
            load_nvfp4_fp8_tileA_sub16_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
            mma_block_scaled_mxfp4_e2m1_fp8_e4m3(C_sub, A, B_pre, scaleA, scaleB_pre);
            if constexpr (exact_scale) {
                fixup_nvfp4_fp8_tileC_mmvq(C_sub, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
            }
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] += C_sub.x[l];
            }
        }
    }

    __shared__ float partial[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial[row_part + 0][warp_id] = C.x[0];
        partial[row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial[row_in_tile][w];
        }
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row_base + row_in_tile] =
            tensor_scale * sum;
    }
}

template <int warps_per_tile>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_nvfp4_q8_to_fp8_cols(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int nvfp4_blocks_per_row, const int blocks_per_row_q8,
        const int stride_channel_x, const int stride_col_y,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_col_dst, const int stride_channel_dst, const int stride_sample_dst,
        const int ncols_dst) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles +
        (sample_dst*int(gridDim.y) + channel_dst)*stride_channel_x;
    const block_q8_1 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_dst] : tensor->weight_scale;
    const int row_base = tile_row * tile_C::I;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx32 = warp_id; kbx32 < blocks_per_row_q8; kbx32 += warps_per_tile) {
        tile_B B_pre;
        uint32_t scaleB_pre;
        load_q8_1_as_fp8_tileB_cols_mmvq(B_pre, scaleB_pre, y_cur + kbx32, stride_col_y, ncols_dst);

#pragma unroll
        for (int sub16 = 0; sub16 < 2; ++sub16) {
            tile_A A;
            tile_C C_sub = {};
            uint32_t scaleA;
            load_nvfp4_fp8_tileA_sub16_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
            mma_block_scaled_mxfp4_e2m1_fp8_e4m3(C_sub, A, B_pre, scaleA, scaleB_pre);
            fixup_nvfp4_fp8_tileC_mmvq(C_sub, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] += C_sub.x[l];
            }
        }
    }

    __shared__ float partial[16][8][warps_per_tile];
#pragma unroll
    for (int l = 0; l < tile_C::ne; ++l) {
        partial[tile_C::get_i(l)][tile_C::get_j(l)][warp_id] = C.x[l];
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = row_base + tile_C::get_i(l);
            const int col = tile_C::get_j(l);
            if (col < ncols_dst) {
                float sum = 0.0f;
#pragma unroll
                for (int w = 0; w < warps_per_tile; ++w) {
                    sum += partial[row - row_base][col][w];
                }
                dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row] =
                    tensor_scale * sum;
            }
        }
    }
}

template <int warps_per_tile, bool exact_scale>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_nvfp4_fp8_direct_1col(
        const void * __restrict__ vx, const block_fp8 * __restrict__ y,
        float * __restrict__ dst, const int nvfp4_blocks_per_row, const int k32_per_row,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles + sample_x*stride_sample_x + channel_x*stride_channel_x;
    const block_fp8 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    const int row_base = tile_row * tile_C::I;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx32 = warp_id; kbx32 < k32_per_row; kbx32 += warps_per_tile) {
        tile_B B_pre;
        uint32_t scaleB_pre;
        load_fp8_tileB_1col_mmvq(B_pre, scaleB_pre, y_cur, kbx32);

#pragma unroll
        for (int sub16 = 0; sub16 < 2; ++sub16) {
            tile_A A;
            tile_C C_sub = {};
            uint32_t scaleA;
            load_nvfp4_fp8_tileA_sub16_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
            mma_block_scaled_mxfp4_e2m1_fp8_e4m3(C_sub, A, B_pre, scaleA, scaleB_pre);
            if constexpr (exact_scale) {
                fixup_nvfp4_fp8_tileC_mmvq(C_sub, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
            }
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] += C_sub.x[l];
            }
        }
    }

    __shared__ float partial[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial[row_part + 0][warp_id] = C.x[0];
        partial[row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial[row_in_tile][w];
        }
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row_base + row_in_tile] =
            tensor_scale * sum;
    }
}

template <int warps_per_tile, int split_k, bool exact_scale>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_nvfp4_fp8_direct_1col_splitk(
        const void * __restrict__ vx, const block_fp8 * __restrict__ y,
        float * __restrict__ partial, const int nvfp4_blocks_per_row, const int k32_per_row,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int channel_ratio, const int sample_ratio,
        const int ne01, const int nchannels_dst) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int split_id = blockIdx.z % split_k;
    const int sample_dst = blockIdx.z / split_k;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles + sample_x*stride_sample_x + channel_x*stride_channel_x;
    const block_fp8 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;
    const int row_base = tile_row * tile_C::I;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx32 = split_id*warps_per_tile + warp_id; kbx32 < k32_per_row; kbx32 += warps_per_tile*split_k) {
        tile_B B_pre;
        uint32_t scaleB_pre;
        load_fp8_tileB_1col_mmvq(B_pre, scaleB_pre, y_cur, kbx32);

#pragma unroll
        for (int sub16 = 0; sub16 < 2; ++sub16) {
            tile_A A;
            tile_C C_sub = {};
            uint32_t scaleA;
            load_nvfp4_fp8_tileA_sub16_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
            mma_block_scaled_mxfp4_e2m1_fp8_e4m3(C_sub, A, B_pre, scaleA, scaleB_pre);
            if constexpr (exact_scale) {
                fixup_nvfp4_fp8_tileC_mmvq(C_sub, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
            }
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] += C_sub.x[l];
            }
        }
    }

    __shared__ float partial_warp[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial_warp[row_part + 0][warp_id] = C.x[0];
        partial_warp[row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial_warp[row_in_tile][w];
        }
        const int row = row_base + row_in_tile;
        partial[((sample_dst*nchannels_dst + channel_dst)*split_k + split_id)*ne01 + row] = tensor_scale * sum;
    }
}

template <int split_k>
static __global__ void reduce_nvfp4_fp8_direct_1col_splitk(
        const float * __restrict__ partial, float * __restrict__ dst,
        const int ne01, const int nchannels_dst,
        const int stride_channel_dst, const int stride_sample_dst) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    if (row >= ne01) {
        return;
    }

    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < split_k; ++s) {
        sum += partial[((sample_dst*nchannels_dst + channel_dst)*split_k + s)*ne01 + row];
    }
    dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = sum;
}

template <int warps_per_tile, bool exact_scale>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_nvfp4_fp8_direct_cols(
        const void * __restrict__ vx, const block_fp8 * __restrict__ y,
        float * __restrict__ dst, const int nvfp4_blocks_per_row, const int k32_per_row,
        const int stride_channel_x, const int stride_col_y,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_col_dst, const int stride_channel_dst, const int stride_sample_dst,
        const int ncols_dst) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;

    const block_nvfp4_blackwell_tensor * tensor = (const block_nvfp4_blackwell_tensor *) vx;
    const block_nvfp4_blackwell * x = tensor->tiles +
        (sample_dst*int(gridDim.y) + channel_dst)*stride_channel_x;
    const block_fp8 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_dst] : tensor->weight_scale;
    const int row_base = tile_row * tile_C::I;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx32 = warp_id; kbx32 < k32_per_row; kbx32 += warps_per_tile) {
        tile_B B_pre;
        uint32_t scaleB_pre;
        load_fp8_tileB_cols_mmvq(B_pre, scaleB_pre, y_cur, stride_col_y, ncols_dst, kbx32);

#pragma unroll
        for (int sub16 = 0; sub16 < 2; ++sub16) {
            tile_A A;
            tile_C C_sub = {};
            uint32_t scaleA;
            load_nvfp4_fp8_tileA_sub16_mmvq(A, scaleA, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
            mma_block_scaled_mxfp4_e2m1_fp8_e4m3(C_sub, A, B_pre, scaleA, scaleB_pre);
            if constexpr (exact_scale) {
                fixup_nvfp4_fp8_tileC_mmvq(C_sub, x, nvfp4_blocks_per_row, row_base, kbx32, sub16);
            }
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] += C_sub.x[l];
            }
        }
    }

    __shared__ float partial[16][8][warps_per_tile];
#pragma unroll
    for (int l = 0; l < tile_C::ne; ++l) {
        partial[tile_C::get_i(l)][tile_C::get_j(l)][warp_id] = C.x[l];
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = row_base + tile_C::get_i(l);
            const int col = tile_C::get_j(l);
            if (col < ncols_dst) {
                float sum = 0.0f;
#pragma unroll
                for (int w = 0; w < warps_per_tile; ++w) {
                    sum += partial[row - row_base][col][w];
                }
                dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row] =
                    tensor_scale * sum;
            }
        }
    }
}

static __device__ __forceinline__ uint8_t fp8_fastq_scale_code_from_amax_mmvq(const float amax) {
    if (!(amax > 0.0f) || !isfinite(amax)) {
        return 127;
    }
    const int code = int(ceilf(log2f(amax * (1.0f / 448.0f)))) + 127;
    return uint8_t(max(0, min(254, code)));
}

static __global__ void quantize_row_fp8_fast_mmvq(
        const float * __restrict__ x,
        block_fp8 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i1 = blockIdx.x;
    const int64_t k_block = blockIdx.y;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t blocks_per_col = (ne0 + QK_FP8 - 1) / QK_FP8;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t base_idx = i3 * s03 + i2 * s02 + i1 * s01;
    const int64_t batch_offset = (int64_t) blockIdx.z * (ne1 * blocks_per_col);
    block_fp8 * yb = y + batch_offset + i1 * blocks_per_col + k_block;

    if (warp >= QK_FP8_FRAGS) {
        return;
    }

    const int frag = warp;
    const int elem0 = 2 * lane + 0;
    const int elem1 = 2 * lane + 1;
    const int64_t i00 = k_block * QK_FP8 + frag * QK_FP8_SUB;
    const float v0_raw = (lane < 16 && i00 + elem0 < ne00) ? x[base_idx + i00 + elem0] : 0.0f;
    const float v1_raw = (lane < 16 && i00 + elem1 < ne00) ? x[base_idx + i00 + elem1] : 0.0f;
    const float v0 = isfinite(v0_raw) ? v0_raw : 0.0f;
    const float v1 = isfinite(v1_raw) ? v1_raw : 0.0f;
    const float lane_amax = lane < 16 ? fmaxf(fabsf(v0), fabsf(v1)) : 0.0f;
    const float amax = warp_reduce_max<32>(lane_amax);
    const uint8_t scale_code = fp8_fastq_scale_code_from_amax_mmvq(amax);
    const float scale = ggml_cuda_e8m0_to_fp32(scale_code);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

    if (lane == 0) {
        yb->e[frag] = scale_code;
        if (frag == 0) {
            yb->pad[0] = 0;
        }
    }

    if (lane < 16) {
        const uint32_t fp8x2 = uint32_t(__nv_cvt_float2_to_fp8x2(
            make_float2(v0 * inv_scale, v1 * inv_scale), __NV_SATFINITE, __NV_E4M3));
        const int word = lane >> 1;
        const uint32_t packed = fp8x2 | (__shfl_sync(0xFFFFFFFFu, fp8x2, lane | 1) << 16);
        if ((lane & 1) == 0) {
            ggml_cuda_fp8_set4_u8containers(*yb, frag, word, packed);
        }
    }
#else
    GGML_UNUSED_VARS(x, y, ne00, s01, s02, s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif
}

static void quantize_row_fp8_fast_mmvq_cuda(
        const float * x, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        cudaStream_t stream) {
    const int64_t block_num_y = (ne0 + QK_FP8 - 1) / QK_FP8;
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(4 * 32, 1, 1);
    quantize_row_fp8_fast_mmvq<<<num_blocks, block_size, 0, stream>>>(
        x, (block_fp8 *) vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

template <int warps_per_tile>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_mxfp6_q8_to_fp8_1col(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int blocks_per_row_x,
        const int stride_channel_x, const int stride_sample_x,
        const int stride_channel_y, const int stride_sample_y,
        const int stride_channel_dst, const int stride_sample_dst,
        const int channel_ratio, const int sample_ratio) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;
    const int channel_x = channel_dst / channel_ratio;
    const int sample_x = sample_dst / sample_ratio;

    const tile_mxfp6_e2m3_blackwell_tensor * tensor = (const tile_mxfp6_e2m3_blackwell_tensor *) vx;
    const tile_mxfp6_e2m3_blackwell * x = tensor->tiles +
        sample_x*stride_sample_x + channel_x*stride_channel_x + tile_row*blocks_per_row_x;
    const block_q8_1 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx = warp_id; kbx < blocks_per_row_x; kbx += warps_per_tile) {
        tile_A A;
        tile_B B;
        int * bx = (int *) B.x;
        const int lane = int(threadIdx.x) & 31;
        const int tid  = lane & 3;
        if (lane < 4) {
            const block_q8_1 & bq8 = y_cur[kbx];
            const float d = __low2float(bq8.ds);
            bx[0] = int(uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 0]) * d, float(bq8.qs[4*tid + 1]) * d), __NV_SATFINITE, __NV_E4M3)) |
                        uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 2]) * d, float(bq8.qs[4*tid + 3]) * d), __NV_SATFINITE, __NV_E4M3)) << 16);
            bx[1] = int(uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 16]) * d, float(bq8.qs[4*tid + 17]) * d), __NV_SATFINITE, __NV_E4M3)) |
                        uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 18]) * d, float(bq8.qs[4*tid + 19]) * d), __NV_SATFINITE, __NV_E4M3)) << 16);
        } else {
            bx[0] = 0;
            bx[1] = 0;
        }

        uint32_t scaleA;
        load_mxfp6_e2m3_tileA_416(A, scaleA, x[kbx].tiles[0]);
        mma_block_scaled_mxfp6_e2m3_fp8_e4m3(C, A, B, scaleA, 0x7Fu);
    }

    __shared__ float partial[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial[row_part + 0][warp_id] = C.x[0];
        partial[row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        const int row = tile_row * tile_C::I + row_in_tile;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial[row_in_tile][w];
        }
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = tensor_scale * sum;
    }
}

template <int warps_per_tile>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_mxfp6_q8_to_fp8_cols(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int blocks_per_row_x, const int stride_channel_x,
        const int stride_col_y, const int stride_channel_y, const int stride_sample_y,
        const int stride_col_dst, const int stride_channel_dst, const int stride_sample_dst,
        const int ncols_dst) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst = blockIdx.z;

    const tile_mxfp6_e2m3_blackwell_tensor * tensor = (const tile_mxfp6_e2m3_blackwell_tensor *) vx;
    const tile_mxfp6_e2m3_blackwell * x = tensor->tiles +
        (sample_dst*int(gridDim.y) + channel_dst)*stride_channel_x + tile_row*blocks_per_row_x;
    const block_q8_1 * y_cur = y + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_dst] : tensor->weight_scale;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx = warp_id; kbx < blocks_per_row_x; kbx += warps_per_tile) {
        tile_A A;
        tile_B B;
        int * bx = (int *) B.x;
        const int lane = int(threadIdx.x) & 31;
        const int col  = lane >> 2;
        const int tid  = lane & 3;
        if (col < ncols_dst) {
            const block_q8_1 & bq8 = y_cur[kbx + col*stride_col_y];
            const float d = __low2float(bq8.ds);
            bx[0] = int(uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 0]) * d, float(bq8.qs[4*tid + 1]) * d), __NV_SATFINITE, __NV_E4M3)) |
                        uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 2]) * d, float(bq8.qs[4*tid + 3]) * d), __NV_SATFINITE, __NV_E4M3)) << 16);
            bx[1] = int(uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 16]) * d, float(bq8.qs[4*tid + 17]) * d), __NV_SATFINITE, __NV_E4M3)) |
                        uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 18]) * d, float(bq8.qs[4*tid + 19]) * d), __NV_SATFINITE, __NV_E4M3)) << 16);
        } else {
            bx[0] = 0;
            bx[1] = 0;
        }

        uint32_t scaleA;
        load_mxfp6_e2m3_tileA_416(A, scaleA, x[kbx].tiles[0]);
        mma_block_scaled_mxfp6_e2m3_fp8_e4m3(C, A, B, scaleA, 0x7Fu);
    }

    __shared__ float partial[16][8][warps_per_tile];
#pragma unroll
    for (int l = 0; l < tile_C::ne; ++l) {
        partial[tile_C::get_i(l)][tile_C::get_j(l)][warp_id] = C.x[l];
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = tile_row * tile_C::I + tile_C::get_i(l);
            const int col = tile_C::get_j(l);
            if (col < ncols_dst) {
                float sum = 0.0f;
#pragma unroll
                for (int w = 0; w < warps_per_tile; ++w) {
                    sum += partial[row - tile_row * tile_C::I][col][w];
                }
                dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row] =
                    tensor_scale * sum;
            }
        }
    }
}

template <int warps_per_tile>
static __global__ __launch_bounds__(32*warps_per_tile, 1) void mul_mat_vec_mxfp6_q8_to_fp8_moe(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int32_t * __restrict__ ids,
        float * __restrict__ dst, const int blocks_per_row_x, const int stride_channel_x,
        const int stride_col_y, const int stride_channel_y, const int stride_col_dst,
        const int stride_channel_dst, const int ids_stride, const int nchannels_y) {
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8, 8, int>    tile_B;
    typedef tile<16, 8, float> tile_C;

    const int tile_row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int token_idx = blockIdx.z;
    const int channel_x = ids[channel_dst + token_idx*ids_stride];
    const int channel_y = channel_dst % nchannels_y;

    const tile_mxfp6_e2m3_blackwell_tensor * tensor = (const tile_mxfp6_e2m3_blackwell_tensor *) vx;
    const tile_mxfp6_e2m3_blackwell * x = tensor->tiles + channel_x*stride_channel_x + tile_row*blocks_per_row_x;
    const block_q8_1 * y_cur = y + channel_y*stride_channel_y + token_idx*stride_col_y;
    const float tensor_scale = tensor->weight_scales ? tensor->weight_scales[channel_x] : tensor->weight_scale;

    const int warp_id = threadIdx.y;
    tile_C C = {};
    for (int kbx = warp_id; kbx < blocks_per_row_x; kbx += warps_per_tile) {
        tile_A A;
        tile_B B;
        int * bx = (int *) B.x;
        const int lane = int(threadIdx.x) & 31;
        const int tid  = lane & 3;
        if (lane < 4) {
            const block_q8_1 & bq8 = y_cur[kbx];
            const float d = __low2float(bq8.ds);
            bx[0] = int(uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 0]) * d, float(bq8.qs[4*tid + 1]) * d), __NV_SATFINITE, __NV_E4M3)) |
                        uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 2]) * d, float(bq8.qs[4*tid + 3]) * d), __NV_SATFINITE, __NV_E4M3)) << 16);
            bx[1] = int(uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 16]) * d, float(bq8.qs[4*tid + 17]) * d), __NV_SATFINITE, __NV_E4M3)) |
                        uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(float(bq8.qs[4*tid + 18]) * d, float(bq8.qs[4*tid + 19]) * d), __NV_SATFINITE, __NV_E4M3)) << 16);
        } else {
            bx[0] = 0;
            bx[1] = 0;
        }

        uint32_t scaleA;
        load_mxfp6_e2m3_tileA_416(A, scaleA, x[kbx].tiles[0]);
        mma_block_scaled_mxfp6_e2m3_fp8_e4m3(C, A, B, scaleA, 0x7Fu);
    }

    __shared__ float partial[16][warps_per_tile];
    if ((threadIdx.x & 3) == 0) {
        const int row_part = threadIdx.x >> 2;
        partial[row_part + 0][warp_id] = C.x[0];
        partial[row_part + 8][warp_id] = C.x[2];
    }
    __syncthreads();

    if (warp_id == 0 && threadIdx.x < 16) {
        const int row_in_tile = threadIdx.x;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < warps_per_tile; ++w) {
            sum += partial[row_in_tile][w];
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + tile_row*tile_C::I + row_in_tile] =
            tensor_scale * sum;
    }
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool use_nvfp4 = src0->type == GGML_TYPE_NVFP4;
    const bool use_mxfp6 = src0->type == GGML_TYPE_MXFP6_E2M3 && src0->view_src == nullptr &&
        ne00 % QK_MXFP6_E2M3 == 0 &&
        blackwell_mma_available(cc);
    const bool use_mxfp8_mma_1col =
        !ids && ggml_cuda_should_use_mxfp8_mma_1col(src0->type, cc, ne11, ne01);
#endif // defined(BLACKWELL_MMA_AVAILABLE)

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }
#if defined(BLACKWELL_MMA_AVAILABLE)
    const bool has_mmvq_fusion =
        fusion_local.gate != nullptr || fusion_local.x_bias != nullptr || fusion_local.gate_bias != nullptr;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    // If src0 is a temporary compute buffer, clear any potential padding.
    if (src0->type != GGML_TYPE_NVFP4 &&
            src0->type != GGML_TYPE_MXFP6_E2M3 &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_channel_dst = ids ? s1   : s2;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    int64_t stride_row_x = s01;
    int64_t stride_channel_x = s02;
    int64_t stride_sample_x = s03;
    int64_t physical_rows_x = ne01;
#if defined(BLACKWELL_MMA_AVAILABLE)
    if (use_nvfp4) {
        const int64_t blocks_per_row_x = ggml_cuda_nvfp4_blocks_per_row(ne00);
        const int64_t tiles_per_channel_x = ggml_cuda_bw_div_up(ne01, 16);
        stride_row_x = blocks_per_row_x;
        stride_channel_x = tiles_per_channel_x * blocks_per_row_x;
        stride_sample_x = (s03 / s02) * stride_channel_x;
    } else if (use_mxfp6) {
        const int64_t blocks_per_row_x = ggml_cuda_mxfp6_e2m3_blocks_per_row(ne00);
        const int64_t tiles_per_channel_x = ggml_cuda_bw_div_up(ne01, 16);
        stride_row_x = blocks_per_row_x;
        stride_channel_x = tiles_per_channel_x * blocks_per_row_x;
        stride_sample_x = (s03 / s02) * stride_channel_x;
    }
#endif // defined(BLACKWELL_MMA_AVAILABLE)

	    const int64_t s11_src1 = src1->nb[1] / ts_src1;
	    const int64_t s12_src1 = src1->nb[2] / ts_src1;
	    const int64_t s13_src1 = src1->nb[3] / ts_src1;
	    const void * src0_data = src0->data;
#if defined(BLACKWELL_MMA_AVAILABLE)
            const bool nvfp4_mmvq_1col_shape =
                !ids && !has_mmvq_fusion && ncols_dst == 1 && ne01 % 16 == 0;
            const bool nvfp4_mmvq_cols_shape =
                !ids && !has_mmvq_fusion && ncols_dst > 1 && ncols_dst <= 8 &&
                ne01 % 16 == 0 && nchannels_dst == ne02 && ne3 == ne03;
            const bool use_native_nvfp4_acts = use_nvfp4 && nvfp4_mmvq_1col_shape &&
                ggml_cuda_nvfp4_mmvq_native_acts();
            const bool use_direct_nvfp4_fp8_acts_requested = use_nvfp4 && ggml_cuda_nvfp4_mmvq_direct_fp8_acts();
            const bool use_direct_nvfp4_fp8_fulln_only = use_direct_nvfp4_fp8_acts_requested &&
                ggml_cuda_nvfp4_mmvq_direct_fp8_fulln_only();
            const bool use_direct_nvfp4_fp8_acts = use_direct_nvfp4_fp8_acts_requested &&
                (nvfp4_mmvq_1col_shape || nvfp4_mmvq_cols_shape) &&
                (!use_direct_nvfp4_fp8_fulln_only || ncols_dst > 1);
            // Full-K scalar FP8 is correct but slower than Q8 MMVQ; keep it out of the default path.
            const bool use_scalar_nvfp4_fp8_acts = false;
            const bool use_direct_nvfp4_fp8_fast_quant = (use_direct_nvfp4_fp8_acts || use_scalar_nvfp4_fp8_acts) &&
                (use_scalar_nvfp4_fp8_acts || ggml_cuda_nvfp4_mmvq_direct_fp8_fast_quant());
            const int use_direct_nvfp4_fp8_split_k = use_direct_nvfp4_fp8_acts ? ggml_cuda_nvfp4_mmvq_direct_fp8_split_k() : 0;
            const ggml_tensor * scale_x_t = use_native_nvfp4_acts ? ggml_cuda_mul_mat_input_scale(dst) : nullptr;
            const float * scale_x_d = scale_x_t ? (const float *) scale_x_t->data : nullptr;
            const int64_t scale_x_ne = scale_x_t ? ggml_nelements(scale_x_t) : 0;
            const ggml_tensor * scale_x_src = use_native_nvfp4_acts ? src0->src[1] : nullptr;
            const bool scale_x_in_header = use_native_nvfp4_acts &&
                scale_x_src != nullptr && ggml_is_scalar(scale_x_src);
            const float * scale_x_q_d = scale_x_d != nullptr ? scale_x_d :
                scale_x_in_header ? &((const block_nvfp4_blackwell_tensor *) src0_data)->input_scale : nullptr;
            const int64_t scale_x_q_ne = scale_x_d != nullptr ? scale_x_ne :
                scale_x_in_header ? 1 : 0;
            const bool use_fp8_acts = use_direct_nvfp4_fp8_acts || use_scalar_nvfp4_fp8_acts;
            const bool use_native_mxfp6_q8 = use_mxfp6;
            const bool use_native_fp8_q8 = false;
#else
        const bool use_mxfp8_mma_1col = false;
        const bool use_native_nvfp4_acts = false;
        const bool use_direct_nvfp4_fp8_fast_quant = false;
        const int use_direct_nvfp4_fp8_split_k = 0;
        const float * scale_x_q_d = nullptr;
        const int64_t scale_x_q_ne = 0;
	    const bool use_fp8_acts = false;
	    const bool use_native_mxfp6_q8 = false;
            const bool use_native_fp8_q8 = false;
#endif // defined(BLACKWELL_MMA_AVAILABLE)

#if defined(BLACKWELL_MMA_AVAILABLE)
        if (!ids && use_native_nvfp4_acts && ggml_cuda_nvfp4_mmvq_native_fused() && ncols_dst == 1 &&
                ne01 % 16 == 0 &&
                fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
            const int nvfp4_blocks_per_row = int(ggml_cuda_nvfp4_blocks_per_row(ne00));
            const int tile_rows_x = int(ggml_cuda_bw_div_up(ne01, 16));
            const int channel_ratio_simple = int(nchannels_dst / ne02);
            const int sample_ratio_simple = int(ne3 / ne03);
            const int warps_per_tile = ggml_cuda_nvfp4_mmvq_warps();
            const int split_k = ggml_cuda_nvfp4_mmvq_native_split_k();
            const bool exact_scale = !ggml_cuda_nvfp4_mmvq_approx_scale();
            GGML_ASSERT(split_k == 1 || split_k == 2 || split_k == 4 || split_k == 8 || split_k == 16 || split_k == 32);
            GGML_ASSERT(warps_per_tile == 4 || warps_per_tile == 8 || warps_per_tile == 16);
            if (split_k == 1) {
#define GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_NOSPLIT(W, RB, EXACT) \
                mul_mat_vec_nvfp4_native_fused_1col_rows<W, RB, EXACT><<< \
                    dim3((tile_rows_x + RB - 1) / RB, nchannels_dst, ne3), dim3(32, W, RB), 0, stream>>>( \
                    src0_data, src1_d, dst_d, nvfp4_blocks_per_row, \
                    int(stride_channel_x), int(stride_sample_x), int(s12_src1), int(s13_src1), \
                    int(stride_channel_dst), int(s3), channel_ratio_simple, sample_ratio_simple, \
                    int(ne10), int(ne01))
#define GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_NOSPLIT_EXACT(W, RB) \
                if (exact_scale) { \
                    GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_NOSPLIT(W, RB, true); \
                } else { \
                    GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_NOSPLIT(W, RB, false); \
                }
                if (warps_per_tile == 4) {
                    GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_NOSPLIT_EXACT(4, 8);
                } else if (warps_per_tile == 8) {
                    GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_NOSPLIT_EXACT(8, 4);
                } else {
                    GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_NOSPLIT_EXACT(16, 2);
                }
#undef GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_NOSPLIT_EXACT
#undef GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_NOSPLIT
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("NVFP4 fused native-activation MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }
            ggml_cuda_pool_alloc<float> split_tmp(ctx.pool(), size_t(ne3*nchannels_dst*split_k*ne01));
            const dim3 block_nums_reduce((ne01 + 255) / 256, nchannels_dst, ne3);
            const dim3 block_dims_reduce(256, 1, 1);
#define GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK(W, SPLIT, RB, EXACT) \
            mul_mat_vec_nvfp4_native_fused_1col_splitk_rows<W, SPLIT, RB, EXACT><<< \
                dim3((tile_rows_x + RB - 1) / RB, nchannels_dst, ne3*split_k), dim3(32, W, RB), 0, stream>>>( \
                src0_data, src1_d, split_tmp.ptr, nvfp4_blocks_per_row, \
                int(stride_channel_x), int(stride_sample_x), int(s12_src1), int(s13_src1), \
                channel_ratio_simple, sample_ratio_simple, int(ne10), int(ne01), int(nchannels_dst)); \
            reduce_nvfp4_fp8_direct_1col_splitk<SPLIT><<<block_nums_reduce, block_dims_reduce, 0, stream>>>( \
                split_tmp.ptr, dst_d, int(ne01), int(nchannels_dst), int(stride_channel_dst), int(s3))
#define GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_EXACT(W, SPLIT, RB) \
            if (exact_scale) { \
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK(W, SPLIT, RB, true); \
            } else { \
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK(W, SPLIT, RB, false); \
            }
#define GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_WARPS(SPLIT) \
            if (warps_per_tile == 4) { \
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_EXACT(4, SPLIT, 8); \
            } else if (warps_per_tile == 8) { \
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_EXACT(8, SPLIT, 4); \
            } else { \
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_EXACT(16, SPLIT, 2); \
            }
            if (split_k == 2) {
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_WARPS(2);
            } else if (split_k == 4) {
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_WARPS(4);
            } else if (split_k == 8) {
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_WARPS(8);
            } else if (split_k == 16) {
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_WARPS(16);
            } else {
                GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_WARPS(32);
            }
#undef GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_WARPS
#undef GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK_EXACT
#undef GGML_CUDA_LAUNCH_NVFP4_NATIVE_FUSED_SPLITK
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                GGML_ABORT("NVFP4 fused native-activation StreamK MMVQ launch failed: %s", cudaGetErrorString(err));
            }
            return;
        }
#endif // defined(BLACKWELL_MMA_AVAILABLE)

	    const int64_t blocks_per_row_y = use_mxfp8_mma_1col ?
            ne10_padded / (4 * QK8_1) : use_native_nvfp4_acts ?
            ggml_cuda_nvfp4_blocks_per_row(ne10_padded) : use_fp8_acts ?
	        ggml_cuda_fp8_blocks_per_row(ne10_padded) : ne10_padded / QK8_1;
	    const size_t nbytes_src1 = size_t(ne13*ne12 * ne11 * blocks_per_row_y *
	        (use_mxfp8_mma_1col ? sizeof(block_mxfp8_mmq) :
             use_native_nvfp4_acts ? sizeof(block_nvfp4_mmq) :
             use_fp8_acts ? sizeof(block_fp8) : sizeof(block_q8_1)));
	    ggml_cuda_pool_alloc<char> src1_t(ctx.pool(), nbytes_src1);

	    int64_t stride_col_y = ids ? ne11 * blocks_per_row_y : blocks_per_row_y;
	    int64_t stride_channel_y = ids ? blocks_per_row_y : ne11 * blocks_per_row_y;
	    int64_t stride_sample_y = ne12 * ne11 * blocks_per_row_y;

        if (use_mxfp8_mma_1col) {
            quantize_mmq_mxfp8_cuda(src1_d, nullptr, src1_t.get(), src0->type, ne10,
                    s11_src1, s12_src1, s13_src1, ne10_padded, ne11, ne12, ne13, stream);
        } else if (use_native_nvfp4_acts) {
            quantize_mmq_nvfp4_cuda(src1_d, nullptr, src1_t.get(), src0->type, ne10, s11_src1, s12_src1, s13_src1,
                    ne10_padded, ne11, ne12, ne13, scale_x_q_d, scale_x_q_ne, stream);
	    } else if (use_direct_nvfp4_fp8_fast_quant) {
	        quantize_row_fp8_fast_mmvq_cuda(src1_d, src1_t.get(), ne10, s11_src1, s12_src1, s13_src1,
	                ne10_padded, ne11, ne12, ne13, stream);
	    } else if (use_fp8_acts) {
	        quantize_row_fp8_cuda(src1_d, nullptr, src1_t.get(), src0->type, ne10, s11_src1, s12_src1, s13_src1,
	                ne10_padded, ne11, ne12, ne13,
                    src0->type == GGML_TYPE_MXFP6_E2M3 ? scale_x_q_d : nullptr,
                    src0->type == GGML_TYPE_MXFP6_E2M3 ? scale_x_q_ne : 0, stream);
	    } else {
	        quantize_row_q8_1_cuda(src1_d, nullptr, src1_t.get(), src0->type, ne10, s11_src1, s12_src1, s13_src1,
	                ne10_padded, ne11, ne12, ne13, stream);
	    }
		    {
		        const cudaError_t err = cudaGetLastError();
		        if (err != cudaSuccess) {
		            GGML_ABORT("MMVQ activation quantization failed: %s", cudaGetErrorString(err));
		        }
		    }

#if defined(BLACKWELL_MMA_AVAILABLE)
            if (use_mxfp8_mma_1col) {
                GGML_ASSERT(ncols_dst == 1 && ne00 % QK_MXFP8 == 0);
                GGML_ASSERT(nchannels_dst % ne02 == 0 && ne3 % ne03 == 0);
                constexpr int warps_per_tile = 16;
                const int     tiles_per_row = int(ggml_cuda_mxfp8_blocks_per_row(ne00));
                const int     tile_stride_channel = int(ggml_cuda_mxfp8_tile_rows(ne01)) * tiles_per_row;
                const int     tile_stride_sample  = int(ne02) * tile_stride_channel;
                launch_mul_mat_vec_mxfp8_mma_1col<warps_per_tile>(
                    (const mxfp8_tile *) src0_data, fusion_local,
                    (const block_mxfp8_mmq *) src1_t.get(), dst_d,
                    int(ne00 / QK_MXFP8_SUB), int(ne01), int(nchannels_dst), int(ne3),
                    tiles_per_row, tile_stride_channel, tile_stride_sample,
                    int(stride_channel_y), int(stride_sample_y),
                    int(stride_channel_dst), int(s3), int(nchannels_dst / ne02), int(ne3 / ne03), stream);
                return;
            }

            const bool use_nvfp4_q8_to_fp8_mmvq = use_nvfp4 && ggml_cuda_nvfp4_enable_q8_to_fp8_mmvq();
            const bool use_mxfp6_q8_to_fp8_mmvq = use_mxfp6 && !ggml_cuda_mxfp6_disable_q8_to_fp8_mmvq();

            if (!ids && !use_native_nvfp4_acts && !use_fp8_acts && use_nvfp4 && ggml_cuda_nvfp4_mmvq_q8_vdr() != 0 &&
                    ncols_dst == 1 &&
                    fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
                const dim3 block_nums(ne01, nchannels_dst, ne3);
                const int channel_ratio_simple = int(nchannels_dst / ne02);
                const int sample_ratio_simple = int(ne3 / ne03);
                const int warps_per_row = ggml_cuda_nvfp4_mmvq_warps();
#define GGML_CUDA_LAUNCH_NVFP4_Q8_VDR(W, VDR) \
                mul_mat_vec_nvfp4_q8_vdr_1col<W, VDR><<<block_nums, dim3(32, W, 1), 0, stream>>>( \
                    src0_data, (const block_q8_1 *) src1_t.get(), dst_d, int(ne00), \
                    int(stride_row_x), int(stride_channel_x), int(stride_sample_x), \
                    int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3), \
                    channel_ratio_simple, sample_ratio_simple)
#define GGML_CUDA_LAUNCH_NVFP4_Q8_VDR_WARPS(VDR) \
                if (warps_per_row == 1) { \
                    GGML_CUDA_LAUNCH_NVFP4_Q8_VDR(1, VDR); \
                } else if (warps_per_row == 2) { \
                    GGML_CUDA_LAUNCH_NVFP4_Q8_VDR(2, VDR); \
                } else if (warps_per_row == 4) { \
                    GGML_CUDA_LAUNCH_NVFP4_Q8_VDR(4, VDR); \
                } else if (warps_per_row == 8) { \
                    GGML_CUDA_LAUNCH_NVFP4_Q8_VDR(8, VDR); \
                } else { \
                    GGML_CUDA_LAUNCH_NVFP4_Q8_VDR(16, VDR); \
                }
                const int q8_vdr = ggml_cuda_nvfp4_mmvq_q8_vdr();
                if (q8_vdr == 2) {
                    GGML_CUDA_LAUNCH_NVFP4_Q8_VDR_WARPS(2);
                } else if (q8_vdr == 4) {
                    GGML_CUDA_LAUNCH_NVFP4_Q8_VDR_WARPS(4);
                } else if (q8_vdr == 8) {
                    GGML_CUDA_LAUNCH_NVFP4_Q8_VDR_WARPS(8);
                } else {
                    GGML_CUDA_LAUNCH_NVFP4_Q8_VDR_WARPS(1);
                }
#undef GGML_CUDA_LAUNCH_NVFP4_Q8_VDR_WARPS
#undef GGML_CUDA_LAUNCH_NVFP4_Q8_VDR
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("NVFP4 Q8 VDR MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }

            if (!ids && use_native_nvfp4_acts && ncols_dst == 1 &&
                    ne01 % 16 == 0 &&
                    fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
                const int nvfp4_blocks_per_row = int(ggml_cuda_nvfp4_blocks_per_row(ne00));
                const int tile_rows_x = int(ggml_cuda_bw_div_up(ne01, 16));
                const int channel_ratio_simple = int(nchannels_dst / ne02);
                const int sample_ratio_simple = int(ne3 / ne03);
                const int warps_per_tile = ggml_cuda_nvfp4_mmvq_warps();
#define GGML_CUDA_LAUNCH_NVFP4_NATIVE_1COL(W, RB) \
                mul_mat_vec_nvfp4_native_1col_rows<W, RB><<< \
                    dim3((tile_rows_x + RB - 1) / RB, nchannels_dst, ne3), dim3(32, W, RB), 0, stream>>>( \
                    src0_data, (const block_nvfp4_mmq *) src1_t.get(), dst_d, \
                    nvfp4_blocks_per_row, int(stride_channel_x), int(stride_sample_x), \
                    int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3), \
                    channel_ratio_simple, sample_ratio_simple, int(ne01))
                if (warps_per_tile == 4) {
                    GGML_CUDA_LAUNCH_NVFP4_NATIVE_1COL(4, 8);
                } else if (warps_per_tile == 8) {
                    GGML_CUDA_LAUNCH_NVFP4_NATIVE_1COL(8, 4);
                } else {
                    GGML_CUDA_LAUNCH_NVFP4_NATIVE_1COL(16, 2);
                }
#undef GGML_CUDA_LAUNCH_NVFP4_NATIVE_1COL
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("NVFP4 native-activation 1col MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }

            if (!ids && use_direct_nvfp4_fp8_acts && ncols_dst == 1 &&
                    ne01 % 16 == 0 &&
                    fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
                const int nvfp4_blocks_per_row = int(ggml_cuda_nvfp4_blocks_per_row(ne00));
                const int k32_per_row = int(ne10_padded / QK8_1);
                const int tile_rows_x = int(ggml_cuda_bw_div_up(ne01, 16));
                const int channel_ratio_simple = int(nchannels_dst / ne02);
                const int sample_ratio_simple = int(ne3 / ne03);
                const bool approx_scale = ggml_cuda_nvfp4_mmvq_approx_scale();
                const int warps_per_tile = ggml_cuda_nvfp4_mmvq_warps();
                const int split_k = use_direct_nvfp4_fp8_split_k;
                if (split_k != 0) {
                    GGML_ASSERT(split_k == 2 || split_k == 4 || split_k == 8);
                    GGML_ASSERT(warps_per_tile == 4 || warps_per_tile == 8 || warps_per_tile == 16);
                    ggml_cuda_pool_alloc<float> split_tmp(ctx.pool(), size_t(ne3*nchannels_dst*split_k*ne01));
                    const dim3 block_nums_split(tile_rows_x, nchannels_dst, ne3*split_k);
                    const dim3 block_nums_reduce((ne01 + 255) / 256, nchannels_dst, ne3);
                    const dim3 block_dims_reduce(256, 1, 1);
#define GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK(W, SPLIT, EXACT) \
                    mul_mat_vec_nvfp4_fp8_direct_1col_splitk<W, SPLIT, EXACT><<<block_nums_split, dim3(32, W, 1), 0, stream>>>( \
                        src0_data, (const block_fp8 *) src1_t.get(), split_tmp.ptr, \
                        nvfp4_blocks_per_row, k32_per_row, int(stride_channel_x), int(stride_sample_x), \
                        int(stride_channel_y), int(stride_sample_y), channel_ratio_simple, sample_ratio_simple, \
                        int(ne01), int(nchannels_dst)); \
                    reduce_nvfp4_fp8_direct_1col_splitk<SPLIT><<<block_nums_reduce, block_dims_reduce, 0, stream>>>( \
                        split_tmp.ptr, dst_d, int(ne01), int(nchannels_dst), int(stride_channel_dst), int(s3))
#define GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_EXACT(W, SPLIT) \
                    if (approx_scale) { \
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK(W, SPLIT, false); \
                    } else { \
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK(W, SPLIT, true); \
                    }
#define GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_WARPS(SPLIT) \
                    if (warps_per_tile == 4) { \
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_EXACT(4, SPLIT); \
                    } else if (warps_per_tile == 8) { \
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_EXACT(8, SPLIT); \
                    } else { \
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_EXACT(16, SPLIT); \
                    }
                    if (split_k == 2) {
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_WARPS(2);
                    } else if (split_k == 4) {
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_WARPS(4);
                    } else {
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_WARPS(8);
                    }
#undef GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_WARPS
#undef GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK_EXACT
#undef GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_SPLITK
                    const cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        GGML_ABORT("NVFP4 direct-FP8 split-K 1col MMVQ launch failed: %s", cudaGetErrorString(err));
                    }
                    return;
                }
                const dim3 block_nums(tile_rows_x, nchannels_dst, ne3);
#define GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_1COL(W, EXACT) \
                mul_mat_vec_nvfp4_fp8_direct_1col<W, EXACT><<<block_nums, dim3(32, W, 1), 0, stream>>>( \
                    src0_data, (const block_fp8 *) src1_t.get(), dst_d, \
                    nvfp4_blocks_per_row, k32_per_row, int(stride_channel_x), int(stride_sample_x), \
                    int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3), \
                    channel_ratio_simple, sample_ratio_simple)
                if (approx_scale) {
                    if (warps_per_tile == 4) {
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_1COL(4, false);
                    } else if (warps_per_tile == 8) {
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_1COL(8, false);
                    } else {
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_1COL(16, false);
                    }
                } else {
                    if (warps_per_tile == 4) {
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_1COL(4, true);
                    } else if (warps_per_tile == 8) {
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_1COL(8, true);
                    } else {
                        GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_1COL(16, true);
                    }
                }
#undef GGML_CUDA_LAUNCH_NVFP4_DIRECT_FP8_1COL
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("NVFP4 direct-FP8 1col MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }

            if (!ids && use_direct_nvfp4_fp8_acts && ncols_dst > 1 && ncols_dst <= 8 &&
                    ne01 % 16 == 0 &&
                    nchannels_dst == ne02 && ne3 == ne03 &&
                    fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
                const int nvfp4_blocks_per_row = int(ggml_cuda_nvfp4_blocks_per_row(ne00));
                const int k32_per_row = int(ne10_padded / QK8_1);
                const int tile_rows_x = int(ggml_cuda_bw_div_up(ne01, 16));
                const int stride_channel_x_tiles = tile_rows_x * nvfp4_blocks_per_row;
                const dim3 block_nums(tile_rows_x, nchannels_dst, ne3);
                constexpr int warps_per_tile = 16;
                const dim3 block_dims(32, warps_per_tile, 1);
                if (ggml_cuda_nvfp4_mmvq_approx_scale()) {
                    mul_mat_vec_nvfp4_fp8_direct_cols<warps_per_tile, false><<<block_nums, block_dims, 0, stream>>>(
                        src0_data, (const block_fp8 *) src1_t.get(), dst_d,
                        nvfp4_blocks_per_row, k32_per_row, stride_channel_x_tiles,
                        int(stride_col_y), int(stride_channel_y), int(stride_sample_y),
                        int(stride_col_dst), int(stride_channel_dst), int(s3), int(ncols_dst));
                } else {
                    mul_mat_vec_nvfp4_fp8_direct_cols<warps_per_tile, true><<<block_nums, block_dims, 0, stream>>>(
                        src0_data, (const block_fp8 *) src1_t.get(), dst_d,
                        nvfp4_blocks_per_row, k32_per_row, stride_channel_x_tiles,
                        int(stride_col_y), int(stride_channel_y), int(stride_sample_y),
                        int(stride_col_dst), int(stride_channel_dst), int(s3), int(ncols_dst));
                }
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("NVFP4 direct-FP8 cols MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }

            if (!ids && !use_fp8_acts && use_nvfp4_q8_to_fp8_mmvq && ncols_dst == 1 &&
                    ne01 % 16 == 0 &&
                    fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
                const int nvfp4_blocks_per_row = int(ggml_cuda_nvfp4_blocks_per_row(ne00));
                const int blocks_per_row_q8 = int(ne10_padded / QK8_1);
                const int tile_rows_x = int(ggml_cuda_bw_div_up(ne01, 16));
                const dim3 block_nums(tile_rows_x, nchannels_dst, ne3);
                const int channel_ratio_simple = int(nchannels_dst / ne02);
                const int sample_ratio_simple = int(ne3 / ne03);
                const bool approx_scale = ggml_cuda_nvfp4_mmvq_approx_scale();
                const int warps_per_tile = ggml_cuda_nvfp4_mmvq_warps();
#define GGML_CUDA_LAUNCH_NVFP4_FP8_1COL(W, EXACT) \
                mul_mat_vec_nvfp4_q8_to_fp8_1col<W, EXACT><<<block_nums, dim3(32, W, 1), 0, stream>>>( \
                    src0_data, (const block_q8_1 *) src1_t.get(), dst_d, \
                    nvfp4_blocks_per_row, blocks_per_row_q8, int(stride_channel_x), int(stride_sample_x), \
                    int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3), \
                    channel_ratio_simple, sample_ratio_simple)
                if (approx_scale) {
                    if (warps_per_tile == 4) {
                        GGML_CUDA_LAUNCH_NVFP4_FP8_1COL(4, false);
                    } else if (warps_per_tile == 8) {
                        GGML_CUDA_LAUNCH_NVFP4_FP8_1COL(8, false);
                    } else {
                        GGML_CUDA_LAUNCH_NVFP4_FP8_1COL(16, false);
                    }
                } else {
                    if (warps_per_tile == 4) {
                        GGML_CUDA_LAUNCH_NVFP4_FP8_1COL(4, true);
                    } else if (warps_per_tile == 8) {
                        GGML_CUDA_LAUNCH_NVFP4_FP8_1COL(8, true);
                    } else {
                        GGML_CUDA_LAUNCH_NVFP4_FP8_1COL(16, true);
                    }
                }
#undef GGML_CUDA_LAUNCH_NVFP4_FP8_1COL
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("NVFP4 Q8-to-FP8 1col MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }

            if (!ids && !use_fp8_acts && use_nvfp4_q8_to_fp8_mmvq && ncols_dst > 1 && ncols_dst <= 8 &&
                    ne01 % 16 == 0 &&
                    nchannels_dst == ne02 && ne3 == ne03 &&
                    fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
                const int nvfp4_blocks_per_row = int(ggml_cuda_nvfp4_blocks_per_row(ne00));
                const int blocks_per_row_q8 = int(ne10_padded / QK8_1);
                const int tile_rows_x = int(ggml_cuda_bw_div_up(ne01, 16));
                const int stride_channel_x_tiles = tile_rows_x * nvfp4_blocks_per_row;
                const dim3 block_nums(tile_rows_x, nchannels_dst, ne3);
                constexpr int warps_per_tile = 16;
                const dim3 block_dims(32, warps_per_tile, 1);
                mul_mat_vec_nvfp4_q8_to_fp8_cols<warps_per_tile><<<block_nums, block_dims, 0, stream>>>(
                    src0_data, (const block_q8_1 *) src1_t.get(), dst_d,
                    nvfp4_blocks_per_row, blocks_per_row_q8, stride_channel_x_tiles,
                    int(stride_col_y), int(stride_channel_y), int(stride_sample_y),
                    int(stride_col_dst), int(stride_channel_dst), int(s3), int(ncols_dst));
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("NVFP4 Q8-to-FP8 MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }

            if (ids && !use_fp8_acts && use_mxfp6_q8_to_fp8_mmvq && ncols_dst > 1 && ne01 % 16 == 0 &&
                    fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
                const int blocks_per_row_x = int(ggml_cuda_mxfp6_e2m3_blocks_per_row(ne00));
                const int tile_rows_x = int(ggml_cuda_bw_div_up(ne01, 16));
                const int stride_channel_x_tiles = tile_rows_x * blocks_per_row_x;
                const dim3 block_nums(tile_rows_x, nchannels_dst, ncols_dst);
                constexpr int warps_per_tile = 16;
                const dim3 block_dims(32, warps_per_tile, 1);
                mul_mat_vec_mxfp6_q8_to_fp8_moe<warps_per_tile><<<block_nums, block_dims, 0, stream>>>(
                    src0_data, (const block_q8_1 *) src1_t.get(), ids_d, dst_d,
                    blocks_per_row_x, stride_channel_x_tiles, int(stride_col_y), int(stride_channel_y),
                    int(stride_col_dst), int(stride_channel_dst), int(ids_stride), int(nchannels_y));
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("MXFP6 Q8-to-FP8 MoE MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }

            if (!ids && !use_fp8_acts && use_mxfp6_q8_to_fp8_mmvq && ncols_dst == 1 &&
                    ne01 % 16 == 0 &&
                    fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
                const int blocks_per_row_x = int(ggml_cuda_mxfp6_e2m3_blocks_per_row(ne00));
                const int tile_rows_x = int(ggml_cuda_bw_div_up(ne01, 16));
                const int stride_channel_x_tiles = tile_rows_x * blocks_per_row_x;
                const dim3 block_nums(tile_rows_x, nchannels_dst, ne3);
                constexpr int warps_per_tile = 16;
                const dim3 block_dims(32, warps_per_tile, 1);
                const int channel_ratio_simple = int(nchannels_dst / ne02);
                const int sample_ratio_simple = int(ne3 / ne03);
                mul_mat_vec_mxfp6_q8_to_fp8_1col<warps_per_tile><<<block_nums, block_dims, 0, stream>>>(
                    src0_data, (const block_q8_1 *) src1_t.get(), dst_d,
                    blocks_per_row_x, stride_channel_x_tiles, int(stride_sample_x),
                    int(stride_channel_y), int(stride_sample_y), int(stride_channel_dst), int(s3),
                    channel_ratio_simple, sample_ratio_simple);
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("MXFP6 Q8-to-FP8 MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }

            if (!ids && !use_fp8_acts && use_mxfp6_q8_to_fp8_mmvq && ncols_dst <= 8 &&
                    ne01 % 16 == 0 &&
                    nchannels_dst == ne02 && ne3 == ne03 &&
                    fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr) {
                const int blocks_per_row_x = int(ggml_cuda_mxfp6_e2m3_blocks_per_row(ne00));
                const int tile_rows_x = int(ggml_cuda_bw_div_up(ne01, 16));
                const int stride_channel_x_tiles = tile_rows_x * blocks_per_row_x;
                const dim3 block_nums(tile_rows_x, nchannels_dst, ne3);
                constexpr int warps_per_tile = 16;
                const dim3 block_dims(32, warps_per_tile, 1);
                mul_mat_vec_mxfp6_q8_to_fp8_cols<warps_per_tile><<<block_nums, block_dims, 0, stream>>>(
                    src0_data, (const block_q8_1 *) src1_t.get(), dst_d,
                    blocks_per_row_x, stride_channel_x_tiles, int(stride_col_y), int(stride_channel_y), int(stride_sample_y),
                    int(stride_col_dst), int(stride_channel_dst), int(s3), int(ncols_dst));
                const cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    GGML_ABORT("MXFP6 Q8-to-FP8 MMVQ launch failed: %s", cudaGetErrorString(err));
                }
                return;
            }
#endif // defined(BLACKWELL_MMA_AVAILABLE)

		    if (use_fp8_acts) {
		        const int blocks_per_row_x = src0->type == GGML_TYPE_NVFP4 ?
		            int(ggml_cuda_nvfp4_blocks_per_row(ne00)) : src0->type == GGML_TYPE_MXFP6_E2M3 ?
	            int(ggml_cuda_mxfp6_e2m3_blocks_per_row(ne00)) : int(ne00 / ggml_blck_size(src0->type));
	        constexpr int warps_per_row = 4;
        constexpr int rows_per_block = 1;
	        const dim3 block_nums((ne01 + rows_per_block - 1) / rows_per_block, nchannels_dst, ne3);
	        const dim3 block_dims(32, warps_per_row, rows_per_block);
	        const int channel_ratio_simple = int(nchannels_dst / ne02);
	        const int sample_ratio_simple = int(ne3 / ne03);
	        if (src0->type == GGML_TYPE_MXFP6_E2M3) {
	            if (rows_per_block == 4) {
	                mul_mat_vec_fp8_acts_1col<GGML_TYPE_MXFP6_E2M3, 4, warps_per_row><<<block_nums, block_dims, 0, stream>>>(
	                    src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
	                    (const float *) fusion_local.x_bias, (const float *) fusion_local.gate_bias, dst_d, fusion_local.glu_op,
	                    ne01, blocks_per_row_x, stride_row_x, stride_channel_x, stride_sample_x,
	                    stride_channel_y, stride_sample_y, stride_channel_dst, s3,
	                    channel_ratio_simple, sample_ratio_simple);
	            } else if (rows_per_block == 2) {
	                mul_mat_vec_fp8_acts_1col<GGML_TYPE_MXFP6_E2M3, 2, warps_per_row><<<block_nums, block_dims, 0, stream>>>(
	                    src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
	                    (const float *) fusion_local.x_bias, (const float *) fusion_local.gate_bias, dst_d, fusion_local.glu_op,
	                    ne01, blocks_per_row_x, stride_row_x, stride_channel_x, stride_sample_x,
	                    stride_channel_y, stride_sample_y, stride_channel_dst, s3,
	                    channel_ratio_simple, sample_ratio_simple);
	            } else {
	                mul_mat_vec_fp8_acts_1col<GGML_TYPE_MXFP6_E2M3, 1, warps_per_row><<<block_nums, block_dims, 0, stream>>>(
	                    src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
	                    (const float *) fusion_local.x_bias, (const float *) fusion_local.gate_bias, dst_d, fusion_local.glu_op,
	                    ne01, blocks_per_row_x, stride_row_x, stride_channel_x, stride_sample_x,
	                    stride_channel_y, stride_sample_y, stride_channel_dst, s3,
	                    channel_ratio_simple, sample_ratio_simple);
	            }
	        } else if (src0->type == GGML_TYPE_NVFP4) {
	            if (rows_per_block == 4) {
	                mul_mat_vec_fp8_acts_1col<GGML_TYPE_NVFP4, 4, warps_per_row><<<block_nums, block_dims, 0, stream>>>(
	                    src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
	                    (const float *) fusion_local.x_bias, (const float *) fusion_local.gate_bias, dst_d, fusion_local.glu_op,
	                    ne01, blocks_per_row_x, stride_row_x, stride_channel_x, stride_sample_x,
	                    stride_channel_y, stride_sample_y, stride_channel_dst, s3,
	                    channel_ratio_simple, sample_ratio_simple);
	            } else if (rows_per_block == 2) {
	                mul_mat_vec_fp8_acts_1col<GGML_TYPE_NVFP4, 2, warps_per_row><<<block_nums, block_dims, 0, stream>>>(
	                    src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
	                    (const float *) fusion_local.x_bias, (const float *) fusion_local.gate_bias, dst_d, fusion_local.glu_op,
	                    ne01, blocks_per_row_x, stride_row_x, stride_channel_x, stride_sample_x,
	                    stride_channel_y, stride_sample_y, stride_channel_dst, s3,
	                    channel_ratio_simple, sample_ratio_simple);
	            } else {
	                mul_mat_vec_fp8_acts_1col<GGML_TYPE_NVFP4, 1, warps_per_row><<<block_nums, block_dims, 0, stream>>>(
	                    src0_data, fusion_local.gate, (const block_fp8 *) src1_t.get(),
	                    (const float *) fusion_local.x_bias, (const float *) fusion_local.gate_bias, dst_d, fusion_local.glu_op,
	                    ne01, blocks_per_row_x, stride_row_x, stride_channel_x, stride_sample_x,
	                    stride_channel_y, stride_sample_y, stride_channel_dst, s3,
	                    channel_ratio_simple, sample_ratio_simple);
	            }
	        } else {
	            GGML_ABORT("unexpected FP8 activation MMVQ type: %s", ggml_type_name(src0->type));
	        }
	        {
	            const cudaError_t err = cudaGetLastError();
	            if (err != cudaSuccess) {
	                GGML_ABORT("simple FP8 activation MMVQ launch failed: %s", cudaGetErrorString(err));
	            }
	        }
	        return;
	    }

	    mul_mat_vec_q_switch_type(
	        src0_data, src0->type, src1_t.get(), ids_d, fusion_local, dst_d, ne00,
	        ne01,              ncols_dst,     stride_row_x, stride_col_y,     stride_col_dst,
	        ne02, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
		        ne03,              ne3,           stride_sample_x, stride_sample_y, s3,                ids_stride, stream,
            use_native_mxfp6_q8, use_fp8_acts, use_native_fp8_q8, physical_rows_x);
	    {
	        const cudaError_t err = cudaGetLastError();
	        if (err != cudaSuccess) {
	            fprintf(stderr,
	                "MMVQ launch debug: tensor=%s type=%s fp8_acts=%d ne00=%ld ne01=%ld ne02=%ld ne03=%ld "
	                "ne10=%ld ne11=%ld ne12=%ld ne13=%ld ncols_dst=%ld strides x={%ld,%ld,%ld} y={%ld,%ld,%ld}\n",
	                src0->name, ggml_type_name(src0->type), (int) use_fp8_acts,
	                (long) ne00, (long) ne01, (long) ne02, (long) ne03,
	                (long) ne10, (long) ne11, (long) ne12, (long) ne13, (long) ncols_dst,
	                (long) stride_row_x, (long) stride_channel_x, (long) stride_sample_x,
	                (long) stride_col_y, (long) stride_channel_y, (long) stride_sample_y);
	            GGML_ABORT("MMVQ launch failed: %s", cudaGetErrorString(err));
	        }
	    }
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    int stride_row_x = ne00 / ggml_blck_size(src0->type);
    int stride_col_y = src1_padded_row_size / QK8_1;

    const int cc = ggml_cuda_info().devices[id].cc;
#if defined(BLACKWELL_MMA_AVAILABLE)
    if (src0->type == GGML_TYPE_NVFP4) {
        stride_row_x = ggml_cuda_nvfp4_blocks_per_row(ne00);
    } else if (src0->type == GGML_TYPE_MXFP6_E2M3 && src0->view_src == nullptr &&
            ne00 % QK_MXFP6_E2M3 == 0 && blackwell_mma_available(cc)) {
        stride_row_x = ggml_cuda_mxfp6_e2m3_blocks_per_row(ne00);
    }
#endif // defined(BLACKWELL_MMA_AVAILABLE)

    ggml_cuda_mm_fusion_args_device fusion_local{};
    const char * src0_q = src0_dd_i;
#if defined(BLACKWELL_MMA_AVAILABLE)
    const bool use_native_mxfp6_q8 = src0->type == GGML_TYPE_MXFP6_E2M3 && src0->view_src == nullptr &&
        ne00 % QK_MXFP6_E2M3 == 0 && blackwell_mma_available(cc);
    const bool use_native_fp8_q8 = false;
#else
    const bool use_native_mxfp6_q8 = false;
    const bool use_native_fp8_q8 = false;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
    mul_mat_vec_q_switch_type(
        src0_q, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream,
        use_native_mxfp6_q8, false, use_native_fp8_q8, row_diff);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
