#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#define NVFP4_A0 0.9918823242f
#define NVFP4_B0 0.9864501953f
#define NVFP4_STEP (1.0f / 32768.0f)

// Four-over-six support is inspired by MIT HAN Lab's Four Over Six work by
// Jack Cook et al. Keep the attribution and MIT license notice in NOTICE.md
// when modifying NVFP4 4/6 controls, presets, or encoder behavior.
enum nvfp4_cuda_choose46_mode {
    NVFP4_CUDA_CHOOSE46_ADAPTIVE = 0,
    NVFP4_CUDA_CHOOSE46_FORCE_M6 = 1,
    NVFP4_CUDA_CHOOSE46_FORCE_M4 = 2,
};

#define NVFP4_CUDA_FLAG_RSF           (1 << 0)
#define NVFP4_CUDA_RSF_MODE_SHIFT     4
#define NVFP4_CUDA_RSF_MODE_MASK      (3 << NVFP4_CUDA_RSF_MODE_SHIFT)
#define NVFP4_CUDA_RSF_DEPTH_SHIFT    8
#define NVFP4_CUDA_RSF_DEPTH_MASK     (3 << NVFP4_CUDA_RSF_DEPTH_SHIFT)

enum nvfp4_cuda_rsf_mode {
    NVFP4_CUDA_RSF_MODE_TENSOR = 0,
    NVFP4_CUDA_RSF_MODE_SLICE  = 1,
    NVFP4_CUDA_RSF_MODE_EXPERT = 2,
    NVFP4_CUDA_RSF_MODE_GROUP  = 3,
};

enum nvfp4_cuda_rsf_depth {
    NVFP4_CUDA_RSF_DEPTH_NORMAL     = 0,
    NVFP4_CUDA_RSF_DEPTH_DEEP       = 1,
    NVFP4_CUDA_RSF_DEPTH_DEEPER     = 2,
    NVFP4_CUDA_RSF_DEPTH_EXHAUSTIVE = 3,
};

typedef struct nvfp4_cuda_runtime_cfg {
    int32_t choose46_mode;
    int32_t refit_iters;
    int32_t use_compand_sat;
    int32_t reserved_i32;
    float cap_m6;
    float cap_m4;
} nvfp4_cuda_runtime_cfg;

typedef struct nvfp4_cuda_tune_result {
    float a;
    float b;
    float scale_mul;
    nvfp4_cuda_runtime_cfg cfg;
    int32_t has_cfg;
} nvfp4_cuda_tune_result;

#define GGML_NVFP4_CUDA_PRESET_LIST(X) \
    X("baseline",                         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  1, 448.0f, 256.0f) \
    X("baseline_gentler_m4",              NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  1, 448.0f, 224.0f) \
    X("baseline_refit4",                  NVFP4_CUDA_CHOOSE46_ADAPTIVE, 4,  1, 448.0f, 256.0f) \
    X("baseline_recover_320_224",         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  1, 320.0f, 224.0f) \
    X("baseline_recover_320_224_refit4",  NVFP4_CUDA_CHOOSE46_ADAPTIVE, 4,  1, 320.0f, 224.0f) \
    X("baseline_recover_352_224",         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  1, 352.0f, 224.0f) \
    X("baseline_recover_384_256",         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  1, 384.0f, 256.0f) \
    X("baseline_recover_416_256",         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  1, 416.0f, 256.0f) \
    X("baseline_recover_448_320",         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  1, 448.0f, 320.0f) \
    X("baseline_recover_448_r4",          NVFP4_CUDA_CHOOSE46_ADAPTIVE, 4,  1, 448.0f, 256.0f) \
    X("baseline_refit24_384_256",         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 24, 1, 384.0f, 256.0f) \
    X("asym_tail",                        NVFP4_CUDA_CHOOSE46_ADAPTIVE, 16, 1, 448.0f, 224.0f) \
    X("asym_tail_strict",                 NVFP4_CUDA_CHOOSE46_ADAPTIVE, 24, 1, 416.0f, 192.0f) \
    X("asym_tail_relaxed",                NVFP4_CUDA_CHOOSE46_ADAPTIVE, 12, 1, 480.0f, 256.0f) \
    X("asym_tail_moe",                    NVFP4_CUDA_CHOOSE46_ADAPTIVE, 16, 1, 512.0f, 288.0f) \
    X("asym_tail_moe_refit4",             NVFP4_CUDA_CHOOSE46_ADAPTIVE, 4,  1, 512.0f, 288.0f) \
    X("adaptive_nocompand_352_224",       NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  0, 352.0f, 224.0f) \
    X("adaptive_nocompand_448_256",       NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  0, 448.0f, 256.0f) \
    X("adaptive_nocompand_384_256",       NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8,  0, 384.0f, 256.0f) \
    X("adaptive_refit12_384_224",         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 12, 1, 384.0f, 224.0f) \
    X("adaptive_refit16_352_192",         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 16, 1, 352.0f, 192.0f) \
    X("adaptive_refit24_448_320",         NVFP4_CUDA_CHOOSE46_ADAPTIVE, 24, 1, 448.0f, 320.0f)

typedef struct nvfp4_cuda_eval_result {
    double sum_sq;
    double sum_abs;
    double max_abs;
    int64_t count;
} nvfp4_cuda_eval_result;

typedef struct nvfp4_cuda_kld_result {
    double sum_nll;
    double sum_nll2;
    double sum_nll_base;
    double sum_nll_base2;
    double sum_nll_nll_base;
    double sum_kld;
    double sum_kld2;
    double max_kld;
    double sum_p_diff2;
    double sum_p_diff4;
    double sum_entropy_diff2;
    double sum_top_prob_diff2;
    double sum_top_flip_weight;
    int64_t same_top;
    int64_t count;
} nvfp4_cuda_kld_result;

GGML_BACKEND_API bool nvfp4_autotune(const float * x, const float * qw, int64_t n, float * best_a, float * best_b);
GGML_BACKEND_API bool nvfp4_autotune_cuda(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, void * stream);
GGML_BACKEND_API bool nvfp4_autotune_cuda_cfg(
        const float * x, const float * qw, int64_t n,
        const nvfp4_cuda_runtime_cfg * cfg_hint,
        nvfp4_cuda_tune_result * result,
        void * stream);
GGML_BACKEND_API bool nvfp4_sample_cache_cuda_create(
        const void * x, int32_t x_type,
        int64_t nrow, int64_t n_per_row,
        const float * qw,
        int64_t sample_nb, int64_t sample_phase,
        float tune_x_mul,
        void ** cache,
        const float ** x_device,
        const float ** tune_x_device,
        const float ** qw_device,
        int64_t * n_device,
        void * stream);
GGML_BACKEND_API void nvfp4_sample_cache_cuda_free(void * cache);
GGML_BACKEND_API void nvfp4_set_ab(float a, float b);
GGML_BACKEND_API void nvfp4_clear_ab(void);
GGML_BACKEND_API void nvfp4_set_runtime_cfg(const nvfp4_cuda_runtime_cfg * cfg);
GGML_BACKEND_API bool nvfp4_get_runtime_cfg(nvfp4_cuda_runtime_cfg * cfg, const nvfp4_cuda_runtime_cfg * cfg_hint);
GGML_BACKEND_API void nvfp4_clear_runtime_cfg(void);
GGML_BACKEND_API void nvfp4_set_autotune_threads(int32_t n_threads);
GGML_BACKEND_API void nvfp4_clear_cuda_stream_cache(void);

GGML_BACKEND_API bool nvfp4_quantize_cuda(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, void * stream);
GGML_BACKEND_API bool nvfp4_quantize_cuda_ab(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, float a, float b, void * stream);
GGML_BACKEND_API bool nvfp4_quantize_cuda_ab_cfg(
        const void * x, bool x_bf16, void * vy,
        int64_t nrow, int64_t n_per_row, const float * qw,
        float x_scale, float a, float b,
        const nvfp4_cuda_runtime_cfg * cfg, void * stream);
GGML_BACKEND_API bool nvfp4_quantize_cuda_ab_eval_cfg(
        const void * x, bool x_bf16, void * vy,
        int64_t nrow, int64_t n_per_row, const float * qw,
        float x_scale, float a, float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        nvfp4_cuda_eval_result * eval,
        void * stream);
GGML_BACKEND_API bool nvfp4_quantize_cuda_ab_eval_to_tensor_cfg(
        const void * x, bool x_bf16, struct ggml_tensor * tensor,
        int64_t plane_index,
        int64_t nrow, int64_t n_per_row, const float * qw,
        float x_scale, float a, float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        nvfp4_cuda_eval_result * eval,
        void * stream);
GGML_BACKEND_API bool nvfp4_quantize_cuda_cfg(
        const void * x, bool x_bf16, void * vy,
        int64_t nrow, int64_t n_per_row, const float * qw,
        float x_scale, const nvfp4_cuda_runtime_cfg * cfg, void * stream);
GGML_BACKEND_API bool mxfp6_e2m3_quantize_cuda(
        const void * x, bool x_bf16, void * vy,
        int64_t nrow, int64_t n_per_row, const float * qw,
        float x_scale, void * stream);
GGML_BACKEND_API bool mxfp6_e2m3_quantize_cuda_eval(
        const void * x, bool x_bf16, void * vy,
        int64_t nrow, int64_t n_per_row, const float * qw,
        float x_scale, nvfp4_cuda_eval_result * eval, void * stream);
GGML_BACKEND_API bool mxfp6_e2m3_quantize_cuda_eval_to_tensor(
        const void * x, bool x_bf16, struct ggml_tensor * tensor,
        int64_t plane_index,
        int64_t nrow, int64_t n_per_row, const float * qw,
        float x_scale, float header_weight_scale, float header_input_scale,
        nvfp4_cuda_eval_result * eval, void * stream);
GGML_BACKEND_API bool ggml_cuda_quantize_classic(
        int32_t type,
        const float * x,
        void * vy,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        int32_t rsf_mode,
        void * stream);
GGML_BACKEND_API bool ggml_cuda_tensor_set_host(
        struct ggml_tensor * tensor, const void * src, size_t nbytes, void * stream);
GGML_BACKEND_API bool ggml_cuda_nvfp4_tensor_set_header_scales(
        struct ggml_tensor * tensor, float weight_scale, float input_scale, void * stream);
GGML_BACKEND_API bool ggml_cuda_tensor_get_host(
        const struct ggml_tensor * tensor, void * dst, size_t nbytes, void * stream);
GGML_BACKEND_API bool ggml_cuda_tensor_snapshot(
        const struct ggml_tensor * tensor, size_t nbytes, void ** snapshot, void * stream);
GGML_BACKEND_API bool ggml_cuda_tensor_restore(
        struct ggml_tensor * tensor, const void * snapshot, size_t nbytes, void * stream);
GGML_BACKEND_API void ggml_cuda_tensor_snapshot_free(void * snapshot);
GGML_BACKEND_API bool nvfp4_kld_reduce_cuda_tensor(
        const struct ggml_tensor * logits,
        const uint16_t * base_logp_u16,
        const int32_t * token_ids,
        int32_t logits_row_offset,
        int32_t n_eval,
        int32_t n_vocab,
        int32_t nv,
        nvfp4_cuda_kld_result * result,
        double * kld_values,
        void * stream);

#ifdef  __cplusplus
}
#endif
