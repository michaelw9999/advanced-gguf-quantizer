#include "common.h"
#include "build-info.h"
#include "ggml-cuda.h"
#include "llama.h"
#include "quantize-imatrix.h"
#include "quantize-kld.h"
#include "quantize-mxfp6.h"
#include "quantize-options.h"
#include "quantize-selector-ledger.h"
#include "quantize-selector-runtime.h"
#include "../../src/llama-context.h"
#include "../../src/llama-model.h"
#include "../../src/llama-model-loader.h"
#include "../../src/llama-quant.h"
#include "../../ggml/src/ggml-impl.h"
#include "../../ggml/src/ggml-quants.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <cinttypes>
#include <cstdint>
#include <atomic>
#include <filesystem>
#include <thread>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#ifndef QK_MXFP6_E2M3
#define QK_MXFP6_E2M3 QK_MXFP6_SUB
#endif

static const char * const LLM_KV_QUANTIZE_IMATRIX_FILE       = "quantize.imatrix.file";
static const char * const LLM_KV_QUANTIZE_IMATRIX_DATASET    = "quantize.imatrix.dataset";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES  = "quantize.imatrix.entries_count";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS   = "quantize.imatrix.chunks_count";

// satisfies -Wmissing-declarations
int llama_quantize(int argc, char ** argv);

static constexpr int64_t SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE = 64;
static constexpr int     SELECTOR_ENCODER_EVIDENCE_VERSION = 4;
static constexpr int     SELECTOR_CLASS_BUCKET_LIMIT = 2;
static constexpr int     SELECTOR_SPECIAL_LIMIT = 4;
static constexpr int     SELECTOR_EXCEPTION_LIMIT = 6;
static constexpr double  SELECTOR_FIRST_TENSOR_SOFT_FACTOR = 1.35;
static constexpr double  SELECTOR_FIRST_TENSOR_HARD_FACTOR = 1.85;
static constexpr double  SELECTOR_FIRST_TENSOR_PENALTY = 8.0;
static constexpr double  SELECTOR_HOLDOUT_WEIGHT = 0.35;
static constexpr double  SELECTOR_PPL_SIGMA = 2.0;
static constexpr double  SELECTOR_KLD_SIGMA = 2.0;
static constexpr double  SELECTOR_RMS_DP_SIGMA = 2.0;
static constexpr double  SELECTOR_SAME_TOP_SIGMA = 1.0;
static constexpr double  SELECTOR_LN_RATIO_ABS_FLOOR = 5e-4;
static constexpr double  SELECTOR_MEAN_KLD_ABS_FLOOR = 5e-5;
static constexpr double  SELECTOR_RMS_DP_ABS_FLOOR = 2e-4;
static constexpr double  SELECTOR_SAME_TOP_ABS_FLOOR = 2.5e-4;
static constexpr double  SELECTOR_ENTROPY_RMSE_ABS_FLOOR = 2e-4;
static constexpr double  SELECTOR_TOP_PROB_RMSE_ABS_FLOOR = 1.5e-4;
static constexpr double  SELECTOR_TOP_FLIP_WEIGHT_ABS_FLOOR = 5e-6;
static constexpr double  SELECTOR_P99_ABS_MARGIN = 7.5e-4;
static constexpr double  SELECTOR_P99_REL_MARGIN = 0.02;
static constexpr double  SELECTOR_P999_ABS_MARGIN = 1.5e-3;
static constexpr double  SELECTOR_P999_REL_MARGIN = 0.02;
static constexpr double  SELECTOR_MAX_KLD_ABS_MARGIN = 0.05;
static constexpr double  SELECTOR_MAX_KLD_REL_MARGIN = 0.05;
static constexpr double  SELECTOR_RSF_SEARCH_SCORE_ABS_MARGIN = 1.20;
static constexpr double  SELECTOR_RSF_SEARCH_SCORE_REL_MARGIN = 0.10;
static constexpr double  SELECTOR_RSF_COMBINED_SCORE_ABS_GAIN = 0.02;
static constexpr double  SELECTOR_RSF_COMBINED_SCORE_REL_GAIN = 0.0015;
static constexpr double  SELECTOR_TENSOR_POLICY_PROXY_FAST_ABS_GAIN = 5e-4;
static constexpr double  SELECTOR_TENSOR_POLICY_PROXY_DEFAULT_ABS_GAIN = 2e-4;
static constexpr double  SELECTOR_TENSOR_POLICY_PROXY_DEEP_ABS_GAIN = 1e-4;
static constexpr double  SELECTOR_TENSOR_POLICY_PROXY_REL_GAIN = 5e-4;
static constexpr int     SELECTOR_N_SEQ_AUTO_MAX = 32;
static constexpr double  SELECTOR_N_SEQ_CUDA_RESERVE_MIN_GIB = 4.0;
static constexpr double  SELECTOR_N_SEQ_CUDA_RESERVE_MAX_GIB = 16.0;
static constexpr double  SELECTOR_N_SEQ_CUDA_RESERVE_FRACTION = 0.20;
static constexpr double  SELECTOR_MXFP6_PPL_ABS_TOL = 2e-4;
static constexpr double  SELECTOR_MXFP6_PPL_REL_TOL = 0.01;
static constexpr double  SELECTOR_MXFP6_MEAN_KLD_ABS_TOL = 7.5e-5;
static constexpr double  SELECTOR_MXFP6_MEAN_KLD_REL_TOL = 0.01;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_SAMPLE_MULT = 12;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_MIN_BLOCKS = 8192;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_REFINE_MULT = 3;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_REFINE_MIN_BLOCKS = 16384;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_GUARD_MULT = 2;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_GUARD_MIN_BLOCKS = 32768;
static constexpr int     SELECTOR_RESCUE_NVFP4_POLICY_BUDGET = 12;
static constexpr double  SELECTOR_RESCUE_NVFP4_PHASE_TAIL_WEIGHT = 0.35;
static constexpr int     SELECTOR_RESCUE_NVFP4_REFINE_TOP = 2;
static constexpr int     SELECTOR_RESCUE_NVFP4_GUARD_TOP = 1;
static constexpr double  SELECTOR_RESCUE_NVFP4_MIN_GAIN = 5e-4;

static bool agq_file_has_gguf_magic(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    char magic[4] = {};
    in.read(magic, sizeof(magic));
    return in.gcount() == (std::streamsize) sizeof(magic) &&
        magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F';
}
static constexpr double  SELECTOR_RESCUE_NVFP4_MIN_REL_GAIN = 0.0025;
static constexpr double  SELECTOR_RESCUE_NVFP4_PREFER_GAIN = 9e-4;
static constexpr double  SELECTOR_RESCUE_NVFP4_PREFER_REL_GAIN = 0.005;
static constexpr double  SELECTOR_RESCUE_SPEED_WEIGHT = 1.0;
static constexpr double  SELECTOR_RESCUE_Q8_SPEED_PENALTY = 0.35;
static constexpr double  SELECTOR_RESCUE_BF16_SPEED_PENALTY = 1.5;
static constexpr double  SELECTOR_RESCUE_Q8_GAIN = 1.0;
static constexpr double  SELECTOR_RESCUE_BF16_GAIN = 1.35;
static constexpr double  SELECTOR_RESCUE_BF16_MARGIN = 1.20;
static constexpr double  SELECTOR_TYPE_CANDIDATE_MIN_COST_MB = 0.25;
static constexpr double  SELECTOR_TYPE_CANDIDATE_NVFP4_BONUS = 0.10;
static constexpr double  SELECTOR_TYPE_CANDIDATE_EXPERT_NVFP4_BONUS = 0.20;
static constexpr double  SELECTOR_TYPE_CANDIDATE_FAST_OUTLIER_RATIO = 1.35;
static constexpr double  SELECTOR_TYPE_CANDIDATE_NORMAL_OUTLIER_RATIO = 1.15;
static constexpr double  SELECTOR_TYPE_CANDIDATE_DEEP_OUTLIER_RATIO = 1.03;
static constexpr double  SELECTOR_TYPE_CANDIDATE_FAST_MIN_ERROR = 0.030;
static constexpr double  SELECTOR_TYPE_CANDIDATE_NORMAL_MIN_ERROR = 0.020;
static constexpr double  SELECTOR_TYPE_CANDIDATE_DEEP_MIN_ERROR = 0.010;
static constexpr int     MXFP6_SELECTOR_SCALE_POOL_PER_TENSOR = 2;
static constexpr int     MXFP6_SELECTOR_SCALE_POOL_TOP = 48;
static constexpr int     MXFP6_SELECTOR_SCALE_POOL_RESCORE_TOP = 24;
static constexpr int     MXFP6_SELECTOR_SCALE_POOL_APPLY_TOP = 0;
static constexpr double  MXFP6_SELECTOR_SCALE_POOL_SCORE_SLACK = 0.05;

static bool nvfp4_cfg_has_rsf(const nvfp4_cuda_runtime_cfg & cfg) {
    return (cfg.reserved_i32 & NVFP4_CUDA_FLAG_RSF) != 0;
}

static int nvfp4_cfg_rsf_mode(const nvfp4_cuda_runtime_cfg & cfg) {
    return (cfg.reserved_i32 & NVFP4_CUDA_RSF_MODE_MASK) >> NVFP4_CUDA_RSF_MODE_SHIFT;
}

static int nvfp4_cfg_rsf_depth(const nvfp4_cuda_runtime_cfg & cfg) {
    return (cfg.reserved_i32 & NVFP4_CUDA_RSF_DEPTH_MASK) >> NVFP4_CUDA_RSF_DEPTH_SHIFT;
}

static void nvfp4_cfg_set_rsf_mode(nvfp4_cuda_runtime_cfg & cfg, int mode) {
    mode = std::max((int) NVFP4_CUDA_RSF_MODE_TENSOR, std::min((int) NVFP4_CUDA_RSF_MODE_GROUP, mode));
    cfg.reserved_i32 &= ~NVFP4_CUDA_RSF_MODE_MASK;
    cfg.reserved_i32 |= mode << NVFP4_CUDA_RSF_MODE_SHIFT;
}

static void nvfp4_cfg_set_rsf_depth(nvfp4_cuda_runtime_cfg & cfg, int depth) {
    depth = std::max((int) NVFP4_CUDA_RSF_DEPTH_NORMAL, std::min((int) NVFP4_CUDA_RSF_DEPTH_EXHAUSTIVE, depth));
    cfg.reserved_i32 &= ~NVFP4_CUDA_RSF_DEPTH_MASK;
    cfg.reserved_i32 |= depth << NVFP4_CUDA_RSF_DEPTH_SHIFT;
}

static const char * nvfp4_rsf_mode_name(int mode) {
    switch (mode) {
        case NVFP4_CUDA_RSF_MODE_TENSOR: return "tensor";
        case NVFP4_CUDA_RSF_MODE_SLICE:  return "slice";
        case NVFP4_CUDA_RSF_MODE_EXPERT: return "expert";
        case NVFP4_CUDA_RSF_MODE_GROUP:  return "group";
    }
    return "tensor";
}

static bool nvfp4_parse_rsf_mode(const char * value, int & out) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    if (striequals(value, "tensor")) {
        out = NVFP4_CUDA_RSF_MODE_TENSOR;
        return true;
    }
    if (striequals(value, "slice")) {
        out = NVFP4_CUDA_RSF_MODE_SLICE;
        return true;
    }
    if (striequals(value, "expert")) {
        out = NVFP4_CUDA_RSF_MODE_EXPERT;
        return true;
    }
    if (striequals(value, "group")) {
        out = NVFP4_CUDA_RSF_MODE_GROUP;
        return true;
    }
    return false;
}

static const char * nvfp4_rsf_depth_name(int depth) {
    switch (depth) {
        case NVFP4_CUDA_RSF_DEPTH_NORMAL:     return "normal";
        case NVFP4_CUDA_RSF_DEPTH_DEEP:       return "deep";
        case NVFP4_CUDA_RSF_DEPTH_DEEPER:     return "deeper";
        case NVFP4_CUDA_RSF_DEPTH_EXHAUSTIVE: return "exhaustive";
    }
    return "normal";
}

static bool nvfp4_parse_rsf_depth(const char * value, int & out) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    if (striequals(value, "normal") || striequals(value, "default")) {
        out = NVFP4_CUDA_RSF_DEPTH_NORMAL;
        return true;
    }
    if (striequals(value, "deep")) {
        out = NVFP4_CUDA_RSF_DEPTH_DEEP;
        return true;
    }
    if (striequals(value, "deeper")) {
        out = NVFP4_CUDA_RSF_DEPTH_DEEPER;
        return true;
    }
    if (striequals(value, "exhaustive") || striequals(value, "max")) {
        out = NVFP4_CUDA_RSF_DEPTH_EXHAUSTIVE;
        return true;
    }
    return false;
}

enum class quantize_work_mode {
    UNSET,
    FAST,
    NORMAL,
    DEEP,
};

static const char * quantize_work_mode_name(quantize_work_mode mode) {
    switch (mode) {
        case quantize_work_mode::FAST:   return "fast";
        case quantize_work_mode::NORMAL: return "normal";
        case quantize_work_mode::DEEP:   return "deep";
        case quantize_work_mode::UNSET:  break;
    }
    return "";
}

static bool parse_quantize_work_mode(const char * value, quantize_work_mode & out) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    if (striequals(value, "fast")) {
        out = quantize_work_mode::FAST;
        return true;
    }
    if (striequals(value, "normal")) {
        out = quantize_work_mode::NORMAL;
        return true;
    }
    if (striequals(value, "deep")) {
        out = quantize_work_mode::DEEP;
        return true;
    }
    return false;
}

static std::string selector_generic_control_key(const char * suffix) {
    return std::string("LLAMA_SELECTOR_") + (suffix != nullptr ? suffix : "");
}

static std::string selector_legacy_control_key(const char * suffix) {
    return std::string("LLAMA_NVFP4_SELECTOR_") + (suffix != nullptr ? suffix : "");
}

static bool selector_control_has(const char * suffix) {
    const std::string generic_key = selector_generic_control_key(suffix);
    const std::string legacy_key = selector_legacy_control_key(suffix);
    return quantize_control_has(generic_key.c_str()) || quantize_control_has(legacy_key.c_str());
}

static std::string selector_control_string(const char * suffix, std::string fallback = {}) {
    const std::string generic_key = selector_generic_control_key(suffix);
    const std::string legacy_key = selector_legacy_control_key(suffix);
    const std::string legacy_value = quantize_control_string(legacy_key.c_str(), fallback);
    return quantize_control_string(generic_key.c_str(), legacy_value);
}

static int64_t selector_control_i64(const char * suffix, int64_t fallback) {
    const std::string generic_key = selector_generic_control_key(suffix);
    const std::string legacy_key = selector_legacy_control_key(suffix);
    const int64_t legacy_value = quantize_control_i64(legacy_key.c_str(), fallback);
    return quantize_control_i64(generic_key.c_str(), legacy_value);
}

static double selector_control_f64(const char * suffix, double fallback) {
    const std::string generic_key = selector_generic_control_key(suffix);
    const std::string legacy_key = selector_legacy_control_key(suffix);
    const double legacy_value = quantize_control_f64(legacy_key.c_str(), fallback);
    return quantize_control_f64(generic_key.c_str(), legacy_value);
}

static bool selector_control_bool(const char * suffix, bool fallback) {
    const std::string generic_key = selector_generic_control_key(suffix);
    const std::string legacy_key = selector_legacy_control_key(suffix);
    const bool legacy_value = quantize_control_bool(legacy_key.c_str(), fallback);
    return quantize_control_bool(generic_key.c_str(), legacy_value);
}

static bool selector_reset_cuda_device(int device) {
#if defined(GGML_USE_CUDA) && (defined(__unix__) || defined(__APPLE__))
    nvfp4_clear_cuda_stream_cache();
    void * handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) {
        handle = dlopen("libcudart.so.13", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (handle == nullptr) {
        handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (handle == nullptr) {
        handle = dlopen("libcudart.so", RTLD_LAZY);
    }
    if (handle == nullptr) {
        fprintf(stderr, "%s: selector stage-b CUDA reset skipped: libcudart is not available through dlopen\n", __func__);
        return false;
    }

    using cuda_set_device_fn = int (*)(int);
    using cuda_noarg_fn = int (*)();
    using cuda_get_error_string_fn = const char * (*)(int);
    auto cuda_set_device = reinterpret_cast<cuda_set_device_fn>(dlsym(handle, "cudaSetDevice"));
    auto cuda_device_synchronize = reinterpret_cast<cuda_noarg_fn>(dlsym(handle, "cudaDeviceSynchronize"));
    auto cuda_device_reset = reinterpret_cast<cuda_noarg_fn>(dlsym(handle, "cudaDeviceReset"));
    auto cuda_get_error_string = reinterpret_cast<cuda_get_error_string_fn>(dlsym(handle, "cudaGetErrorString"));
    if (cuda_set_device == nullptr || cuda_device_synchronize == nullptr || cuda_device_reset == nullptr) {
        fprintf(stderr, "%s: selector stage-b CUDA reset skipped: required libcudart symbols are unavailable\n", __func__);
        return false;
    }

    int err = cuda_set_device(device);
    if (err == 0) {
        err = cuda_device_synchronize();
    }
    if (err == 0) {
        err = cuda_device_reset();
    }
    if (err != 0) {
        fprintf(stderr,
            "%s: selector stage-b CUDA reset failed: %s (%d)\n",
            __func__,
            cuda_get_error_string != nullptr ? cuda_get_error_string(err) : "unknown CUDA error",
            err);
        return false;
    }
    nvfp4_clear_cuda_stream_cache();
    return true;
#else
    (void) device;
    return false;
#endif
}

struct selector_rank_config {
    struct mxfp6_config {
        double ppl_abs_tol = SELECTOR_MXFP6_PPL_ABS_TOL;
        double ppl_rel_tol = SELECTOR_MXFP6_PPL_REL_TOL;
        double mean_kld_abs_tol = SELECTOR_MXFP6_MEAN_KLD_ABS_TOL;
        double mean_kld_rel_tol = SELECTOR_MXFP6_MEAN_KLD_REL_TOL;
    };

    double kld_threshold = -1.0;
    double p99_threshold = -1.0;
    double p999_threshold = -1.0;
    double max_kld_threshold = -1.0;
    double kld_penalty = 0.0;
    double p99_penalty = 0.0;
    double p999_penalty = 0.0;
    double max_kld_penalty = 0.0;
    double holdout_weight = SELECTOR_HOLDOUT_WEIGHT;
    bool kld_hard_gate = false;
    bool p99_hard_gate = false;
    bool p999_hard_gate = false;
    bool max_kld_hard_gate = false;
    double ppl_sigma = SELECTOR_PPL_SIGMA;
    double kld_sigma = SELECTOR_KLD_SIGMA;
    double rms_dp_sigma = SELECTOR_RMS_DP_SIGMA;
    double same_top_sigma = SELECTOR_SAME_TOP_SIGMA;
    double ln_ratio_abs_floor = SELECTOR_LN_RATIO_ABS_FLOOR;
    double mean_kld_abs_floor = SELECTOR_MEAN_KLD_ABS_FLOOR;
    double rms_dp_abs_floor = SELECTOR_RMS_DP_ABS_FLOOR;
    double same_top_abs_floor = SELECTOR_SAME_TOP_ABS_FLOOR;
    double entropy_rmse_abs_floor = SELECTOR_ENTROPY_RMSE_ABS_FLOOR;
    double top_prob_rmse_abs_floor = SELECTOR_TOP_PROB_RMSE_ABS_FLOOR;
    double top_flip_weight_abs_floor = SELECTOR_TOP_FLIP_WEIGHT_ABS_FLOOR;
    double p99_abs_margin = SELECTOR_P99_ABS_MARGIN;
    double p99_rel_margin = SELECTOR_P99_REL_MARGIN;
    double p999_abs_margin = SELECTOR_P999_ABS_MARGIN;
    double p999_rel_margin = SELECTOR_P999_REL_MARGIN;
    double max_kld_abs_margin = SELECTOR_MAX_KLD_ABS_MARGIN;
    double max_kld_rel_margin = SELECTOR_MAX_KLD_REL_MARGIN;
    mxfp6_config mxfp6;
};

enum class selector_tensor_class : uint8_t;

struct selector_rsf_tensor_record {
    std::string tensor;
    std::string policy;
    selector_tensor_class cls = (selector_tensor_class) 0;
    int bucket = -1;
    int32_t layer = -1;
    int64_t slices = 0;
    double scale_mul_sum = 0.0;
    double proxy_score = std::numeric_limits<double>::infinity();
    bool rsf_changed = false;
    std::vector<float> scale_muls;
};

struct selector_policy {
    std::string name;
    nvfp4_cuda_runtime_cfg cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    double proxy_score = std::numeric_limits<double>::infinity();
    double proxy_rmse = std::numeric_limits<double>::infinity();
    double proxy_abs_mean = std::numeric_limits<double>::infinity();
    double proxy_max_abs = std::numeric_limits<double>::infinity();
    double survey_proxy_score = std::numeric_limits<double>::infinity();
    double survey_proxy_rmse = std::numeric_limits<double>::infinity();
    double survey_proxy_abs_mean = std::numeric_limits<double>::infinity();
    double survey_proxy_max_abs = std::numeric_limits<double>::infinity();
    double first_tensor_rmse = std::numeric_limits<double>::infinity();
    double first_tensor_abs_mean = std::numeric_limits<double>::infinity();
    double first_tensor_max_abs = std::numeric_limits<double>::infinity();
    bool proxy_rejected = false;
    bool has_survey = false;
    selector_derived_metrics measured;
    selector_derived_metrics measured_holdout;
    bool has_holdout = false;
    bool measured_pass = false;
    double measured_score = std::numeric_limits<double>::infinity();
    std::vector<selector_rsf_tensor_record> tensor_records;
};

enum class selector_tensor_class : uint8_t {
    EMBEDDING_OUTPUT = 0,
    ATTN_QKV,
    ATTN_OUT,
    ROUTER,
    EXPERT_UP_GATE,
    EXPERT_DOWN,
    DENSE_MLP,
    SSM,
    OTHER,
};

struct selector_device_snapshot {
    void * data = nullptr;
    size_t nbytes = 0;

    selector_device_snapshot() = default;
    selector_device_snapshot(const selector_device_snapshot &) = delete;
    selector_device_snapshot & operator=(const selector_device_snapshot &) = delete;

    selector_device_snapshot(selector_device_snapshot && other) noexcept {
        data = other.data;
        nbytes = other.nbytes;
        other.data = nullptr;
        other.nbytes = 0;
    }

    selector_device_snapshot & operator=(selector_device_snapshot && other) noexcept {
        if (this != &other) {
            reset();
            data = other.data;
            nbytes = other.nbytes;
            other.data = nullptr;
            other.nbytes = 0;
        }
        return *this;
    }

    ~selector_device_snapshot() {
        reset();
    }

    void reset() {
        if (data != nullptr) {
            ggml_cuda_tensor_snapshot_free(data);
            data = nullptr;
            nbytes = 0;
        }
    }

    bool valid_for(size_t bytes) const {
        return data != nullptr && nbytes == bytes && bytes > 0;
    }

    bool capture(const ggml_tensor * tensor, size_t bytes) {
        reset();
        void * snapshot = nullptr;
        if (!ggml_cuda_tensor_snapshot(tensor, bytes, &snapshot, nullptr)) {
            return false;
        }
        data = snapshot;
        nbytes = bytes;
        return true;
    }

    bool restore(ggml_tensor * tensor, size_t bytes) const {
        return valid_for(bytes) && ggml_cuda_tensor_restore(tensor, data, bytes, nullptr);
    }
};

struct nvfp4_selector_device_sample_entry {
    int64_t slice_index = -1;
    int64_t sample_nb = 0;
    int64_t sample_phase = 0;
    int32_t source_type = GGML_TYPE_COUNT;
    const void * source_ptr = nullptr;
    const float * imatrix_ptr = nullptr;
    float tune_x_mul = 1.0f;
    void * cache = nullptr;
    const float * x_device = nullptr;
    const float * tune_x_device = nullptr;
    const float * qw_device = nullptr;
    int64_t n_device = 0;

    ~nvfp4_selector_device_sample_entry() {
        if (cache != nullptr) {
            nvfp4_sample_cache_cuda_free(cache);
        }
    }

    bool matches(
            int64_t slice,
            int64_t nb,
            int64_t phase,
            int32_t type,
            const void * src,
            const float * imat,
            float tune_mul) const {
        return
            slice_index == slice &&
            sample_nb == nb &&
            sample_phase == phase &&
            source_type == type &&
            source_ptr == src &&
            imatrix_ptr == imat &&
            std::fabs(tune_x_mul - tune_mul) <= 1e-12f * std::max(1.0f, std::max(std::fabs(tune_x_mul), std::fabs(tune_mul)));
    }
};

struct nvfp4_selector_device_sample_bank {
    std::mutex mutex;
    std::vector<std::unique_ptr<nvfp4_selector_device_sample_entry>> entries;
};

struct nvfp4_selector_device_sample_view {
    const float * x_device = nullptr;
    const float * tune_x_device = nullptr;
    const float * qw_device = nullptr;
    int64_t n_device = 0;
    float x_scale = 1.0f;
};

struct selector_binding {
    std::string name;
    selector_tensor_class cls = selector_tensor_class::OTHER;
    int bucket = -1;
    int32_t layer = -1;
    ggml_tensor * target = nullptr;
    ggml_tensor * target_scale = nullptr;
    ggml_tensor * target_input_scale = nullptr;
    ggml_tensor * source = nullptr;
    const float * imatrix_row = nullptr;
    float source_tensor_scale = 1.0f;
    std::vector<float> quant_tensor_scales;
    mutable std::vector<float> working_quant_tensor_scales;
    size_t source_nbytes = 0;
    size_t target_nbytes = 0;
    size_t target_scale_nbytes = 0;
    size_t target_input_scale_nbytes = 0;
    std::vector<uint8_t> original_target_bytes;
    std::vector<uint8_t> working_target_bytes;
    std::vector<uint8_t> original_scale_bytes;
    std::vector<uint8_t> working_scale_bytes;
    std::vector<uint8_t> original_input_scale_bytes;
    std::vector<uint8_t> working_input_scale_bytes;
    selector_device_snapshot original_target_device;
    selector_device_snapshot original_scale_device;
    selector_device_snapshot original_input_scale_device;
    std::shared_ptr<nvfp4_selector_device_sample_bank> device_samples;
};

struct selector_type_candidate {
    ggml_type type = GGML_TYPE_COUNT;
    size_t nbytes = 0;
    size_t delta_bytes = 0;
    double roi = 0.0;
    double speed_penalty = 0.0;
    double type_gain = 1.0;
};

struct selector_tensor_sensitivity {
    std::string name;
    std::string role;
    selector_tensor_class cls = selector_tensor_class::OTHER;
    int bucket = -1;
    int32_t layer = -1;
    size_t binding_index = 0;
    size_t target_nbytes = 0;
    size_t best_candidate_nbytes = 0;
    size_t bf16_nbytes = 0;
    std::vector<selector_type_candidate> type_candidates;
    double proxy_score = 0.0;
    double proxy_rmse = 0.0;
    double proxy_abs_mean = 0.0;
    double proxy_max_abs = 0.0;
    double best_candidate_roi = 0.0;
    double bf16_roi = 0.0;
    size_t best_candidate_delta_bytes = 0;
    size_t bf16_delta_bytes = 0;
    double best_candidate_speed_penalty = 0.0;
    double bf16_speed_penalty = 0.0;
    ggml_type best_candidate_type = GGML_TYPE_Q8_0;
    bool has_alt_nvfp4_encoder_cfg = false;
    nvfp4_cuda_runtime_cfg alt_nvfp4_encoder_cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    int64_t alt_nvfp4_encoder_sample_blocks = 0;
    std::string alt_nvfp4_encoder_policy_name;
    double alt_nvfp4_encoder_proxy_score = std::numeric_limits<double>::infinity();
    double alt_nvfp4_encoder_gain = 0.0;
    double alt_nvfp4_encoder_gain_rel = 0.0;
    double suggested_priority = 0.0;
    ggml_type suggested_type = GGML_TYPE_COUNT;
    size_t suggested_nbytes = 0;
    size_t suggested_delta_bytes = 0;
    double suggested_speed_penalty = 0.0;
    double suggested_type_gain = 1.0;
    double suggested_gate_floor = 0.0;
    double suggested_error_ratio = 0.0;
    bool has_suggested_k_rsf_mode = false;
    int32_t suggested_k_rsf_mode = 0;
};

static std::string format_selector_suggested_type(const selector_tensor_sensitivity & s) {
    tensor_type_option opt;
    opt.type = s.suggested_type;
    if (s.suggested_type == GGML_TYPE_NVFP4 && s.has_alt_nvfp4_encoder_cfg) {
        opt.has_nvfp4_cfg = true;
        opt.nvfp4_cfg = s.alt_nvfp4_encoder_cfg;
        opt.nvfp4_sample_blocks = s.alt_nvfp4_encoder_sample_blocks;
        opt.nvfp4_policy_name = s.alt_nvfp4_encoder_policy_name;
    } else if (s.has_suggested_k_rsf_mode && s.suggested_k_rsf_mode > 0) {
        opt.has_k_rsf_mode = true;
        opt.k_rsf_mode = s.suggested_k_rsf_mode;
    }
    return format_tensor_type_value(opt);
}

static int64_t selector_sample_blocks(int64_t nb_total, int64_t override_cap);

enum class quantize_stream_key_mode {
    HASHED_SLOT,
    LINEAR_SLOT,
};

struct quantize_cuda_eval_params_t {
    const void * x = nullptr;
    bool x_bf16 = false;
    uint8_t * out = nullptr;
    ggml_tensor * target = nullptr;
    int64_t slice_index = 0;
    int64_t nrow = 0;
    int64_t n_per_row = 0;
    const float * qw = nullptr;
    float x_scale = 1.0f;
    float a = 1.0f;
    float b = 1.0f;
    float weight_scale = 1.0f;
    float input_scale = 1.0f;
    const nvfp4_cuda_runtime_cfg * nvfp4_cfg = nullptr;
    nvfp4_cuda_eval_result * eval = nullptr;
    void * stream_key = nullptr;
};

template <ggml_type type>
struct ggml_quantize_type_traits;

template <>
struct ggml_quantize_type_traits<GGML_TYPE_NVFP4> {
    static constexpr ggml_type quant_type = GGML_TYPE_NVFP4;
    static constexpr int64_t block_size = SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
    static constexpr bool require_scale_bytes = false;
    static constexpr bool supports_sample_only = true;
    static constexpr const char * thread_control_key = nullptr;
    static constexpr quantize_stream_key_mode stream_key_mode = quantize_stream_key_mode::HASHED_SLOT;
    static constexpr uintptr_t single_stream_slot = 2;
    static constexpr uintptr_t worker_stream_slot = 16;
    static constexpr bool single_stream_slot_add_slice = false;

    static bool quantize_eval(const quantize_cuda_eval_params_t & p) {
        if (p.target != nullptr) {
            return nvfp4_quantize_cuda_ab_eval_to_tensor_cfg(
                p.x, p.x_bf16, p.target, p.slice_index, p.nrow, p.n_per_row, p.qw,
                p.x_scale, p.a, p.b, p.nvfp4_cfg, p.eval, p.stream_key);
        }
        return nvfp4_quantize_cuda_ab_eval_cfg(
            p.x, p.x_bf16, p.out, p.nrow, p.n_per_row, p.qw,
            p.x_scale, p.a, p.b, p.nvfp4_cfg, p.eval, p.stream_key);
    }
};

template <>
struct ggml_quantize_type_traits<GGML_TYPE_MXFP6_E2M3> {
    static constexpr ggml_type quant_type = GGML_TYPE_MXFP6_E2M3;
    static constexpr int64_t block_size = QK_MXFP6_E2M3;
    static constexpr bool require_scale_bytes = true;
    static constexpr bool supports_sample_only = false;
    static constexpr const char * thread_control_key = nullptr;
    static constexpr quantize_stream_key_mode stream_key_mode = quantize_stream_key_mode::LINEAR_SLOT;
    static constexpr uintptr_t single_stream_slot = 0x3000;
    static constexpr uintptr_t worker_stream_slot = 0x3000;
    static constexpr bool single_stream_slot_add_slice = true;

    static bool quantize_eval(const quantize_cuda_eval_params_t & p) {
        if (p.target != nullptr) {
            return mxfp6_e2m3_quantize_cuda_eval_to_tensor(
                p.x, p.x_bf16, p.target, p.slice_index,
                p.nrow, p.n_per_row, p.qw, p.weight_scale,
                p.weight_scale, p.input_scale,
                p.eval, p.stream_key);
        }
        return mxfp6_e2m3_quantize_cuda_eval(
            p.x, p.x_bf16, p.out,
            p.nrow, p.n_per_row, p.qw, p.weight_scale,
            p.eval, p.stream_key);
    }
};

struct quantize_binding_slice_accum_t {
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double max_abs = 0.0;
    int64_t count = 0;
    bool ok = true;
};

struct quantize_binding_shape_t {
    int64_t n_per_row = 0;
    int64_t nrows = 0;
    int64_t n_slices = 0;
    int64_t nb_total = 0;
    bool sample_only = false;
    int64_t sample_nb_for_bytes = 0;
    int host_threads = 1;
    int64_t out_nrow = 0;
    int64_t out_n_per_row = 0;
    size_t out_slice_bytes = 0;
};

template <ggml_type type>
static bool quantize_binding_prepare_shape(
    const selector_binding & binding,
    int64_t sample_blocks_override,
    int64_t sample_phase,
    const char * caller,
    quantize_binding_shape_t & shape) {
    using traits = ggml_quantize_type_traits<type>;
    shape.n_per_row = binding.source->ne[0];
    shape.nrows = binding.source->ne[1];
    shape.n_slices = std::max<int64_t>(1, binding.source->ne[2]);

    if (shape.n_per_row <= 0 || shape.nrows <= 0 || shape.n_per_row % traits::block_size != 0) {
        fprintf(stderr,
            "%s: invalid %s quantize shape for %s (rows=%" PRId64 " cols=%" PRId64 " block=%" PRId64 ")\n",
            caller,
            ggml_type_name(traits::quant_type),
            binding.name.c_str(),
            shape.nrows,
            shape.n_per_row,
            traits::block_size);
        return false;
    }

    shape.sample_only = sample_blocks_override > 0;
    if constexpr (!traits::supports_sample_only) {
        if (shape.sample_only) {
            fprintf(stderr,
                "%s: %s quantize binding does not support sampled output for %s\n",
                caller,
                ggml_type_name(traits::quant_type),
                binding.name.c_str());
            return false;
        }
    }

    shape.nb_total = (shape.nrows * shape.n_per_row) / traits::block_size;
    if constexpr (traits::supports_sample_only) {
        shape.sample_nb_for_bytes = shape.sample_only
            ? selector_sample_blocks(shape.nb_total, sample_blocks_override)
            : 0;
    } else {
        shape.sample_nb_for_bytes = 0;
    }
    if (shape.sample_only && shape.sample_nb_for_bytes <= 0) {
        fprintf(stderr,
            "%s: empty selector sample for %s (rows=%" PRId64 " cols=%" PRId64 " slices=%" PRId64 " blocks=%" PRId64 " requested=%" PRId64 " phase=%" PRId64 ")\n",
            caller,
            binding.name.c_str(),
            shape.nrows,
            shape.n_per_row,
            shape.n_slices,
            shape.nb_total,
            sample_blocks_override,
            sample_phase);
        return false;
    }

    shape.out_nrow = shape.sample_only ? 1 : shape.nrows;
    shape.out_n_per_row = shape.sample_only ? shape.sample_nb_for_bytes * traits::block_size : shape.n_per_row;
    shape.out_slice_bytes = (size_t) shape.out_nrow * ggml_row_size(traits::quant_type, shape.out_n_per_row);
    return true;
}

template <ggml_type type>
static int quantize_binding_thread_count(int64_t n_slices, int nthread) {
    using traits = ggml_quantize_type_traits<type>;
    int64_t requested = std::max(1, nthread);
    if constexpr (traits::quant_type == GGML_TYPE_NVFP4) {
        requested = selector_control_i64("THREADS", requested);
    } else if constexpr (traits::thread_control_key != nullptr) {
        requested = quantize_control_i64(traits::thread_control_key, requested);
    }
    return (int) std::max<int64_t>(1, std::min<int64_t>(std::max<int64_t>(1, n_slices), requested));
}

static int quantize_host_range_worker_count(int64_t n_items, int nthread, int64_t min_items_per_worker) {
    if (n_items <= 1 || nthread <= 1) {
        return 1;
    }

    const int64_t min_items = std::max<int64_t>(1, min_items_per_worker);
    const int64_t work_limited = (n_items + min_items - 1) / min_items;
    return (int) std::max<int64_t>(1, std::min<int64_t>({
        n_items,
        (int64_t) nthread,
        work_limited,
    }));
}

template <typename RangeFn>
static bool quantize_parallel_for_ranges(int64_t n_items, int nthread, int64_t min_items_per_worker, RangeFn && fn) {
    const int worker_count = quantize_host_range_worker_count(n_items, nthread, min_items_per_worker);
    if (worker_count <= 1) {
        return fn(0, 0, n_items);
    }

    std::atomic_bool ok { true };
    std::vector<std::thread> workers;
    workers.reserve((size_t) worker_count);
    for (int wi = 0; wi < worker_count; ++wi) {
        workers.emplace_back([&, wi]() {
            const int64_t begin = (n_items * wi) / worker_count;
            const int64_t end = (n_items * (wi + 1)) / worker_count;
            if (!fn(wi, begin, end)) {
                ok.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    return ok.load(std::memory_order_relaxed);
}

template <ggml_type type>
static void * quantize_binding_stream_key(
    uintptr_t stream_seed,
    int worker_index,
    int64_t slice_index,
    bool single_worker) {
    using traits = ggml_quantize_type_traits<type>;
    const uintptr_t slot = single_worker
        ? traits::single_stream_slot + (traits::single_stream_slot_add_slice ? (uintptr_t) slice_index : 0)
        : traits::worker_stream_slot + (uintptr_t) worker_index;

    switch (traits::stream_key_mode) {
        case quantize_stream_key_mode::HASHED_SLOT:
            return reinterpret_cast<void *>(stream_seed ^ (slot * UINT64_C(0x9E3779B97F4A7C15)));
        case quantize_stream_key_mode::LINEAR_SLOT:
            return reinterpret_cast<void *>(slot);
    }

    GGML_ABORT("unreachable");
}

template <ggml_type type, typename SliceFn>
static bool quantize_run_binding_slices(
    int64_t n_slices,
    int nthread,
    uintptr_t stream_seed,
    SliceFn && run_slice,
    double & sum_sq,
    double & sum_abs,
    double & max_abs,
    int64_t & count) {
    static_assert(ggml_quantize_type_traits<type>::quant_type == type, "quantize type trait mismatch");

    sum_sq = 0.0;
    sum_abs = 0.0;
    max_abs = 0.0;
    count = 0;

    const int worker_count = (int) std::max<int64_t>(1, std::min<int64_t>(
        std::max<int64_t>(1, n_slices),
        nthread > 0 ? nthread : 1));
    if (worker_count <= 1) {
        quantize_binding_slice_accum_t acc;
        for (int64_t i03 = 0; i03 < n_slices; ++i03) {
            run_slice(i03, quantize_binding_stream_key<type>(stream_seed, 0, i03, true), acc);
            if (!acc.ok) {
                return false;
            }
        }
        sum_sq = acc.sum_sq;
        sum_abs = acc.sum_abs;
        max_abs = acc.max_abs;
        count = acc.count;
        return true;
    }

    std::atomic<int64_t> next_slice { 0 };
    std::vector<std::thread> workers;
    std::vector<quantize_binding_slice_accum_t> accs((size_t) worker_count);
    workers.reserve((size_t) worker_count);
    for (int wi = 0; wi < worker_count; ++wi) {
        workers.emplace_back([&, wi]() {
            void * stream_key = quantize_binding_stream_key<type>(stream_seed, wi, -1, false);
            while (accs[(size_t) wi].ok) {
                const int64_t i03 = next_slice.fetch_add(1, std::memory_order_relaxed);
                if (i03 >= n_slices) {
                    break;
                }
                run_slice(i03, stream_key, accs[(size_t) wi]);
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    for (const auto & acc : accs) {
        if (!acc.ok) {
            return false;
        }
        sum_sq += acc.sum_sq;
        sum_abs += acc.sum_abs;
        max_abs = std::max(max_abs, acc.max_abs);
        count += acc.count;
    }
    return true;
}

template <ggml_type type>
static bool quantize_binding_has_target(
    const selector_binding & binding,
    const char * caller) {
    using traits = ggml_quantize_type_traits<type>;

    if (binding.source != nullptr &&
            binding.target != nullptr &&
            binding.target->type == traits::quant_type &&
            (!traits::require_scale_bytes || binding.target_scale_nbytes != 0)) {
        return true;
    }

    fprintf(stderr,
        "%s: bad %s quantize binding for %s (source=%p target=%p target_type=%s scale=%p scale_bytes=%zu)\n",
        caller,
        ggml_type_name(traits::quant_type),
        binding.name.c_str(),
        (void *) binding.source,
        (void *) binding.target,
        binding.target ? ggml_type_name(binding.target->type) : "null",
        (void *) binding.target_scale,
        binding.target_scale_nbytes);
    return false;
}

template <ggml_type type, typename Binding, typename PrepareFn, typename SliceFn>
static bool quantize_binding_quantize(
    Binding & binding,
    int nthread,
    int64_t sample_blocks_override,
    int64_t sample_phase,
    uintptr_t stream_seed,
    const char * caller,
    PrepareFn && prepare,
    SliceFn && run_slice,
    double & sum_sq,
    double & sum_abs,
    double & max_abs,
    int64_t & count) {
    using traits = ggml_quantize_type_traits<type>;

    if (!quantize_binding_has_target<traits::quant_type>(binding, caller)) {
        return false;
    }

    quantize_binding_shape_t shape;
    if (!quantize_binding_prepare_shape<traits::quant_type>(
            binding, sample_blocks_override, sample_phase, caller, shape)) {
        return false;
    }

    if (binding.source->data == nullptr) {
        fprintf(stderr,
            "%s: %s source tensor %s has null data pointer (type=%s rows=%" PRId64 " cols=%" PRId64 " slices=%" PRId64 ")\n",
            caller,
            ggml_type_name(traits::quant_type),
            binding.name.c_str(),
            ggml_type_name(binding.source->type),
            shape.nrows,
            shape.n_per_row,
            shape.n_slices);
        return false;
    }

    const int selector_threads = quantize_binding_thread_count<traits::quant_type>(shape.n_slices, nthread);
    shape.host_threads = std::max(1, nthread / std::max(1, selector_threads));

    if (!prepare(shape)) {
        return false;
    }

    auto run_slice_with_shape = [&](int64_t i03, void * stream_key, quantize_binding_slice_accum_t & acc) {
        run_slice(i03, stream_key, shape, acc);
    };
    return quantize_run_binding_slices<traits::quant_type>(
        shape.n_slices,
        selector_threads,
        stream_seed,
        run_slice_with_shape,
        sum_sq,
        sum_abs,
        max_abs,
        count);
}

static int64_t selector_sample_phase_stride(
    int64_t nb_total,
    selector_tensor_class cls) {
    int64_t stride = std::max<int64_t>(1, nb_total / 11);
    switch (cls) {
        case selector_tensor_class::ATTN_QKV:
        case selector_tensor_class::ATTN_OUT:
        case selector_tensor_class::SSM:
            stride = std::max<int64_t>(stride, std::max<int64_t>(1, nb_total / 7));
            break;
        case selector_tensor_class::EXPERT_UP_GATE:
        case selector_tensor_class::EXPERT_DOWN:
            stride = std::max<int64_t>(stride, std::max<int64_t>(1, nb_total / 5));
            break;
        default:
            break;
    }
    return std::max<int64_t>(1, stride);
}

static float gguf_get_nvfp4_tensor_scale(const gguf_context * ctx_gguf, const char * tensor_name) {
    if (!ctx_gguf || !tensor_name) {
        return 1.0f;
    }

    float weight_scale = 1.0f;
    float weight_scale_2 = 1.0f;

    const std::string key = std::string(tensor_name) + ".weight_scale";
    const int key_id = gguf_find_key(ctx_gguf, key.c_str());
    if (key_id >= 0 && gguf_get_kv_type(ctx_gguf, key_id) == GGUF_TYPE_FLOAT32) {
        weight_scale = gguf_get_val_f32(ctx_gguf, key_id);
    }

    const std::string key_2 = std::string(tensor_name) + ".weight_scale_2";
    const int key_id_2 = gguf_find_key(ctx_gguf, key_2.c_str());
    if (key_id_2 >= 0 && gguf_get_kv_type(ctx_gguf, key_id_2) == GGUF_TYPE_FLOAT32) {
        weight_scale_2 = gguf_get_val_f32(ctx_gguf, key_id_2);
    }

    const float out = weight_scale * weight_scale_2;
    return (std::isfinite(out) && out > 0.0f) ? out : 1.0f;
}

static bool selector_gpu_kld_parity_ok(
        const selector_kld_metrics & cpu,
        const selector_kld_metrics & gpu,
        std::string * reason) {
    auto fail = [&](const std::string & why) {
        if (reason != nullptr) {
            *reason = why;
        }
        return false;
    };
    if (cpu.count != gpu.count) {
        return fail("count mismatch cpu=" + std::to_string(cpu.count) + " gpu=" + std::to_string(gpu.count));
    }
    if (cpu.count <= 0) {
        return fail("empty metric set");
    }

    auto close = [](double a, double b, double abs_tol, double rel_tol) {
        const double diff = std::fabs(a - b);
        const double scale = std::max({1.0, std::fabs(a), std::fabs(b)});
        return diff <= abs_tol || diff <= rel_tol * scale;
    };
    const double inv = 1.0 / (double) cpu.count;
    const double cpu_ln  = cpu.sum_nll * inv;
    const double gpu_ln  = gpu.sum_nll * inv;
    const double cpu_kld = cpu.sum_kld * inv;
    const double gpu_kld = gpu.sum_kld * inv;
    if (!close(cpu_ln, gpu_ln, 1e-4, 5e-4)) {
        char buf[192];
        snprintf(buf, sizeof(buf), "mean ln mismatch cpu=%.9f gpu=%.9f", cpu_ln, gpu_ln);
        return fail(buf);
    }
    if (!close(cpu_kld, gpu_kld, 1e-4, 5e-4)) {
        char buf[192];
        snprintf(buf, sizeof(buf), "mean kld mismatch cpu=%.9f gpu=%.9f", cpu_kld, gpu_kld);
        return fail(buf);
    }
    if (!close(cpu.max_kld, gpu.max_kld, 1e-3, 1e-3)) {
        char buf[192];
        snprintf(buf, sizeof(buf), "max kld mismatch cpu=%.9f gpu=%.9f", cpu.max_kld, gpu.max_kld);
        return fail(buf);
    }
    return true;
}

class selector_kld_worker_pool {
public:
    explicit selector_kld_worker_pool(int n_threads) {
        const int capped = std::max(0, n_threads);
        workers.reserve((size_t) capped);
        for (int i = 0; i < capped; ++i) {
            workers.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    ~selector_kld_worker_pool() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stopping = true;
            generation++;
        }
        cv.notify_all();
        for (std::thread & worker : workers) {
            worker.join();
        }
    }

    int capacity() const {
        return (int) workers.size();
    }

    void run(int n_tasks, std::function<void(int)> fn) {
        if (n_tasks <= 0) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            task = std::move(fn);
            active = std::min(n_tasks, capacity());
            remaining = active;
            generation++;
        }
        cv.notify_all();

        std::unique_lock<std::mutex> lock(mu);
        done_cv.wait(lock, [this]() { return remaining == 0; });
        task = {};
    }

private:
    void worker_loop(int worker_index) {
        uint64_t seen_generation = 0;
        for (;;) {
            std::function<void(int)> local_task;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [this, &seen_generation]() {
                    return stopping || generation != seen_generation;
                });
                if (stopping) {
                    return;
                }
                seen_generation = generation;
                if (worker_index >= active) {
                    continue;
                }
                local_task = task;
            }

            local_task(worker_index);

            {
                std::lock_guard<std::mutex> lock(mu);
                if (--remaining == 0) {
                    done_cv.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv;
    std::condition_variable done_cv;
    std::function<void(int)> task;
    int active = 0;
    int remaining = 0;
    uint64_t generation = 0;
    bool stopping = false;
};

static bool selector_eval_kld_subset(
    llama_context * ctx,
    const selector_kld_subset & kld,
    int32_t n_batch,
    selector_kld_metrics & out,
    bool collect_kld_distribution,
    const std::string & progress_label = "selector kld eval") {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    if (kld.n_vocab != llama_vocab_n_tokens(vocab)) {
        fprintf(stderr, "%s: selector vocab mismatch\n", __func__);
        return false;
    }
    if (kld.n_ctx > (int32_t) llama_n_ctx(ctx)) {
        fprintf(stderr, "%s: selector ctx too small\n", __func__);
        return false;
    }

    if (n_batch <= 0) {
        n_batch = kld.n_ctx;
    }
    int n_seq = std::max(1, n_batch / kld.n_ctx);
    n_seq = std::min(n_seq, (int) llama_n_seq_max(ctx));
    n_seq = std::min(n_seq, std::max(1, (int) llama_n_ctx(ctx) / kld.n_ctx));
    if (n_seq < 1) {
        n_seq = 1;
    }
    n_batch = std::max(1, n_batch);
    const int num_batches = (kld.n_ctx + n_batch - 1) / n_batch;
    out.collect_kld_values = collect_kld_distribution;
    if (collect_kld_distribution) {
        out.kld_values.clear();
    }
    const int64_t metric_thread_cap = std::max<int64_t>(1, kld.n_chunk * kld.n_score);
    int64_t metric_thread_request = selector_kld_threads_override();
    if (metric_thread_request <= 0) {
        metric_thread_request = std::max<unsigned int>(1, std::thread::hardware_concurrency());
    }
    const int metric_threads = (int) std::max<int64_t>(1, std::min<int64_t>(metric_thread_cap, metric_thread_request));
    const bool progress = true;
    const int64_t max_decode_tokens =
        (int64_t) std::max(1, std::min(n_batch, kld.n_ctx)) * (int64_t) n_seq;
    const bool gpu_kld_enabled = llama_n_ubatch(ctx) >= (uint32_t) max_decode_tokens;
    if (!gpu_kld_enabled && progress) {
        fprintf(stderr,
            "%s: selector GPU KLD disabled for this eval because n_ubatch=%u < decode_tokens=%" PRId64
            "; using CPU reduction to avoid raw-logits row mismatch\n",
            __func__,
            llama_n_ubatch(ctx),
            max_decode_tokens);
    }
    static std::atomic<int> gpu_kld_state { 0 };
    struct scoped_selector_gpu_kld_active {
        llama_context * ctx = nullptr;
        bool active = false;
        scoped_selector_gpu_kld_active(llama_context * ctx_, bool active_) : ctx(ctx_), active(active_) {
            if (active) {
                llama_internal_set_selector_gpu_kld_active(ctx, true);
            }
        }
        ~scoped_selector_gpu_kld_active() {
            if (active) {
                llama_internal_set_selector_gpu_kld_active(ctx, false);
            }
        }
    };

    std::unique_ptr<selector_kld_worker_pool> cpu_kld_pool;
    llama_batch batch = llama_batch_init(std::min(n_batch, kld.n_ctx * n_seq), 0, 1);
    const int64_t progress_total =
        (int64_t) ((kld.n_chunk + n_seq - 1) / n_seq) * (int64_t) std::max(1, num_batches);
    int64_t progress_done = 0;
    selector_progress_heartbeat heartbeat(progress_label, progress_total);
    {
        char detail[192];
        snprintf(detail, sizeof(detail),
            "chunks=%d n_seq=%d ctx=%d batch=%d splits=%d score_tokens=%d threads=%d",
            kld.n_chunk, n_seq, kld.n_ctx, n_batch, num_batches, kld.n_score, metric_threads);
        heartbeat.update(progress_done, progress_total, detail, true);
    }
    for (int ci = 0; ci < kld.n_chunk; ci += n_seq) {
        const int n_seq_batch = std::min(n_seq, kld.n_chunk - ci);
        {
            char detail[160];
            snprintf(detail, sizeof(detail), "chunk [%d/%d] preparing", ci + 1, kld.n_chunk);
            heartbeat.detail(detail);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<float> logits_all;
        if (num_batches > 1) {
            logits_all.reserve((size_t) n_seq_batch * (size_t) kld.n_score * (size_t) kld.n_vocab);
        }

        bool chunk_skipped_host_logits = false;
        bool chunk_gpu_complete = gpu_kld_enabled && gpu_kld_state.load(std::memory_order_relaxed) >= 0;
        bool chunk_gpu_failed = false;
        int chunk_gpu_rows = 0;
        selector_kld_metrics chunk_gpu_metrics;
        chunk_gpu_metrics.collect_kld_values = collect_kld_distribution;
        for (int jb = 0; jb < num_batches; ++jb) {
            const int batch_start = jb * n_batch;
            const int batch_size = std::min(kld.n_ctx - batch_start, n_batch);
            const int s_begin = std::max(0, batch_start - kld.first);
            const int s_end = std::min(kld.n_score, batch_start + batch_size - kld.first);
            const int score_count = std::max(0, s_end - s_begin);
            {
                char detail[192];
                snprintf(detail, sizeof(detail), "chunk [%d/%d] split [%d/%d] decoding", ci + 1, kld.n_chunk, jb + 1, num_batches);
                heartbeat.detail(detail);
            }
            common_batch_clear(batch);
            int n_outputs = 0;
            for (int seq = 0; seq < n_seq_batch; ++seq) {
                const llama_token * chunk_tokens = kld.tokens.data() + (size_t) (ci + seq) * (size_t) kld.n_ctx;
                for (int t = 0; t < batch_size; ++t) {
                    const int pos = batch_start + t;
                    llama_token tok = chunk_tokens[pos];
                    if (add_bos && jb == 0 && t == 0) {
                        tok = llama_vocab_bos(vocab);
                    }
                    const bool need_logits = pos >= kld.first && (pos - kld.first) < kld.n_score;
                    common_batch_add(batch, tok, pos, { seq }, need_logits);
                    n_outputs += need_logits;
                }
            }
            if (n_outputs != n_seq_batch * score_count) {
                fprintf(stderr,
                    "%s: selector logits row mismatch chunk=%d split=%d outputs=%d expected=%d\n",
                    __func__, ci, jb, n_outputs, n_seq_batch * score_count);
                llama_batch_free(batch);
                return false;
            }
            const bool skip_host_logits = gpu_kld_enabled && gpu_kld_state.load(std::memory_order_relaxed) > 0;
            if (skip_host_logits) {
                chunk_skipped_host_logits = true;
            }
            scoped_selector_gpu_kld_active gpu_kld_guard(ctx, skip_host_logits);
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s: selector llama_decode failed chunk=%d batch=%d\n", __func__, ci, jb);
                llama_batch_free(batch);
                return false;
            }
            ++progress_done;
            {
                char detail[192];
                snprintf(detail, sizeof(detail), "chunk [%d/%d] split [%d/%d] decoded", ci + 1, kld.n_chunk, jb + 1, num_batches);
                heartbeat.update(progress_done, progress_total, detail);
            }
            if (!skip_host_logits && num_batches > 1 && n_outputs > 0) {
                const float * l = llama_get_logits(ctx);
                if (l == nullptr) {
                    fprintf(stderr, "%s: selector missing split logits chunk=%d split=%d\n", __func__, ci, jb);
                    llama_batch_free(batch);
                    return false;
                }
                if (logits_all.empty()) {
                    logits_all.resize((size_t) n_seq_batch * (size_t) kld.n_score * (size_t) kld.n_vocab);
                }
                for (int seq = 0; seq < n_seq_batch; ++seq) {
                    const float * src = l + (size_t) seq * (size_t) score_count * (size_t) kld.n_vocab;
                    float * dst = logits_all.data() +
                        ((size_t) seq * (size_t) kld.n_score + (size_t) s_begin) * (size_t) kld.n_vocab;
                    memcpy(dst, src, (size_t) score_count * (size_t) kld.n_vocab * sizeof(float));
                }
            }
            if (chunk_gpu_complete && !chunk_gpu_failed && n_outputs > 0) {
                {
                    char detail[192];
                    snprintf(detail, sizeof(detail), "chunk [%d/%d] split [%d/%d] reducing GPU KLD", ci + 1, kld.n_chunk, jb + 1, num_batches);
                    heartbeat.detail(detail);
                }
                if (score_count > 0) {
                    ctx->synchronize();
                    const ggml_tensor * logits_tensor = llama_internal_get_logits_tensor(ctx);
                    for (int seq = 0; seq < n_seq_batch; ++seq) {
                        std::vector<int32_t> token_ids((size_t) score_count);
                        const llama_token * chunk_tokens = kld.tokens.data() + (size_t) (ci + seq) * (size_t) kld.n_ctx;
                        for (int s = 0; s < score_count; ++s) {
                            const int t = kld.first + s_begin + s;
                            token_ids[(size_t) s] = (int32_t) chunk_tokens[t + 1];
                        }
                        const uint16_t * base_seq =
                            selector_kld_log_probs_data(kld) +
                            ((size_t) (ci + seq) * (size_t) kld.n_score + (size_t) s_begin) * (size_t) kld.nv;
                        nvfp4_cuda_kld_result cuda_km{};
                        std::vector<double> cuda_kld_values;
                        if (collect_kld_distribution) {
                            cuda_kld_values.resize((size_t) score_count);
                        }
                        if (!nvfp4_kld_reduce_cuda_tensor(
                                logits_tensor,
                                base_seq,
                                token_ids.data(),
                                seq * score_count,
                                score_count,
                                kld.n_vocab,
                                kld.nv,
                                &cuda_km,
                                collect_kld_distribution ? cuda_kld_values.data() : nullptr,
                                nullptr)) {
                            chunk_gpu_failed = true;
                            chunk_gpu_complete = false;
                            if (chunk_skipped_host_logits) {
                                fprintf(stderr, "%s: selector GPU KLD failed after host-logits copy was skipped; CPU fallback is unavailable for this chunk\n", __func__);
                                llama_batch_free(batch);
                                return false;
                            }
                            gpu_kld_state.store(-1, std::memory_order_relaxed);
                            break;
                        }
                        selector_merge_cuda_kld_metrics(chunk_gpu_metrics, cuda_km, std::move(cuda_kld_values));
                        chunk_gpu_rows += score_count;
                    }
                }
            }
        }
        const int n_eval = n_seq_batch * kld.n_score;
        const bool chunk_gpu_ready = chunk_gpu_complete && !chunk_gpu_failed && chunk_gpu_rows == n_eval;
        const bool validate_gpu_chunk = chunk_gpu_ready && gpu_kld_state.load(std::memory_order_relaxed) == 0;
        if (chunk_gpu_ready && !validate_gpu_chunk) {
            {
                char detail[160];
                snprintf(detail, sizeof(detail), "chunk [%d/%d] complete via GPU KLD", ci + 1, kld.n_chunk);
                heartbeat.detail(detail);
            }
            gpu_kld_state.store(1, std::memory_order_relaxed);
            selector_merge_kld_metrics(out, std::move(chunk_gpu_metrics));
            continue;
        }
        if (chunk_skipped_host_logits) {
            fprintf(stderr, "%s: selector GPU KLD did not cover the full chunk after host-logits copy was skipped\n", __func__);
            llama_batch_free(batch);
            return false;
        }
        const float * logits_contiguous = nullptr;
        if (num_batches == 1) {
            // Synchronize exactly once per decoded chunk. Worker threads must not
            // call llama_get_logits_ith(), because that public API synchronizes
            // the whole context on every token row and collapses KLD reduction
            // back into a mostly single-threaded bottleneck.
            logits_contiguous = llama_get_logits(ctx);
            if (logits_contiguous == nullptr) {
                fprintf(stderr, "%s: selector missing contiguous logits chunk=%d\n", __func__, ci);
                llama_batch_free(batch);
                return false;
            }
        }

        {
            char detail[160];
            snprintf(detail, sizeof(detail), "chunk [%d/%d] reducing CPU KLD", ci + 1, kld.n_chunk);
            heartbeat.detail(detail);
        }
        auto eval_range = [&](int begin, int end, selector_kld_metrics & local) -> bool {
            local.collect_kld_values = collect_kld_distribution;
            if (collect_kld_distribution) {
                local.kld_values.reserve((size_t) std::max(0, end - begin));
            }
            for (int item = begin; item < end; ++item) {
                const int seq = item / kld.n_score;
                const int s = item - seq * kld.n_score;
                const int t = kld.first + s;
                const llama_token * chunk_tokens = kld.tokens.data() + (size_t) (ci + seq) * (size_t) kld.n_ctx;
                const uint16_t * base_chunk = selector_kld_log_probs_data(kld) + (size_t) (ci + seq) * (size_t) kld.n_score * (size_t) kld.nv;
                const float * logits_t = num_batches > 1
                    ? logits_all.data() + ((size_t) seq * (size_t) kld.n_score + (size_t) s) * (size_t) kld.n_vocab
                    : logits_contiguous + ((size_t) seq * (size_t) kld.n_score + (size_t) s) * (size_t) kld.n_vocab;
                const uint16_t * base_t = base_chunk + (size_t) s * (size_t) kld.nv;
                const llama_token tok = chunk_tokens[t + 1];
                selector_eval_one_token(kld.n_vocab, logits_t, base_t, tok, local);
            }
            return true;
        };

        selector_kld_metrics chunk_cpu_metrics;
        chunk_cpu_metrics.collect_kld_values = collect_kld_distribution;
        const int threads_for_chunk = std::min(metric_threads, std::max(1, n_eval));
        if (threads_for_chunk <= 1 || n_eval < 2) {
            selector_kld_metrics local;
            if (!eval_range(0, n_eval, local)) {
                fprintf(stderr, "%s: selector missing logits chunk=%d token=%d\n", __func__, ci, kld.first);
                llama_batch_free(batch);
                return false;
            }
            selector_merge_kld_metrics(chunk_cpu_metrics, std::move(local));
        } else {
            std::vector<selector_kld_metrics> locals((size_t) threads_for_chunk);
            if (!cpu_kld_pool || cpu_kld_pool->capacity() < threads_for_chunk) {
                cpu_kld_pool = std::make_unique<selector_kld_worker_pool>(threads_for_chunk);
            }
            std::atomic_bool ok{true};
            cpu_kld_pool->run(threads_for_chunk, [&](int ti) {
                const int begin = (n_eval * ti) / threads_for_chunk;
                const int end = (n_eval * (ti + 1)) / threads_for_chunk;
                if (!eval_range(begin, end, locals[(size_t) ti])) {
                    ok.store(false, std::memory_order_relaxed);
                }
            });
            if (!ok.load(std::memory_order_relaxed)) {
                fprintf(stderr, "%s: selector missing logits chunk=%d token=%d\n", __func__, ci, kld.first);
                llama_batch_free(batch);
                return false;
            }
            for (auto & local : locals) {
                selector_merge_kld_metrics(chunk_cpu_metrics, std::move(local));
            }
        }
        if (validate_gpu_chunk) {
            std::string parity_reason;
            if (selector_gpu_kld_parity_ok(chunk_cpu_metrics, chunk_gpu_metrics, &parity_reason)) {
                fprintf(stderr,
                    "%s: selector GPU KLD parity passed; enabling host-logits skip for later chunks\n",
                    __func__);
                gpu_kld_state.store(1, std::memory_order_relaxed);
                selector_merge_kld_metrics(out, std::move(chunk_gpu_metrics));
            } else {
                fprintf(stderr,
                    "%s: selector GPU KLD parity failed (%s); disabling CUDA reducer for this process\n",
                    __func__, parity_reason.c_str());
                gpu_kld_state.store(-1, std::memory_order_relaxed);
                selector_merge_kld_metrics(out, std::move(chunk_cpu_metrics));
            }
        } else {
            selector_merge_kld_metrics(out, std::move(chunk_cpu_metrics));
        }
        {
            char detail[160];
            snprintf(detail, sizeof(detail), "chunk [%d/%d] complete", ci + 1, kld.n_chunk);
            heartbeat.detail(detail);
        }
    }
    llama_batch_free(batch);
    heartbeat.finish("complete");

    return true;
}

static std::string selector_format_metrics(
        const char * label,
        const selector_derived_metrics & dm) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
        "%s{ln=%.6f+-%.6f kld=%.6f+-%.6f p95=%.6f p99=%.6f p999=%.6f tail99=%.6f rms=%.6f+-%.6f max=%.6f top=%.4f+-%.4f top_flip_w=%.6f top_p_rmse=%.6f entropy_rmse=%.6f}",
        label,
        dm.ln_ratio,
        dm.ln_ratio_unc,
        dm.mean_kld,
        dm.mean_kld_unc,
        dm.kld_p95,
        dm.kld_p99,
        dm.kld_p999,
        dm.kld_tail_mean,
        dm.rms_dp,
        dm.rms_dp_unc,
        dm.max_kld,
        dm.same_top,
        dm.same_top_unc,
        dm.top_flip_weight,
        dm.top_prob_rmse,
        dm.entropy_rmse);
    return buf;
}

static std::string selector_json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out.push_back(c); break;
        }
    }
    return out;
}

static void selector_write_json_number(std::ostream & out, double value) {
    if (std::isfinite(value)) {
        out << value;
    } else {
        out << "null";
    }
}

static void selector_write_json_number_field(std::ostream & out, const char * name, double value) {
    out << ", \"" << name << "\": ";
    selector_write_json_number(out, value);
}

static void selector_write_metric_json_fields(
        std::ostream & out,
        const selector_derived_metrics & dm,
        const char * prefix) {
    out
        << ", \"" << prefix << "valid\": " << (dm.ok ? "true" : "false");
    out << ", \"" << prefix << "ppl\": "; selector_write_json_number(out, dm.ppl_q);
    out << ", \"" << prefix << "ppl_base\": "; selector_write_json_number(out, dm.ppl_base);
    out << ", \"" << prefix << "ln_ratio\": "; selector_write_json_number(out, dm.ln_ratio);
    out << ", \"" << prefix << "mean_kld\": "; selector_write_json_number(out, dm.mean_kld);
    out << ", \"" << prefix << "kld_p95\": "; selector_write_json_number(out, dm.kld_p95);
    out << ", \"" << prefix << "kld_p99\": "; selector_write_json_number(out, dm.kld_p99);
    out << ", \"" << prefix << "kld_p999\": "; selector_write_json_number(out, dm.kld_p999);
    out << ", \"" << prefix << "kld_tail_mean\": "; selector_write_json_number(out, dm.kld_tail_mean);
    out << ", \"" << prefix << "max_kld\": "; selector_write_json_number(out, dm.max_kld);
    out << ", \"" << prefix << "rms_dp\": "; selector_write_json_number(out, dm.rms_dp);
    out << ", \"" << prefix << "same_top\": "; selector_write_json_number(out, dm.same_top);
    out << ", \"" << prefix << "entropy_rmse\": "; selector_write_json_number(out, dm.entropy_rmse);
    out << ", \"" << prefix << "top_prob_rmse\": "; selector_write_json_number(out, dm.top_prob_rmse);
    out << ", \"" << prefix << "top_flip_weight\": "; selector_write_json_number(out, dm.top_flip_weight);
}

static void selector_write_delta_metric_json_fields(
        std::ostream & out,
        const selector_derived_metrics & after,
        const selector_derived_metrics & before) {
    auto write_delta = [&](const char * name, double value) {
        out << ", \"" << name << "\": ";
        if (after.ok && before.ok && std::isfinite(value)) {
            selector_write_json_number(out, value);
        } else {
            out << "null";
        }
    };
    write_delta("delta_ppl", after.ppl_q - before.ppl_q);
    write_delta("delta_ln_ratio", after.ln_ratio - before.ln_ratio);
    write_delta("delta_mean_kld", after.mean_kld - before.mean_kld);
    write_delta("delta_kld_p95", after.kld_p95 - before.kld_p95);
    write_delta("delta_kld_p99", after.kld_p99 - before.kld_p99);
    write_delta("delta_kld_p999", after.kld_p999 - before.kld_p999);
    write_delta("delta_kld_tail_mean", after.kld_tail_mean - before.kld_tail_mean);
    write_delta("delta_max_kld", after.max_kld - before.max_kld);
    write_delta("delta_rms_dp", after.rms_dp - before.rms_dp);
    write_delta("delta_same_top", after.same_top - before.same_top);
    write_delta("delta_entropy_rmse", after.entropy_rmse - before.entropy_rmse);
    write_delta("delta_top_prob_rmse", after.top_prob_rmse - before.top_prob_rmse);
    write_delta("delta_top_flip_weight", after.top_flip_weight - before.top_flip_weight);
}

struct selector_stageb_digest {
    uint64_t value = 1469598103934665603ull;

    void add_bytes(const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            value ^= (uint64_t) bytes[i];
            value *= 1099511628211ull;
        }
    }

    void add_string(const std::string & text) {
        add_bytes(text.data(), text.size());
        const uint8_t nul = 0;
        add_bytes(&nul, sizeof(nul));
    }

    std::string hex() const {
        char buf[32];
        snprintf(buf, sizeof(buf), "%016" PRIx64, value);
        return buf;
    }
};

static nlohmann::ordered_json selector_stageb_metric_json(const selector_derived_metrics & dm) {
    nlohmann::ordered_json out = nlohmann::ordered_json::object();
    out["ok"] = dm.ok;
    out["ppl_q"] = dm.ppl_q;
    out["ppl_base"] = dm.ppl_base;
    out["ln_ratio"] = dm.ln_ratio;
    out["ln_ratio_unc"] = dm.ln_ratio_unc;
    out["mean_kld"] = dm.mean_kld;
    out["mean_kld_unc"] = dm.mean_kld_unc;
    out["kld_p95"] = dm.kld_p95;
    out["kld_p99"] = dm.kld_p99;
    out["kld_p999"] = dm.kld_p999;
    out["kld_tail_mean"] = dm.kld_tail_mean;
    out["max_kld"] = dm.max_kld;
    out["rms_dp"] = dm.rms_dp;
    out["rms_dp_unc"] = dm.rms_dp_unc;
    out["same_top"] = dm.same_top;
    out["same_top_unc"] = dm.same_top_unc;
    out["entropy_rmse"] = dm.entropy_rmse;
    out["top_prob_rmse"] = dm.top_prob_rmse;
    out["top_flip_weight"] = dm.top_flip_weight;
    return out;
}

static bool selector_stageb_metric_from_json(
        const nlohmann::ordered_json & in,
        selector_derived_metrics & dm) {
    if (!in.is_object()) {
        return false;
    }
    dm.ok = in.value("ok", false);
    dm.ppl_q = in.value("ppl_q", 0.0);
    dm.ppl_base = in.value("ppl_base", 0.0);
    dm.ln_ratio = in.value("ln_ratio", 0.0);
    dm.ln_ratio_unc = in.value("ln_ratio_unc", 0.0);
    dm.mean_kld = in.value("mean_kld", 0.0);
    dm.mean_kld_unc = in.value("mean_kld_unc", 0.0);
    dm.kld_p95 = in.value("kld_p95", 0.0);
    dm.kld_p99 = in.value("kld_p99", 0.0);
    dm.kld_p999 = in.value("kld_p999", 0.0);
    dm.kld_tail_mean = in.value("kld_tail_mean", 0.0);
    dm.max_kld = in.value("max_kld", 0.0);
    dm.rms_dp = in.value("rms_dp", 0.0);
    dm.rms_dp_unc = in.value("rms_dp_unc", 0.0);
    dm.same_top = in.value("same_top", 0.0);
    dm.same_top_unc = in.value("same_top_unc", 0.0);
    dm.entropy_rmse = in.value("entropy_rmse", 0.0);
    dm.top_prob_rmse = in.value("top_prob_rmse", 0.0);
    dm.top_flip_weight = in.value("top_flip_weight", 0.0);
    return dm.ok;
}

static nlohmann::ordered_json selector_stageb_cfg_json(const nvfp4_cuda_runtime_cfg & cfg) {
    nlohmann::ordered_json out = nlohmann::ordered_json::object();
    out["choose46"] = cfg.choose46_mode;
    out["refit"] = cfg.refit_iters;
    out["compand"] = cfg.use_compand_sat;
    out["reserved"] = cfg.reserved_i32;
    out["cap6"] = cfg.cap_m6;
    out["cap4"] = cfg.cap_m4;
    return out;
}

static bool selector_stageb_cfg_from_json(
        const nlohmann::ordered_json & in,
        nvfp4_cuda_runtime_cfg & cfg) {
    if (!in.is_object()) {
        return false;
    }
    cfg.choose46_mode = in.value("choose46", NVFP4_CUDA_CHOOSE46_ADAPTIVE);
    cfg.refit_iters = in.value("refit", 8);
    cfg.use_compand_sat = in.value("compand", 1);
    cfg.reserved_i32 = in.value("reserved", 0);
    cfg.cap_m6 = in.value("cap6", 448.0f);
    cfg.cap_m4 = in.value("cap4", 256.0f);
    return std::isfinite(cfg.cap_m6) && std::isfinite(cfg.cap_m4) &&
           cfg.refit_iters > 0 && cfg.cap_m6 > 0.0f && cfg.cap_m4 > 0.0f;
}

static nlohmann::ordered_json selector_stageb_rank_json(const selector_rank_config & cfg) {
    nlohmann::ordered_json out = nlohmann::ordered_json::object();
    out["kld_threshold"] = cfg.kld_threshold;
    out["p99_threshold"] = cfg.p99_threshold;
    out["p999_threshold"] = cfg.p999_threshold;
    out["max_kld_threshold"] = cfg.max_kld_threshold;
    out["kld_penalty"] = cfg.kld_penalty;
    out["p99_penalty"] = cfg.p99_penalty;
    out["p999_penalty"] = cfg.p999_penalty;
    out["max_kld_penalty"] = cfg.max_kld_penalty;
    out["holdout_weight"] = cfg.holdout_weight;
    out["kld_hard_gate"] = cfg.kld_hard_gate;
    out["p99_hard_gate"] = cfg.p99_hard_gate;
    out["p999_hard_gate"] = cfg.p999_hard_gate;
    out["max_kld_hard_gate"] = cfg.max_kld_hard_gate;
    out["ppl_sigma"] = cfg.ppl_sigma;
    out["kld_sigma"] = cfg.kld_sigma;
    out["rms_dp_sigma"] = cfg.rms_dp_sigma;
    out["same_top_sigma"] = cfg.same_top_sigma;
    out["ln_ratio_abs_floor"] = cfg.ln_ratio_abs_floor;
    out["mean_kld_abs_floor"] = cfg.mean_kld_abs_floor;
    out["rms_dp_abs_floor"] = cfg.rms_dp_abs_floor;
    out["same_top_abs_floor"] = cfg.same_top_abs_floor;
    out["entropy_rmse_abs_floor"] = cfg.entropy_rmse_abs_floor;
    out["top_prob_rmse_abs_floor"] = cfg.top_prob_rmse_abs_floor;
    out["top_flip_weight_abs_floor"] = cfg.top_flip_weight_abs_floor;
    out["p99_abs_margin"] = cfg.p99_abs_margin;
    out["p99_rel_margin"] = cfg.p99_rel_margin;
    out["p999_abs_margin"] = cfg.p999_abs_margin;
    out["p999_rel_margin"] = cfg.p999_rel_margin;
    out["max_kld_abs_margin"] = cfg.max_kld_abs_margin;
    out["max_kld_rel_margin"] = cfg.max_kld_rel_margin;
    out["mxfp6_ppl_abs_tol"] = cfg.mxfp6.ppl_abs_tol;
    out["mxfp6_ppl_rel_tol"] = cfg.mxfp6.ppl_rel_tol;
    out["mxfp6_mean_kld_abs_tol"] = cfg.mxfp6.mean_kld_abs_tol;
    out["mxfp6_mean_kld_rel_tol"] = cfg.mxfp6.mean_kld_rel_tol;
    return out;
}

static nlohmann::ordered_json selector_stageb_file_json(const std::string & path) {
    nlohmann::ordered_json out = nlohmann::ordered_json::object();
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(path, ec);
    out["path"] = ec ? path : canonical.string();
    const uintmax_t size = std::filesystem::file_size(path, ec);
    out["size"] = ec ? 0 : size;
    const auto write_time = std::filesystem::last_write_time(path, ec);
    out["mtime"] = ec ? 0 : write_time.time_since_epoch().count();
    return out;
}

static nlohmann::ordered_json selector_stageb_kld_json(const selector_kld_subset & kld) {
    nlohmann::ordered_json out = nlohmann::ordered_json::object();
    out["path"] = kld.source_path;
    out["size"] = kld.source_size;
    out["mtime"] = kld.source_mtime;
    out["ctx"] = kld.n_ctx;
    out["vocab"] = kld.n_vocab;
    out["chunk_total"] = kld.n_chunk_total;
    out["chunk_count"] = kld.n_chunk;
    out["first"] = kld.first;
    out["score_tokens"] = kld.n_score;
    out["nv"] = kld.nv;
    return out;
}

struct selector_stageb_cache_entry {
    std::string event;
    std::string policy;
    selector_derived_metrics search;
    selector_derived_metrics validation;
    bool has_validation = false;
    bool pass = false;
    double score = std::numeric_limits<double>::infinity();
};

struct selector_stageb_cache {
    static constexpr const char * schema = "advanced-gguf-selector-stageb-result-v1";
    std::string path;
    const selector_ledger * ledger = nullptr;
    std::unordered_map<std::string, selector_stageb_cache_entry> entries;
    std::unordered_set<std::string> sidecar_keys;
    std::unordered_set<std::string> policies;
    std::unordered_set<std::string> mismatch_logged;

    void load_rows(const std::string & rows_path, bool from_ledger) {
        if (rows_path.empty()) {
            return;
        }
        std::ifstream in(rows_path);
        if (!in) {
            return;
        }

        std::string line;
        size_t loaded = 0;
        size_t ignored = 0;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            try {
                const auto record = nlohmann::ordered_json::parse(line);
                const std::string record_schema = record.value("schema", std::string());
                if (!record.is_object()) {
                    ++ignored;
                    continue;
                }
                if (from_ledger) {
                    if (record_schema != selector_ledger::schema ||
                            record.value("row_type", std::string()) != "exact_eval") {
                        ++ignored;
                        continue;
                    }
                } else if (record_schema != schema) {
                    ++ignored;
                    continue;
                }
                const auto key_it = record.find("key");
                const auto search_it = record.find("search");
                if (key_it == record.end() || search_it == record.end()) {
                    ++ignored;
                    continue;
                }

                selector_stageb_cache_entry entry;
                entry.event = record.value("event", std::string());
                entry.policy = record.value("policy", std::string());
                entry.pass = record.value("pass", false);
                entry.score = record.value("score", std::numeric_limits<double>::infinity());
                entry.has_validation = record.value("has_validation", false);
                if (entry.policy.empty() ||
                        !selector_stageb_metric_from_json(*search_it, entry.search)) {
                    ++ignored;
                    continue;
                }
                if (entry.has_validation) {
                    const auto validation_it = record.find("validation");
                    if (validation_it == record.end() ||
                            !selector_stageb_metric_from_json(*validation_it, entry.validation)) {
                        ++ignored;
                        continue;
                    }
                }
                const std::string key = key_it->dump();
                if (from_ledger && sidecar_keys.find(key) != sidecar_keys.end()) {
                    continue;
                }
                entries[key] = entry;
                if (!from_ledger) {
                    sidecar_keys.insert(key);
                }
                policies.insert(entry.policy);
                ++loaded;
            } catch (const std::exception &) {
                ++ignored;
            }
        }

        fprintf(stderr,
            "%s: selector stage-b result cache loaded %s entries=%zu total=%zu ignored=%zu path=%s\n",
            __func__, from_ledger ? "ledger" : "sidecar", loaded, entries.size(), ignored, rows_path.c_str());
    }

    void load(const std::string & cache_path) {
        path = cache_path;
        load_rows(path, false);
    }

    void load_exact_eval_ledger(const std::string & ledger_path) {
        load_rows(ledger_path, true);
    }

    bool find(
            const nlohmann::ordered_json & key,
            const std::string & policy,
            selector_stageb_cache_entry & out) {
        const auto it = entries.find(key.dump());
        if (it != entries.end()) {
            out = it->second;
            fprintf(stderr,
                "%s: selector stage-b cache accepted policy=%s event=%s\n",
                __func__, policy.c_str(), out.event.c_str());
            return true;
        }
        if (key.contains("encoder") && !key.contains("autotune")) {
            nlohmann::ordered_json compat_key = nlohmann::ordered_json::object();
            for (auto it_key = key.begin(); it_key != key.end(); ++it_key) {
                if (it_key.key() == "encoder") {
                    compat_key["autotune"] = it_key.value();
                } else {
                    compat_key[it_key.key()] = it_key.value();
                }
            }
            const auto compat_it = entries.find(compat_key.dump());
            if (compat_it != entries.end()) {
                out = compat_it->second;
                fprintf(stderr,
                    "%s: selector stage-b cache accepted policy=%s event=%s key_migration=encoder\n",
                    __func__, policy.c_str(), out.event.c_str());
                return true;
            }
        }
        if (policies.find(policy) != policies.end() && mismatch_logged.insert(policy).second) {
            fprintf(stderr,
                "%s: selector stage-b cache ignored policy=%s due to key mismatch\n",
                __func__, policy.c_str());
        }
        return false;
    }

    void append(
            const char * event,
            const nlohmann::ordered_json & key,
            const std::string & policy,
            const selector_derived_metrics & search,
            const selector_derived_metrics * validation,
            bool pass,
            double score) {
        if (path.empty() || !search.ok) {
            return;
        }

        std::filesystem::path cache_path(path);
        std::error_code ec;
        if (cache_path.has_parent_path()) {
            std::filesystem::create_directories(cache_path.parent_path(), ec);
            if (ec) {
                fprintf(stderr, "%s: selector stage-b cache directory unavailable %s: %s\n",
                    __func__, cache_path.parent_path().string().c_str(), ec.message().c_str());
                return;
            }
        }

        nlohmann::ordered_json record = nlohmann::ordered_json::object();
        record["schema"] = schema;
        record["event"] = event;
        record["policy"] = policy;
        record["key"] = key;
        record["search"] = selector_stageb_metric_json(search);
        record["has_validation"] = validation != nullptr && validation->ok;
        if (validation != nullptr && validation->ok) {
            record["validation"] = selector_stageb_metric_json(*validation);
        }
        record["pass"] = pass;
        record["score"] = score;

        std::ofstream out(path, std::ios::app);
        if (!out) {
            fprintf(stderr, "%s: selector stage-b cache append failed path=%s\n", __func__, path.c_str());
            return;
        }
        out << record.dump() << '\n';

        if (ledger != nullptr) {
            nlohmann::ordered_json ledger_record = nlohmann::ordered_json::object();
            ledger_record["event"] = event;
            ledger_record["policy"] = policy;
            ledger_record["cache_path"] = path;
            ledger_record["key"] = key;
            ledger_record["search"] = selector_stageb_metric_json(search);
            ledger_record["has_validation"] = validation != nullptr && validation->ok;
            if (validation != nullptr && validation->ok) {
                ledger_record["validation"] = selector_stageb_metric_json(*validation);
            }
            ledger_record["pass"] = pass;
            ledger_record["score"] = score;
            ledger->append_exact_eval(std::move(ledger_record));
        }

        selector_stageb_cache_entry entry;
        entry.event = event;
        entry.policy = policy;
        entry.search = search;
        if (validation != nullptr) {
            entry.validation = *validation;
        }
        entry.has_validation = validation != nullptr && validation->ok;
        entry.pass = pass;
        entry.score = score;
        entries[key.dump()] = entry;
        policies.insert(policy);
    }
};

struct selector_tensor_rescue_cache_entry {
    std::string policy;
    nvfp4_cuda_runtime_cfg cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    int64_t sample_blocks = 0;
    double base_score = std::numeric_limits<double>::infinity();
    double best_score = std::numeric_limits<double>::infinity();
};

struct selector_tensor_rescue_cache {
    static constexpr const char * schema = "advanced-gguf-selector-tensor-rescue-result-v1";
    std::string path;
    std::unordered_map<std::string, selector_tensor_rescue_cache_entry> entries;

    void load(const std::string & cache_path) {
        path = cache_path;
        entries.clear();
        if (path.empty()) {
            return;
        }
        std::ifstream in(path);
        if (!in) {
            return;
        }

        std::string line;
        size_t loaded = 0;
        size_t ignored = 0;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            try {
                const auto record = nlohmann::ordered_json::parse(line);
                if (!record.is_object() ||
                        record.value("schema", std::string()) != schema) {
                    ++ignored;
                    continue;
                }
                const auto key_it = record.find("key");
                const auto cfg_it = record.find("cfg");
                if (key_it == record.end() || cfg_it == record.end()) {
                    ++ignored;
                    continue;
                }

                selector_tensor_rescue_cache_entry entry;
                entry.policy = record.value("policy", std::string());
                entry.sample_blocks = record.value("sample_blocks", (int64_t) 0);
                entry.base_score = record.value("base_score", std::numeric_limits<double>::infinity());
                entry.best_score = record.value("best_score", std::numeric_limits<double>::infinity());
                if (entry.policy.empty() ||
                        entry.sample_blocks <= 0 ||
                        !std::isfinite(entry.base_score) ||
                        !std::isfinite(entry.best_score) ||
                        !selector_stageb_cfg_from_json(*cfg_it, entry.cfg)) {
                    ++ignored;
                    continue;
                }
                entries[key_it->dump()] = entry;
                ++loaded;
            } catch (const std::exception &) {
                ++ignored;
            }
        }

        fprintf(stderr,
            "%s: selector tensor rescue cache loaded entries=%zu ignored=%zu path=%s\n",
            __func__, loaded, ignored, path.c_str());
    }

    bool find(
            const nlohmann::ordered_json & key,
            const std::string & tensor,
            selector_tensor_rescue_cache_entry & out) const {
        const auto it = entries.find(key.dump());
        if (it == entries.end()) {
            return false;
        }
        out = it->second;
        fprintf(stderr,
            "%s: selector tensor rescue cache accepted tensor=%s policy=%s\n",
            __func__, tensor.c_str(), out.policy.c_str());
        return true;
    }

    void append(
            const nlohmann::ordered_json & key,
            const std::string & tensor,
            const selector_tensor_rescue_cache_entry & entry) {
        if (path.empty() || tensor.empty() || entry.policy.empty() ||
                entry.sample_blocks <= 0 ||
                !std::isfinite(entry.base_score) ||
                !std::isfinite(entry.best_score)) {
            return;
        }

        std::filesystem::path cache_path(path);
        std::error_code ec;
        if (cache_path.has_parent_path()) {
            std::filesystem::create_directories(cache_path.parent_path(), ec);
            if (ec) {
                fprintf(stderr, "%s: selector tensor rescue cache directory unavailable %s: %s\n",
                    __func__, cache_path.parent_path().string().c_str(), ec.message().c_str());
                return;
            }
        }

        nlohmann::ordered_json record = nlohmann::ordered_json::object();
        record["schema"] = schema;
        record["event"] = "tensor_rescue";
        record["tensor"] = tensor;
        record["key"] = key;
        record["policy"] = entry.policy;
        record["cfg"] = selector_stageb_cfg_json(entry.cfg);
        record["sample_blocks"] = entry.sample_blocks;
        record["base_score"] = entry.base_score;
        record["best_score"] = entry.best_score;

        std::ofstream out(path, std::ios::app);
        if (!out) {
            fprintf(stderr, "%s: selector tensor rescue cache append failed path=%s\n",
                __func__, path.c_str());
            return;
        }
        out << record.dump() << '\n';
        entries[key.dump()] = entry;
    }
};

static double selector_combined_unc(double ua, double ub) {
    return std::sqrt(std::max(0.0, ua * ua + ub * ub));
}

static int selector_compare_lower_noise(double a, double a_unc, double b, double b_unc, double sigma, double abs_floor) {
    const double margin = std::max(abs_floor, std::max(0.0, sigma) * selector_combined_unc(a_unc, b_unc));
    if (a + margin < b) {
        return -1;
    }
    if (b + margin < a) {
        return 1;
    }
    return 0;
}

static int selector_compare_higher_noise(double a, double a_unc, double b, double b_unc, double sigma, double abs_floor) {
    const double margin = std::max(abs_floor, std::max(0.0, sigma) * selector_combined_unc(a_unc, b_unc));
    if (a > b + margin) {
        return -1;
    }
    if (b > a + margin) {
        return 1;
    }
    return 0;
}

static int selector_compare_lower_margin(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    if (a + margin < b) {
        return -1;
    }
    if (b + margin < a) {
        return 1;
    }
    return 0;
}

static double selector_threshold_penalty(double value, double threshold, double penalty) {
    if (!(penalty > 0.0) || !std::isfinite(penalty) ||
            !(threshold > 0.0) || !std::isfinite(threshold)) {
        return 0.0;
    }
    const double excess = std::max(0.0, value - threshold);
    return penalty * std::max(0.0, excess);
}

static double mxfp6_selector_metric_tol(double base, double abs_floor, double rel_floor) {
    return std::max(std::max(0.0, abs_floor), std::max(0.0, rel_floor) * std::max(std::fabs(base), 1e-12));
}

static double mxfp6_selector_rel_excess(double value, double base, double abs_floor, double rel_floor) {
    const double excess = value - base - mxfp6_selector_metric_tol(base, abs_floor, rel_floor);
    if (!(excess > 0.0) || !std::isfinite(excess)) {
        return 0.0;
    }
    return excess / std::max(std::fabs(base), 1e-12);
}

enum class selector_metric_mode {
    POLICY,
    MXFP6_SCALE,
};

struct selector_metric_score {
    bool pass = false;
    double score = std::numeric_limits<double>::infinity();
};

static selector_metric_score selector_score_metrics(
    const selector_derived_metrics & dm,
    const selector_derived_metrics * base,
    const selector_rank_config & rank_cfg,
    selector_metric_mode mode) {
    selector_metric_score out;
    if (!dm.ok) {
        return out;
    }

    const bool have_base = base != nullptr && base->ok;
    bool pass = true;
    if (have_base) {
        const double kld_thr = rank_cfg.kld_threshold > 0.0 ? std::max(rank_cfg.kld_threshold, base->mean_kld) : base->mean_kld;
        const double p99_thr = rank_cfg.p99_threshold > 0.0 ? std::max(rank_cfg.p99_threshold, base->kld_p99) : base->kld_p99;
        const double p999_thr = rank_cfg.p999_threshold > 0.0 ? std::max(rank_cfg.p999_threshold, base->kld_p999) : base->kld_p999;
        const double max_kld_thr = rank_cfg.max_kld_threshold > 0.0 ? std::max(rank_cfg.max_kld_threshold, base->max_kld) : base->max_kld;

        pass =
            (!rank_cfg.kld_hard_gate || dm.mean_kld <= kld_thr) &&
            (!rank_cfg.p99_hard_gate || dm.kld_p99 <= p99_thr) &&
            (!rank_cfg.p999_hard_gate || dm.kld_p999 <= p999_thr) &&
            (!rank_cfg.max_kld_hard_gate || dm.max_kld <= max_kld_thr);
    }

    const bool scale_mode = mode == selector_metric_mode::MXFP6_SCALE;
    if (scale_mode && !have_base) {
        return out;
    }

    const double same_top_deficit = std::max(0.0, 1.0 - dm.same_top);
    out.score =
        4.00 * dm.mean_kld +
        (scale_mode ? 0.35 : 0.45) * dm.kld_p95 +
        0.90 * dm.kld_p99 +
        0.18 * dm.kld_p999 +
        (scale_mode ? 0.70 : 0.75) * dm.kld_tail_mean +
        1.25 * dm.rms_dp +
        (scale_mode ? 2.00 : 2.40) * same_top_deficit +
        (scale_mode ? 2.80 : 3.20) * dm.top_flip_weight +
        (scale_mode ? 1.10 : 1.25) * dm.top_prob_rmse +
        (scale_mode ? 0.45 : 0.50) * dm.entropy_rmse +
        0.003 * dm.max_kld +
        (scale_mode ? 0.050 : 0.005) * dm.ln_ratio;

    if (!scale_mode) {
        out.score +=
            selector_threshold_penalty(dm.mean_kld, rank_cfg.kld_threshold, rank_cfg.kld_penalty) +
            selector_threshold_penalty(dm.kld_p99, rank_cfg.p99_threshold, rank_cfg.p99_penalty) +
            selector_threshold_penalty(dm.kld_p999, rank_cfg.p999_threshold, rank_cfg.p999_penalty) +
            selector_threshold_penalty(dm.max_kld, rank_cfg.max_kld_threshold, rank_cfg.max_kld_penalty);
        out.pass = pass;
        return out;
    }

    // MXFP6_E2M3 scale refinement is a per-tensor surgical edit.  KLD and
    // token-distribution shape are the primary guards; PPL only prevents a
    // clearly worse teacher-token loss from slipping through.
    pass = pass &&
        dm.mean_kld <= base->mean_kld + mxfp6_selector_metric_tol(base->mean_kld, rank_cfg.mxfp6.mean_kld_abs_tol, rank_cfg.mxfp6.mean_kld_rel_tol) &&
        dm.kld_p99 <= base->kld_p99 + mxfp6_selector_metric_tol(base->kld_p99, 0.01, 0.15) &&
        dm.kld_p999 <= base->kld_p999 + mxfp6_selector_metric_tol(base->kld_p999, 0.50, 0.75) &&
        dm.max_kld <= base->max_kld + mxfp6_selector_metric_tol(base->max_kld, 2.0, 2.0) &&
        dm.rms_dp <= base->rms_dp + mxfp6_selector_metric_tol(base->rms_dp, 4e-4, 0.01) &&
        dm.same_top + mxfp6_selector_metric_tol(base->same_top, 3e-4, 0.0015) >= base->same_top &&
        dm.top_flip_weight <= base->top_flip_weight + mxfp6_selector_metric_tol(base->top_flip_weight, 1e-5, 0.005) &&
        dm.entropy_rmse <= base->entropy_rmse + mxfp6_selector_metric_tol(base->entropy_rmse, 5e-4, 0.02) &&
        dm.ln_ratio <= base->ln_ratio + mxfp6_selector_metric_tol(base->ln_ratio, rank_cfg.mxfp6.ppl_abs_tol, rank_cfg.mxfp6.ppl_rel_tol);

    out.score +=
        1.0  * mxfp6_selector_rel_excess(dm.mean_kld, base->mean_kld, rank_cfg.mxfp6.mean_kld_abs_tol, rank_cfg.mxfp6.mean_kld_rel_tol) +
        0.35 * mxfp6_selector_rel_excess(dm.kld_p99,  base->kld_p99,  0.01, 0.15) +
        0.20 * mxfp6_selector_rel_excess(dm.kld_p999, base->kld_p999, 0.50, 0.75) +
        0.05 * mxfp6_selector_rel_excess(dm.max_kld,  base->max_kld,  2.0, 2.0) +
        0.50 * mxfp6_selector_rel_excess(dm.rms_dp,   base->rms_dp,   4e-4, 0.01) +
        1.00 * mxfp6_selector_rel_excess(base->same_top, dm.same_top, 3e-4, 0.0015) +
        1.25 * mxfp6_selector_rel_excess(dm.top_flip_weight, base->top_flip_weight, 1e-5, 0.005) +
        0.40 * mxfp6_selector_rel_excess(dm.entropy_rmse, base->entropy_rmse, 5e-4, 0.02) +
        0.25 * mxfp6_selector_rel_excess(dm.ln_ratio, base->ln_ratio, rank_cfg.mxfp6.ppl_abs_tol, rank_cfg.mxfp6.ppl_rel_tol);
    out.pass = pass;
    return out;
}

struct selector_metric_rank {
    bool pass = false;
    double score = std::numeric_limits<double>::infinity();
};

static selector_metric_rank selector_rank_policy_metrics(
        const selector_derived_metrics & main_dm,
        const selector_derived_metrics * holdout_dm,
        const selector_derived_metrics & baseline_main,
        const selector_derived_metrics * baseline_holdout,
        const selector_rank_config & rank_cfg) {
    const selector_metric_score main_score =
        selector_score_metrics(main_dm, &baseline_main, rank_cfg, selector_metric_mode::POLICY);

    selector_metric_rank out;
    out.pass = main_score.pass;
    out.score = main_score.score;

    if (holdout_dm != nullptr && holdout_dm->ok) {
        const bool have_holdout_base = baseline_holdout != nullptr && baseline_holdout->ok;
        const selector_metric_score holdout_score =
            selector_score_metrics(
                *holdout_dm,
                have_holdout_base ? baseline_holdout : nullptr,
                rank_cfg,
                selector_metric_mode::POLICY);
        if (have_holdout_base) {
            out.pass = out.pass && holdout_score.pass;
        }
        const double w = std::clamp(rank_cfg.holdout_weight, 0.0, 0.95);
        out.score = (1.0 - w) * out.score + w * holdout_score.score;
    }

    return out;
}

static int selector_compare_one(
    const selector_derived_metrics & a,
    const selector_derived_metrics & b,
    const selector_rank_config & rank_cfg) {
    if (!a.ok || !b.ok) {
        if (a.ok != b.ok) {
            return a.ok ? -1 : 1;
        }
        return 0;
    }

    const int same_top_cmp = selector_compare_higher_noise(
        a.same_top, a.same_top_unc,
        b.same_top, b.same_top_unc,
        rank_cfg.same_top_sigma,
        rank_cfg.same_top_abs_floor);

    int cmp = selector_compare_lower_noise(
        a.mean_kld, a.mean_kld_unc,
        b.mean_kld, b.mean_kld_unc,
        rank_cfg.kld_sigma,
        rank_cfg.mean_kld_abs_floor);
    if (cmp != 0) {
        return cmp;
    }

    if (same_top_cmp != 0) {
        return same_top_cmp;
    }

    cmp = selector_compare_lower_margin(a.kld_p95, b.kld_p95, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    cmp = selector_compare_lower_margin(a.kld_p99, b.kld_p99, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    cmp = selector_compare_lower_margin(a.kld_p999, b.kld_p999, rank_cfg.p999_abs_margin, rank_cfg.p999_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    cmp = selector_compare_lower_margin(a.kld_tail_mean, b.kld_tail_mean, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    cmp = selector_compare_lower_noise(
        a.rms_dp, a.rms_dp_unc,
        b.rms_dp, b.rms_dp_unc,
        rank_cfg.rms_dp_sigma,
        rank_cfg.rms_dp_abs_floor);
    if (cmp != 0) {
        return cmp;
    }

    cmp = selector_compare_lower_margin(a.top_flip_weight, b.top_flip_weight, rank_cfg.top_flip_weight_abs_floor, 0.0);
    if (cmp != 0) {
        return cmp;
    }

    cmp = selector_compare_lower_margin(a.top_prob_rmse, b.top_prob_rmse, rank_cfg.top_prob_rmse_abs_floor, 0.0);
    if (cmp != 0) {
        return cmp;
    }

    cmp = selector_compare_lower_margin(a.entropy_rmse, b.entropy_rmse, rank_cfg.entropy_rmse_abs_floor, 0.0);
    if (cmp != 0) {
        return cmp;
    }

    cmp = selector_compare_lower_margin(a.max_kld, b.max_kld, rank_cfg.max_kld_abs_margin, rank_cfg.max_kld_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    return selector_compare_lower_noise(
        a.ln_ratio, a.ln_ratio_unc,
        b.ln_ratio, b.ln_ratio_unc,
        rank_cfg.ppl_sigma,
        rank_cfg.ln_ratio_abs_floor);
}

static double selector_measured_score_one(
        const selector_derived_metrics & dm,
        const selector_rank_config & rank_cfg) {
    return selector_score_metrics(
        dm, nullptr, rank_cfg, selector_metric_mode::POLICY).score;
}

static double selector_measured_score(
    const selector_derived_metrics & main_dm,
    const selector_derived_metrics * holdout_dm,
    const selector_rank_config & rank_cfg) {
    const double main_score = selector_measured_score_one(main_dm, rank_cfg);
    if (holdout_dm == nullptr || !holdout_dm->ok) {
        return main_score;
    }
    const double holdout_score = selector_measured_score_one(*holdout_dm, rank_cfg);
    const double w = std::clamp(rank_cfg.holdout_weight, 0.0, 0.95);
    return (1.0 - w) * main_score + w * holdout_score;
}

static int selector_compare_measured_score(double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        if (std::isfinite(a) != std::isfinite(b)) {
            return std::isfinite(a) ? -1 : 1;
        }
        return 0;
    }

    const double margin = std::max(1e-5, 0.001 * std::max(std::fabs(a), std::fabs(b)));
    if (a + margin < b) {
        return -1;
    }
    if (b + margin < a) {
        return 1;
    }
    return 0;
}

struct selector_best_objectives {
    bool ok = false;
    double ln_ratio = 0.0;
    double mean_kld = 0.0;
    double kld_p95 = 0.0;
    double kld_p99 = 0.0;
    double kld_p999 = 0.0;
    double kld_tail_mean = 0.0;
    double max_kld = 0.0;
    double rms_dp = 0.0;
    double same_top = 0.0;
    double entropy_rmse = 0.0;
    double top_prob_rmse = 0.0;
    double top_flip_weight = 0.0;
};

static selector_best_objectives selector_best_objectives_for(
        const selector_policy & policy,
        double holdout_weight) {
    selector_best_objectives obj;
    if (!policy.measured.ok) {
        return obj;
    }

    obj.ok = true;
    obj.ln_ratio = policy.measured.ln_ratio;
    obj.mean_kld = policy.measured.mean_kld;
    obj.kld_p95 = policy.measured.kld_p95;
    obj.kld_p99 = policy.measured.kld_p99;
    obj.kld_p999 = policy.measured.kld_p999;
    obj.kld_tail_mean = policy.measured.kld_tail_mean;
    obj.max_kld = policy.measured.max_kld;
    obj.rms_dp = policy.measured.rms_dp;
    obj.same_top = policy.measured.same_top;
    obj.entropy_rmse = policy.measured.entropy_rmse;
    obj.top_prob_rmse = policy.measured.top_prob_rmse;
    obj.top_flip_weight = policy.measured.top_flip_weight;

    if (policy.has_holdout && policy.measured_holdout.ok) {
        const double w = std::clamp(holdout_weight, 0.0, 0.95);
        auto mix_lower = [w](double main_v, double holdout_v) {
            return (1.0 - w) * main_v + w * holdout_v;
        };
        obj.ln_ratio = mix_lower(obj.ln_ratio, policy.measured_holdout.ln_ratio);
        obj.mean_kld = mix_lower(obj.mean_kld, policy.measured_holdout.mean_kld);
        obj.kld_p95 = mix_lower(obj.kld_p95, policy.measured_holdout.kld_p95);
        obj.kld_p99 = mix_lower(obj.kld_p99, policy.measured_holdout.kld_p99);
        obj.kld_p999 = mix_lower(obj.kld_p999, policy.measured_holdout.kld_p999);
        obj.kld_tail_mean = mix_lower(obj.kld_tail_mean, policy.measured_holdout.kld_tail_mean);
        obj.max_kld = mix_lower(obj.max_kld, policy.measured_holdout.max_kld);
        obj.rms_dp = mix_lower(obj.rms_dp, policy.measured_holdout.rms_dp);
        obj.same_top = mix_lower(obj.same_top, policy.measured_holdout.same_top);
        obj.entropy_rmse = mix_lower(obj.entropy_rmse, policy.measured_holdout.entropy_rmse);
        obj.top_prob_rmse = mix_lower(obj.top_prob_rmse, policy.measured_holdout.top_prob_rmse);
        obj.top_flip_weight = mix_lower(obj.top_flip_weight, policy.measured_holdout.top_flip_weight);
    }

    return obj;
}

static bool selector_best_lower_no_worse(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    return a <= b + margin;
}

static bool selector_best_lower_better(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    return a + margin < b;
}

static bool selector_best_higher_no_worse(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    return a + margin >= b;
}

static bool selector_best_higher_better(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    return a > b + margin;
}

static bool selector_rsf_holdout_guard(
        const selector_policy & policy,
        const selector_policy & base,
        const selector_rank_config & rank_cfg,
        std::string & reason) {
    const bool is_rsf =
        nvfp4_cfg_has_rsf(policy.cfg) ||
        (policy.name.size() >= 4 && policy.name.compare(policy.name.size() - 4, 4, "_rsf") == 0);
    if (!is_rsf || !policy.measured.ok || !base.measured.ok) {
        return true;
    }

    auto score_metrics = [&](const selector_derived_metrics & dm) {
        return selector_score_metrics(
            dm, nullptr, rank_cfg, selector_metric_mode::POLICY).score;
    };
    auto score_margin = [](double a, double b, double abs_margin, double rel_margin) {
        return std::max(std::max(0.0, abs_margin),
            std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    };
    auto score_better = [&](double a, double b, double abs_margin, double rel_margin) {
        const double margin = score_margin(a, b, abs_margin, rel_margin);
        return std::isfinite(a) && std::isfinite(b) && a + margin < b;
    };
    auto score_no_worse = [&](double a, double b, double abs_margin, double rel_margin) {
        const double margin = score_margin(a, b, abs_margin, rel_margin);
        return std::isfinite(a) && std::isfinite(b) && a <= b + margin;
    };
    auto metric_tol = [](double base_v, double abs_floor, double rel_floor) {
        return std::max(std::max(0.0, abs_floor),
            std::max(0.0, rel_floor) * std::max(std::fabs(base_v), 1e-12));
    };
    auto lower_gain = [&](double value, double base_v, double abs_floor, double rel_floor) {
        const double gain = base_v - value;
        if (!(gain > 0.0) || !std::isfinite(gain)) {
            return 0.0;
        }
        return gain / metric_tol(base_v, abs_floor, rel_floor);
    };
    auto higher_gain = [&](double value, double base_v, double abs_floor, double rel_floor) {
        const double gain = value - base_v;
        if (!(gain > 0.0) || !std::isfinite(gain)) {
            return 0.0;
        }
        return gain / metric_tol(base_v, abs_floor, rel_floor);
    };
    auto lower_excess = [&](double value, double base_v, double abs_floor, double rel_floor) {
        const double excess = value - base_v - metric_tol(base_v, abs_floor, rel_floor);
        if (!(excess > 0.0) || !std::isfinite(excess)) {
            return 0.0;
        }
        return excess / std::max(std::fabs(base_v), 1e-12);
    };
    auto higher_excess = [&](double value, double base_v, double abs_floor, double rel_floor) {
        const double excess = base_v - value - metric_tol(base_v, abs_floor, rel_floor);
        if (!(excess > 0.0) || !std::isfinite(excess)) {
            return 0.0;
        }
        return excess / std::max(std::fabs(base_v), 1e-12);
    };
    auto utility_gain = [&](const selector_derived_metrics & dm, const selector_derived_metrics & bm) {
        return
            3.25 * lower_gain(dm.mean_kld, bm.mean_kld, rank_cfg.mean_kld_abs_floor, 0.0) +
            0.85 * lower_gain(dm.kld_p95, bm.kld_p95, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin) +
            2.10 * lower_gain(dm.kld_p99, bm.kld_p99, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin) +
            1.45 * lower_gain(dm.kld_p999, bm.kld_p999, rank_cfg.p999_abs_margin, rank_cfg.p999_rel_margin) +
            1.75 * lower_gain(dm.kld_tail_mean, bm.kld_tail_mean, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin) +
            1.25 * lower_gain(dm.rms_dp, bm.rms_dp, rank_cfg.rms_dp_abs_floor, 0.0) +
            2.20 * higher_gain(dm.same_top, bm.same_top, rank_cfg.same_top_abs_floor, 0.0) +
            2.70 * lower_gain(dm.top_flip_weight, bm.top_flip_weight, rank_cfg.top_flip_weight_abs_floor, 0.0) +
            1.35 * lower_gain(dm.top_prob_rmse, bm.top_prob_rmse, rank_cfg.top_prob_rmse_abs_floor, 0.0) +
            0.75 * lower_gain(dm.entropy_rmse, bm.entropy_rmse, rank_cfg.entropy_rmse_abs_floor, 0.0) +
            0.30 * lower_gain(dm.max_kld, bm.max_kld, rank_cfg.max_kld_abs_margin, rank_cfg.max_kld_rel_margin) +
            0.06 * lower_gain(dm.ln_ratio, bm.ln_ratio, rank_cfg.ln_ratio_abs_floor, 0.0);
    };
    auto broad_regression = [&](const selector_derived_metrics & dm, const selector_derived_metrics & bm, bool validation) {
        const double mean_abs = validation ? 0.0030 : 0.0040;
        const double mean_rel = validation ? 0.0500 : 0.0700;
        const double p99_abs = validation ? 0.0800 : 0.1000;
        const double p99_rel = validation ? 0.1800 : 0.2400;
        const double p999_abs = validation ? 0.4500 : 0.5500;
        const double p999_rel = validation ? 0.3000 : 0.4000;
        const double tail_abs = validation ? 0.1800 : 0.2200;
        const double tail_rel = validation ? 0.2400 : 0.3000;
        const double max_abs = validation ? 1.2500 : 1.5000;
        const double max_rel = validation ? 0.7500 : 1.0000;
        const double rms_abs = validation ? 0.0075 : 0.0100;
        const double rms_rel = validation ? 0.0800 : 0.1200;
        const double top_abs = validation ? 0.0100 : 0.0120;
        const double top_rel = validation ? 0.0120 : 0.0160;
        const double flip_abs = validation ? 0.0025 : 0.0035;
        const double flip_rel = validation ? 0.1200 : 0.1800;
        const double prob_abs = validation ? 0.0075 : 0.0100;
        const double prob_rel = validation ? 0.0800 : 0.1200;
        const double ent_abs = validation ? 0.0350 : 0.0450;
        const double ent_rel = validation ? 0.0800 : 0.1200;
        const double ln_abs = validation ? 0.0300 : 0.0400;
        const double ln_rel = validation ? 0.1800 : 0.2400;

        return
            4.00 * lower_excess(dm.mean_kld, bm.mean_kld, mean_abs, mean_rel) +
            0.90 * lower_excess(dm.kld_p95, bm.kld_p95, p99_abs, p99_rel) +
            2.25 * lower_excess(dm.kld_p99, bm.kld_p99, p99_abs, p99_rel) +
            1.35 * lower_excess(dm.kld_p999, bm.kld_p999, p999_abs, p999_rel) +
            1.80 * lower_excess(dm.kld_tail_mean, bm.kld_tail_mean, tail_abs, tail_rel) +
            1.20 * lower_excess(dm.rms_dp, bm.rms_dp, rms_abs, rms_rel) +
            2.20 * higher_excess(dm.same_top, bm.same_top, top_abs, top_rel) +
            2.50 * lower_excess(dm.top_flip_weight, bm.top_flip_weight, flip_abs, flip_rel) +
            1.20 * lower_excess(dm.top_prob_rmse, bm.top_prob_rmse, prob_abs, prob_rel) +
            0.75 * lower_excess(dm.entropy_rmse, bm.entropy_rmse, ent_abs, ent_rel) +
            0.20 * lower_excess(dm.max_kld, bm.max_kld, max_abs, max_rel) +
            0.15 * lower_excess(dm.ln_ratio, bm.ln_ratio, ln_abs, ln_rel);
    };
    auto format_guard = [&](double search_delta, double combined_delta, double search_gain, double search_regression,
                            double holdout_delta, double holdout_gain, double holdout_regression) {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "score_delta{search=%.6f combined=%.6f validation=%.6f} utility{search_gain=%.3f search_reg=%.3f validation_gain=%.3f validation_reg=%.3f}",
            search_delta,
            combined_delta,
            holdout_delta,
            search_gain,
            search_regression,
            holdout_gain,
            holdout_regression);
        return std::string(buf);
    };

    const double search_score = score_metrics(policy.measured);
    const double base_search_score = score_metrics(base.measured);
    const bool have_holdout =
        policy.has_holdout && policy.measured_holdout.ok &&
        base.has_holdout && base.measured_holdout.ok;
    const double combined_score =
        selector_measured_score(
            policy.measured,
            have_holdout ? &policy.measured_holdout : nullptr,
            rank_cfg);
    const double base_combined_score =
        selector_measured_score(
            base.measured,
            have_holdout ? &base.measured_holdout : nullptr,
            rank_cfg);
    const double search_gain = utility_gain(policy.measured, base.measured);
    const double search_regression = broad_regression(policy.measured, base.measured, false);
    const double search_delta = search_score - base_search_score;
    const double combined_delta = combined_score - base_combined_score;
    const bool search_improved =
        score_better(search_score, base_search_score, 1e-6, 1e-6) ||
        selector_compare_one(policy.measured, base.measured, rank_cfg) < 0;
    const bool search_utility_improved = search_gain > search_regression * 0.65 + 0.25;
    const bool combined_improved =
        score_better(
            combined_score,
            base_combined_score,
            SELECTOR_RSF_COMBINED_SCORE_ABS_GAIN,
            SELECTOR_RSF_COMBINED_SCORE_REL_GAIN);
    const bool search_not_catastrophic =
        score_no_worse(
            search_score,
            base_search_score,
            SELECTOR_RSF_SEARCH_SCORE_ABS_MARGIN,
            SELECTOR_RSF_SEARCH_SCORE_REL_MARGIN) ||
        search_regression <= 2.0 ||
        search_utility_improved;

    double holdout_score = std::numeric_limits<double>::quiet_NaN();
    double base_holdout_score = std::numeric_limits<double>::quiet_NaN();
    double holdout_gain = 0.0;
    double holdout_regression = 0.0;
    double holdout_delta = std::numeric_limits<double>::quiet_NaN();
    bool holdout_improved = false;
    bool holdout_utility_improved = false;
    bool holdout_not_catastrophic = true;
    if (have_holdout) {
        holdout_score = score_metrics(policy.measured_holdout);
        base_holdout_score = score_metrics(base.measured_holdout);
        holdout_delta = holdout_score - base_holdout_score;
        holdout_gain = utility_gain(policy.measured_holdout, base.measured_holdout);
        holdout_regression = broad_regression(policy.measured_holdout, base.measured_holdout, true);
        holdout_improved =
            score_better(holdout_score, base_holdout_score, 0.08, 0.01) ||
            selector_compare_one(policy.measured_holdout, base.measured_holdout, rank_cfg) < 0;
        holdout_utility_improved = holdout_gain > holdout_regression * 0.60 + 0.20;
        holdout_not_catastrophic =
            score_no_worse(holdout_score, base_holdout_score, 0.60, 0.08) ||
            holdout_regression <= 1.25 ||
            holdout_utility_improved;
    }
    const std::string guard_summary = format_guard(
        search_delta,
        combined_delta,
        search_gain,
        search_regression,
        holdout_delta,
        holdout_gain,
        holdout_regression);

    if (!search_improved && !search_utility_improved && !combined_improved) {
        reason = "search and combined utility did not improve; " + guard_summary;
        return false;
    }
    if (!search_improved && !search_not_catastrophic) {
        reason = "search regression exceeded RSF budget; " + guard_summary;
        return false;
    }

    if (have_holdout) {
        if (!holdout_not_catastrophic) {
            reason = "validation holdout regression exceeded RSF budget; " + guard_summary;
            return false;
        }
        if (!search_improved &&
                !search_utility_improved &&
                !holdout_improved &&
                !holdout_utility_improved &&
                !combined_improved) {
            reason = "search regression was not offset by validation utility; " + guard_summary;
            return false;
        }
    } else if (!search_improved && !search_utility_improved) {
        reason = "search utility did not improve and no validation holdout is available; " + guard_summary;
        return false;
    }

    reason = guard_summary;
    return true;
}

static bool selector_best_dominates(
        const selector_policy & a,
        const selector_policy & b,
        const selector_rank_config & rank_cfg) {
    const auto ao = selector_best_objectives_for(a, rank_cfg.holdout_weight);
    const auto bo = selector_best_objectives_for(b, rank_cfg.holdout_weight);
    if (!ao.ok || !bo.ok) {
        return false;
    }

    bool strictly_better = false;
    auto lower = [&](double av, double bv, double abs_margin, double rel_margin) {
        if (!selector_best_lower_no_worse(av, bv, abs_margin, rel_margin)) {
            return false;
        }
        strictly_better = strictly_better || selector_best_lower_better(av, bv, abs_margin, rel_margin);
        return true;
    };
    auto higher = [&](double av, double bv, double abs_margin, double rel_margin) {
        if (!selector_best_higher_no_worse(av, bv, abs_margin, rel_margin)) {
            return false;
        }
        strictly_better = strictly_better || selector_best_higher_better(av, bv, abs_margin, rel_margin);
        return true;
    };

    return
        lower(ao.ln_ratio, bo.ln_ratio, rank_cfg.ln_ratio_abs_floor, 0.0) &&
        lower(ao.mean_kld, bo.mean_kld, rank_cfg.mean_kld_abs_floor, 0.0) &&
        lower(ao.kld_p95, bo.kld_p95, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin) &&
        lower(ao.kld_p99, bo.kld_p99, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin) &&
        lower(ao.kld_p999, bo.kld_p999, rank_cfg.p999_abs_margin, rank_cfg.p999_rel_margin) &&
        lower(ao.kld_tail_mean, bo.kld_tail_mean, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin) &&
        lower(ao.max_kld, bo.max_kld, rank_cfg.max_kld_abs_margin, rank_cfg.max_kld_rel_margin) &&
        lower(ao.rms_dp, bo.rms_dp, rank_cfg.rms_dp_abs_floor, 0.0) &&
        lower(ao.entropy_rmse, bo.entropy_rmse, rank_cfg.entropy_rmse_abs_floor, 0.0) &&
        lower(ao.top_prob_rmse, bo.top_prob_rmse, rank_cfg.top_prob_rmse_abs_floor, 0.0) &&
        lower(ao.top_flip_weight, bo.top_flip_weight, rank_cfg.top_flip_weight_abs_floor, 0.0) &&
        higher(ao.same_top, bo.same_top, rank_cfg.same_top_abs_floor, 0.0) &&
        strictly_better;
}

static std::vector<size_t> selector_best_set(
        const std::vector<selector_policy> & policies,
        const std::vector<size_t> & candidates,
        const selector_rank_config & rank_cfg) {
    std::vector<size_t> best_set;
    best_set.reserve(candidates.size());

    for (size_t idx : candidates) {
        if (idx >= policies.size() || !std::isfinite(policies[idx].measured_score) || !policies[idx].measured.ok) {
            continue;
        }
        bool dominated = false;
        for (size_t other : candidates) {
            if (other == idx || other >= policies.size()) {
                continue;
            }
            if (selector_best_dominates(policies[other], policies[idx], rank_cfg)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            best_set.push_back(idx);
        }
    }

    return best_set;
}

struct selector_best_set_stats {
    bool ok = false;
    double min_ln_ratio = std::numeric_limits<double>::infinity();
    double max_ln_ratio = -std::numeric_limits<double>::infinity();
    double min_mean_kld = std::numeric_limits<double>::infinity();
    double max_mean_kld = -std::numeric_limits<double>::infinity();
    double min_kld_p95 = std::numeric_limits<double>::infinity();
    double max_kld_p95 = -std::numeric_limits<double>::infinity();
    double min_kld_p99 = std::numeric_limits<double>::infinity();
    double max_kld_p99 = -std::numeric_limits<double>::infinity();
    double min_kld_p999 = std::numeric_limits<double>::infinity();
    double max_kld_p999 = -std::numeric_limits<double>::infinity();
    double min_kld_tail_mean = std::numeric_limits<double>::infinity();
    double max_kld_tail_mean = -std::numeric_limits<double>::infinity();
    double min_max_kld = std::numeric_limits<double>::infinity();
    double max_max_kld = -std::numeric_limits<double>::infinity();
    double min_rms_dp = std::numeric_limits<double>::infinity();
    double max_rms_dp = -std::numeric_limits<double>::infinity();
    double min_same_top = std::numeric_limits<double>::infinity();
    double max_same_top = -std::numeric_limits<double>::infinity();
    double min_entropy_rmse = std::numeric_limits<double>::infinity();
    double max_entropy_rmse = -std::numeric_limits<double>::infinity();
    double min_top_prob_rmse = std::numeric_limits<double>::infinity();
    double max_top_prob_rmse = -std::numeric_limits<double>::infinity();
    double min_top_flip_weight = std::numeric_limits<double>::infinity();
    double max_top_flip_weight = -std::numeric_limits<double>::infinity();
};

static void selector_best_set_stats_add(
        selector_best_set_stats & stats,
        const selector_best_objectives & obj) {
    if (!obj.ok ||
            !std::isfinite(obj.ln_ratio) ||
            !std::isfinite(obj.mean_kld) ||
            !std::isfinite(obj.kld_p95) ||
            !std::isfinite(obj.kld_p99) ||
            !std::isfinite(obj.kld_p999) ||
            !std::isfinite(obj.kld_tail_mean) ||
            !std::isfinite(obj.max_kld) ||
            !std::isfinite(obj.rms_dp) ||
            !std::isfinite(obj.same_top) ||
            !std::isfinite(obj.entropy_rmse) ||
            !std::isfinite(obj.top_prob_rmse) ||
            !std::isfinite(obj.top_flip_weight)) {
        return;
    }
    stats.ok = true;
    stats.min_ln_ratio = std::min(stats.min_ln_ratio, obj.ln_ratio);
    stats.max_ln_ratio = std::max(stats.max_ln_ratio, obj.ln_ratio);
    stats.min_mean_kld = std::min(stats.min_mean_kld, obj.mean_kld);
    stats.max_mean_kld = std::max(stats.max_mean_kld, obj.mean_kld);
    stats.min_kld_p95 = std::min(stats.min_kld_p95, obj.kld_p95);
    stats.max_kld_p95 = std::max(stats.max_kld_p95, obj.kld_p95);
    stats.min_kld_p99 = std::min(stats.min_kld_p99, obj.kld_p99);
    stats.max_kld_p99 = std::max(stats.max_kld_p99, obj.kld_p99);
    stats.min_kld_p999 = std::min(stats.min_kld_p999, obj.kld_p999);
    stats.max_kld_p999 = std::max(stats.max_kld_p999, obj.kld_p999);
    stats.min_kld_tail_mean = std::min(stats.min_kld_tail_mean, obj.kld_tail_mean);
    stats.max_kld_tail_mean = std::max(stats.max_kld_tail_mean, obj.kld_tail_mean);
    stats.min_max_kld = std::min(stats.min_max_kld, obj.max_kld);
    stats.max_max_kld = std::max(stats.max_max_kld, obj.max_kld);
    stats.min_rms_dp = std::min(stats.min_rms_dp, obj.rms_dp);
    stats.max_rms_dp = std::max(stats.max_rms_dp, obj.rms_dp);
    stats.min_same_top = std::min(stats.min_same_top, obj.same_top);
    stats.max_same_top = std::max(stats.max_same_top, obj.same_top);
    stats.min_entropy_rmse = std::min(stats.min_entropy_rmse, obj.entropy_rmse);
    stats.max_entropy_rmse = std::max(stats.max_entropy_rmse, obj.entropy_rmse);
    stats.min_top_prob_rmse = std::min(stats.min_top_prob_rmse, obj.top_prob_rmse);
    stats.max_top_prob_rmse = std::max(stats.max_top_prob_rmse, obj.top_prob_rmse);
    stats.min_top_flip_weight = std::min(stats.min_top_flip_weight, obj.top_flip_weight);
    stats.max_top_flip_weight = std::max(stats.max_top_flip_weight, obj.top_flip_weight);
}

static selector_best_set_stats selector_best_set_stats_for(
        const std::vector<selector_policy> & policies,
        const std::vector<size_t> & candidates,
        const selector_rank_config & rank_cfg) {
    selector_best_set_stats stats;
    for (size_t idx : candidates) {
        if (idx >= policies.size()) {
            continue;
        }
        selector_best_set_stats_add(
            stats,
            selector_best_objectives_for(policies[idx], rank_cfg.holdout_weight));
    }
    return stats;
}

static double selector_range_floor(double best, double worst, double abs_floor, double rel_floor) {
    return std::max({
        std::fabs(worst - best),
        std::max(0.0, abs_floor),
        std::max(0.0, rel_floor) * std::max(std::fabs(best), 1e-12),
        1e-12,
    });
}

static double selector_norm_lower(double value, double best, double worst, double abs_floor, double rel_floor) {
    if (!std::isfinite(value) || !std::isfinite(best) || !std::isfinite(worst)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, value - best) / selector_range_floor(best, worst, abs_floor, rel_floor);
}

static double selector_norm_higher(double value, double best, double worst, double abs_floor, double rel_floor) {
    if (!std::isfinite(value) || !std::isfinite(best) || !std::isfinite(worst)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, best - value) / selector_range_floor(best, worst, abs_floor, rel_floor);
}

static double selector_best_set_utility(
        const selector_policy & policy,
        const selector_best_set_stats & stats,
        const selector_rank_config & rank_cfg) {
    if (!stats.ok || !policy.measured.ok) {
        return std::numeric_limits<double>::infinity();
    }
    const auto obj = selector_best_objectives_for(policy, rank_cfg.holdout_weight);
    if (!obj.ok) {
        return std::numeric_limits<double>::infinity();
    }

    const double ppl = selector_norm_lower(
        obj.ln_ratio, stats.min_ln_ratio, stats.max_ln_ratio,
        rank_cfg.ln_ratio_abs_floor, 0.0);
    const double mean = selector_norm_lower(
        obj.mean_kld, stats.min_mean_kld, stats.max_mean_kld,
        rank_cfg.mean_kld_abs_floor, 0.0);
    const double p95 = selector_norm_lower(
        obj.kld_p95, stats.min_kld_p95, stats.max_kld_p95,
        rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    const double p99 = selector_norm_lower(
        obj.kld_p99, stats.min_kld_p99, stats.max_kld_p99,
        rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    const double p999 = selector_norm_lower(
        obj.kld_p999, stats.min_kld_p999, stats.max_kld_p999,
        rank_cfg.p999_abs_margin, rank_cfg.p999_rel_margin);
    const double tail_mean = selector_norm_lower(
        obj.kld_tail_mean, stats.min_kld_tail_mean, stats.max_kld_tail_mean,
        rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    const double max_kld = selector_norm_lower(
        obj.max_kld, stats.min_max_kld, stats.max_max_kld,
        rank_cfg.max_kld_abs_margin, rank_cfg.max_kld_rel_margin);
    const double rms = selector_norm_lower(
        obj.rms_dp, stats.min_rms_dp, stats.max_rms_dp,
        rank_cfg.rms_dp_abs_floor, 0.0);
    const double same_top = selector_norm_higher(
        obj.same_top, stats.max_same_top, stats.min_same_top,
        rank_cfg.same_top_abs_floor, 0.0);
    const double entropy = selector_norm_lower(
        obj.entropy_rmse, stats.min_entropy_rmse, stats.max_entropy_rmse,
        rank_cfg.entropy_rmse_abs_floor, 0.0);
    const double top_prob = selector_norm_lower(
        obj.top_prob_rmse, stats.min_top_prob_rmse, stats.max_top_prob_rmse,
        rank_cfg.top_prob_rmse_abs_floor, 0.0);
    const double top_flip = selector_norm_lower(
        obj.top_flip_weight, stats.min_top_flip_weight, stats.max_top_flip_weight,
        rank_cfg.top_flip_weight_abs_floor, 0.0);

    const double weighted_sum =
        3.25 * mean +
        0.85 * p95 +
        2.10 * p99 +
        1.45 * p999 +
        1.75 * tail_mean +
        1.25 * rms +
        2.20 * same_top +
        2.70 * top_flip +
        1.35 * top_prob +
        0.75 * entropy +
        0.30 * max_kld +
        0.06 * ppl;

    const double worst_kld_axis = std::max({
        3.25 * mean,
        0.85 * p95,
        2.10 * p99,
        1.45 * p999,
        1.75 * tail_mean,
        1.25 * rms,
        2.20 * same_top,
        2.70 * top_flip,
    });

    // Distinguish "one bad token but good average" from a broader tail failure.
    // A max-only spike is a small penalty; p99/p999 degradation relative to mean
    // indicates many tokens moved and should dominate the final utility.
    const double tail_shape_penalty =
        0.30 * std::max(0.0, p95 - mean) +
        0.60 * std::max(0.0, p99 - mean) +
        0.45 * std::max(0.0, p999 - mean) +
        0.55 * std::max(0.0, tail_mean - mean) +
        0.08 * std::max(0.0, max_kld - p999);
    const double behavior_shape_penalty =
        0.95 * std::max(0.0, top_flip - same_top) +
        0.20 * std::max(0.0, entropy - mean);

    return 0.62 * weighted_sum + 0.38 * worst_kld_axis + tail_shape_penalty + behavior_shape_penalty;
}

static int selector_compare_policy_measured(
    const selector_policy & a,
    const selector_policy & b,
    const selector_derived_metrics & baseline_eval,
    const selector_derived_metrics * baseline_holdout_eval,
    const selector_rank_config & rank_cfg) {
    const selector_metric_rank a_rank =
        selector_rank_policy_metrics(
            a.measured,
            a.has_holdout ? &a.measured_holdout : nullptr,
            baseline_eval,
            baseline_holdout_eval,
            rank_cfg);
    const selector_metric_rank b_rank =
        selector_rank_policy_metrics(
            b.measured,
            b.has_holdout ? &b.measured_holdout : nullptr,
            baseline_eval,
            baseline_holdout_eval,
            rank_cfg);
    const bool a_pass = a_rank.pass;
    const bool b_pass = b_rank.pass;

    if (a_pass != b_pass) {
        return a_pass ? -1 : 1;
    }

    const int cmp_main = selector_compare_one(a.measured, b.measured, rank_cfg);
    if (cmp_main != 0) {
        if (baseline_holdout_eval != nullptr && a.has_holdout && b.has_holdout) {
            const int cmp_holdout = selector_compare_one(a.measured_holdout, b.measured_holdout, rank_cfg);
            if (cmp_holdout != 0 && cmp_holdout != cmp_main) {
                return cmp_holdout;
            }
        }
        return cmp_main;
    }

    if (baseline_holdout_eval != nullptr && a.has_holdout && b.has_holdout) {
        const int cmp_holdout = selector_compare_one(a.measured_holdout, b.measured_holdout, rank_cfg);
        if (cmp_holdout != 0) {
            return cmp_holdout;
        }
    }

    const int score_cmp = selector_compare_measured_score(a.measured_score, b.measured_score);
    if (score_cmp != 0) {
        return score_cmp;
    }

    if (a.proxy_score != b.proxy_score) {
        return a.proxy_score < b.proxy_score ? -1 : 1;
    }
    if (a.name != b.name) {
        return a.name < b.name ? -1 : 1;
    }
    return 0;
}

static int32_t selector_parse_layer(const std::string & name) {
    if (name.rfind("blk.", 0) != 0) {
        return -1;
    }
    size_t pos = 4;
    int32_t layer = 0;
    bool have_digit = false;
    while (pos < name.size() && std::isdigit((unsigned char) name[pos])) {
        have_digit = true;
        layer = layer * 10 + (name[pos] - '0');
        ++pos;
    }
    return have_digit ? layer : -1;
}

static selector_tensor_class selector_classify_tensor(const std::string & name) {
    if (name == "token_embd.weight" || name == "output.weight" || name == "token_embd_norm.weight" || name == "output_norm.weight") {
        return selector_tensor_class::EMBEDDING_OUTPUT;
    }
    if (name.find(".ffn_gate_inp") != std::string::npos || name.find(".altup_router") != std::string::npos || name.find(".router") != std::string::npos) {
        return selector_tensor_class::ROUTER;
    }
    if (name.find(".attn_gate.weight") != std::string::npos) {
        return selector_tensor_class::ROUTER;
    }
    if (name.find(".attn_output.weight") != std::string::npos || name.find(".wo.weight") != std::string::npos) {
        return selector_tensor_class::ATTN_OUT;
    }
    if (name.find(".attn_q.weight") != std::string::npos ||
        name.find(".attn_k.weight") != std::string::npos ||
        name.find(".attn_v.weight") != std::string::npos ||
        name.find(".attn_qkv.weight") != std::string::npos) {
        return selector_tensor_class::ATTN_QKV;
    }
    if (name.find(".ffn_up_exps.weight") != std::string::npos ||
        name.find(".ffn_gate.weight") != std::string::npos ||
        name.find(".ffn_up.weight") != std::string::npos) {
        return selector_tensor_class::EXPERT_UP_GATE;
    }
    if (name.find(".ffn_down_exps.weight") != std::string::npos) {
        return selector_tensor_class::EXPERT_DOWN;
    }
    if (name.find(".ffn_down.weight") != std::string::npos) {
        return selector_tensor_class::DENSE_MLP;
    }
    if (name.find(".ssm_out.weight") != std::string::npos ||
        name.find(".ssm_in.weight") != std::string::npos ||
        name.find(".ssm_alpha.weight") != std::string::npos ||
        name.find(".ssm_beta.weight") != std::string::npos ||
        name.find(".ssm_conv1d.weight") != std::string::npos ||
        name.find(".ssm_") != std::string::npos) {
        return selector_tensor_class::SSM;
    }
    return selector_tensor_class::OTHER;
}

static const char * selector_tensor_class_name(selector_tensor_class cls) {
    switch (cls) {
        case selector_tensor_class::EMBEDDING_OUTPUT: return "embedding_output";
        case selector_tensor_class::ATTN_QKV:         return "attn_qkv";
        case selector_tensor_class::ATTN_OUT:         return "attn_out";
        case selector_tensor_class::ROUTER:           return "router";
        case selector_tensor_class::EXPERT_UP_GATE:   return "expert_up_gate";
        case selector_tensor_class::EXPERT_DOWN:      return "expert_down";
        case selector_tensor_class::DENSE_MLP:        return "dense_mlp";
        case selector_tensor_class::SSM:              return "ssm";
        case selector_tensor_class::OTHER:            return "other";
    }
    return "other";
}

static std::string selector_tensor_policy_unit_key(const std::string & name, const std::string & mode) {
    if (striequals(mode.c_str(), "tensor") ||
            striequals(mode.c_str(), "off") ||
            striequals(mode.c_str(), "none")) {
        return name;
    }

    const int32_t layer = selector_parse_layer(name);
    if (layer >= 0) {
        const std::string prefix = "blk." + std::to_string(layer);
        if (name.find(".attn_q.weight") != std::string::npos ||
                name.find(".attn_k.weight") != std::string::npos ||
                name.find(".attn_v.weight") != std::string::npos) {
            return prefix + ".attn_qkv";
        }
        if (name.find(".ffn_gate.weight") != std::string::npos ||
                name.find(".ffn_up.weight") != std::string::npos) {
            return prefix + ".ffn_gate_up";
        }
        if (name.find(".ffn_gate_exps.weight") != std::string::npos ||
                name.find(".ffn_up_exps.weight") != std::string::npos) {
            return prefix + ".ffn_gate_up_exps";
        }
    }
    if (name == "token_embd.weight" || name == "output.weight") {
        return "embedding_output";
    }
    return name;
}

static std::string selector_rsf_hist_class(const std::string & name, selector_tensor_class cls) {
    if (name.find("mtp") != std::string::npos || name.find("nextn") != std::string::npos) {
        return "MTP/NextN";
    }
    if (name.find(".attn_q.weight") != std::string::npos) {
        return "attn_q";
    }
    if (name.find(".attn_k.weight") != std::string::npos) {
        return "attn_k";
    }
    if (name.find(".attn_v.weight") != std::string::npos) {
        return "attn_v";
    }
    if (name.find(".attn_output.weight") != std::string::npos || name.find(".wo.weight") != std::string::npos) {
        return "attn_output";
    }
    if (name.find("_exps.weight") != std::string::npos) {
        return "experts";
    }
    if (name.find(".ffn_gate.weight") != std::string::npos) {
        return "ffn_gate";
    }
    if (name.find(".ffn_up.weight") != std::string::npos) {
        return "ffn_up";
    }
    if (name.find(".ffn_down.weight") != std::string::npos) {
        return "ffn_down";
    }
    return selector_tensor_class_name(cls);
}

static double selector_percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    q = std::clamp(q, 0.0, 1.0);
    const double pos = q * (double) (values.size() - 1);
    const size_t lo = (size_t) std::floor(pos);
    const size_t hi = std::min(values.size() - 1, lo + 1);
    const double frac = pos - (double) lo;
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

static double selector_rsf_record_scale_mul(const selector_rsf_tensor_record & r) {
    if (r.scale_muls.empty()) {
        if (r.slices > 0 && std::isfinite(r.scale_mul_sum) && r.scale_mul_sum > 0.0) {
            return r.scale_mul_sum / (double) r.slices;
        }
        return 1.0;
    }
    double sum = 0.0;
    for (float v : r.scale_muls) {
        if (std::isfinite(v) && v > 0.0f) {
            sum += v;
        }
    }
    return sum / (double) std::max<size_t>(1, r.scale_muls.size());
}

static std::unordered_map<std::string, const selector_rsf_tensor_record *>
selector_rsf_record_map(const selector_policy & policy) {
    std::unordered_map<std::string, const selector_rsf_tensor_record *> out;
    out.reserve(policy.tensor_records.size());
    for (const auto & r : policy.tensor_records) {
        out.emplace(r.tensor, &r);
    }
    return out;
}

static int selector_layer_bucket(int32_t layer, int32_t n_layer) {
    if (layer < 0 || n_layer <= 0) {
        return -1;
    }
    if (n_layer <= 3) {
        return std::clamp(layer, 0, n_layer - 1);
    }
    if (layer < n_layer / 3) {
        return 0;
    }
    if (layer < (2 * n_layer) / 3) {
        return 1;
    }
    return 2;
}

static int32_t selector_detect_n_layer(const std::vector<std::pair<std::string, ggml_tensor *>> & tmap) {
    int32_t max_layer = -1;
    for (const auto & kv : tmap) {
        max_layer = std::max(max_layer, selector_parse_layer(kv.first));
    }
    return max_layer + 1;
}

static std::string selector_regex_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() * 2 + 2);
    out.push_back('^');
    for (char ch : s) {
        switch (ch) {
            case '\\':
            case '^':
            case '$':
            case '.':
            case '|':
            case '?':
            case '*':
            case '+':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
                out.push_back('\\');
                break;
            default:
                break;
        }
        out.push_back(ch);
    }
    out.push_back('$');
    return out;
}

static void selector_attach_exact_scales(
        tensor_type_option & opt,
        const selector_binding & binding) {
    const int64_t n_slices = binding.source != nullptr ? std::max<int64_t>(1, binding.source->ne[2]) : 1;
    if (binding.quant_tensor_scales.size() != (size_t) n_slices ||
            binding.working_quant_tensor_scales.size() != (size_t) n_slices) {
        return;
    }
    opt.nvfp4_base_scale_overrides = binding.quant_tensor_scales;
    opt.nvfp4_scale_overrides = binding.working_quant_tensor_scales;
}

static bool quantize_tensor_type_shape_compatible(const ggml_tensor * tensor, ggml_type type) {
    if (tensor == nullptr || type == GGML_TYPE_COUNT) {
        return false;
    }
    const int64_t ne0 = std::max<int64_t>(1, tensor->ne[0]);
    const int64_t blck = ggml_blck_size(type);
    return blck <= 1 || (ne0 % blck) == 0;
}

static size_t quantize_tensor_nbytes_as_type(const ggml_tensor * tensor, ggml_type type) {
    if (tensor == nullptr || type == GGML_TYPE_COUNT) {
        return 0;
    }
    const int64_t ne0 = std::max<int64_t>(1, tensor->ne[0]);
    if (!quantize_tensor_type_shape_compatible(tensor, type)) {
        return 0;
    }
    const int64_t rows = std::max<int64_t>(1, tensor->ne[1]) * std::max<int64_t>(1, tensor->ne[2]) * std::max<int64_t>(1, tensor->ne[3]);
    return ggml_row_size(type, ne0) * (size_t) rows;
}

static float nvfp4_selector_slice_absmax(const ggml_tensor * tensor, int64_t slice_index) {
    if (tensor == nullptr || tensor->data == nullptr || slice_index < 0 || slice_index >= std::max<int64_t>(1, tensor->ne[2])) {
        return 0.0f;
    }

    const int64_t n_per_row = tensor->ne[0];
    const int64_t nrows = tensor->ne[1];
    const size_t row_size = ggml_row_size(tensor->type, n_per_row);
    const uint8_t * src_slice = (const uint8_t *) tensor->data + (size_t) slice_index * (size_t) nrows * row_size;

    float amax = 0.0f;
    std::vector<float> tmp;

    for (int64_t row = 0; row < nrows; ++row) {
        const uint8_t * row_ptr = src_slice + (size_t) row * row_size;
        if (tensor->type == GGML_TYPE_F32) {
            const float * x = (const float *) row_ptr;
            for (int64_t i = 0; i < n_per_row; ++i) {
                const float a = std::fabs(x[i]);
                if (a > amax && std::isfinite(a)) {
                    amax = a;
                }
            }
        } else if (tensor->type == GGML_TYPE_BF16) {
            const ggml_bf16_t * x = (const ggml_bf16_t *) row_ptr;
            for (int64_t i = 0; i < n_per_row; ++i) {
                const float a = std::fabs(ggml_bf16_to_fp32(x[i]));
                if (a > amax && std::isfinite(a)) {
                    amax = a;
                }
            }
        } else if (tensor->type == GGML_TYPE_F16) {
            const ggml_fp16_t * x = (const ggml_fp16_t *) row_ptr;
            for (int64_t i = 0; i < n_per_row; ++i) {
                const float a = std::fabs(ggml_fp16_to_fp32(x[i]));
                if (a > amax && std::isfinite(a)) {
                    amax = a;
                }
            }
        } else {
            const auto * traits = ggml_get_type_traits(tensor->type);
            if (traits == nullptr || traits->to_float == nullptr) {
                return 0.0f;
            }
            tmp.resize((size_t) n_per_row);
            traits->to_float(row_ptr, tmp.data(), n_per_row);
            for (float v : tmp) {
                const float a = std::fabs(v);
                if (a > amax && std::isfinite(a)) {
                    amax = a;
                }
            }
        }
    }

    return amax;
}

static std::vector<float> nvfp4_selector_quant_tensor_scales(const ggml_tensor * tensor, float correction_denom) {
    std::vector<float> scales;
    if (tensor == nullptr || !(correction_denom > 0.0f) || !std::isfinite(correction_denom)) {
        return scales;
    }

    const int64_t n_slices = std::max<int64_t>(1, tensor->ne[2]);
    scales.resize((size_t) n_slices, 1.0f);
    for (int64_t is = 0; is < n_slices; ++is) {
        const float amax = nvfp4_selector_slice_absmax(tensor, is);
        if (amax > 0.0f && std::isfinite(amax)) {
            const float scale = amax / correction_denom;
            if (scale > 0.0f && std::isfinite(scale)) {
                scales[(size_t) is] = scale;
            }
        }
    }
    return scales;
}

static bool quantize_tensor_copy_out(const ggml_tensor * tensor, void * dst, size_t nbytes) {
    if (tensor == nullptr || dst == nullptr) {
        return false;
    }
    if (tensor->buffer != nullptr) {
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(tensor), dst, 0, nbytes);
        return true;
    }
    if (tensor->data != nullptr) {
        if (ggml_cuda_tensor_get_host(tensor, dst, nbytes, nullptr)) {
            return true;
        }
        memcpy(dst, tensor->data, nbytes);
        return true;
    }
    return false;
}

static bool quantize_tensor_copy_in(ggml_tensor * tensor, const void * src, size_t nbytes) {
    if (tensor == nullptr || src == nullptr) {
        return false;
    }
    if (ggml_cuda_tensor_set_host(tensor, src, nbytes, nullptr)) {
        return true;
    }
    if (tensor->buffer != nullptr) {
        if (ggml_backend_buffer_is_host(tensor->buffer)) {
            return false;
        }
        const char * buffer_name = ggml_backend_buffer_name(tensor->buffer);
        if (buffer_name != nullptr && std::strstr(buffer_name, "CUDA") != nullptr) {
            return false;
        }
        ggml_backend_tensor_set(tensor, const_cast<void *>(src), 0, nbytes);
        return true;
    }
    if (tensor->data != nullptr) {
        if (ggml_cuda_tensor_set_host(tensor, src, nbytes, nullptr)) {
            return true;
        }
    }
    return false;
}

static bool quantize_tensor_host_buffer(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer);
}

static bool quantize_binding_ensure_target_bytes(selector_binding & binding) {
    if (binding.target == nullptr || binding.target_nbytes == 0) {
        return false;
    }
    if (!binding.original_target_device.valid_for(binding.target_nbytes)) {
        (void) binding.original_target_device.capture(binding.target, binding.target_nbytes);
    }
    if (binding.original_target_bytes.size() != binding.target_nbytes) {
        binding.original_target_bytes.resize(binding.target_nbytes);
        if (!quantize_tensor_copy_out(binding.target, binding.original_target_bytes.data(), binding.target_nbytes)) {
            return false;
        }
    }
    if (binding.working_target_bytes.size() != binding.target_nbytes) {
        binding.working_target_bytes.resize(binding.target_nbytes);
    }
    return true;
}

static bool quantize_binding_ensure_scale_bytes(selector_binding & binding) {
    if (binding.target_scale_nbytes == 0) {
        return false;
    }
    if (binding.target_scale == nullptr) {
        if (binding.target == nullptr || binding.target->type != GGML_TYPE_MXFP6_E2M3 ||
                binding.target_nbytes < MXFP6_HEADER_OFFSET ||
                binding.target_scale_nbytes != sizeof(float)) {
            return false;
        }
        if (!quantize_binding_ensure_target_bytes(binding)) {
            return false;
        }
        if (binding.original_scale_bytes.size() != binding.target_scale_nbytes) {
            binding.original_scale_bytes.resize(binding.target_scale_nbytes);
            const auto * header = (const tensor_mxfp6 *) binding.original_target_bytes.data();
            const float weight_scale = header->weight_scale > 0.0f && std::isfinite(header->weight_scale) ? header->weight_scale : 1.0f;
            memcpy(binding.original_scale_bytes.data(), &weight_scale, sizeof(float));
        }
        if (binding.working_scale_bytes.size() != binding.target_scale_nbytes) {
            binding.working_scale_bytes.resize(binding.target_scale_nbytes);
        }
        return true;
    }
    if (binding.original_scale_bytes.size() != binding.target_scale_nbytes) {
        binding.original_scale_bytes.resize(binding.target_scale_nbytes);
        if (!quantize_tensor_copy_out(binding.target_scale, binding.original_scale_bytes.data(), binding.target_scale_nbytes)) {
            return false;
        }
    }
    if (!binding.original_scale_device.valid_for(binding.target_scale_nbytes)) {
        (void) binding.original_scale_device.capture(binding.target_scale, binding.target_scale_nbytes);
    }
    if (binding.working_scale_bytes.size() != binding.target_scale_nbytes) {
        binding.working_scale_bytes.resize(binding.target_scale_nbytes);
    }
    return true;
}

static bool quantize_binding_ensure_input_scale_bytes(selector_binding & binding) {
    if (binding.target_input_scale == nullptr || binding.target_input_scale_nbytes == 0) {
        return false;
    }
    if (binding.original_input_scale_bytes.size() != binding.target_input_scale_nbytes) {
        binding.original_input_scale_bytes.resize(binding.target_input_scale_nbytes);
        if (!quantize_tensor_copy_out(
                binding.target_input_scale,
                binding.original_input_scale_bytes.data(),
                binding.target_input_scale_nbytes)) {
            return false;
        }
    }
    if (!binding.original_input_scale_device.valid_for(binding.target_input_scale_nbytes)) {
        (void) binding.original_input_scale_device.capture(binding.target_input_scale, binding.target_input_scale_nbytes);
    }
    if (binding.working_input_scale_bytes.size() != binding.target_input_scale_nbytes) {
        binding.working_input_scale_bytes.resize(binding.target_input_scale_nbytes);
    }
    return true;
}

static bool nvfp4_selector_prepare_nvfp4_runtime_scales(
        selector_binding & binding,
        int32_t input_scale_policy,
        float * header_weight_scale,
        float * header_input_scale) {
    const int64_t n_slices = binding.source != nullptr ? std::max<int64_t>(1, binding.source->ne[2]) : 1;
    const int64_t n_per_row = binding.source != nullptr ? binding.source->ne[0] : 0;
    const std::vector<float> & tensor_scales =
        binding.working_quant_tensor_scales.size() == (size_t) n_slices
        ? binding.working_quant_tensor_scales
        : binding.quant_tensor_scales;

    std::vector<float> weight_scales((size_t) n_slices, 1.0f);
    for (int64_t is = 0; is < n_slices; ++is) {
        if ((size_t) is < tensor_scales.size() &&
                std::isfinite(tensor_scales[(size_t) is]) &&
                tensor_scales[(size_t) is] > 0.0f) {
            weight_scales[(size_t) is] = tensor_scales[(size_t) is];
        }
    }

    std::vector<float> input_scales((size_t) n_slices, 1.0f);
    for (int64_t is = 0; is < n_slices; ++is) {
        const float * imatrix_slice =
            binding.imatrix_row != nullptr && n_per_row > 0 ? binding.imatrix_row + is * n_per_row : nullptr;
        input_scales[(size_t) is] =
            llama_nvfp4_input_scale_from_imatrix(imatrix_slice, n_per_row, input_scale_policy);
    }

    if (binding.target_scale != nullptr) {
        if (!quantize_binding_ensure_scale_bytes(binding) ||
                binding.working_scale_bytes.size() != binding.target_scale_nbytes ||
                binding.target_scale_nbytes < weight_scales.size() * sizeof(float)) {
            return false;
        }
        memcpy(binding.working_scale_bytes.data(), weight_scales.data(), weight_scales.size() * sizeof(float));
    }
    if (binding.target_input_scale != nullptr) {
        if (!quantize_binding_ensure_input_scale_bytes(binding) ||
                binding.working_input_scale_bytes.size() != binding.target_input_scale_nbytes ||
                binding.target_input_scale_nbytes < input_scales.size() * sizeof(float)) {
            return false;
        }
        memcpy(binding.working_input_scale_bytes.data(), input_scales.data(), input_scales.size() * sizeof(float));
    }

    if (header_weight_scale != nullptr) {
        *header_weight_scale = weight_scales.empty() ? 1.0f : weight_scales[0];
    }
    if (header_input_scale != nullptr) {
        *header_input_scale = input_scales.empty() ? 1.0f : input_scales[0];
    }
    return true;
}

struct selector_restore_counters {
    int64_t device = 0;
    int64_t host = 0;
    int64_t failed = 0;
};

static bool quantize_restore_from_snapshot_or_host(
        ggml_tensor * tensor,
        const selector_device_snapshot & snapshot,
        const std::vector<uint8_t> & host_bytes,
        size_t nbytes,
        selector_restore_counters * counters) {
    if (tensor == nullptr || nbytes == 0) {
        if (counters != nullptr) {
            ++counters->failed;
        }
        return false;
    }
    if (!snapshot.valid_for(nbytes) && host_bytes.size() != nbytes) {
        return false;
    }
    if (snapshot.restore(tensor, nbytes)) {
        if (counters != nullptr) {
            ++counters->device;
        }
        return true;
    }
    if (host_bytes.size() == nbytes && quantize_tensor_copy_in(tensor, host_bytes.data(), nbytes)) {
        if (counters != nullptr) {
            ++counters->host;
        }
        return true;
    }
    if (counters != nullptr) {
        ++counters->failed;
    }
    return false;
}

static bool quantize_restore_from_host(
        ggml_tensor * tensor,
        const std::vector<uint8_t> & host_bytes,
        size_t nbytes,
        selector_restore_counters * counters) {
    if (tensor == nullptr || nbytes == 0 || host_bytes.size() != nbytes ||
            !quantize_tensor_copy_in(tensor, host_bytes.data(), nbytes)) {
        if (counters != nullptr) {
            ++counters->failed;
        }
        return false;
    }
    if (counters != nullptr) {
        ++counters->host;
    }
    return true;
}

static std::vector<std::string> selector_build_stress_tensors(
    const std::vector<std::pair<std::string, ggml_tensor *>> & tmap,
    int32_t n_layer) {
    std::vector<std::string> out;
    if (tmap.empty()) {
        return out;
    }

    struct tensor_pick {
        std::string name;
        selector_tensor_class cls = selector_tensor_class::OTHER;
        int bucket = -1;
        int32_t layer = -1;
        size_t nbytes = 0;
    };

    std::vector<tensor_pick> picks;
    picks.reserve(tmap.size());
    for (const auto & kv : tmap) {
        if (kv.second == nullptr || kv.second->type != GGML_TYPE_NVFP4) {
            continue;
        }
        tensor_pick pick;
        pick.name = kv.first;
        pick.cls = selector_classify_tensor(kv.first);
        pick.layer = selector_parse_layer(kv.first);
        pick.bucket = selector_layer_bucket(pick.layer, n_layer);
        pick.nbytes = ggml_nbytes(kv.second);
        picks.push_back(std::move(pick));
    }

    auto append_unique = [&](const std::string & name) {
        if (std::find(out.begin(), out.end(), name) == out.end()) {
            out.push_back(name);
        }
    };

    const int per_bucket = (int) std::max<int64_t>(1, SELECTOR_CLASS_BUCKET_LIMIT);
    const int special_limit = (int) std::max<int64_t>(1, SELECTOR_SPECIAL_LIMIT);
    const int exception_limit = (int) std::max<int64_t>(0, SELECTOR_EXCEPTION_LIMIT);

    const std::vector<std::string> special_names = {
        "token_embd.weight",
        "output.weight",
        "token_embd_norm.weight",
        "output_norm.weight",
    };
    for (const std::string & special : special_names) {
        if ((int) out.size() >= special_limit) {
            break;
        }
        for (const auto & pick : picks) {
            if (pick.name == special) {
                append_unique(pick.name);
                break;
            }
        }
    }

    std::stable_sort(picks.begin(), picks.end(), [](const tensor_pick & a, const tensor_pick & b) {
        if (a.cls != b.cls) return a.cls < b.cls;
        if (a.bucket != b.bucket) return a.bucket < b.bucket;
        if (a.nbytes != b.nbytes) return a.nbytes > b.nbytes;
        if (a.layer != b.layer) return a.layer < b.layer;
        return a.name < b.name;
    });

    const std::vector<selector_tensor_class> class_order = {
        selector_tensor_class::EMBEDDING_OUTPUT,
        selector_tensor_class::ROUTER,
        selector_tensor_class::ATTN_QKV,
        selector_tensor_class::ATTN_OUT,
        selector_tensor_class::EXPERT_UP_GATE,
        selector_tensor_class::EXPERT_DOWN,
        selector_tensor_class::DENSE_MLP,
        selector_tensor_class::SSM,
    };
    for (selector_tensor_class cls : class_order) {
        for (int bucket = -1; bucket < 3; ++bucket) {
            int added = 0;
            for (const auto & pick : picks) {
                if (pick.cls != cls || pick.bucket != bucket) {
                    continue;
                }
                append_unique(pick.name);
                if (++added >= per_bucket) {
                    break;
                }
            }
        }
    }

    if (exception_limit > 0) {
        std::vector<tensor_pick> by_size = picks;
        std::stable_sort(by_size.begin(), by_size.end(), [](const tensor_pick & a, const tensor_pick & b) {
            if (a.nbytes != b.nbytes) return a.nbytes > b.nbytes;
            return a.name < b.name;
        });
        int added = 0;
        for (const auto & pick : by_size) {
            if (pick.cls == selector_tensor_class::OTHER) {
                continue;
            }
            const size_t before = out.size();
            append_unique(pick.name);
            if (out.size() != before && ++added >= exception_limit) {
                break;
            }
        }
    }

    return out;
}

static double selector_class_hotness(selector_tensor_class cls) {
    switch (cls) {
        case selector_tensor_class::ATTN_QKV:        return 1.00;
        case selector_tensor_class::ATTN_OUT:        return 0.95;
        case selector_tensor_class::EXPERT_UP_GATE:  return 0.90;
        case selector_tensor_class::EXPERT_DOWN:     return 0.90;
        case selector_tensor_class::SSM:             return 1.00;
        case selector_tensor_class::EMBEDDING_OUTPUT:return 0.80;
        case selector_tensor_class::ROUTER:          return 0.70;
        case selector_tensor_class::DENSE_MLP:       return 0.75;
        case selector_tensor_class::OTHER:           return 0.60;
    }
    return 0.75;
}

static bool selector_type_is_k_rsf_capable(ggml_type type) {
    return type == GGML_TYPE_Q2_K ||
           type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K;
}

static const char * selector_tensor_role_name(const std::string & name, selector_tensor_class cls) {
    if (name.find(".attn_gate.weight") != std::string::npos) {
        return "attn_gate";
    }
    if (name.find(".attn_qkv.weight") != std::string::npos) {
        return "attn_qkv";
    }
    if (name.find(".attn_q.weight") != std::string::npos) {
        return "attn_q";
    }
    if (name.find(".attn_k.weight") != std::string::npos) {
        return "attn_k";
    }
    if (name.find(".attn_v.weight") != std::string::npos) {
        return "attn_v";
    }
    if (name.find(".attn_output.weight") != std::string::npos || name.find(".wo.weight") != std::string::npos) {
        return "attn_output";
    }
    if (name.find(".ffn_down.weight") != std::string::npos || name.find(".ffn_down_exps.weight") != std::string::npos) {
        return "ffn_down";
    }
    if (name.find(".ffn_gate.weight") != std::string::npos || name.find(".ffn_gate_exps.weight") != std::string::npos) {
        return "ffn_gate";
    }
    if (name.find(".ffn_up.weight") != std::string::npos || name.find(".ffn_up_exps.weight") != std::string::npos) {
        return "ffn_up";
    }
    if (name.find(".ssm_out.weight") != std::string::npos) {
        return "ssm_out";
    }
    if (name.find(".ssm_alpha.weight") != std::string::npos || name.find(".ssm_beta.weight") != std::string::npos) {
        return "ssm_control";
    }
    if (name.find(".ssm_") != std::string::npos) {
        return "ssm";
    }
    return selector_tensor_class_name(cls);
}

static double selector_tensor_role_importance(const std::string & name, selector_tensor_class cls) {
    const char * role = selector_tensor_role_name(name, cls);
    if (striequals(role, "attn_gate"))    return 1.25;
    if (striequals(role, "ssm_out"))      return 1.24;
    if (striequals(role, "ffn_down"))     return 1.20;
    if (striequals(role, "attn_qkv"))     return 1.16;
    if (striequals(role, "attn_output"))  return 1.12;
    if (striequals(role, "attn_k"))       return 1.10;
    if (striequals(role, "attn_v"))       return 1.10;
    if (striequals(role, "ssm_control"))  return 1.10;
    if (striequals(role, "attn_q"))       return 0.98;
    if (striequals(role, "ffn_gate"))     return 1.06;
    if (striequals(role, "ffn_up"))       return 1.02;

    switch (cls) {
        case selector_tensor_class::ATTN_QKV:         return 1.10;
        case selector_tensor_class::ATTN_OUT:         return 1.08;
        case selector_tensor_class::EXPERT_DOWN:      return 1.08;
        case selector_tensor_class::EXPERT_UP_GATE:   return 0.90;
        case selector_tensor_class::SSM:              return 1.08;
        case selector_tensor_class::ROUTER:           return 0.95;
        case selector_tensor_class::DENSE_MLP:        return 1.05;
        case selector_tensor_class::EMBEDDING_OUTPUT: return 0.70;
        case selector_tensor_class::OTHER:            return 0.75;
    }
    return 0.85;
}

static double selector_tensor_role_gate_mul(const std::string & name, selector_tensor_class cls) {
    const char * role = selector_tensor_role_name(name, cls);
    if (striequals(role, "ssm_control"))  return 0.70;
    if (striequals(role, "attn_gate"))    return 0.78;
    if (striequals(role, "ssm_out"))      return 0.80;
    if (striequals(role, "ffn_down"))     return 0.84;
    if (striequals(role, "attn_qkv"))     return 0.86;
    if (striequals(role, "attn_output"))  return 0.90;
    if (striequals(role, "attn_k"))       return 0.92;
    if (striequals(role, "attn_v"))       return 0.92;
    if (striequals(role, "attn_q"))       return 1.00;
    if (striequals(role, "ffn_gate"))     return 1.00;
    if (striequals(role, "ffn_up"))       return 1.06;

    switch (cls) {
        case selector_tensor_class::ATTN_QKV:
        case selector_tensor_class::ATTN_OUT:
        case selector_tensor_class::EXPERT_DOWN:
        case selector_tensor_class::SSM:
            return 0.92;
        case selector_tensor_class::EXPERT_UP_GATE:
            return 1.12;
        case selector_tensor_class::ROUTER:
            return 1.00;
        case selector_tensor_class::DENSE_MLP:
            return 0.95;
        case selector_tensor_class::EMBEDDING_OUTPUT:
            return 1.25;
        case selector_tensor_class::OTHER:
            return 1.15;
    }
    return 1.0;
}

static bool selector_bf16_allowed(selector_tensor_class cls) {
    switch (cls) {
        case selector_tensor_class::ROUTER:
        case selector_tensor_class::SSM:
        case selector_tensor_class::EMBEDDING_OUTPUT:
        case selector_tensor_class::ATTN_OUT:
        case selector_tensor_class::ATTN_QKV:
            return true;
        default:
            return false;
    }
}

static bool selector_expert_class(selector_tensor_class cls) {
    return
        cls == selector_tensor_class::EXPERT_UP_GATE ||
        cls == selector_tensor_class::EXPERT_DOWN;
}

static double selector_rescue_roi(double proxy_score, size_t delta_bytes, double speed_penalty, double type_gain, double speed_weight) {
    if (delta_bytes == 0 || !std::isfinite(proxy_score) || proxy_score <= 0.0) {
        return 0.0;
    }

    const double delta_mb = std::max(1e-6, (double) delta_bytes / (1024.0 * 1024.0));
    const double denom = delta_mb * (1.0 + std::max(0.0, speed_weight) * std::max(0.0, speed_penalty));
    return (proxy_score * std::max(0.0, type_gain)) / denom;
}

static double selector_type_candidate_roi(
        double proxy_score,
        size_t delta_bytes,
        double speed_penalty,
        double type_gain,
        double speed_weight) {
    if (!std::isfinite(proxy_score) || proxy_score <= 0.0) {
        return 0.0;
    }

    const double delta_mb = (double) delta_bytes / (1024.0 * 1024.0);
    const double size_cost_mb = std::max(SELECTOR_TYPE_CANDIDATE_MIN_COST_MB, delta_mb);
    const double speed_cost = 1.0 + std::max(0.0, speed_weight) * std::max(0.0, speed_penalty);
    return (proxy_score * std::max(0.0, type_gain)) / (size_cost_mb * speed_cost);
}

static double selector_type_speed_penalty(ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP6_E2M3: return 0.03;
        case GGML_TYPE_Q5_0:       return 0.08;
        case GGML_TYPE_Q4_K:       return 0.12;
        case GGML_TYPE_Q5_K:       return 0.14;
        case GGML_TYPE_Q6_K:       return 0.20;
        case GGML_TYPE_Q8_0:       return SELECTOR_RESCUE_Q8_SPEED_PENALTY;
        case GGML_TYPE_F16:        return 1.00;
        case GGML_TYPE_BF16:       return SELECTOR_RESCUE_BF16_SPEED_PENALTY;
        default:                   return 0.30;
    }
}

static double selector_type_quality_gain(ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP6_E2M3: return 1.28;
        case GGML_TYPE_Q5_0:       return 1.13;
        case GGML_TYPE_Q4_K:       return 1.16;
        case GGML_TYPE_Q5_K:       return 1.22;
        case GGML_TYPE_Q6_K:       return 1.25;
        case GGML_TYPE_Q8_0:       return SELECTOR_RESCUE_Q8_GAIN;
        case GGML_TYPE_F16:        return 1.30;
        case GGML_TYPE_BF16:       return SELECTOR_RESCUE_BF16_GAIN;
        default:                   return 1.0;
    }
}

static bool selector_type_candidate_allowed(ggml_type type, selector_tensor_class cls) {
    if (type == GGML_TYPE_COUNT || type == GGML_TYPE_NVFP4) {
        return false;
    }
    if ((type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) && !selector_bf16_allowed(cls)) {
        return false;
    }
    return true;
}

static double selector_tensor_role_type_gain(const std::string & name, selector_tensor_class cls, ggml_type type) {
    const char * role = selector_tensor_role_name(name, cls);
    double pref = 1.0;

    if (striequals(role, "ssm_control")) {
        pref =
            type == GGML_TYPE_Q8_0 ? 1.55 :
            selector_type_is_k_rsf_capable(type) ? 0.62 : 0.85;
    } else if (striequals(role, "attn_gate") || striequals(role, "ssm_out") || striequals(role, "ffn_down")) {
        pref =
            type == GGML_TYPE_Q5_K ? 1.36 :
            type == GGML_TYPE_Q6_K ? 1.24 :
            type == GGML_TYPE_Q4_K ? 1.12 :
            type == GGML_TYPE_Q5_0 ? 1.02 :
            type == GGML_TYPE_Q8_0 ? 0.90 : 1.0;
    } else if (striequals(role, "attn_qkv") ||
               striequals(role, "attn_k") ||
               striequals(role, "attn_v") ||
               striequals(role, "attn_output")) {
        pref =
            type == GGML_TYPE_Q5_K ? 1.24 :
            type == GGML_TYPE_Q4_K ? 1.14 :
            type == GGML_TYPE_Q6_K ? 1.12 :
            type == GGML_TYPE_Q5_0 ? 1.04 :
            type == GGML_TYPE_Q8_0 ? 0.88 : 1.0;
    } else if (striequals(role, "ffn_gate") || striequals(role, "ffn_up")) {
        pref =
            type == GGML_TYPE_Q4_K ? 1.14 :
            type == GGML_TYPE_Q5_0 ? 1.08 :
            type == GGML_TYPE_Q5_K ? 1.04 :
            type == GGML_TYPE_Q6_K ? 0.98 :
            type == GGML_TYPE_Q8_0 ? 0.72 : 0.92;
    }

    return selector_type_quality_gain(type) *
           selector_tensor_role_importance(name, cls) *
           pref;
}

struct selector_stage1_prior_item {
    std::string name;
    std::string role;
    selector_tensor_class cls = selector_tensor_class::OTHER;
    ggml_tensor * tensor = nullptr;
    size_t nvfp4_nbytes = 0;
    double imatrix_mean = 0.0;
    double imatrix_max = 0.0;
    double risk = 0.0;
    bool has_imatrix = false;
};

static bool selector_stage1_exact_seen(
        const std::unordered_set<std::string> & seen,
        const std::string & tensor_name) {
    return seen.find(selector_regex_escape(tensor_name)) != seen.end();
}

static ggml_type selector_stage1_compatible_type(
        const ggml_tensor * tensor,
        selector_tensor_class cls,
        ggml_type requested) {
    auto allowed = [&](ggml_type type) {
        return selector_type_candidate_allowed(type, cls) &&
               quantize_tensor_nbytes_as_type(tensor, type) > 0;
    };

    if (allowed(requested)) {
        return requested;
    }
    if (requested == GGML_TYPE_Q8_0) {
        return GGML_TYPE_COUNT;
    }
    for (ggml_type fallback : { GGML_TYPE_Q5_0, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 }) {
        if (allowed(fallback)) {
            return fallback;
        }
    }
    return GGML_TYPE_COUNT;
}

static bool selector_stage1_conditional_role(const std::string & role, quantize_work_mode mode) {
    if (striequals(role.c_str(), "attn_gate") ||
        striequals(role.c_str(), "ssm_out") ||
        striequals(role.c_str(), "ssm_control") ||
        striequals(role.c_str(), "ffn_down") ||
        striequals(role.c_str(), "ffn_gate") ||
        striequals(role.c_str(), "ffn_up") ||
        striequals(role.c_str(), "attn_qkv") ||
        striequals(role.c_str(), "attn_k") ||
        striequals(role.c_str(), "attn_v") ||
        striequals(role.c_str(), "attn_output")) {
        return true;
    }
    return mode == quantize_work_mode::DEEP && striequals(role.c_str(), "attn_q");
}

static ggml_type selector_stage1_requested_type(
        const std::string & role,
        quantize_work_mode mode,
        double risk,
        double deep_gate) {
    if (striequals(role.c_str(), "ssm_control")) {
        return GGML_TYPE_Q8_0;
    }

    if (mode == quantize_work_mode::DEEP &&
            std::isfinite(risk) &&
            std::isfinite(deep_gate) &&
            deep_gate > 0.0 &&
            risk >= deep_gate * 1.65 &&
            (striequals(role.c_str(), "ffn_down") ||
             striequals(role.c_str(), "attn_output") ||
             striequals(role.c_str(), "attn_k") ||
             striequals(role.c_str(), "attn_v"))) {
        return GGML_TYPE_Q6_K;
    }

    if (striequals(role.c_str(), "ffn_gate") || striequals(role.c_str(), "ffn_up")) {
        return mode == quantize_work_mode::DEEP ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
    }

    return GGML_TYPE_Q5_K;
}

static std::vector<tensor_type_option> selector_stage1_type_priors(
        const std::string & source_model_path,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        quantize_work_mode mode,
        const std::vector<tensor_type_option> & existing_options) {
    if (mode == quantize_work_mode::UNSET) {
        mode = quantize_work_mode::NORMAL;
    }

    std::unordered_set<std::string> exact_seen;
    exact_seen.reserve(existing_options.size());
    for (const auto & opt : existing_options) {
        exact_seen.insert(opt.name);
    }

    std::vector<std::string> src_splits;
    std::vector<selector_stage1_prior_item> items;
    size_t stage1_nvfp4_total_bytes = 0;
    try {
        llama_model_loader src_ml(
            /*metadata=*/ nullptr,
            /*set_tensor_data=*/ nullptr,
            /*set_tensor_data_ud=*/ nullptr,
            source_model_path,
            src_splits,
            /*file=*/ nullptr,
            /*use_mmap=*/ true,
            /*use_direct_io=*/ false,
            /*check_tensors=*/ false,
            /*no_alloc=*/ true,
            /*param_overrides_p=*/ nullptr,
            /*param_tensor_buft_overrides_p=*/ nullptr);
        src_ml.init_mappings(false);

        items.reserve(src_ml.weights_map.size());
        for (const auto & kv : src_ml.weights_map) {
            ggml_tensor * tensor = kv.second.tensor;
            if (tensor == nullptr) {
                continue;
            }
            const std::string & name = kv.first;
            if (name.find("mtp") != std::string::npos || name.find("nextn") != std::string::npos) {
                continue;
            }
            const selector_tensor_class cls = selector_classify_tensor(name);
            const size_t nvfp4_nbytes = quantize_tensor_nbytes_as_type(tensor, GGML_TYPE_NVFP4);
            if (cls != selector_tensor_class::EMBEDDING_OUTPUT && nvfp4_nbytes > 0) {
                stage1_nvfp4_total_bytes += nvfp4_nbytes;
            }
            if (cls == selector_tensor_class::EMBEDDING_OUTPUT ||
                    cls == selector_tensor_class::OTHER ||
                    nvfp4_nbytes == 0 ||
                    selector_stage1_exact_seen(exact_seen, name)) {
                continue;
            }

            selector_stage1_prior_item item;
            item.name = name;
            item.role = selector_tensor_role_name(name, cls);
            item.cls = cls;
            item.tensor = tensor;
            item.nvfp4_nbytes = nvfp4_nbytes;

            const auto im_it = imatrix_data.find(name);
            const size_t expected = (size_t) std::max<int64_t>(1, tensor->ne[0]) *
                (size_t) std::max<int64_t>(1, tensor->ne[2]);
            if (im_it != imatrix_data.end() && im_it->second.size() >= expected && expected > 0) {
                double sum = 0.0;
                double mx = 0.0;
                size_t n = 0;
                for (size_t i = 0; i < expected; ++i) {
                    const float v = im_it->second[i];
                    if (!std::isfinite(v)) {
                        continue;
                    }
                    const double av = std::fabs((double) v);
                    sum += av;
                    mx = std::max(mx, av);
                    ++n;
                }
                if (n > 0) {
                    item.has_imatrix = true;
                    item.imatrix_mean = sum / (double) n;
                    item.imatrix_max = mx;
                    item.risk = std::sqrt(std::max(0.0, item.imatrix_mean)) *
                        selector_tensor_role_importance(name, cls);
                }
            }

            if (selector_stage1_conditional_role(item.role, mode)) {
                items.push_back(std::move(item));
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr,
            "%s: selector stage1 type priors unavailable for %s: %s\n",
            __func__,
            source_model_path.c_str(),
            e.what());
        return {};
    }

    std::vector<double> risks;
    risks.reserve(items.size());
    for (const auto & item : items) {
        if (item.has_imatrix && std::isfinite(item.risk) && item.risk > 0.0) {
            risks.push_back(item.risk);
        }
    }

    const double p50 = selector_percentile(risks, 0.50);
    const double p75 = selector_percentile(risks, 0.75);
    const double p90 = selector_percentile(risks, 0.90);
    double gate = std::numeric_limits<double>::infinity();
    if (!risks.empty()) {
        switch (mode) {
            case quantize_work_mode::FAST:
                gate = std::max(p50 * 1.35, p75 * 1.10);
                break;
            case quantize_work_mode::DEEP:
                gate = std::max(p50 * 1.03, p75 * 0.70);
                break;
            case quantize_work_mode::NORMAL:
            case quantize_work_mode::UNSET:
                gate = std::max(p50 * 1.15, p75 * 0.95);
                break;
        }
    }

    std::vector<tensor_type_option> out;
    out.reserve(items.size());
    std::unordered_map<std::string, int> role_counts;
    std::unordered_map<std::string, int> type_counts;

    struct stage1_proposal {
        tensor_type_option opt;
        std::string role;
        ggml_type type = GGML_TYPE_COUNT;
        size_t delta_bytes = 0;
        double priority = 0.0;
        double risk = 0.0;
    };

    std::vector<stage1_proposal> proposals;
    proposals.reserve(items.size());
    for (const auto & item : items) {
        if (!item.has_imatrix || !std::isfinite(item.risk) || !std::isfinite(gate)) {
            continue;
        }
        const double role_gate = gate * selector_tensor_role_gate_mul(item.name, item.cls);
        if (item.risk < role_gate) {
            continue;
        }

        const ggml_type requested = selector_stage1_requested_type(item.role, mode, item.risk, gate);
        const ggml_type type = selector_stage1_compatible_type(item.tensor, item.cls, requested);
        if (type == GGML_TYPE_COUNT) {
            continue;
        }
        const size_t type_nbytes = quantize_tensor_nbytes_as_type(item.tensor, type);
        if (type_nbytes == 0) {
            continue;
        }
        const size_t delta = type_nbytes > item.nvfp4_nbytes ? type_nbytes - item.nvfp4_nbytes : 0;

        tensor_type_option opt;
        opt.name = selector_regex_escape(item.name);
        opt.type = type;
        if (selector_type_is_k_rsf_capable(type)) {
            opt.has_k_rsf_mode = true;
            opt.k_rsf_mode = 2;
        }
        const double delta_mb = std::max(0.25, (double) delta / (1024.0 * 1024.0));
        stage1_proposal proposal;
        proposal.opt = std::move(opt);
        proposal.role = item.role;
        proposal.type = type;
        proposal.delta_bytes = delta;
        proposal.risk = item.risk;
        proposal.priority =
            (item.risk * selector_tensor_role_type_gain(item.name, item.cls, type)) / delta_mb;
        proposals.push_back(std::move(proposal));
    }

    std::stable_sort(proposals.begin(), proposals.end(), [](const stage1_proposal & a, const stage1_proposal & b) {
        if (a.priority != b.priority) return a.priority > b.priority;
        if (a.risk != b.risk) return a.risk > b.risk;
        if (a.delta_bytes != b.delta_bytes) return a.delta_bytes < b.delta_bytes;
        return a.opt.name < b.opt.name;
    });

    const double budget_fraction =
        mode == quantize_work_mode::FAST   ? 0.012 :
        mode == quantize_work_mode::DEEP   ? 0.035 :
                                             0.022;
    const size_t budget_cap_bytes =
        mode == quantize_work_mode::FAST   ? 256ull * 1024ull * 1024ull :
        mode == quantize_work_mode::DEEP   ? 768ull * 1024ull * 1024ull :
                                             512ull * 1024ull * 1024ull;
    const size_t stage1_budget_bytes =
        stage1_nvfp4_total_bytes > 0 ?
        std::min(budget_cap_bytes, (size_t) std::floor((double) stage1_nvfp4_total_bytes * budget_fraction)) :
        0;
    size_t budget_used = 0;
    for (auto & proposal : proposals) {
        if (proposal.delta_bytes > 0 &&
                (stage1_budget_bytes == 0 || budget_used + proposal.delta_bytes > stage1_budget_bytes)) {
            continue;
        }
        budget_used += proposal.delta_bytes;
        exact_seen.insert(proposal.opt.name);
        role_counts[proposal.role]++;
        type_counts[ggml_type_name(proposal.type)]++;
        out.push_back(std::move(proposal.opt));
    }

    auto counts_to_string = [](const std::unordered_map<std::string, int> & counts) {
        std::vector<std::pair<std::string, int>> ordered(counts.begin(), counts.end());
        std::sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });
        std::ostringstream ss;
        for (size_t i = 0; i < ordered.size(); ++i) {
            if (i > 0) {
                ss << ',';
            }
            ss << ordered[i].first << '=' << ordered[i].second;
        }
        return ss.str();
    };
    fprintf(stderr,
        "%s: selector stage1 type priors considered=%zu proposed=%zu added=%zu mode=%s imatrix_risk_p50=%.6g p75=%.6g p90=%.6g gate=%.6g budget_used=%zu budget_cap=%zu types={%s} roles={%s}\n",
        __func__,
        items.size(),
        proposals.size(),
        out.size(),
        quantize_work_mode_name(mode),
        p50,
        p75,
        p90,
        gate,
        budget_used,
        stage1_budget_bytes,
        counts_to_string(type_counts).c_str(),
        counts_to_string(role_counts).c_str());

    return out;
}

static int64_t selector_sample_blocks(int64_t nb_total, int64_t override_cap = 0) {
    if (nb_total <= 0) {
        return 0;
    }
    const int64_t control_cap = quantize_control_i64("LLAMA_NVFP4_ENCODER_MAX_BLOCKS", 0);
    const int64_t cap_hint = override_cap > 0 ? override_cap : control_cap;
    if (cap_hint > 0) {
        return std::min(nb_total, cap_hint);
    }
    const int64_t cap =
        nb_total >= 262144 ? 4096 :
        nb_total >= 65536  ? 3072 :
        nb_total >= 16384  ? 2048 :
        nb_total >= 4096   ? 1024 :
        nb_total >= 1024   ? 512  :
        nb_total >= 256    ? 256  : 64;
    return std::min(nb_total, cap);
}

static bool quantize_tensor_slice_to_f32(
    const ggml_tensor * src_tensor,
    int64_t slice_index,
    float tensor_scale,
    std::vector<float> & out,
    int nthread = 1) {
    if (src_tensor == nullptr || slice_index < 0 || slice_index >= src_tensor->ne[2]) {
        return false;
    }

    const int64_t n_per_row = src_tensor->ne[0];
    const int64_t nrows = src_tensor->ne[1];
    const size_t n = (size_t) n_per_row * (size_t) nrows;
    out.resize(n);

    const size_t src_row_size = ggml_row_size(src_tensor->type, n_per_row);
    const auto * src_traits = ggml_get_type_traits(src_tensor->type);
    if (src_tensor->type != GGML_TYPE_F32 && src_tensor->type != GGML_TYPE_F16 && src_tensor->type != GGML_TYPE_BF16 &&
        (!src_traits || src_traits->to_float == nullptr)) {
        return false;
    }

    const bool apply_tensor_scale =
        std::isfinite(tensor_scale) && tensor_scale > 0.0f && std::fabs(tensor_scale - 1.0f) > 1e-12f;
    const int64_t min_rows_per_worker =
        std::max<int64_t>(1, (1024 * 1024) / std::max<int64_t>(1, (int64_t) src_row_size));
    return quantize_parallel_for_ranges(nrows, nthread, min_rows_per_worker, [&](int, int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const uint8_t * src_ptr = (const uint8_t *) src_tensor->data + ((size_t) slice_index * (size_t) nrows + (size_t) row) * src_row_size;
            float * dst_ptr = out.data() + (size_t) row * (size_t) n_per_row;
            if (src_tensor->type == GGML_TYPE_F32) {
                memcpy(dst_ptr, src_ptr, (size_t) n_per_row * sizeof(float));
            } else if (src_tensor->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) src_ptr, dst_ptr, n_per_row);
            } else if (src_tensor->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) src_ptr, dst_ptr, n_per_row);
            } else {
                src_traits->to_float(src_ptr, dst_ptr, n_per_row);
            }
            if (apply_tensor_scale) {
                for (int64_t i = 0; i < n_per_row; ++i) {
                    dst_ptr[i] *= tensor_scale;
                }
            }
        }
        return true;
    });
}

struct selector_source_slice_view {
    const void * raw = nullptr;
    bool raw_bf16 = false;
    float x_scale = 1.0f;
    std::vector<float> owned_f32;
};

static bool selector_prepare_sample_from_f32(
    const float * src_f32,
    const float * imatrix_row,
    int64_t nrows,
    int64_t n_per_row,
    std::vector<float> & sample_x,
    std::vector<float> & sample_qw,
    int64_t sample_blocks_override = 0,
    int64_t sample_phase = 0) {
    const int64_t nb_total = (nrows * n_per_row) / SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
    const int64_t sample_nb = selector_sample_blocks(nb_total, sample_blocks_override);
    if (sample_nb <= 0) {
        return false;
    }
    sample_x.resize((size_t) sample_nb * (size_t) SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE);
    sample_qw.clear();

    const int64_t row_blocks = n_per_row / SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
    const int64_t slice_elems = nrows * n_per_row;
    const bool build_qw = imatrix_row != nullptr && row_blocks > 0;
    if (build_qw) {
        sample_qw.resize(sample_x.size());
    }

    for (int64_t ib = 0; ib < sample_nb; ++ib) {
        const int64_t src_block = llama_nvfp4_sample_block_index(ib, sample_nb, nb_total, row_blocks, sample_phase);
        const int64_t src_off = src_block * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
        if (src_off < 0 || src_off + SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE > slice_elems) {
            return false;
        }
        const int64_t dst_off = ib * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
        memcpy(sample_x.data() + dst_off, src_f32 + src_off, (size_t) SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE * sizeof(float));
        if (build_qw) {
            const int64_t block_in_row = src_block % row_blocks;
            memcpy(sample_qw.data() + dst_off, imatrix_row + block_in_row * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE, (size_t) SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE * sizeof(float));
        }
    }

    return true;
}

static float selector_qw_at(const float * qw, int idx) {
    if (qw == nullptr) {
        return 1.0f;
    }
    const float w = qw[idx];
    return std::isfinite(w) && w > 0.0f ? w : 0.0f;
}

static bool selector_prepare_sample(
    const selector_binding & binding,
    int64_t slice_index,
    std::vector<float> & sample_x,
    std::vector<float> & sample_qw,
    selector_source_slice_view & source_view,
    int64_t sample_blocks_override = 0,
    int64_t sample_phase = 0,
    int host_threads = 1) {
    const ggml_tensor * src_tensor = binding.source;
    if (src_tensor == nullptr || slice_index < 0 || slice_index >= src_tensor->ne[2]) {
        return false;
    }

    const int64_t n_per_row = src_tensor->ne[0];
    const int64_t nrows = src_tensor->ne[1];
    const float * imatrix_row = binding.imatrix_row ? (binding.imatrix_row + slice_index * n_per_row) : nullptr;
    const float quant_tensor_scale =
        slice_index >= 0 && (size_t) slice_index < binding.quant_tensor_scales.size() &&
        std::isfinite(binding.quant_tensor_scales[(size_t) slice_index]) &&
        binding.quant_tensor_scales[(size_t) slice_index] > 0.0f
        ? binding.quant_tensor_scales[(size_t) slice_index]
        : 1.0f;
    const bool tensor_scale_identity =
        !(std::isfinite(binding.source_tensor_scale) &&
          binding.source_tensor_scale > 0.0f &&
          std::fabs(binding.source_tensor_scale - 1.0f) > 1e-12f);

    const size_t src_row_size = ggml_row_size(src_tensor->type, n_per_row);
    const uint8_t * src_slice = (const uint8_t *) src_tensor->data + (size_t) slice_index * (size_t) nrows * src_row_size;
    const int64_t slice_elems = nrows * n_per_row;

    source_view = {};

    if (tensor_scale_identity && src_tensor->type == GGML_TYPE_F32) {
        source_view.raw = src_slice;
        source_view.raw_bf16 = false;
        source_view.x_scale = quant_tensor_scale;
        return selector_prepare_sample_from_f32((const float *) src_slice, imatrix_row, nrows, n_per_row, sample_x, sample_qw, sample_blocks_override, sample_phase);
    }

    const int64_t nb_total = (nrows * n_per_row) / SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
    const int64_t sample_nb = selector_sample_blocks(nb_total, sample_blocks_override);
    if (sample_nb <= 0) {
        return false;
    }
    sample_x.resize((size_t) sample_nb * (size_t) SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE);
    sample_qw.clear();

    const int64_t row_blocks = n_per_row / SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
    const bool build_qw = imatrix_row != nullptr && row_blocks > 0;
    if (build_qw) {
        sample_qw.resize(sample_x.size());
    }

    if (tensor_scale_identity && src_tensor->type == GGML_TYPE_BF16) {
        source_view.raw = src_slice;
        source_view.raw_bf16 = true;
        source_view.x_scale = quant_tensor_scale;
        const auto * src_bf16 = (const ggml_bf16_t *) src_slice;
        for (int64_t ib = 0; ib < sample_nb; ++ib) {
            const int64_t src_block = llama_nvfp4_sample_block_index(ib, sample_nb, nb_total, row_blocks, sample_phase);
            const int64_t src_off = src_block * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
            if (src_off < 0 || src_off + SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE > slice_elems) {
                return false;
            }
            const int64_t dst_off = ib * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
            ggml_bf16_to_fp32_row(src_bf16 + src_off, sample_x.data() + dst_off, SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE);
            if (build_qw) {
                const int64_t block_in_row = src_block % row_blocks;
                memcpy(sample_qw.data() + dst_off, imatrix_row + block_in_row * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE, (size_t) SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE * sizeof(float));
            }
        }
        return true;
    }

    if (tensor_scale_identity && src_tensor->type == GGML_TYPE_F16) {
        const auto * src_f16 = (const ggml_fp16_t *) src_slice;
        for (int64_t ib = 0; ib < sample_nb; ++ib) {
            const int64_t src_block = llama_nvfp4_sample_block_index(ib, sample_nb, nb_total, row_blocks, sample_phase);
            const int64_t src_off = src_block * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
            if (src_off < 0 || src_off + SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE > slice_elems) {
                return false;
            }
            const int64_t dst_off = ib * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
            ggml_fp16_to_fp32_row(src_f16 + src_off, sample_x.data() + dst_off, SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE);
            if (build_qw) {
                const int64_t block_in_row = src_block % row_blocks;
                memcpy(sample_qw.data() + dst_off, imatrix_row + block_in_row * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE, (size_t) SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE * sizeof(float));
            }
        }
        if (sample_blocks_override > 0) {
            source_view.x_scale = quant_tensor_scale;
            return true;
        }
        if (!quantize_tensor_slice_to_f32(src_tensor, slice_index, binding.source_tensor_scale, source_view.owned_f32, host_threads)) {
            return false;
        }
        source_view.raw = source_view.owned_f32.data();
        source_view.raw_bf16 = false;
        source_view.x_scale = quant_tensor_scale;
        return true;
    }

    if (!quantize_tensor_slice_to_f32(src_tensor, slice_index, binding.source_tensor_scale, source_view.owned_f32, host_threads)) {
        return false;
    }
    source_view.raw = source_view.owned_f32.data();
    source_view.raw_bf16 = false;
    source_view.x_scale = quant_tensor_scale;
    return selector_prepare_sample_from_f32(source_view.owned_f32.data(), imatrix_row, nrows, n_per_row, sample_x, sample_qw, sample_blocks_override, sample_phase);
}

static bool nvfp4_selector_device_sample_supported(const ggml_tensor * src_tensor, float source_tensor_scale) {
    if (src_tensor == nullptr || src_tensor->data == nullptr) {
        return false;
    }
    if (std::isfinite(source_tensor_scale) && source_tensor_scale > 0.0f &&
            std::fabs(source_tensor_scale - 1.0f) > 1e-12f) {
        return false;
    }
    return
        src_tensor->type == GGML_TYPE_F32 ||
        src_tensor->type == GGML_TYPE_F16 ||
        src_tensor->type == GGML_TYPE_BF16;
}

static bool nvfp4_selector_get_device_sample(
    const selector_binding & binding,
    int64_t slice_index,
    int64_t sample_nb,
    int64_t sample_phase,
    void * stream_key,
    nvfp4_selector_device_sample_view & out) {
    out = {};
    const ggml_tensor * src_tensor = binding.source;
    if (sample_nb <= 0 || !binding.device_samples ||
            !nvfp4_selector_device_sample_supported(src_tensor, binding.source_tensor_scale) ||
            slice_index < 0 || slice_index >= src_tensor->ne[2]) {
        return false;
    }

    const int64_t n_per_row = src_tensor->ne[0];
    const int64_t nrows = src_tensor->ne[1];
    if (n_per_row <= 0 || nrows <= 0 || (n_per_row % SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE) != 0) {
        return false;
    }
    const int64_t nb_total = (nrows * n_per_row) / SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
    if (sample_nb > nb_total || sample_nb * 32 < nb_total) {
        return false;
    }

    const size_t src_row_size = ggml_row_size(src_tensor->type, n_per_row);
    const uint8_t * src_slice = (const uint8_t *) src_tensor->data + (size_t) slice_index * (size_t) nrows * src_row_size;
    const float * imatrix_row = binding.imatrix_row ? (binding.imatrix_row + slice_index * n_per_row) : nullptr;
    const float quant_tensor_scale =
        slice_index >= 0 && (size_t) slice_index < binding.quant_tensor_scales.size() &&
        std::isfinite(binding.quant_tensor_scales[(size_t) slice_index]) &&
        binding.quant_tensor_scales[(size_t) slice_index] > 0.0f
        ? binding.quant_tensor_scales[(size_t) slice_index]
        : 1.0f;
    const float tune_x_mul =
        std::isfinite(quant_tensor_scale) && quant_tensor_scale > 0.0f &&
        std::fabs(quant_tensor_scale - 1.0f) > 1e-12f
        ? 1.0f / quant_tensor_scale
        : 1.0f;

    std::lock_guard<std::mutex> lock(binding.device_samples->mutex);
    for (const auto & entry : binding.device_samples->entries) {
        if (entry && entry->matches(slice_index, sample_nb, sample_phase, (int32_t) src_tensor->type, src_slice, imatrix_row, tune_x_mul)) {
            out.x_device = entry->x_device;
            out.tune_x_device = entry->tune_x_device;
            out.qw_device = entry->qw_device;
            out.n_device = entry->n_device;
            out.x_scale = quant_tensor_scale;
            return out.x_device != nullptr && out.tune_x_device != nullptr && out.n_device == sample_nb * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
        }
    }

    auto entry = std::make_unique<nvfp4_selector_device_sample_entry>();
    entry->slice_index = slice_index;
    entry->sample_nb = sample_nb;
    entry->sample_phase = sample_phase;
    entry->source_type = (int32_t) src_tensor->type;
    entry->source_ptr = src_slice;
    entry->imatrix_ptr = imatrix_row;
    entry->tune_x_mul = tune_x_mul;

    if (!nvfp4_sample_cache_cuda_create(
            src_slice,
            (int32_t) src_tensor->type,
            nrows,
            n_per_row,
            imatrix_row,
            sample_nb,
            sample_phase,
            tune_x_mul,
            &entry->cache,
            &entry->x_device,
            &entry->tune_x_device,
            &entry->qw_device,
            &entry->n_device,
            stream_key)) {
        return false;
    }

    out.x_device = entry->x_device;
    out.tune_x_device = entry->tune_x_device;
    out.qw_device = entry->qw_device;
    out.n_device = entry->n_device;
    out.x_scale = quant_tensor_scale;
    const bool ok = out.x_device != nullptr && out.tune_x_device != nullptr && out.n_device == sample_nb * SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
    if (ok) {
        binding.device_samples->entries.push_back(std::move(entry));
    }
    return ok;
}

static bool nvfp4_selector_quantize_binding(
    const selector_binding & binding,
    const nvfp4_cuda_runtime_cfg & cfg,
    int nthread,
    std::vector<uint8_t> & out_bytes,
    double & sum_sq,
    double & sum_abs,
    double & max_abs,
    int64_t & count,
    int64_t sample_blocks_override = 0,
    int64_t sample_phase = 0,
    bool direct_runtime_patch = false,
    selector_rsf_tensor_record * rsf_record = nullptr,
    bool collect_eval_metrics = true) {
    using qtype = ggml_quantize_type_traits<GGML_TYPE_NVFP4>;

    std::mutex rsf_record_mutex;
    auto note_rsf_scale_mul = [&](int64_t slice, float scale_mul) {
        if (rsf_record == nullptr || !nvfp4_cfg_has_rsf(cfg)) {
            return;
        }
        const float safe_mul = std::isfinite(scale_mul) && scale_mul > 0.0f ? scale_mul : 1.0f;
        std::lock_guard<std::mutex> lock(rsf_record_mutex);
        rsf_record->scale_muls.push_back(safe_mul);
        rsf_record->scale_mul_sum += safe_mul;
        rsf_record->slices = std::max<int64_t>(rsf_record->slices, slice + 1);
        rsf_record->rsf_changed = rsf_record->rsf_changed || std::fabs(safe_mul - 1.0f) > 1e-6f;
    };

    auto prepare = [&](const quantize_binding_shape_t & shape) -> bool {
        direct_runtime_patch = direct_runtime_patch && !shape.sample_only;
        if (nvfp4_cfg_has_rsf(cfg) && binding.quant_tensor_scales.size() == (size_t) shape.n_slices) {
            binding.working_quant_tensor_scales = binding.quant_tensor_scales;
        } else {
            binding.working_quant_tensor_scales.clear();
        }
        if (quantize_control_i64("LLAMA_NVFP4_TRACE", 0) != 0) {
            fprintf(stderr,
                "%s: tensor=%s type=%s rows=%" PRId64 " cols=%" PRId64 " slices=%" PRId64 " sample_blocks=%" PRId64 " cfg={choose46=%d refit=%d compand=%d flags=%d cap6=%.1f cap4=%.1f}\n",
                __func__,
                binding.name.c_str(),
                ggml_type_name(binding.source->type),
                shape.nrows,
                shape.n_per_row,
                shape.n_slices,
                sample_blocks_override,
                cfg.choose46_mode,
                cfg.refit_iters,
                cfg.use_compand_sat,
                cfg.reserved_i32,
                (double) cfg.cap_m6,
                (double) cfg.cap_m4);
        }
        if (direct_runtime_patch) {
            out_bytes.clear();
        } else {
            out_bytes.resize((size_t) shape.n_slices * shape.out_slice_bytes);
        }
        return true;
    };

    auto run_slice = [&](int64_t i03, void * stream_key, const quantize_binding_shape_t & shape, quantize_binding_slice_accum_t & acc) {
        const int64_t n_per_row = shape.n_per_row;
        const int64_t nrows = shape.nrows;
        const bool sample_only = shape.sample_only;
        const int64_t sample_nb_for_bytes = shape.sample_nb_for_bytes;
        const size_t out_slice_bytes = shape.out_slice_bytes;
        if (sample_only) {
            nvfp4_selector_device_sample_view device_sample;
            if (nvfp4_selector_get_device_sample(binding, i03, sample_nb_for_bytes, sample_phase, stream_key, device_sample)) {
                nvfp4_cuda_tune_result tune = {
                    NVFP4_A0,
                    NVFP4_B0,
                    1.0f,
                    cfg,
                    1,
                };
                if (nvfp4_autotune_cuda_cfg(
                        device_sample.tune_x_device,
                        device_sample.qw_device,
                        device_sample.n_device,
                        &cfg,
                        &tune,
                        stream_key)) {
                    const float tuned_x_scale =
                        nvfp4_cfg_has_rsf(cfg) &&
                        std::isfinite(tune.scale_mul) && tune.scale_mul > 0.0f
                        ? device_sample.x_scale * tune.scale_mul
                        : device_sample.x_scale;
                    note_rsf_scale_mul(i03, nvfp4_cfg_has_rsf(cfg) ? tune.scale_mul : 1.0f);
                    if (nvfp4_cfg_has_rsf(cfg) &&
                            binding.working_quant_tensor_scales.size() == binding.quant_tensor_scales.size() &&
                            (size_t) i03 < binding.working_quant_tensor_scales.size()) {
                        binding.working_quant_tensor_scales[(size_t) i03] = tuned_x_scale;
                    }
                    nvfp4_cuda_eval_result eval = {};
                    quantize_cuda_eval_params_t eval_params;
                    eval_params.x = device_sample.x_device;
                    eval_params.out = out_bytes.data() + (size_t) i03 * out_slice_bytes;
                    eval_params.nrow = 1;
                    eval_params.n_per_row = device_sample.n_device;
                    eval_params.qw = device_sample.qw_device;
                    eval_params.x_scale = tuned_x_scale;
                    eval_params.a = tune.a;
                    eval_params.b = tune.b;
                    eval_params.nvfp4_cfg = &cfg;
                    eval_params.eval = &eval;
                    eval_params.stream_key = stream_key;
                    if (qtype::quantize_eval(eval_params)) {
                        acc.sum_sq += eval.sum_sq;
                        acc.sum_abs += eval.sum_abs;
                        acc.max_abs = std::max(acc.max_abs, eval.max_abs);
                        acc.count += eval.count;
                        return;
                    }
                }
            }
        }

        selector_source_slice_view source_view;
        std::vector<float> sample_x;
        std::vector<float> sample_qw;
        if (!selector_prepare_sample(
                binding, i03, sample_x, sample_qw, source_view,
                sample_blocks_override, sample_phase, shape.host_threads)) {
            fprintf(stderr,
                "%s: failed preparing selector sample for %s slice=%" PRId64 " source_type=%s rows=%" PRId64 " cols=%" PRId64 " sample_blocks=%" PRId64 " phase=%" PRId64 "\n",
                __func__,
                binding.name.c_str(),
                i03,
                ggml_type_name(binding.source->type),
                nrows,
                n_per_row,
                sample_blocks_override,
                sample_phase);
            acc.ok = false;
            return;
        }
        const float * imatrix_slice = binding.imatrix_row ? (binding.imatrix_row + i03 * n_per_row) : nullptr;

        nvfp4_cuda_tune_result tune = {
            NVFP4_A0,
            NVFP4_B0,
            1.0f,
            cfg,
            1,
        };
        const float sample_inv_scale =
            std::isfinite(source_view.x_scale) && source_view.x_scale > 0.0f &&
            std::fabs(source_view.x_scale - 1.0f) > 1e-12f
            ? 1.0f / source_view.x_scale
            : 1.0f;
        std::vector<float> tune_x;
        const float * tune_x_ptr = sample_x.data();
        if (sample_inv_scale != 1.0f) {
            tune_x.resize(sample_x.size());
            for (size_t i = 0; i < sample_x.size(); ++i) {
                tune_x[i] = sample_x[i] * sample_inv_scale;
            }
            tune_x_ptr = tune_x.data();
        }
        if (!nvfp4_autotune_cuda_cfg(tune_x_ptr, sample_qw.empty() ? nullptr : sample_qw.data(), (int64_t) sample_x.size(), &cfg, &tune, stream_key)) {
            fprintf(stderr,
                "%s: CUDA encoder search failed for %s slice=%" PRId64 " sample_elems=%zu sample_qw=%s choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f\n",
                __func__,
                binding.name.c_str(),
                i03,
                sample_x.size(),
                sample_qw.empty() ? "no" : "yes",
                cfg.choose46_mode,
                cfg.refit_iters,
                cfg.use_compand_sat,
                (double) cfg.cap_m6,
                (double) cfg.cap_m4);
            acc.ok = false;
            return;
        }

        const float tuned_x_scale =
            nvfp4_cfg_has_rsf(cfg) &&
            std::isfinite(tune.scale_mul) && tune.scale_mul > 0.0f
            ? source_view.x_scale * tune.scale_mul
            : source_view.x_scale;
        note_rsf_scale_mul(i03, nvfp4_cfg_has_rsf(cfg) ? tune.scale_mul : 1.0f);
        if (nvfp4_cfg_has_rsf(cfg) &&
                binding.working_quant_tensor_scales.size() == binding.quant_tensor_scales.size() &&
                (size_t) i03 < binding.working_quant_tensor_scales.size()) {
            binding.working_quant_tensor_scales[(size_t) i03] = tuned_x_scale;
        }

        const void * eval_x = source_view.raw;
        bool eval_bf16 = source_view.raw_bf16;
        float eval_x_scale = tuned_x_scale;
        int64_t eval_nrow = nrows;
        int64_t eval_n_per_row = n_per_row;
        const float * eval_qw = imatrix_slice;

        if (sample_only) {
            eval_x = sample_x.data();
            eval_bf16 = false;
            eval_x_scale = tuned_x_scale;
            eval_nrow = 1;
            eval_n_per_row = (int64_t) sample_x.size();
            eval_qw = sample_qw.empty() ? nullptr : sample_qw.data();
        }

        nvfp4_cuda_eval_result eval = {};
        quantize_cuda_eval_params_t eval_params;
        eval_params.x = eval_x;
        eval_params.x_bf16 = eval_bf16;
        eval_params.out = direct_runtime_patch ? nullptr : out_bytes.data() + (size_t) i03 * out_slice_bytes;
        eval_params.target = direct_runtime_patch ? binding.target : nullptr;
        eval_params.slice_index = i03;
        eval_params.nrow = eval_nrow;
        eval_params.n_per_row = eval_n_per_row;
        eval_params.qw = eval_qw;
        eval_params.x_scale = eval_x_scale;
        eval_params.a = tune.a;
        eval_params.b = tune.b;
        eval_params.nvfp4_cfg = &cfg;
        eval_params.eval = collect_eval_metrics ? &eval : nullptr;
        eval_params.stream_key = stream_key;
        const bool quant_ok = qtype::quantize_eval(eval_params);
        if (!quant_ok) {
            fprintf(stderr,
                "%s: CUDA quant/eval failed for %s slice=%" PRId64 " eval_rows=%" PRId64 " eval_cols=%" PRId64 " eval_bf16=%d eval_qw=%s x_scale=%g a=%g b=%g choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f\n",
                __func__,
                binding.name.c_str(),
                i03,
                eval_nrow,
                eval_n_per_row,
                eval_bf16 ? 1 : 0,
                eval_qw ? "yes" : "no",
                (double) eval_x_scale,
                (double) tune.a,
                (double) tune.b,
                cfg.choose46_mode,
                cfg.refit_iters,
                cfg.use_compand_sat,
                (double) cfg.cap_m6,
                (double) cfg.cap_m4);
            acc.ok = false;
            return;
        }
        if (collect_eval_metrics) {
            acc.sum_sq += eval.sum_sq;
            acc.sum_abs += eval.sum_abs;
            acc.max_abs = std::max(acc.max_abs, eval.max_abs);
            acc.count += eval.count;
        }
    };

    return quantize_binding_quantize<qtype::quant_type>(
        binding,
        nthread,
        sample_blocks_override,
        sample_phase,
        reinterpret_cast<uintptr_t>(&out_bytes),
        __func__,
        prepare,
        run_slice,
        sum_sq,
        sum_abs,
        max_abs,
        count);
}

static bool mxfp6_selector_quantize_binding(
    selector_binding & binding,
    float scale_mul,
    int nthread,
    std::vector<uint8_t> & out_bytes,
    std::vector<uint8_t> & out_scale_bytes,
    double & sum_sq,
    double & sum_abs,
    double & max_abs,
    int64_t & count,
    bool direct_runtime_patch = false,
    bool * direct_applied = nullptr) {
    if (direct_applied != nullptr) {
        *direct_applied = false;
    }
    using qtype = ggml_quantize_type_traits<GGML_TYPE_MXFP6_E2M3>;
    if (!(scale_mul > 0.0f) || !std::isfinite(scale_mul)) {
        return false;
    }

    bool scale_in_header = false;
    bool direct_patch = false;
    size_t out_header_bytes = 0;
    size_t out_slice_bytes = 0;
    const float * base_scales = nullptr;
    float * tuned_scales = nullptr;
    float input_scale = 1.0f;

    auto prepare = [&](const quantize_binding_shape_t & shape) -> bool {
        if (!quantize_binding_ensure_scale_bytes(binding)) {
            fprintf(stderr, "%s: missing MXFP6_E2M3 scale tensor for %s\n", __func__, binding.name.c_str());
            return false;
        }

        scale_in_header = binding.target_scale == nullptr;
        if (scale_in_header && shape.n_slices != 1) {
            fprintf(stderr, "%s: embedded MXFP6_E2M3 scale tuning only supports scalar header scale for %s (slices=%" PRId64 ")\n",
                    __func__, binding.name.c_str(), shape.n_slices);
            return false;
        }

        out_slice_bytes = shape.out_slice_bytes;
        out_header_bytes = scale_in_header ? MXFP6_HEADER_OFFSET : 0;
        direct_patch = direct_runtime_patch && binding.target != nullptr;
        if (direct_patch) {
            out_bytes.clear();
        } else {
            out_bytes.resize(out_header_bytes + (size_t) shape.n_slices * out_slice_bytes);
        }
        out_scale_bytes.resize((size_t) shape.n_slices * sizeof(float));
        base_scales = (const float *) binding.original_scale_bytes.data();
        tuned_scales = (float *) out_scale_bytes.data();
        input_scale = 1.0f;
        if (binding.original_target_bytes.size() >= MXFP6_HEADER_OFFSET) {
            const auto * header = (const tensor_mxfp6 *) binding.original_target_bytes.data();
            input_scale = header->input_scale > 0.0f && std::isfinite(header->input_scale) ? header->input_scale : 1.0f;
        }
        return true;
    };

    auto run_slice = [&](int64_t i03, void * stream_key, const quantize_binding_shape_t & shape, quantize_binding_slice_accum_t & acc) {
        const int64_t n_per_row = shape.n_per_row;
        const int64_t nrows = shape.nrows;
        std::vector<float> src_f32;
        if (!quantize_tensor_slice_to_f32(binding.source, i03, binding.source_tensor_scale, src_f32, shape.host_threads)) {
            fprintf(stderr, "%s: failed dequantizing source slice for %s slice=%" PRId64 "\n", __func__, binding.name.c_str(), i03);
            acc.ok = false;
            return;
        }
        if (src_f32.size() != (size_t) nrows * (size_t) n_per_row) {
            acc.ok = false;
            return;
        }

        float tensor_scale = base_scales[i03] * scale_mul;
        if (!(tensor_scale > 0.0f) || !std::isfinite(tensor_scale)) {
            tensor_scale = 1.0f;
        }
        tuned_scales[i03] = tensor_scale;

        const float * imatrix_slice = binding.imatrix_row ? (binding.imatrix_row + i03 * n_per_row) : nullptr;

        nvfp4_cuda_eval_result eval{};
        quantize_cuda_eval_params_t eval_params;
        eval_params.x = src_f32.data();
        eval_params.out = direct_patch ? nullptr : out_bytes.data() + out_header_bytes + (size_t) i03 * out_slice_bytes;
        eval_params.target = direct_patch ? binding.target : nullptr;
        eval_params.slice_index = i03;
        eval_params.nrow = nrows;
        eval_params.n_per_row = n_per_row;
        eval_params.qw = imatrix_slice;
        eval_params.weight_scale = tensor_scale;
        eval_params.input_scale = input_scale;
        eval_params.eval = &eval;
        eval_params.stream_key = stream_key;
        const bool ok = qtype::quantize_eval(eval_params);
        if (!ok) {
            fprintf(stderr, "%s: CUDA MXFP6_E2M3 quantize/eval failed for %s slice=%" PRId64 "\n",
                    __func__, binding.name.c_str(), i03);
            acc.ok = false;
            return;
        }
        acc.sum_sq += eval.sum_sq;
        acc.sum_abs += eval.sum_abs;
        acc.max_abs = std::max(acc.max_abs, eval.max_abs);
        acc.count += eval.count;
    };

    if (!quantize_binding_quantize<qtype::quant_type>(
            binding,
            nthread,
            0,
            0,
            0,
            __func__,
            prepare,
            run_slice,
            sum_sq,
            sum_abs,
            max_abs,
            count)) {
        return false;
    }

    if (!direct_patch && scale_in_header && count > 0) {
        auto * header = (tensor_mxfp6 *) out_bytes.data();
        header->weight_scale  = tuned_scales[0];
        header->input_scale   = input_scale;
        header->weight_scales = nullptr;
        header->input_scales  = nullptr;
    }

    if (direct_patch && count > 0 && direct_applied != nullptr) {
        *direct_applied = true;
    }
    return count > 0;
}

struct selector_proxy_metrics {
    bool ok = false;
    double rmse = std::numeric_limits<double>::infinity();
    double abs_mean = std::numeric_limits<double>::infinity();
    double max_abs = std::numeric_limits<double>::infinity();
    double score = std::numeric_limits<double>::infinity();
};

static selector_proxy_metrics selector_proxy_score(
        double sum_sq,
        double sum_abs,
        double max_abs,
        int64_t count) {
    selector_proxy_metrics out;
    if (count <= 0) {
        return out;
    }
    const double inv_count = 1.0 / (double) count;
    out.rmse = std::sqrt(std::max(0.0, sum_sq * inv_count));
    out.abs_mean = sum_abs * inv_count;
    out.max_abs = max_abs;
    out.score = out.rmse + 4.0 * out.abs_mean + 0.2 * out.max_abs;
    out.ok =
        std::isfinite(out.rmse) &&
        std::isfinite(out.abs_mean) &&
        std::isfinite(out.max_abs) &&
        std::isfinite(out.score);
    return out;
}

static selector_proxy_metrics selector_policy_proxy_metrics(
        const selector_policy & policy,
        bool prefer_survey) {
    selector_proxy_metrics out;
    if (prefer_survey && policy.has_survey) {
        out.ok = std::isfinite(policy.survey_proxy_score);
        out.rmse = policy.survey_proxy_rmse;
        out.abs_mean = policy.survey_proxy_abs_mean;
        out.max_abs = policy.survey_proxy_max_abs;
        out.score = policy.survey_proxy_score;
        return out;
    }
    out.ok = std::isfinite(policy.proxy_score);
    out.rmse = policy.proxy_rmse;
    out.abs_mean = policy.proxy_abs_mean;
    out.max_abs = policy.proxy_max_abs;
    out.score = policy.proxy_score;
    return out;
}

static bool selector_metric_close(double a, double b, double abs_tol, double rel_tol);

static int selector_policy_proxy_tie_rank(const selector_policy & policy) {
    int rank = 0;
    if (policy.name.find("recipe_") == 0) {
        rank -= 300;
    }
    if (policy.name == "baseline") {
        rank -= 250;
    } else if (policy.name.find("baseline") != std::string::npos) {
        rank -= 150;
    }
    if (policy.cfg.use_compand_sat == 0) {
        rank += 1000;
    }
    rank += 2 * std::abs(policy.cfg.refit_iters - 8);
    rank += (int) std::llround(std::fabs((double) policy.cfg.cap_m6 - 448.0) / 16.0);
    rank += (int) std::llround(std::fabs((double) policy.cfg.cap_m4 - 256.0) / 16.0);
    return rank;
}

static bool selector_policy_proxy_less(const selector_policy & a, const selector_policy & b) {
    if (a.has_survey != b.has_survey) return a.has_survey > b.has_survey;
    const bool prefer_survey = a.has_survey && b.has_survey;
    const selector_proxy_metrics am = selector_policy_proxy_metrics(a, prefer_survey);
    const selector_proxy_metrics bm = selector_policy_proxy_metrics(b, prefer_survey);
    if (!selector_metric_close(am.score, bm.score, 1e-8, 1e-7)) return am.score < bm.score;
    const int ar = selector_policy_proxy_tie_rank(a);
    const int br = selector_policy_proxy_tie_rank(b);
    if (ar != br) return ar < br;
    if (am.score != bm.score) return am.score < bm.score;
    return a.name < b.name;
}

static bool selector_metric_close(double a, double b, double abs_tol, double rel_tol) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return a == b;
    }
    const double tol = std::max(abs_tol, rel_tol * std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= tol;
}

static bool selector_policy_is_recipe(const std::string & name) {
    return name == "recipe_current" || name.rfind("recipe_", 0) == 0;
}

static bool selector_proxy_equivalent(
        const selector_policy & a,
        const selector_policy & b,
        bool prefer_survey) {
    if (selector_policy_is_recipe(a.name) != selector_policy_is_recipe(b.name)) {
        return false;
    }
    if (nvfp4_cfg_has_rsf(a.cfg) != nvfp4_cfg_has_rsf(b.cfg)) {
        return false;
    }
    const bool require_cfg_match = selector_control_i64("DEDUP_REQUIRE_CFG", 0) != 0;
    if (require_cfg_match && (
            a.cfg.choose46_mode != b.cfg.choose46_mode ||
            a.cfg.refit_iters != b.cfg.refit_iters ||
            a.cfg.use_compand_sat != b.cfg.use_compand_sat ||
            a.cfg.reserved_i32 != b.cfg.reserved_i32 ||
            a.cfg.cap_m6 != b.cfg.cap_m6 ||
            a.cfg.cap_m4 != b.cfg.cap_m4)) {
        return false;
    }
    const bool use_survey = prefer_survey && a.has_survey && b.has_survey;
    const selector_proxy_metrics am = selector_policy_proxy_metrics(a, use_survey);
    const selector_proxy_metrics bm = selector_policy_proxy_metrics(b, use_survey);

    return
        selector_metric_close(am.score,    bm.score,    1e-8, 1e-7) &&
        selector_metric_close(am.rmse,     bm.rmse,     1e-8, 1e-7) &&
        selector_metric_close(am.abs_mean, bm.abs_mean, 1e-8, 1e-7) &&
        selector_metric_close(am.max_abs,  bm.max_abs,  1e-7, 1e-7);
}

static bool selector_cfg_equal(const nvfp4_cuda_runtime_cfg & a, const nvfp4_cuda_runtime_cfg & b) {
    return
        a.choose46_mode == b.choose46_mode &&
        a.refit_iters == b.refit_iters &&
        a.use_compand_sat == b.use_compand_sat &&
        a.reserved_i32 == b.reserved_i32 &&
        a.cap_m6 == b.cap_m6 &&
        a.cap_m4 == b.cap_m4;
}

static nvfp4_cuda_runtime_cfg selector_cfg_without_rsf(nvfp4_cuda_runtime_cfg cfg) {
    cfg.reserved_i32 &= ~(NVFP4_CUDA_FLAG_RSF | NVFP4_CUDA_RSF_MODE_MASK | NVFP4_CUDA_RSF_DEPTH_MASK);
    return cfg;
}

static nvfp4_cuda_runtime_cfg selector_cfg_with_rsf(nvfp4_cuda_runtime_cfg cfg, int rsf_mode, int rsf_depth) {
    cfg.reserved_i32 |= NVFP4_CUDA_FLAG_RSF;
    nvfp4_cfg_set_rsf_mode(cfg, rsf_mode);
    nvfp4_cfg_set_rsf_depth(cfg, rsf_depth);
    return cfg;
}

static nvfp4_cuda_runtime_cfg selector_cfg_as_rsf(nvfp4_cuda_runtime_cfg cfg, int rsf_mode, int rsf_depth) {
    cfg = selector_cfg_without_rsf(cfg);
    return selector_cfg_with_rsf(cfg, rsf_mode, rsf_depth);
}

static bool selector_policy_name_has_rsf(const std::string & name) {
    return (name.size() >= 4 && name.compare(name.size() - 4, 4, "_rsf") == 0) ||
           name.find("_rsf_") != std::string::npos;
}

static int selector_effective_rsf_mode(const nvfp4_cuda_runtime_cfg * recipe_cfg = nullptr) {
    int rsf_mode = recipe_cfg != nullptr ? nvfp4_cfg_rsf_mode(*recipe_cfg) : NVFP4_CUDA_RSF_MODE_TENSOR;
    const std::string rsf_mode_control = selector_control_string("RSF_MODE");
    if (!rsf_mode_control.empty() && !nvfp4_parse_rsf_mode(rsf_mode_control.c_str(), rsf_mode)) {
        fprintf(stderr,
            "%s: invalid RSF mode '%s'; using tensor\n",
            __func__,
            rsf_mode_control.c_str());
        rsf_mode = NVFP4_CUDA_RSF_MODE_TENSOR;
    }
    return rsf_mode;
}

static int selector_effective_rsf_depth(const nvfp4_cuda_runtime_cfg * recipe_cfg = nullptr) {
    int rsf_depth = recipe_cfg != nullptr && nvfp4_cfg_has_rsf(*recipe_cfg)
        ? nvfp4_cfg_rsf_depth(*recipe_cfg)
        : NVFP4_CUDA_RSF_DEPTH_DEEPER;
    const std::string rsf_depth_control = selector_control_string("RSF_DEPTH");
    if (!rsf_depth_control.empty() && !nvfp4_parse_rsf_depth(rsf_depth_control.c_str(), rsf_depth)) {
        fprintf(stderr,
            "%s: invalid RSF depth '%s'; using deeper\n",
            __func__,
            rsf_depth_control.c_str());
        rsf_depth = NVFP4_CUDA_RSF_DEPTH_DEEPER;
    }
    return rsf_depth;
}

static std::vector<selector_policy> selector_default_policies(
        const nvfp4_cuda_runtime_cfg * recipe_cfg = nullptr,
        const std::string & recipe_policy_name = {}) {
    std::vector<selector_policy> out;
    const int rsf_mode = selector_effective_rsf_mode(recipe_cfg);
    const int rsf_depth = selector_effective_rsf_depth(recipe_cfg);
    auto push = [&](const std::string & name, nvfp4_cuda_runtime_cfg cfg) {
        for (const auto & existing : out) {
            if (selector_cfg_equal(existing.cfg, cfg)) {
                return false;
            }
        }
        selector_policy p;
        p.name = name;
        p.cfg = cfg;
        out.push_back(std::move(p));
        return true;
    };
    auto push_family = [&](const std::string & name, nvfp4_cuda_runtime_cfg cfg) {
        cfg = selector_cfg_as_rsf(cfg, rsf_mode, rsf_depth);
        push(name, cfg);
    };
    if (recipe_cfg != nullptr) {
        push_family(recipe_policy_name.empty() ? "recipe_current" : "recipe_" + recipe_policy_name, *recipe_cfg);
    }
    for (const llama_nvfp4_named_preset & preset : llama_nvfp4_preset_catalog()) {
        push_family(preset.name, preset.cfg);
    }
    return out;
}

static bool selector_policy_is_awq_tail(const std::string & name) {
    return name.find("asym_tail") != std::string::npos ||
           name.find("awq_tail") != std::string::npos;
}

static bool selector_policy_is_rsf_family(const selector_policy & policy) {
    return nvfp4_cfg_has_rsf(policy.cfg) ||
           selector_policy_name_has_rsf(policy.name);
}

static bool selector_ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string selector_rsf_base_policy_name(const std::string & name) {
    if (name.size() >= 4 && name.compare(name.size() - 4, 4, "_rsf") == 0) {
        return name.substr(0, name.size() - 4);
    }
    const size_t infix = name.find("_rsf_");
    if (infix != std::string::npos) {
        std::string out = name;
        out.erase(infix, 4);
        return out;
    }
    return name;
}

static std::vector<std::string> selector_rsf_base_policy_candidates(const std::string & name) {
    std::vector<std::string> out;
    auto push = [&](const std::string & candidate) {
        if (!candidate.empty() &&
                candidate != name &&
                std::find(out.begin(), out.end(), candidate) == out.end()) {
            out.push_back(candidate);
        }
    };

    const std::string direct = selector_rsf_base_policy_name(name);
    push(direct);

    static const std::array<const char *, 5> refine_suffixes = {
        "_recover_m6",
        "_gentler_m4",
        "_refit4",
        "_refit16",
        "_compand_flip",
    };
    for (const char * suffix : refine_suffixes) {
        if (selector_ends_with(direct, suffix)) {
            push(direct.substr(0, direct.size() - std::strlen(suffix)));
        }
        if (selector_ends_with(name, suffix)) {
            const std::string parent = name.substr(0, name.size() - std::strlen(suffix));
            push(selector_rsf_base_policy_name(parent));
        }
    }

    return out;
}

static bool selector_policy_is_fixed_cap_sweep(const std::string & name) {
    const std::string base = selector_rsf_base_policy_name(name);
    return
        base.find("_recover_") != std::string::npos ||
        base.find("asym_tail") != std::string::npos ||
        base.find("adaptive_nocompand_") != std::string::npos ||
        base.find("adaptive_refit") == 0 ||
        base.find("_moe") != std::string::npos ||
        base.find("moe_") != std::string::npos ||
        base == "baseline_refit24_384_256";
}

static const selector_policy * selector_find_rsf_base_policy(
        const std::vector<selector_policy> & policies,
        const selector_policy & policy,
        bool require_measured = false) {
    for (const std::string & candidate : selector_rsf_base_policy_candidates(policy.name)) {
        auto it = std::find_if(policies.begin(), policies.end(), [&](const selector_policy & p) {
            return p.name == candidate &&
                   !selector_policy_is_rsf_family(p) &&
                   (!require_measured || p.measured.ok);
        });
        if (it != policies.end()) {
            return &*it;
        }
    }
    return nullptr;
}

static bool selector_write_rsf_report(
        const std::string & report_path,
        const std::vector<selector_policy> & policies) {
    std::vector<const selector_policy *> rsf_policies;
    for (const auto & policy : policies) {
        if (selector_policy_is_rsf_family(policy) && !policy.tensor_records.empty()) {
            rsf_policies.push_back(&policy);
        }
    }
    if (rsf_policies.empty()) {
        return false;
    }

    std::filesystem::path path(report_path);
    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream out(report_path);
    if (!out) {
        fprintf(stderr, "%s: failed writing RSF report %s\n", __func__, report_path.c_str());
        return false;
    }

    struct tensor_row {
        std::string tensor;
        std::string policy;
        std::string cls;
        double scale_mul = 1.0;
        double proxy_delta = std::numeric_limits<double>::quiet_NaN();
        double delta_mean_kld = std::numeric_limits<double>::quiet_NaN();
        double delta_p99 = std::numeric_limits<double>::quiet_NaN();
        double delta_same_top = std::numeric_limits<double>::quiet_NaN();
        bool win = false;
    };
    std::vector<tensor_row> rows;
    std::map<std::string, std::vector<double>> hist_by_class;

    out << "RSF refined scale fit report\n\n";
    out << "Per-tensor rows use tensor_delta_proxy for tensor-local reconstruction evidence.\n";
    out << "policy_delta_* fields are measured policy-wide KLD/PPL deltas versus the non-RSF companion.\n\n";
    out << "policy                         tensors tested    proxy wins    avg scale_mul    p10/p50/p90 scale_mul\n";
    for (const auto * policy : rsf_policies) {
        const selector_policy * base_policy =
            selector_find_rsf_base_policy(policies, *policy);
        const auto base_records = base_policy != nullptr
            ? selector_rsf_record_map(*base_policy)
            : std::unordered_map<std::string, const selector_rsf_tensor_record *>();

        int tested = 0;
        int wins = 0;
        std::vector<double> scale_muls;
        for (const auto & record : policy->tensor_records) {
            if (!std::isfinite(record.proxy_score)) {
                continue;
            }
            ++tested;
            const double scale_mul = selector_rsf_record_scale_mul(record);
            scale_muls.push_back(scale_mul);
            hist_by_class[selector_rsf_hist_class(record.tensor, record.cls)].push_back(scale_mul);

            double proxy_delta = std::numeric_limits<double>::quiet_NaN();
            auto bit = base_records.find(record.tensor);
            if (bit != base_records.end() && bit->second != nullptr && std::isfinite(bit->second->proxy_score)) {
                proxy_delta = record.proxy_score - bit->second->proxy_score;
            }
            const bool win = record.rsf_changed && std::isfinite(proxy_delta) && proxy_delta < 0.0;
            wins += win ? 1 : 0;

            tensor_row row;
            row.tensor = record.tensor;
            row.policy = policy->name;
            row.cls = selector_rsf_hist_class(record.tensor, record.cls);
            row.scale_mul = scale_mul;
            row.proxy_delta = proxy_delta;
            row.win = win;
            if (base_policy != nullptr && policy->measured.ok && base_policy->measured.ok) {
                row.delta_mean_kld = policy->measured.mean_kld - base_policy->measured.mean_kld;
                row.delta_p99 = policy->measured.kld_p99 - base_policy->measured.kld_p99;
                row.delta_same_top = policy->measured.same_top - base_policy->measured.same_top;
            }
            if (win || std::fabs(scale_mul - 1.0) > 1e-6) {
                rows.push_back(std::move(row));
            }
        }

        const double avg = scale_muls.empty()
            ? std::numeric_limits<double>::quiet_NaN()
            : std::accumulate(scale_muls.begin(), scale_muls.end(), 0.0) / (double) scale_muls.size();
        out << std::left << std::setw(30) << policy->name
            << std::right << std::setw(8) << tested
            << std::setw(14) << wins
            << std::setw(17) << std::fixed << std::setprecision(3) << avg
            << "           "
            << std::fixed << std::setprecision(3)
            << selector_percentile(scale_muls, 0.10) << '/'
            << selector_percentile(scale_muls, 0.50) << '/'
            << selector_percentile(scale_muls, 0.90) << '\n';
    }

    std::stable_sort(rows.begin(), rows.end(), [](const tensor_row & a, const tensor_row & b) {
        const double ap = std::isfinite(a.proxy_delta) ? a.proxy_delta : 0.0;
        const double bp = std::isfinite(b.proxy_delta) ? b.proxy_delta : 0.0;
        if (a.win != b.win) return a.win > b.win;
        if (ap != bp) return ap < bp;
        if (a.policy != b.policy) return a.policy < b.policy;
        return a.tensor < b.tensor;
    });

    out << "\nper tensor RSF rows\n";
    out << "tensor\tpolicy\ttensor_class\tscale_mul\tpolicy_delta_mean_kld\tpolicy_delta_p99\tpolicy_delta_same_top\ttensor_delta_proxy\n";
    auto write_num = [&](double v, int precision) {
        if (std::isfinite(v)) {
            out << std::fixed << std::setprecision(precision) << v;
        } else {
            out << "n/a";
        }
    };
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        out << row.tensor << '\t'
            << row.policy << '\t'
            << row.cls << '\t'
            << std::fixed << std::setprecision(6) << row.scale_mul << '\t';
        write_num(row.delta_mean_kld, 6);
        out << '\t';
        write_num(row.delta_p99, 6);
        out << '\t';
        write_num(row.delta_same_top, 6);
        out << '\t';
        write_num(row.proxy_delta, 6);
        out << '\n';
    }

    out << "\nscale_mul histogram by tensor class:\n";
    const std::vector<std::pair<double, double>> bins = {
        {0.0, 0.875},
        {0.875, 0.9375},
        {0.9375, 0.96875},
        {0.96875, 1.0},
        {1.0, 1.03125},
        {1.03125, 1.0625},
        {1.0625, 1.09375},
        {1.09375, 1.25},
        {1.25, std::numeric_limits<double>::infinity()},
    };
    for (const auto & kv : hist_by_class) {
        out << "  " << kv.first << ":";
        std::vector<int> counts(bins.size(), 0);
        for (double v : kv.second) {
            for (size_t bi = 0; bi < bins.size(); ++bi) {
                if (v >= bins[bi].first && v < bins[bi].second) {
                    ++counts[bi];
                    break;
                }
            }
        }
        for (size_t bi = 0; bi < bins.size(); ++bi) {
            out << ' ' << '[' << std::fixed << std::setprecision(3) << bins[bi].first
                << ',' << (std::isfinite(bins[bi].second) ? std::to_string(bins[bi].second).substr(0, 5) : std::string("inf"))
                << ")=" << counts[bi];
        }
        out << '\n';
    }
    if (hist_by_class.find("MTP/NextN") == hist_by_class.end()) {
        out << "  MTP/NextN: uses explicit MTP tensor policy when quantized\n";
    }

    fprintf(stderr, "%s: wrote RSF report %s\n", __func__, report_path.c_str());
    return true;
}

static std::unordered_set<std::string> selector_policy_set_from_control(const char * suffix) {
    std::unordered_set<std::string> out;
    const std::string raw = selector_control_string(suffix);
    if (raw.empty()) {
        return out;
    }

    std::string normalized(raw);
    for (char & ch : normalized) {
        if (ch == ';' || ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ',';
        }
    }

    std::stringstream ss(normalized);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (!token.empty()) {
            out.insert(std::move(token));
        }
    }
    return out;
}

static std::vector<ggml_type> selector_type_list_from_control(const char * suffix) {
    std::vector<ggml_type> out;
    const std::string raw = selector_control_string(suffix);
    if (raw.empty()) {
        return out;
    }

    std::string normalized(raw);
    for (char & ch : normalized) {
        if (ch == ';' || ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ',';
        }
    }

    std::stringstream ss(normalized);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) {
            continue;
        }
        const ggml_type type = parse_ggml_type(token.c_str());
        if (type == GGML_TYPE_COUNT || type == GGML_TYPE_NVFP4) {
            continue;
        }
        if (std::find(out.begin(), out.end(), type) == out.end()) {
            out.push_back(type);
        }
    }
    return out;
}

static void selector_push_policy_unique(std::vector<selector_policy> & out, std::string name, const nvfp4_cuda_runtime_cfg & cfg) {
    for (const auto & p : out) {
        if (selector_cfg_equal(p.cfg, cfg)) {
            return;
        }
    }
    selector_policy p;
    p.name = std::move(name);
    p.cfg = cfg;
    out.push_back(std::move(p));
}

static std::vector<std::pair<float, float>> selector_cap_pairs_for_class(
    selector_tensor_class cls,
    const nvfp4_cuda_runtime_cfg & base_cfg) {
    std::vector<std::pair<float, float>> out;
    auto add = [&](float c6, float c4) {
        c6 = std::clamp(c6, 256.0f, 544.0f);
        c4 = std::clamp(c4, 160.0f, 352.0f);
        if (c4 > c6) {
            c4 = c6;
        }
        const std::pair<float, float> p { c6, c4 };
        if (std::find(out.begin(), out.end(), p) == out.end()) {
            out.push_back(p);
        }
    };

    add(base_cfg.cap_m6, base_cfg.cap_m4);

    switch (cls) {
        case selector_tensor_class::ATTN_QKV:
        case selector_tensor_class::ATTN_OUT:
        case selector_tensor_class::SSM:
            add(320.0f, 224.0f);
            add(352.0f, 224.0f);
            add(416.0f, 192.0f);
            add(384.0f, 256.0f);
            add(448.0f, 224.0f);
            add(416.0f, 256.0f);
            add(448.0f, 320.0f);
            add(480.0f, 320.0f);
            add(384.0f, 384.0f);
            break;
        case selector_tensor_class::EXPERT_UP_GATE:
        case selector_tensor_class::EXPERT_DOWN:
            add(352.0f, 224.0f);
            add(384.0f, 224.0f);
            add(384.0f, 256.0f);
            add(416.0f, 256.0f);
            add(448.0f, 256.0f);
            add(480.0f, 256.0f);
            add(512.0f, 288.0f);
            add(448.0f, 320.0f);
            add(480.0f, 320.0f);
            add(512.0f, 320.0f);
            add(512.0f, 352.0f);
            add(544.0f, 320.0f);
            add(544.0f, 352.0f);
            break;
        default:
            add(320.0f, 224.0f);
            add(352.0f, 224.0f);
            add(384.0f, 256.0f);
            add(448.0f, 224.0f);
            add(448.0f, 256.0f);
            add(480.0f, 256.0f);
            add(448.0f, 320.0f);
            break;
    }

    return out;
}

static std::vector<int> selector_refit_candidates_for_class(
    selector_tensor_class cls,
    int base_refit) {
    std::vector<int> out;
    auto add = [&](int refit) {
        refit = std::max(2, std::min(32, refit));
        if (std::find(out.begin(), out.end(), refit) == out.end()) {
            out.push_back(refit);
        }
    };

    add(base_refit);
    add(4);
    add(8);
    add(12);
    add(16);
    add(24);
    if (cls == selector_tensor_class::ATTN_QKV ||
        cls == selector_tensor_class::ATTN_OUT ||
        cls == selector_tensor_class::SSM ||
        cls == selector_tensor_class::EXPERT_UP_GATE ||
        cls == selector_tensor_class::EXPERT_DOWN) {
        add(32);
    }
    return out;
}

static std::vector<selector_policy> selector_refine_policies(
        const std::vector<selector_policy> & ranked,
        int rsf_mode = NVFP4_CUDA_RSF_MODE_TENSOR,
        int rsf_depth = NVFP4_CUDA_RSF_DEPTH_DEEPER) {
    const int refine_top = (int) std::max<int64_t>(1, selector_control_i64("REFINE_TOP", 4));
    const int refine_budget = (int) std::max<int64_t>(0, selector_control_i64("REFINE_BUDGET", 24));
    if (refine_budget <= 0 || ranked.empty()) {
        return {};
    }

    auto clamp_cfg = [](nvfp4_cuda_runtime_cfg cfg) {
        cfg = selector_cfg_without_rsf(cfg);
        cfg.refit_iters = std::max(2, std::min(32, cfg.refit_iters));
        cfg.cap_m6 = std::clamp(cfg.cap_m6, 256.0f, 544.0f);
        cfg.cap_m4 = std::clamp(cfg.cap_m4, 160.0f, 352.0f);
        if (cfg.cap_m4 > cfg.cap_m6) {
            cfg.cap_m4 = cfg.cap_m6;
        }
        return cfg;
    };

    std::vector<selector_policy> out;
    auto push_variant = [&](const std::string & name, nvfp4_cuda_runtime_cfg cfg) {
        if ((int) out.size() >= refine_budget) {
            return;
        }
        cfg = clamp_cfg(cfg);
        cfg = selector_cfg_as_rsf(cfg, rsf_mode, rsf_depth);
        for (const selector_policy & existing : ranked) {
            if (selector_cfg_equal(existing.cfg, cfg)) {
                return;
            }
        }
        selector_push_policy_unique(out, name, cfg);
    };

    for (int i = 0; i < refine_top && i < (int) ranked.size() && (int) out.size() < refine_budget; ++i) {
        const selector_policy & seed = ranked[(size_t) i];
        if (seed.proxy_rejected || !std::isfinite(seed.proxy_score)) {
            continue;
        }

        const std::string base_name = selector_rsf_base_policy_name(seed.name);
        const nvfp4_cuda_runtime_cfg base = clamp_cfg(seed.cfg);

        push_variant(base_name + "_frontier", base);

        nvfp4_cuda_runtime_cfg refit16 = base;
        refit16.choose46_mode = NVFP4_CUDA_CHOOSE46_ADAPTIVE;
        refit16.refit_iters = std::max(refit16.refit_iters, 16);
        refit16.use_compand_sat = 1;
        push_variant(base_name + "_refit16", refit16);

        nvfp4_cuda_runtime_cfg flip = base;
        flip.refit_iters = std::max(flip.refit_iters, 8);
        flip.use_compand_sat = flip.use_compand_sat ? 0 : 1;
        push_variant(base_name + "_compand_flip", flip);
    }

    if (!out.empty()) {
        fprintf(stderr,
            "%s: selector added %zu compact proxy-frontier policy candidate(s) from top %d/%zu seed policy candidate(s)\n",
            __func__,
            out.size(),
            std::min<int>(refine_top, (int) ranked.size()),
            ranked.size());
    }

    return out;
}

static int64_t selector_rescue_sample_blocks(int64_t nb_total, selector_tensor_class cls) {
    const int64_t base = selector_sample_blocks(nb_total);
    const int64_t mult = SELECTOR_RESCUE_NVFP4_SAMPLE_MULT;
    const int64_t max_blocks = selector_control_i64("RESCUE_NVFP4_MAX_BLOCKS", 16384);
    int64_t min_blocks = SELECTOR_RESCUE_NVFP4_MIN_BLOCKS;
    switch (cls) {
        case selector_tensor_class::ATTN_QKV:
        case selector_tensor_class::ATTN_OUT:
        case selector_tensor_class::SSM:
            min_blocks = std::max<int64_t>(min_blocks, 8192);
            break;
        case selector_tensor_class::EXPERT_UP_GATE:
        case selector_tensor_class::EXPERT_DOWN:
            min_blocks = std::max<int64_t>(min_blocks, 16384);
            break;
        default:
            break;
    }
    const int64_t target = std::max<int64_t>(base * std::max<int64_t>(1, mult), min_blocks);
    return std::min<int64_t>(nb_total, std::max<int64_t>(1, std::min<int64_t>(target, max_blocks)));
}

static int64_t selector_rescue_refine_blocks(int64_t nb_total, selector_tensor_class cls, int64_t coarse_blocks) {
    const int64_t mult = SELECTOR_RESCUE_NVFP4_REFINE_MULT;
    const int64_t max_blocks = selector_control_i64("RESCUE_NVFP4_REFINE_MAX_BLOCKS", 32768);
    int64_t min_blocks = SELECTOR_RESCUE_NVFP4_REFINE_MIN_BLOCKS;
    switch (cls) {
        case selector_tensor_class::ATTN_QKV:
        case selector_tensor_class::ATTN_OUT:
        case selector_tensor_class::SSM:
            min_blocks = std::max<int64_t>(min_blocks, 16384);
            break;
        case selector_tensor_class::EXPERT_UP_GATE:
        case selector_tensor_class::EXPERT_DOWN:
            min_blocks = std::max<int64_t>(min_blocks, 32768);
            break;
        default:
            break;
    }
    const int64_t target = std::max<int64_t>(coarse_blocks * std::max<int64_t>(1, mult), min_blocks);
    return std::min<int64_t>(nb_total, std::max<int64_t>(1, std::min<int64_t>(target, max_blocks)));
}

static int64_t selector_rescue_guard_blocks(int64_t nb_total, selector_tensor_class cls, int64_t refine_blocks) {
    const int64_t mult = SELECTOR_RESCUE_NVFP4_GUARD_MULT;
    const int64_t max_blocks = selector_control_i64("RESCUE_NVFP4_GUARD_MAX_BLOCKS", 32768);
    int64_t min_blocks = SELECTOR_RESCUE_NVFP4_GUARD_MIN_BLOCKS;
    switch (cls) {
        case selector_tensor_class::ATTN_QKV:
        case selector_tensor_class::ATTN_OUT:
        case selector_tensor_class::SSM:
            min_blocks = std::max<int64_t>(min_blocks, 32768);
            break;
        case selector_tensor_class::EXPERT_UP_GATE:
        case selector_tensor_class::EXPERT_DOWN:
            min_blocks = std::max<int64_t>(min_blocks, 65536);
            break;
        default:
            break;
    }
    const int64_t target = std::max<int64_t>(refine_blocks * std::max<int64_t>(1, mult), min_blocks);
    return std::min<int64_t>(nb_total, std::max<int64_t>(1, std::min<int64_t>(target, max_blocks)));
}

static std::vector<selector_policy> selector_rescue_policies(
    const nvfp4_cuda_runtime_cfg & base_cfg,
    selector_tensor_class cls) {
    std::vector<selector_policy> out;
    const int rsf_mode = selector_effective_rsf_mode(&base_cfg);
    const int rsf_depth = selector_effective_rsf_depth(&base_cfg);
    selector_push_policy_unique(out, "rescue_base",
        selector_cfg_as_rsf(base_cfg, rsf_mode, rsf_depth));

    const int rescue_budget = (int) std::max<int64_t>(
        8,
        selector_control_i64("RESCUE_NVFP4_POLICY_BUDGET", SELECTOR_RESCUE_NVFP4_POLICY_BUDGET));
    auto add = [&](const std::string & name, nvfp4_cuda_runtime_cfg cfg) {
        if ((int) out.size() >= rescue_budget) {
            return;
        }
        cfg = selector_cfg_without_rsf(cfg);
        cfg.refit_iters = std::max(2, std::min(32, cfg.refit_iters));
        cfg.cap_m6 = std::clamp(cfg.cap_m6, 256.0f, 544.0f);
        cfg.cap_m4 = std::clamp(cfg.cap_m4, 160.0f, 352.0f);
        cfg = selector_cfg_as_rsf(cfg, rsf_mode, rsf_depth);
        selector_push_policy_unique(out, name, cfg);
    };

    const auto refits = selector_refit_candidates_for_class(cls, base_cfg.refit_iters);

    for (int refit : refits) {
        if (refit != base_cfg.refit_iters) {
            add("rescue_refit", { base_cfg.choose46_mode, refit, base_cfg.use_compand_sat, base_cfg.reserved_i32, base_cfg.cap_m6, base_cfg.cap_m4 });
        }
    }
    add("rescue_compand_flip", { base_cfg.choose46_mode, std::max(base_cfg.refit_iters, 8), base_cfg.use_compand_sat ? 0 : 1, base_cfg.reserved_i32, base_cfg.cap_m6, base_cfg.cap_m4 });

    if (base_cfg.choose46_mode == NVFP4_CUDA_CHOOSE46_ADAPTIVE) {
        add("rescue_refit16", { NVFP4_CUDA_CHOOSE46_ADAPTIVE, std::max(base_cfg.refit_iters, 16), 1, base_cfg.reserved_i32, base_cfg.cap_m6, base_cfg.cap_m4 });
    } else {
        add("rescue_adaptive", { NVFP4_CUDA_CHOOSE46_ADAPTIVE, std::max(base_cfg.refit_iters, 10), base_cfg.use_compand_sat, base_cfg.reserved_i32, base_cfg.cap_m6, base_cfg.cap_m4 });
    }

    if ((int) out.size() > rescue_budget) {
        out.resize((size_t) rescue_budget);
    }
    return out;
}

static bool selector_find_tensor_rescue_cfg(
    const selector_binding & binding,
    const nvfp4_cuda_runtime_cfg & base_cfg,
    int nthread,
    nvfp4_cuda_runtime_cfg & out_cfg,
    std::string & out_policy_name,
    int64_t & out_sample_blocks,
    double & out_base_score,
    double & out_best_score) {
    const int64_t nb_total = (binding.source->ne[0] * binding.source->ne[1]) / SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
    const int64_t sample_blocks = selector_rescue_sample_blocks(nb_total, binding.cls);
    const int64_t refine_blocks = selector_rescue_refine_blocks(nb_total, binding.cls, sample_blocks);
    const int64_t guard_blocks  = selector_rescue_guard_blocks(nb_total, binding.cls, refine_blocks);
    const int phase_count =
        (binding.cls == selector_tensor_class::EXPERT_UP_GATE || binding.cls == selector_tensor_class::EXPERT_DOWN ||
         binding.cls == selector_tensor_class::ATTN_QKV || binding.cls == selector_tensor_class::ATTN_OUT ||
         binding.cls == selector_tensor_class::SSM) ? 2 :
        1;
    const double phase_tail_weight = SELECTOR_RESCUE_NVFP4_PHASE_TAIL_WEIGHT;
    const int64_t phase_stride = selector_sample_phase_stride(nb_total, binding.cls);
    auto policies = selector_rescue_policies(base_cfg, binding.cls);
    if (policies.empty()) {
        return false;
    }

    out_base_score = std::numeric_limits<double>::infinity();
    out_best_score = std::numeric_limits<double>::infinity();
    out_cfg = base_cfg;
    out_policy_name = "rescue_base";
    out_sample_blocks = sample_blocks;

    struct scored_policy {
        size_t index = 0;
        double score = std::numeric_limits<double>::infinity();
    };

    auto score_policy = [&](const selector_policy & policy, int64_t blocks, std::vector<uint8_t> & tmp_bytes, double & score_out) -> bool {
        double phase_sum = 0.0;
        double phase_worst = 0.0;
        int phases_done = 0;
        for (int phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
            double sum_sq = 0.0;
            double sum_abs = 0.0;
            double max_abs = 0.0;
            int64_t count = 0;
            const int64_t sample_phase =
                phase_idx == 0 ? 0 :
                (phase_stride * phase_idx + std::max<int32_t>(0, binding.layer + 1)) % std::max<int64_t>(1, nb_total);
            if (!nvfp4_selector_quantize_binding(binding, policy.cfg, nthread, tmp_bytes, sum_sq, sum_abs, max_abs, count, blocks, sample_phase) || count <= 0) {
                return false;
            }
            const selector_proxy_metrics phase_metrics =
                selector_proxy_score(sum_sq, sum_abs, max_abs, count);
            if (!phase_metrics.ok) {
                return false;
            }
            const double phase_score = phase_metrics.score;
            phase_sum += phase_score;
            phase_worst = std::max(phase_worst, phase_score);
            ++phases_done;
        }
        if (phases_done <= 0) {
            return false;
        }
        score_out = (phase_sum / (double) phases_done) + phase_tail_weight * phase_worst;
        return std::isfinite(score_out);
    };

    const int refine_top = (int) std::max<int64_t>(
        1,
        selector_control_i64("RESCUE_NVFP4_REFINE_TOP", SELECTOR_RESCUE_NVFP4_REFINE_TOP));
    const int guard_top  = (int) std::max<int64_t>(
        1,
        selector_control_i64("RESCUE_NVFP4_GUARD_TOP", SELECTOR_RESCUE_NVFP4_GUARD_TOP));

    std::vector<uint8_t> tmp_bytes;
    std::vector<scored_policy> coarse_ranked;
    coarse_ranked.reserve(policies.size());
    for (size_t i = 0; i < policies.size(); ++i) {
        double score = std::numeric_limits<double>::infinity();
        if (!score_policy(policies[i], sample_blocks, tmp_bytes, score)) {
            continue;
        }
        coarse_ranked.push_back({ i, score });
        if (selector_cfg_equal(policies[i].cfg, base_cfg)) {
            out_base_score = score;
        }
        if (score < out_best_score) {
            out_best_score = score;
            out_cfg = policies[i].cfg;
            out_policy_name = policies[i].name;
        }
    }

    if (coarse_ranked.empty()) {
        return false;
    }

    std::sort(coarse_ranked.begin(), coarse_ranked.end(), [](const scored_policy & a, const scored_policy & b) {
        if (a.score != b.score) return a.score < b.score;
        return a.index < b.index;
    });

    if (refine_blocks > sample_blocks) {
        std::vector<scored_policy> refine_ranked;
        refine_ranked.reserve((size_t) refine_top + 1);
        auto maybe_push_refine = [&](size_t idx) {
            if (std::find_if(refine_ranked.begin(), refine_ranked.end(), [&](const scored_policy & sp) { return sp.index == idx; }) != refine_ranked.end()) {
                return;
            }
            double score = std::numeric_limits<double>::infinity();
            if (!score_policy(policies[idx], refine_blocks, tmp_bytes, score)) {
                return;
            }
            refine_ranked.push_back({ idx, score });
        };

        for (int i = 0; i < refine_top && i < (int) coarse_ranked.size(); ++i) {
            maybe_push_refine(coarse_ranked[(size_t) i].index);
        }
        for (size_t i = 0; i < policies.size(); ++i) {
            if (selector_cfg_equal(policies[i].cfg, base_cfg)) {
                maybe_push_refine(i);
                break;
            }
        }

        if (!refine_ranked.empty()) {
            std::sort(refine_ranked.begin(), refine_ranked.end(), [](const scored_policy & a, const scored_policy & b) {
                if (a.score != b.score) return a.score < b.score;
                return a.index < b.index;
            });
            coarse_ranked.swap(refine_ranked);
            out_best_score = coarse_ranked.front().score;
            out_cfg = policies[coarse_ranked.front().index].cfg;
            out_policy_name = policies[coarse_ranked.front().index].name;
            for (const auto & ranked : coarse_ranked) {
                if (selector_cfg_equal(policies[ranked.index].cfg, base_cfg)) {
                    out_base_score = ranked.score;
                    break;
                }
            }
        }
    }

    if (guard_blocks > refine_blocks) {
        std::vector<scored_policy> guard_ranked;
        guard_ranked.reserve((size_t) guard_top + 1);
        auto maybe_push_guard = [&](size_t idx) {
            if (std::find_if(guard_ranked.begin(), guard_ranked.end(), [&](const scored_policy & sp) { return sp.index == idx; }) != guard_ranked.end()) {
                return;
            }
            double score = std::numeric_limits<double>::infinity();
            if (!score_policy(policies[idx], guard_blocks, tmp_bytes, score)) {
                return;
            }
            guard_ranked.push_back({ idx, score });
        };

        for (int i = 0; i < guard_top && i < (int) coarse_ranked.size(); ++i) {
            maybe_push_guard(coarse_ranked[(size_t) i].index);
        }
        for (size_t i = 0; i < policies.size(); ++i) {
            if (selector_cfg_equal(policies[i].cfg, base_cfg)) {
                maybe_push_guard(i);
                break;
            }
        }

        if (!guard_ranked.empty()) {
            std::sort(guard_ranked.begin(), guard_ranked.end(), [](const scored_policy & a, const scored_policy & b) {
                if (a.score != b.score) return a.score < b.score;
                return a.index < b.index;
            });
            out_best_score = guard_ranked.front().score;
            out_cfg = policies[guard_ranked.front().index].cfg;
            out_policy_name = policies[guard_ranked.front().index].name;
            for (const auto & ranked : guard_ranked) {
                if (selector_cfg_equal(policies[ranked.index].cfg, base_cfg)) {
                    out_base_score = ranked.score;
                    break;
                }
            }
        }
    }

    if (!std::isfinite(out_base_score)) {
        out_base_score = out_best_score;
    }
    out_sample_blocks = guard_blocks > refine_blocks ? guard_blocks : (refine_blocks > sample_blocks ? refine_blocks : sample_blocks);
    return std::isfinite(out_best_score);
}

static bool selector_choose_policy(
    const std::string & source_model_path,
    const std::string & checkpoint_model_path,
    const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
    int nthread,
    const selector_kld_subset & kld,
    const selector_kld_subset * kld_holdout,
    const selector_rank_config & rank_cfg,
    const nvfp4_cuda_runtime_cfg * recipe_cfg,
    const std::string & recipe_policy_name,
    int32_t nvfp4_input_scale_policy,
    int32_t selector_eval_batch_override,
    int32_t nvfp4_encoder_threads,
    nvfp4_cuda_runtime_cfg & out_cfg,
    std::string & out_name,
        bool * out_kept_seed,
	    std::vector<tensor_type_option> * out_tensor_overrides) {
	    common_init();
    if (out_kept_seed != nullptr) {
        *out_kept_seed = false;
    }
    selector_progress_heartbeat full_quant_eta("full_quant_eta");
    int64_t full_quant_eta_done = 0;
    int64_t full_quant_eta_total = 0;
    int64_t stageb_policy_total = 0;
    int64_t stageb_policy_done = 0;
    double stageb_policy_done_seconds = 0.0;
    int64_t stageb_policy_eta_done = 0;
    double stageb_policy_eta_done_seconds = 0.0;
    bool stageb_policy_active = false;
    std::chrono::steady_clock::time_point stageb_policy_start;
    std::string stageb_policy_active_name;
    size_t stageb_policy_active_index = 0;
    double stageb_policy_active_estimated_total_seconds = 0.0;
    std::string stageb_policy_active_estimate_basis;
    auto note_stageb_policy_active_progress = [&](int64_t completed, int64_t total, const char * basis) {
        if (!stageb_policy_active || completed <= 0 || total <= 0 || completed > total) {
            return;
        }
        const double active_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - stageb_policy_start).count();
        const double fraction = (double) completed / (double) total;
        if (active_s > 0.0 && fraction > 0.0) {
            stageb_policy_active_estimated_total_seconds = active_s / fraction;
            stageb_policy_active_estimate_basis = basis != nullptr ? basis : "active_progress";
        }
    };
    auto stageb_eta_hint = [&]() -> std::string {
        if (full_quant_eta_total > 0 && full_quant_eta_done >= full_quant_eta_total) {
            return "done";
        }
        if (stageb_policy_total <= 0) {
            return {};
        }
        int64_t remaining_policies = std::max<int64_t>(0, stageb_policy_total - stageb_policy_done);
        const double active_s = stageb_policy_active
            ? std::chrono::duration<double>(std::chrono::steady_clock::now() - stageb_policy_start).count()
            : 0.0;
        if (stageb_policy_eta_done > 0 && stageb_policy_eta_done_seconds > 0.0) {
            const double avg_policy_s = stageb_policy_eta_done_seconds / (double) stageb_policy_eta_done;
            double remaining_s = 0.0;
            if (stageb_policy_active && remaining_policies > 0) {
                if (stageb_policy_active_estimated_total_seconds > 0.0) {
                    remaining_s += std::max(0.0, stageb_policy_active_estimated_total_seconds - active_s);
                } else {
                    remaining_s += std::max(0.0, avg_policy_s - active_s);
                }
                --remaining_policies;
            }
            remaining_s += (double) remaining_policies * avg_policy_s;
            return selector_format_duration(remaining_s) + " avg_measured_policy";
        }
        if (stageb_policy_active && stageb_policy_active_estimated_total_seconds > 0.0 && remaining_policies > 0) {
            double remaining_s = std::max(0.0, stageb_policy_active_estimated_total_seconds - active_s);
            --remaining_policies;
            remaining_s += (double) remaining_policies * stageb_policy_active_estimated_total_seconds;
            return "~" + selector_format_duration(remaining_s) + " rough_active_progress";
        }
        if (stageb_policy_active && active_s > 0.0 && remaining_policies > 0) {
            return ">=" + selector_format_duration(active_s * (double) remaining_policies) + " active_policy";
        }
        return "warming-up";
    };
    auto stageb_eta_detail = [&]() -> std::string {
        if (stageb_policy_total <= 0) {
            return "policy_units=0/0";
        }
        char buf[256];
        if (stageb_policy_active) {
            if (stageb_policy_active_estimated_total_seconds > 0.0) {
                snprintf(buf, sizeof(buf),
                    "policy_units=%" PRId64 "/%" PRId64 " measured_eta_units=%" PRId64 " active=[%zu/%" PRId64 "] active_est=%s basis=%s policy=%s",
                    stageb_policy_done,
                    stageb_policy_total,
                    stageb_policy_eta_done,
                    stageb_policy_active_index,
                    stageb_policy_total,
                    selector_format_duration(stageb_policy_active_estimated_total_seconds).c_str(),
                    stageb_policy_active_estimate_basis.c_str(),
                    stageb_policy_active_name.c_str());
            } else {
                snprintf(buf, sizeof(buf),
                    "policy_units=%" PRId64 "/%" PRId64 " measured_eta_units=%" PRId64 " active=[%zu/%" PRId64 "] policy=%s",
                    stageb_policy_done,
                    stageb_policy_total,
                    stageb_policy_eta_done,
                    stageb_policy_active_index,
                    stageb_policy_total,
                    stageb_policy_active_name.c_str());
            }
        } else {
            snprintf(buf, sizeof(buf),
                "policy_units=%" PRId64 "/%" PRId64 " measured_eta_units=%" PRId64,
                stageb_policy_done,
                stageb_policy_total,
                stageb_policy_eta_done);
        }
        return buf;
    };
    auto update_full_quant_eta = [&](const char * phase, bool print_now = false) {
        const std::string eta_detail = stageb_eta_detail();
        char detail[512];
        snprintf(detail, sizeof(detail),
            "phase=%s scope=whole_artifact_estimate basis=selector_stageb_policy_units_plus_final_materialization %s",
            phase,
            eta_detail.c_str());
        full_quant_eta.eta_hint(stageb_eta_hint());
        full_quant_eta.update(full_quant_eta_done, full_quant_eta_total, detail, print_now);
    };
    auto finish_stageb_policy_unit = [&](const char * phase, bool print_now = false, bool count_for_eta = true) {
        if (stageb_policy_active) {
            const double policy_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stageb_policy_start).count();
            stageb_policy_done_seconds += policy_s;
            if (count_for_eta && policy_s > 0.0) {
                stageb_policy_eta_done_seconds += policy_s;
                ++stageb_policy_eta_done;
            }
            ++stageb_policy_done;
            stageb_policy_active = false;
            stageb_policy_active_name.clear();
            stageb_policy_active_index = 0;
            stageb_policy_active_estimated_total_seconds = 0.0;
            stageb_policy_active_estimate_basis.clear();
        }
        full_quant_eta_done = std::min<int64_t>(full_quant_eta_total, 2 + stageb_policy_done);
        update_full_quant_eta(phase, print_now);
    };
    update_full_quant_eta("selector-setup", true);

    selector_ledger ledger(selector_control_string("LEDGER_FILE"));
    ledger.load_local_metrics();
    const bool selector_tensor_plan_enabled =
        out_tensor_overrides != nullptr &&
        selector_control_i64("TENSOR_POLICY_MAP", 1) != 0;
    const bool selector_fresh_tensor_plan =
        selector_tensor_plan_enabled &&
        !selector_control_has("CHECKPOINT_MODEL");

    ledger.append_context("selector_setup", {
        {"source_model", source_model_path},
        {"checkpoint_model", checkpoint_model_path},
        {"nthread", nthread},
        {"input_scale_policy", nvfp4_input_scale_policy},
        {"tensor_plan_primary", selector_fresh_tensor_plan},
    });

    const int eval_top_default = selector_fresh_tensor_plan ? 0 : 16;
    const int eval_top = (int) std::max<int64_t>(0, selector_control_i64("EVAL_TOP", eval_top_default));
    const bool want_measured_eval_requested = eval_top > 0 || selector_fresh_tensor_plan;
    const bool want_measured_eval = want_measured_eval_requested && selector_control_i64("ENABLE_EVAL", 0) != 0;
    const bool require_runtime_cache = selector_control_i64("REQUIRE_RUNTIME_CACHE", 0) != 0;
    selector_stageb_digest imatrix_digest;
    std::vector<std::string> imatrix_keys;
    imatrix_keys.reserve(imatrix_data.size());
    for (const auto & kv : imatrix_data) {
        imatrix_keys.push_back(kv.first);
    }
    std::sort(imatrix_keys.begin(), imatrix_keys.end());
    for (const std::string & name : imatrix_keys) {
        const auto it = imatrix_data.find(name);
        if (it == imatrix_data.end()) {
            continue;
        }
        imatrix_digest.add_string(name);
        const uint64_t row_size = (uint64_t) it->second.size();
        imatrix_digest.add_bytes(&row_size, sizeof(row_size));
        if (!it->second.empty()) {
            imatrix_digest.add_bytes(it->second.data(), it->second.size() * sizeof(float));
        }
    }
    const std::string imatrix_hash = imatrix_digest.hex();

    std::vector<std::string> src_splits;
    llama_model_loader src_ml(
        /*metadata=*/ nullptr,
        /*set_tensor_data=*/ nullptr,
        /*set_tensor_data_ud=*/ nullptr,
        source_model_path,
        src_splits,
        /*file=*/ nullptr,
        /*use_mmap=*/ true,
        /*use_direct_io=*/ false,
        /*check_tensors=*/ false,
        /*no_alloc=*/ true,
        /*param_overrides_p=*/ nullptr,
        /*param_tensor_buft_overrides_p=*/ nullptr);
    src_ml.init_mappings(false);

    std::vector<std::string> seed_splits;
	    llama_model_loader seed_ml(
        /*metadata=*/ nullptr,
        /*set_tensor_data=*/ nullptr,
        /*set_tensor_data_ud=*/ nullptr,
	        checkpoint_model_path,
        seed_splits,
        /*file=*/ nullptr,
        /*use_mmap=*/ true,
        /*use_direct_io=*/ false,
        /*check_tensors=*/ false,
        /*no_alloc=*/ true,
        /*param_overrides_p=*/ nullptr,
        /*param_tensor_buft_overrides_p=*/ nullptr);
    seed_ml.init_mappings(false);

    std::vector<std::pair<std::string, ggml_tensor *>> seed_tmap;
    seed_tmap.reserve(seed_ml.weights_map.size());
    for (const auto & kv : seed_ml.weights_map) {
        if (kv.second.tensor == nullptr) {
            continue;
        }
        seed_tmap.emplace_back(kv.first, kv.second.tensor);
    }
    if (seed_tmap.empty()) {
        fprintf(stderr, "%s: failed to load candidate search checkpoint GGUF %s\n", __func__, checkpoint_model_path.c_str());
        return false;
    }

    const int32_t n_layer = selector_detect_n_layer(seed_tmap);
    std::vector<std::string> tensor_names = selector_build_stress_tensors(seed_tmap, n_layer);
    const int64_t max_tensors = std::max<int64_t>(1, selector_control_i64("MAX_TENSORS", 24));
    if ((int64_t) tensor_names.size() > max_tensors) {
        tensor_names.resize((size_t) max_tensors);
    }

    std::unordered_set<std::string> stress_name_set(tensor_names.begin(), tensor_names.end());
    std::vector<selector_binding> all_bindings;
    all_bindings.reserve(seed_tmap.size());
    std::vector<selector_binding> mxfp6_bindings;
    mxfp6_bindings.reserve(seed_tmap.size());
    std::vector<size_t> stress_binding_indices;
    const float selector_correction_denom =
        (float) selector_control_f64("CORRECTION_DENOM", NVFP4_SELECTOR_CORRECTION_DENOM_DEFAULT);
    for (const auto & kv : seed_tmap) {
        const std::string & tname = kv.first;
        ggml_tensor * target_tensor = kv.second;
        if (target_tensor == nullptr || (target_tensor->type != GGML_TYPE_NVFP4 && target_tensor->type != GGML_TYPE_MXFP6_E2M3)) {
            continue;
        }
        const bool is_mxfp6_target = target_tensor->type == GGML_TYPE_MXFP6_E2M3;
        const auto * src_weight = src_ml.get_weight(tname.c_str());
        ggml_tensor * src_tensor = src_weight ? src_weight->tensor : nullptr;
        if (src_tensor == nullptr) {
            continue;
        }
        src_ml.load_data_for(src_tensor);
        const float * imatrix_row = nullptr;
        auto it = imatrix_data.find(tname);
        if (it != imatrix_data.end() && it->second.size() >= (size_t) src_tensor->ne[0] * (size_t) std::max<int64_t>(1, src_tensor->ne[2])) {
            imatrix_row = it->second.data();
        }

        selector_binding b;
        b.name = tname;
        b.cls = selector_classify_tensor(tname);
        if (b.cls == selector_tensor_class::EMBEDDING_OUTPUT) {
            continue;
        }
        b.layer = selector_parse_layer(tname);
        b.bucket = selector_layer_bucket(b.layer, n_layer);
        b.target = target_tensor;
        if (is_mxfp6_target) {
            const std::string scale_name = llama_nvfp4_scale_tensor_name(tname);
            auto scale_it = seed_ml.weights_map.find(scale_name);
            if (scale_it != seed_ml.weights_map.end() && scale_it->second.tensor != nullptr &&
                    scale_it->second.tensor->type == GGML_TYPE_F32 &&
                    ggml_nelements(scale_it->second.tensor) >= std::max<int64_t>(1, target_tensor->ne[2])) {
                seed_ml.load_data_for(scale_it->second.tensor);
                b.target_scale = scale_it->second.tensor;
                b.target_scale_nbytes = (size_t) std::max<int64_t>(1, target_tensor->ne[2]) * sizeof(float);
            } else if (std::max<int64_t>(1, target_tensor->ne[2]) == 1 && ggml_nbytes(target_tensor) >= MXFP6_HEADER_OFFSET) {
                b.target_scale = nullptr;
                b.target_scale_nbytes = sizeof(float);
            }
        } else {
            const int64_t scale_len = std::max<int64_t>(1, src_tensor->ne[2]);
            const std::string scale_name = llama_nvfp4_scale_tensor_name(tname);
            auto scale_it = seed_ml.weights_map.find(scale_name);
            if (scale_it != seed_ml.weights_map.end() && scale_it->second.tensor != nullptr &&
                    scale_it->second.tensor->type == GGML_TYPE_F32 &&
                    ggml_nelements(scale_it->second.tensor) >= scale_len) {
                seed_ml.load_data_for(scale_it->second.tensor);
                b.target_scale = scale_it->second.tensor;
                b.target_scale_nbytes = (size_t) scale_len * sizeof(float);
            }
            const std::string input_scale_name = llama_nvfp4_input_scale_tensor_name(tname);
            auto input_scale_it = seed_ml.weights_map.find(input_scale_name);
            if (input_scale_it != seed_ml.weights_map.end() && input_scale_it->second.tensor != nullptr &&
                    input_scale_it->second.tensor->type == GGML_TYPE_F32 &&
                    ggml_nelements(input_scale_it->second.tensor) >= scale_len) {
                seed_ml.load_data_for(input_scale_it->second.tensor);
                b.target_input_scale = input_scale_it->second.tensor;
                b.target_input_scale_nbytes = (size_t) scale_len * sizeof(float);
            }
        }
        b.source = src_tensor;
        b.device_samples = std::make_shared<nvfp4_selector_device_sample_bank>();
        b.imatrix_row = imatrix_row;
        b.source_tensor_scale = src_tensor->type == GGML_TYPE_NVFP4 ? gguf_get_nvfp4_tensor_scale(src_ml.metadata, tname.c_str()) : 1.0f;
        if (src_tensor->type != GGML_TYPE_NVFP4) {
            b.quant_tensor_scales = nvfp4_selector_quant_tensor_scales(src_tensor, selector_correction_denom);
            b.working_quant_tensor_scales = b.quant_tensor_scales;
        }
        b.source_nbytes = ggml_nbytes(src_tensor);
        b.target_nbytes = ggml_nbytes(target_tensor);
        if (is_mxfp6_target) {
            if (b.target_scale_nbytes > 0) {
                mxfp6_bindings.push_back(std::move(b));
            }
        } else {
            all_bindings.push_back(std::move(b));
            if (stress_name_set.find(tname) != stress_name_set.end()) {
                stress_binding_indices.push_back(all_bindings.size() - 1);
            }
        }
    }

    auto clear_selector_cuda_caches = [&](bool clear_device_samples) {
        if (clear_device_samples) {
            for (auto & b : all_bindings) {
                if (b.device_samples) {
                    std::lock_guard<std::mutex> lock(b.device_samples->mutex);
                    b.device_samples->entries.clear();
                }
            }
        }
        nvfp4_clear_cuda_stream_cache();
    };
    const bool keep_device_sample_cache =
        selector_control_i64("KEEP_DEVICE_SAMPLE_CACHE", 0) != 0;
    auto finish_selector_cuda_policy_scope = [&]() {
        clear_selector_cuda_caches(!keep_device_sample_cache);
    };

    selector_restore_counters runtime_restore_counters;
    auto restore_all = [&]() {
        if (!want_measured_eval) {
            return;
        }
        for (auto & b : all_bindings) {
            if (!quantize_tensor_host_buffer(b.target) &&
                    b.original_target_bytes.size() == b.target_nbytes) {
                quantize_restore_from_host(
                    b.target, b.original_target_bytes, b.target_nbytes, &runtime_restore_counters);
            }
            if (b.target_scale != nullptr && !quantize_tensor_host_buffer(b.target_scale) &&
                    b.original_scale_bytes.size() == b.target_scale_nbytes) {
                quantize_restore_from_snapshot_or_host(
                    b.target_scale, b.original_scale_device, b.original_scale_bytes, b.target_scale_nbytes,
                    &runtime_restore_counters);
            }
            if (b.target_input_scale != nullptr && !quantize_tensor_host_buffer(b.target_input_scale) &&
                    b.original_input_scale_bytes.size() == b.target_input_scale_nbytes) {
                quantize_restore_from_snapshot_or_host(
                    b.target_input_scale, b.original_input_scale_device, b.original_input_scale_bytes,
                    b.target_input_scale_nbytes, &runtime_restore_counters);
            }
        }
        for (auto & b : mxfp6_bindings) {
            if (!quantize_tensor_host_buffer(b.target)) {
                quantize_restore_from_snapshot_or_host(
                    b.target, b.original_target_device, b.original_target_bytes, b.target_nbytes, &runtime_restore_counters);
            }
            if (b.target_scale != nullptr && !quantize_tensor_host_buffer(b.target_scale)) {
                quantize_restore_from_snapshot_or_host(
                    b.target_scale, b.original_scale_device, b.original_scale_bytes, b.target_scale_nbytes,
                    &runtime_restore_counters);
            }
        }
    };

    auto covers_all_nvfp4_bindings = [&](const std::vector<size_t> & binding_indices) {
        if (binding_indices.size() != all_bindings.size()) {
            return false;
        }
        for (size_t i = 0; i < binding_indices.size(); ++i) {
            if (binding_indices[i] != i) {
                return false;
            }
        }
        return true;
    };

    if (all_bindings.empty() && mxfp6_bindings.empty()) {
        fprintf(stderr, "%s: selector found no NVFP4/MXFP6_E2M3 candidate tensors in %s\n", __func__, checkpoint_model_path.c_str());
        return false;
    }
    if (stress_binding_indices.empty() && !all_bindings.empty()) {
        for (size_t i = 0; i < all_bindings.size() && (int64_t) i < max_tensors; ++i) {
            stress_binding_indices.push_back(i);
        }
    }
    std::vector<size_t> all_binding_indices;
    all_binding_indices.reserve(all_bindings.size());
    for (size_t i = 0; i < all_bindings.size(); ++i) {
        all_binding_indices.push_back(i);
    }

    ledger.append_context("selector_bindings_loaded", {
        {"source_model", source_model_path},
        {"checkpoint_model", checkpoint_model_path},
        {"imatrix", {
            {"entries", imatrix_data.size()},
            {"hash", imatrix_hash},
        }},
        {"search_kld", selector_stageb_kld_json(kld)},
        {"validation_kld", kld_holdout != nullptr ? selector_stageb_kld_json(*kld_holdout) : nlohmann::ordered_json(nullptr)},
        {"eval", {
            {"enabled", want_measured_eval},
            {"requested", want_measured_eval_requested},
            {"eval_top", eval_top},
            {"eval_batch_override", selector_eval_batch_override},
        }},
        {"bindings", {
            {"nvfp4_count", all_bindings.size()},
            {"nvfp4_stress_count", stress_binding_indices.size()},
            {"mxfp6_count", mxfp6_bindings.size()},
        }},
    });

    fprintf(stderr,
	        "%s: selector loaded %zu NVFP4 candidate tensors from checkpoint=%s (stress=%zu, survey=%zu, eval=%s, eval_rank=best)\n",
	        __func__,
	        all_bindings.size(),
	        checkpoint_model_path.c_str(),
        stress_binding_indices.size(),
        all_binding_indices.size(),
        want_measured_eval ? "enabled" : "disabled");
    if (!mxfp6_bindings.empty()) {
	        fprintf(stderr,
	            "%s: selector loaded %zu MXFP6_E2M3 scale-refine candidate tensors from checkpoint=%s\n",
	            __func__, mxfp6_bindings.size(), checkpoint_model_path.c_str());
    }
    if (want_measured_eval_requested && !want_measured_eval) {
        fprintf(stderr,
            "%s: selector measured stage is unavailable because the runtime patch cache is not active\n",
            __func__);
    }

    auto selector_local_metric_key = [&](
            const char * phase,
            const std::string & policy_name,
            const nvfp4_cuda_runtime_cfg & cfg,
            const selector_binding & binding,
            int64_t sample_blocks_requested,
            int64_t sample_phase) {
        nlohmann::ordered_json key = nlohmann::ordered_json::object();
        key["row_type"] = "tensor.local_metric";
        key["objective"] = "weight_reconstruction_sampled";
        key["encoder_evidence_version"] = SELECTOR_ENCODER_EVIDENCE_VERSION;
        key["source_model"] = source_model_path;
        key["checkpoint_model"] = checkpoint_model_path;
        key["imatrix_hash"] = imatrix_hash;
        key["phase"] = phase != nullptr ? phase : "";
        key["policy"] = policy_name;
        key["policy_cfg"] = selector_stageb_cfg_json(cfg);
        key["tensor"] = binding.name;
        key["source_type"] = binding.source != nullptr ? ggml_type_name(binding.source->type) : "unknown";
        key["target_type"] = binding.target != nullptr ? ggml_type_name(binding.target->type) : "unknown";
        key["source_nbytes"] = binding.source_nbytes;
        key["target_nbytes"] = binding.target_nbytes;
        key["sample_blocks_requested"] = sample_blocks_requested;
        key["sample_phase"] = sample_phase;
        return key;
    };

    auto emit_selector_local_metric = [&](
            const char * phase,
            const selector_policy & policy,
            const selector_binding & binding,
            const selector_rsf_tensor_record & rsf_record,
            const nvfp4_cuda_runtime_cfg & cfg,
            int64_t sample_blocks_requested,
            int64_t sample_phase,
            double tensor_sq,
            double tensor_abs,
            double tensor_max,
            int64_t tensor_n) {
        if (tensor_n <= 0) {
            return;
        }
        const selector_proxy_metrics proxy_metrics =
            selector_proxy_score(tensor_sq, tensor_abs, tensor_max, tensor_n);

        nlohmann::ordered_json row = nlohmann::ordered_json::object();
        row["phase"] = phase != nullptr ? phase : "";
        row["objective"] = "weight_reconstruction_sampled";
        row["policy"] = policy.name;
        row["policy_cfg"] = selector_stageb_cfg_json(cfg);
        row["key"] = selector_local_metric_key(
            phase, policy.name, cfg, binding, sample_blocks_requested, sample_phase);
        row["tensor"] = binding.name;
        row["class"] = selector_tensor_class_name(binding.cls);
        row["bucket"] = binding.bucket;
        row["layer"] = binding.layer;
        row["source_type"] = binding.source != nullptr ? ggml_type_name(binding.source->type) : "unknown";
        row["target_type"] = binding.target != nullptr ? ggml_type_name(binding.target->type) : "unknown";
        row["source_nbytes"] = binding.source_nbytes;
        row["target_nbytes"] = binding.target_nbytes;
        row["sample_blocks_requested"] = sample_blocks_requested;
        row["sample_blocks_evaluated"] = (tensor_n + SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE - 1) / SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
        row["sample_phase"] = sample_phase;
        row["count"] = tensor_n;
        row["sum_sq"] = tensor_sq;
        row["sum_abs"] = tensor_abs;
        row["rmse"] = proxy_metrics.rmse;
        row["abs_mean"] = proxy_metrics.abs_mean;
        row["max_abs"] = proxy_metrics.max_abs;
        row["proxy_score"] = proxy_metrics.score;
        row["proxy_ok"] = proxy_metrics.ok;

        nlohmann::ordered_json rsf = nlohmann::ordered_json::object();
        rsf["enabled"] = nvfp4_cfg_has_rsf(cfg);
        rsf["changed"] = rsf_record.rsf_changed;
        rsf["slices"] = rsf_record.slices;
        rsf["scale_mul_count"] = rsf_record.scale_muls.size();
        if (nvfp4_cfg_has_rsf(cfg)) {
            rsf["mode"] = nvfp4_rsf_mode_name(nvfp4_cfg_rsf_mode(cfg));
            rsf["depth"] = nvfp4_rsf_depth_name(nvfp4_cfg_rsf_depth(cfg));
        }
        if (!rsf_record.scale_muls.empty()) {
            const auto minmax = std::minmax_element(rsf_record.scale_muls.begin(), rsf_record.scale_muls.end());
            rsf["scale_mul_mean"] = rsf_record.scale_mul_sum / (double) rsf_record.scale_muls.size();
            rsf["scale_mul_min"] = *minmax.first;
            rsf["scale_mul_max"] = *minmax.second;
        }
        row["rsf"] = std::move(rsf);
        ledger.append_tensor_local_metric(std::move(row));
    };

    auto score_proxy_policy = [&](selector_policy & policy, const std::vector<size_t> & binding_indices) -> bool {
        restore_all();
        double sum_sq = 0.0;
        double sum_abs = 0.0;
        double max_abs = 0.0;
        int64_t count = 0;
        const int64_t stagea_default_sample_blocks = want_measured_eval ? 2048 : 0;
        const int64_t stagea_sample_blocks = std::max<int64_t>(0, selector_control_i64("STAGEA_SAMPLE_BLOCKS", stagea_default_sample_blocks));
        policy.proxy_rejected = false;
        if (binding_indices.empty()) {
            policy.proxy_rmse = 0.0;
            policy.proxy_abs_mean = 0.0;
            policy.proxy_max_abs = 0.0;
            policy.proxy_score = 0.0;
            policy.first_tensor_rmse = 0.0;
            policy.first_tensor_abs_mean = 0.0;
            policy.first_tensor_max_abs = 0.0;
            fprintf(stderr,
                "selector proxy-rank phase=stage-a policy=%s objective=weight_reconstruction_sampled "
                "rank=lower_is_better score=0.000000 rmse=0.000000 mean_abs=0.000000 max_abs=0.000000 "
                "tensors=0 sample_blocks=full measured_kld_ppl=no skipped=no_nvfp4_tensors\n",
                policy.name.c_str());
            return true;
        }
        struct binding_score {
            bool ok = true;
            bool cached = false;
            double tensor_sq = 0.0;
            double tensor_abs = 0.0;
            double tensor_max = 0.0;
            int64_t tensor_n = 0;
        };
        const int default_policy_threads = std::max<int>(1, nthread / 4);
        const int policy_threads = (int) std::max<int64_t>(1, std::min<int64_t>(
            (int64_t) binding_indices.size(),
            selector_control_i64("POLICY_THREADS", default_policy_threads)));
        const int binding_nthread = std::max(1, nthread / std::max(1, policy_threads));
        std::vector<binding_score> binding_scores(binding_indices.size());
        std::vector<selector_rsf_tensor_record> tensor_records(binding_indices.size());
        for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
            const auto & b = all_bindings[binding_indices[ib]];
            tensor_records[ib].tensor = b.name;
            tensor_records[ib].policy = policy.name;
            tensor_records[ib].cls = b.cls;
            tensor_records[ib].bucket = b.bucket;
            tensor_records[ib].layer = b.layer;
        }
        const bool may_use_cached_local_metrics = !(want_measured_eval && stagea_sample_blocks <= 0);
        std::atomic<size_t> ledger_hits{0};
        auto apply_cached_local_metric = [&](size_t ib, const selector_ledger::local_metric & cached) {
            auto & bs = binding_scores[ib];
            auto & rsf = tensor_records[ib];
            bs.cached = true;
            bs.tensor_sq = cached.sum_sq;
            bs.tensor_abs = cached.sum_abs;
            bs.tensor_max = cached.max_abs;
            bs.tensor_n = cached.count;
            rsf.proxy_score = cached.proxy_score;
            rsf.rsf_changed = cached.rsf_changed;
            rsf.slices = cached.rsf_slices;
            rsf.scale_mul_sum = cached.rsf_scale_mul_mean * (double) cached.rsf_scale_mul_count;
            if (cached.rsf_scale_mul_count > 0 &&
                    std::isfinite(cached.rsf_scale_mul_mean) &&
                    cached.rsf_scale_mul_mean > 0.0) {
                rsf.scale_muls.assign((size_t) cached.rsf_scale_mul_count, (float) cached.rsf_scale_mul_mean);
            }
            ledger_hits.fetch_add(1, std::memory_order_relaxed);
        };

        if (policy_threads <= 1 || binding_indices.size() <= 1) {
            std::vector<uint8_t> tmp_bytes;
            for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
                auto & b = all_bindings[binding_indices[ib]];
                std::vector<uint8_t> & quant_bytes = want_measured_eval ? b.working_target_bytes : tmp_bytes;
                selector_rsf_tensor_record * rsf_record = &tensor_records[ib];
                selector_ledger::local_metric cached;
                const auto key = selector_local_metric_key("stage-a", policy.name, policy.cfg, b, stagea_sample_blocks, 0);
                if (may_use_cached_local_metrics && ledger.find_local_metric(key, cached)) {
                    apply_cached_local_metric(ib, cached);
                    continue;
                }
                if (!nvfp4_selector_quantize_binding(b, policy.cfg, binding_nthread, quant_bytes,
                        binding_scores[ib].tensor_sq, binding_scores[ib].tensor_abs, binding_scores[ib].tensor_max, binding_scores[ib].tensor_n,
                        stagea_sample_blocks, 0, false, rsf_record)) {
                    fprintf(stderr, "%s: selector failed patching tensor %s for policy %s\n", __func__, b.name.c_str(), policy.name.c_str());
                    return false;
                }
            }
        } else {
            std::atomic<size_t> next_binding{0};
            std::vector<std::thread> workers_local;
            workers_local.reserve((size_t) policy_threads);
            for (int ti = 0; ti < policy_threads; ++ti) {
                workers_local.emplace_back([&, ti]() {
                    std::vector<uint8_t> tmp_bytes_local;
                    while (true) {
                        const size_t ib = next_binding.fetch_add(1, std::memory_order_relaxed);
                        if (ib >= binding_indices.size()) {
                            break;
                        }
                        auto & b = all_bindings[binding_indices[ib]];
                        std::vector<uint8_t> & quant_bytes = want_measured_eval ? b.working_target_bytes : tmp_bytes_local;
                        selector_rsf_tensor_record * rsf_record = &tensor_records[ib];
                        selector_ledger::local_metric cached;
                        const auto key = selector_local_metric_key("stage-a", policy.name, policy.cfg, b, stagea_sample_blocks, 0);
                        if (may_use_cached_local_metrics && ledger.find_local_metric(key, cached)) {
                            apply_cached_local_metric(ib, cached);
                            continue;
                        }
                        if (!nvfp4_selector_quantize_binding(b, policy.cfg, binding_nthread, quant_bytes,
                                binding_scores[ib].tensor_sq, binding_scores[ib].tensor_abs, binding_scores[ib].tensor_max, binding_scores[ib].tensor_n,
                                stagea_sample_blocks, 0, false, rsf_record)) {
                            binding_scores[ib].ok = false;
                        }
                    }
                });
            }
            for (auto & worker : workers_local) {
                worker.join();
            }
            for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
                if (!binding_scores[ib].ok) {
                    auto & b = all_bindings[binding_indices[ib]];
                    fprintf(stderr, "%s: selector failed patching tensor %s for policy %s\n", __func__, b.name.c_str(), policy.name.c_str());
                    return false;
                }
            }
        }

        for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
            auto & b = all_bindings[binding_indices[ib]];
            const auto & bs = binding_scores[ib];
            if (want_measured_eval && stagea_sample_blocks <= 0 &&
                    !quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                fprintf(stderr, "%s: selector failed writing policy bytes for %s\n", __func__, b.name.c_str());
                return false;
            }
            sum_sq += bs.tensor_sq;
            sum_abs += bs.tensor_abs;
            max_abs = std::max(max_abs, bs.tensor_max);
            count += bs.tensor_n;
            if (bs.tensor_n > 0) {
                const selector_proxy_metrics tensor_proxy =
                    selector_proxy_score(bs.tensor_sq, bs.tensor_abs, bs.tensor_max, bs.tensor_n);
                tensor_records[ib].proxy_score = tensor_proxy.score;
                if (!bs.cached) {
                    emit_selector_local_metric(
                    "stage-a",
                    policy,
                    b,
                    tensor_records[ib],
                    policy.cfg,
                    stagea_sample_blocks,
                    0,
                    bs.tensor_sq,
                    bs.tensor_abs,
                    bs.tensor_max,
                    bs.tensor_n);
                }
            }
            if (ib == 0 && bs.tensor_n > 0) {
                policy.first_tensor_rmse = std::sqrt(bs.tensor_sq / (double) bs.tensor_n);
                policy.first_tensor_abs_mean = bs.tensor_abs / (double) bs.tensor_n;
                policy.first_tensor_max_abs = bs.tensor_max;
            }
        }

        const selector_proxy_metrics proxy_metrics =
            selector_proxy_score(sum_sq, sum_abs, max_abs, count);
        if (proxy_metrics.ok) {
            policy.proxy_rmse = proxy_metrics.rmse;
            policy.proxy_abs_mean = proxy_metrics.abs_mean;
            policy.proxy_max_abs = proxy_metrics.max_abs;
            policy.proxy_score = proxy_metrics.score;
        }
        policy.tensor_records = std::move(tensor_records);

        if (ledger_hits.load(std::memory_order_relaxed) > 0) {
            fprintf(stderr,
                "%s: selector ledger local-metric cache hits phase=stage-a policy=%s hits=%zu/%zu\n",
                __func__,
                policy.name.c_str(),
                ledger_hits.load(std::memory_order_relaxed),
                binding_indices.size());
        }
        fprintf(stderr,
            "selector proxy-rank phase=stage-a policy=%s objective=weight_reconstruction_sampled rank=lower_is_better "
            "score=%.6f rmse=%.6f mean_abs=%.6f max_abs=%.6f tensors=%zu sample_blocks=full measured_kld_ppl=no\n",
            policy.name.c_str(),
            policy.proxy_score,
            policy.proxy_rmse,
            policy.proxy_abs_mean,
            policy.proxy_max_abs,
            binding_indices.size());
        return true;
    };

    auto score_proxy_policy_extended = [&](selector_policy & policy,
                                          const std::vector<size_t> & binding_indices,
                                          int64_t sample_blocks_override,
                                          bool store_survey,
                                          const char * label) -> bool {
        restore_all();
        nvfp4_cuda_runtime_cfg proxy_cfg = policy.cfg;
        const bool materializes_full_policy = want_measured_eval && sample_blocks_override <= 0;
        const bool proxy_only = !materializes_full_policy;
        if (proxy_only &&
                nvfp4_cfg_has_rsf(proxy_cfg) &&
                nvfp4_cfg_rsf_depth(proxy_cfg) > NVFP4_CUDA_RSF_DEPTH_DEEP) {
            nvfp4_cfg_set_rsf_depth(proxy_cfg, NVFP4_CUDA_RSF_DEPTH_DEEP);
        }
        if (binding_indices.empty()) {
            if (store_survey) {
                policy.has_survey = true;
                policy.survey_proxy_rmse = 0.0;
                policy.survey_proxy_abs_mean = 0.0;
                policy.survey_proxy_max_abs = 0.0;
                policy.survey_proxy_score = 0.0;
            } else {
                policy.proxy_rmse = 0.0;
                policy.proxy_abs_mean = 0.0;
                policy.proxy_max_abs = 0.0;
                policy.proxy_score = 0.0;
            }
            fprintf(stderr,
                "selector proxy-rank phase=%s policy=%s objective=weight_reconstruction_sampled rank=lower_is_better "
                "score=0.000000 rmse=0.000000 mean_abs=0.000000 max_abs=0.000000 "
                "tensors=0 sample_blocks_per_tensor=%" PRId64 " measured_kld_ppl=no skipped=no_nvfp4_tensors\n",
                label,
                policy.name.c_str(),
                sample_blocks_override);
            return true;
        }
        double sum_sq = 0.0;
        double sum_abs = 0.0;
        double max_abs = 0.0;
        int64_t count = 0;
        struct binding_score {
            bool ok = true;
            bool cached = false;
            double tensor_sq = 0.0;
            double tensor_abs = 0.0;
            double tensor_max = 0.0;
            int64_t tensor_n = 0;
        };
        const int default_policy_threads = std::max<int>(1, nthread / 4);
        const int policy_threads = (int) std::max<int64_t>(1, std::min<int64_t>(
            (int64_t) binding_indices.size(),
            selector_control_i64("POLICY_THREADS", default_policy_threads)));
        const int binding_nthread = std::max(1, nthread / std::max(1, policy_threads));
        std::vector<binding_score> binding_scores(binding_indices.size());
        std::vector<selector_rsf_tensor_record> tensor_records(binding_indices.size());
        for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
            const auto & b = all_bindings[binding_indices[ib]];
            tensor_records[ib].tensor = b.name;
            tensor_records[ib].policy = policy.name;
            tensor_records[ib].cls = b.cls;
            tensor_records[ib].bucket = b.bucket;
            tensor_records[ib].layer = b.layer;
        }
        std::atomic<size_t> ledger_hits{0};
        auto apply_cached_local_metric = [&](size_t ib, const selector_ledger::local_metric & cached) {
            auto & bs = binding_scores[ib];
            auto & rsf = tensor_records[ib];
            bs.cached = true;
            bs.tensor_sq = cached.sum_sq;
            bs.tensor_abs = cached.sum_abs;
            bs.tensor_max = cached.max_abs;
            bs.tensor_n = cached.count;
            rsf.proxy_score = cached.proxy_score;
            rsf.rsf_changed = cached.rsf_changed;
            rsf.slices = cached.rsf_slices;
            rsf.scale_mul_sum = cached.rsf_scale_mul_mean * (double) cached.rsf_scale_mul_count;
            if (cached.rsf_scale_mul_count > 0 &&
                    std::isfinite(cached.rsf_scale_mul_mean) &&
                    cached.rsf_scale_mul_mean > 0.0) {
                rsf.scale_muls.assign((size_t) cached.rsf_scale_mul_count, (float) cached.rsf_scale_mul_mean);
            }
            ledger_hits.fetch_add(1, std::memory_order_relaxed);
        };

        if (policy_threads <= 1 || binding_indices.size() <= 1) {
            std::vector<uint8_t> tmp_bytes;
            for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
                auto & b = all_bindings[binding_indices[ib]];
                std::vector<uint8_t> & quant_bytes = materializes_full_policy ? b.working_target_bytes : tmp_bytes;
                selector_rsf_tensor_record * rsf_record = &tensor_records[ib];
                selector_ledger::local_metric cached;
                const auto key = selector_local_metric_key(label, policy.name, proxy_cfg, b, sample_blocks_override, 0);
                if (proxy_only && ledger.find_local_metric(key, cached)) {
                    apply_cached_local_metric(ib, cached);
                    continue;
                }
                if (!nvfp4_selector_quantize_binding(b, proxy_cfg, binding_nthread, quant_bytes,
                        binding_scores[ib].tensor_sq, binding_scores[ib].tensor_abs, binding_scores[ib].tensor_max, binding_scores[ib].tensor_n,
                        sample_blocks_override, 0, false, rsf_record)) {
                    fprintf(stderr, "%s: selector failed patching tensor %s for policy %s\n", __func__, b.name.c_str(), policy.name.c_str());
                    return false;
                }
            }
        } else {
            std::atomic<size_t> next_binding{0};
            std::vector<std::thread> workers_local;
            workers_local.reserve((size_t) policy_threads);
            for (int ti = 0; ti < policy_threads; ++ti) {
                workers_local.emplace_back([&, ti]() {
                    std::vector<uint8_t> tmp_bytes_local;
                    while (true) {
                        const size_t ib = next_binding.fetch_add(1, std::memory_order_relaxed);
                        if (ib >= binding_indices.size()) {
                            break;
                        }
                        auto & b = all_bindings[binding_indices[ib]];
                        std::vector<uint8_t> & quant_bytes = materializes_full_policy ? b.working_target_bytes : tmp_bytes_local;
                        selector_rsf_tensor_record * rsf_record = &tensor_records[ib];
                        selector_ledger::local_metric cached;
                        const auto key = selector_local_metric_key(label, policy.name, proxy_cfg, b, sample_blocks_override, 0);
                        if (proxy_only && ledger.find_local_metric(key, cached)) {
                            apply_cached_local_metric(ib, cached);
                            continue;
                        }
                        if (!nvfp4_selector_quantize_binding(b, proxy_cfg, binding_nthread, quant_bytes,
                                binding_scores[ib].tensor_sq, binding_scores[ib].tensor_abs, binding_scores[ib].tensor_max, binding_scores[ib].tensor_n,
                                sample_blocks_override, 0, false, rsf_record)) {
                            binding_scores[ib].ok = false;
                        }
                    }
                });
            }
            for (auto & worker : workers_local) {
                worker.join();
            }
            for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
                if (!binding_scores[ib].ok) {
                    auto & b = all_bindings[binding_indices[ib]];
                    fprintf(stderr, "%s: selector failed patching tensor %s for policy %s\n", __func__, b.name.c_str(), policy.name.c_str());
                    return false;
                }
            }
        }

        for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
            auto & b = all_bindings[binding_indices[ib]];
            const auto & bs = binding_scores[ib];
            if (materializes_full_policy &&
                    !quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                fprintf(stderr, "%s: selector failed writing policy bytes for %s\n", __func__, b.name.c_str());
                return false;
            }
            sum_sq += bs.tensor_sq;
            sum_abs += bs.tensor_abs;
            max_abs = std::max(max_abs, bs.tensor_max);
            count += bs.tensor_n;
            if (bs.tensor_n > 0) {
                const selector_proxy_metrics tensor_proxy =
                    selector_proxy_score(bs.tensor_sq, bs.tensor_abs, bs.tensor_max, bs.tensor_n);
                tensor_records[ib].proxy_score = tensor_proxy.score;
                if (!bs.cached) {
                    emit_selector_local_metric(
                    label,
                    policy,
                    b,
                    tensor_records[ib],
                    proxy_cfg,
                    sample_blocks_override,
                    0,
                    bs.tensor_sq,
                    bs.tensor_abs,
                    bs.tensor_max,
                    bs.tensor_n);
                }
            }
        }

        const selector_proxy_metrics proxy_metrics =
            selector_proxy_score(sum_sq, sum_abs, max_abs, count);
        if (proxy_metrics.ok) {
            if (store_survey) {
                policy.has_survey = true;
                policy.survey_proxy_rmse = proxy_metrics.rmse;
                policy.survey_proxy_abs_mean = proxy_metrics.abs_mean;
                policy.survey_proxy_max_abs = proxy_metrics.max_abs;
                policy.survey_proxy_score = proxy_metrics.score;
            } else {
                policy.proxy_rmse = proxy_metrics.rmse;
                policy.proxy_abs_mean = proxy_metrics.abs_mean;
                policy.proxy_max_abs = proxy_metrics.max_abs;
                policy.proxy_score = proxy_metrics.score;
            }
            fprintf(stderr,
                "selector proxy-rank phase=%s policy=%s objective=weight_reconstruction_sampled rank=lower_is_better "
                "score=%.6f rmse=%.6f mean_abs=%.6f max_abs=%.6f tensors=%zu sample_blocks_per_tensor=%" PRId64
                " rsf_depth=%s measured_kld_ppl=no\n",
                label,
                policy.name.c_str(),
                proxy_metrics.score,
                proxy_metrics.rmse,
                proxy_metrics.abs_mean,
                proxy_metrics.max_abs,
                binding_indices.size(),
                sample_blocks_override,
                nvfp4_cfg_has_rsf(proxy_cfg) ? nvfp4_rsf_depth_name(nvfp4_cfg_rsf_depth(proxy_cfg)) : "off");
        }
        policy.tensor_records = std::move(tensor_records);
        if (ledger_hits.load(std::memory_order_relaxed) > 0) {
            fprintf(stderr,
                "%s: selector ledger local-metric cache hits phase=%s policy=%s hits=%zu/%zu\n",
                __func__,
                label,
                policy.name.c_str(),
                ledger_hits.load(std::memory_order_relaxed),
                binding_indices.size());
        }
        return true;
    };

    auto reject_policy = [&](selector_policy & policy, const char * phase) {
        restore_all();
        policy.proxy_score = std::numeric_limits<double>::infinity();
        policy.proxy_rmse = std::numeric_limits<double>::infinity();
        policy.proxy_abs_mean = std::numeric_limits<double>::infinity();
        policy.proxy_max_abs = std::numeric_limits<double>::infinity();
        policy.measured_score = std::numeric_limits<double>::infinity();
        policy.measured_pass = false;
        policy.proxy_rejected = true;
        fprintf(stderr, "%s: selector rejected policy=%s during %s after quantize/patch failure\n",
            __func__, policy.name.c_str(), phase);
    };

    bool skip_remaining_tuning = false;
    auto note_skip_remaining = [&](const char * phase) {
        if (skip_remaining_tuning) {
            return true;
        }
        if (!selector_skip_requested(phase)) {
            return false;
        }
        skip_remaining_tuning = true;
        fprintf(stderr,
            "%s: selector skip requested after %s; remaining optional tuning will be skipped\n",
            __func__,
            phase != nullptr && phase[0] != '\0' ? phase : "current phase");
        return true;
    };

    const int selector_rsf_mode = selector_effective_rsf_mode(recipe_cfg);
    const int selector_rsf_depth = selector_effective_rsf_depth(recipe_cfg);
    auto policies = selector_default_policies(recipe_cfg, recipe_policy_name);
    const bool has_expert_bindings = std::any_of(all_bindings.begin(), all_bindings.end(), [](const selector_binding & b) {
        return b.name.find(".ffn_gate_exps.weight") != std::string::npos ||
               b.name.find(".ffn_up_exps.weight") != std::string::npos ||
               b.name.find(".ffn_down_exps.weight") != std::string::npos;
    });
    const bool include_moe_policies = selector_control_i64("INCLUDE_MOE_POLICIES", has_expert_bindings ? 1 : 0) != 0;
    if (!include_moe_policies) {
        const size_t before = policies.size();
        policies.erase(std::remove_if(policies.begin(), policies.end(), [](const selector_policy & policy) {
            return !selector_policy_is_recipe(policy.name) &&
                   (policy.name.find("_moe") != std::string::npos ||
                    policy.name.find("moe_") != std::string::npos ||
                    policy.name.find("awq_tail_moe") != std::string::npos);
        }), policies.end());
        if (before != policies.size()) {
            fprintf(stderr,
                "%s: selector pruned %zu MoE-specialized policy candidate(s) for dense/non-expert model\n",
                __func__,
                before - policies.size());
        }
    }
    const std::unordered_set<std::string> include_policy_names =
        selector_policy_set_from_control("INCLUDE_POLICIES");
    const std::unordered_set<std::string> skip_policy_names =
        selector_policy_set_from_control("SKIP_POLICIES");
    auto selector_policy_excluded_by_include = [&](const selector_policy & policy) {
        return policy.name != "seed_keep" &&
               !include_policy_names.empty() &&
               include_policy_names.find(policy.name) == include_policy_names.end();
    };
    auto selector_policy_skipped = [&](const selector_policy & policy) {
        return policy.name != "seed_keep" &&
               (selector_policy_excluded_by_include(policy) ||
                skip_policy_names.find(policy.name) != skip_policy_names.end());
    };
    if (!include_policy_names.empty()) {
        fprintf(stderr, "%s: selector include-only list has %zu named policy candidate(s); seed_keep remains allowed\n",
            __func__, include_policy_names.size());
    }
    if (!skip_policy_names.empty()) {
        fprintf(stderr, "%s: selector will skip %zu named policy candidate(s) during survey/eval\n",
            __func__, skip_policy_names.size());
    }
    if (include_policy_names.empty()) {
        const size_t before = policies.size();
        policies.erase(std::remove_if(policies.begin(), policies.end(), [](const selector_policy & policy) {
            return !selector_policy_is_recipe(policy.name) &&
                   selector_policy_is_fixed_cap_sweep(policy.name);
        }), policies.end());
        if (before != policies.size()) {
            fprintf(stderr,
                "%s: selector pruned %zu fixed-cap sweep policy candidate(s) from tensor-plan auto search; explicit include-policy can still request them\n",
                __func__,
                before - policies.size());
        }
    }
    const int stagea_max_policies = (int) std::max<int64_t>(0, selector_control_i64("STAGEA_MAX_POLICIES", 0));
    if (stagea_max_policies > 0 && (int) policies.size() > stagea_max_policies) {
        const int awq_keep = std::min<int>(
            stagea_max_policies,
            (int) std::max<int64_t>(0, selector_control_i64("AWQ_TOP", 0)));
        if (awq_keep > 0) {
            std::vector<selector_policy> limited;
            limited.reserve((size_t) stagea_max_policies);
            auto push_policy = [&](const selector_policy & policy) {
                for (const auto & existing : limited) {
                    if (existing.name == policy.name) {
                        return false;
                    }
                }
                if ((int) limited.size() >= stagea_max_policies) {
                    return false;
                }
                limited.push_back(policy);
                return true;
            };

            const int primary_keep = std::max(0, stagea_max_policies - awq_keep);
            for (const auto & policy : policies) {
                if ((int) limited.size() >= primary_keep) {
                    break;
                }
                if (!selector_policy_is_awq_tail(policy.name)) {
                    push_policy(policy);
                }
            }
            size_t awq_added = 0;
            for (const auto & policy : policies) {
                if ((int) awq_added >= awq_keep) {
                    break;
                }
                if (selector_policy_is_awq_tail(policy.name)) {
                    if (push_policy(policy)) {
                        ++awq_added;
                    }
                }
            }
            for (const auto & policy : policies) {
                if ((int) limited.size() >= stagea_max_policies) {
                    break;
                }
                push_policy(policy);
            }
            fprintf(stderr,
                "%s: selector capped stage-a policies to %zu/%zu while preserving %zu asym/AWQ-tail candidate(s)\n",
                __func__,
                limited.size(),
                policies.size(),
                awq_added);
            policies.swap(limited);
        } else {
            policies.resize((size_t) stagea_max_policies);
        }
    }
    for (size_t policy_idx = 0; policy_idx < policies.size(); ++policy_idx) {
        auto & policy = policies[policy_idx];
        if (selector_policy_excluded_by_include(policy)) {
            policy.proxy_score = std::numeric_limits<double>::infinity();
            policy.proxy_rejected = true;
            fprintf(stderr,
                "selector stage-a skip [%zu/%zu] policy=%s reason=include-only\n",
                policy_idx + 1,
                policies.size(),
                policy.name.c_str());
            continue;
        }
        fprintf(stderr,
            "selector stage-a start [%zu/%zu] policy=%s tensors=%zu\n",
            policy_idx + 1,
            policies.size(),
            policy.name.c_str(),
            stress_binding_indices.size());
        const bool score_ok = score_proxy_policy(policy, stress_binding_indices);
        finish_selector_cuda_policy_scope();
        if (!score_ok) {
            reject_policy(policy, "stage-a");
            continue;
        }
    }
    note_skip_remaining("stage-a");

    std::sort(policies.begin(), policies.end(), selector_policy_proxy_less);

    if (!skip_remaining_tuning) {
        auto refined = selector_refine_policies(policies, selector_rsf_mode, selector_rsf_depth);
        for (size_t policy_idx = 0; policy_idx < refined.size(); ++policy_idx) {
            auto & policy = refined[policy_idx];
            if (selector_policy_excluded_by_include(policy)) {
                fprintf(stderr,
                    "selector refine skip [%zu/%zu] policy=%s reason=include-only\n",
                    policy_idx + 1,
                    refined.size(),
                    policy.name.c_str());
                continue;
            }
            fprintf(stderr,
                "selector refine start [%zu/%zu] policy=%s tensors=%zu\n",
                policy_idx + 1,
                refined.size(),
                policy.name.c_str(),
                stress_binding_indices.size());
            const bool score_ok = score_proxy_policy(policy, stress_binding_indices);
            finish_selector_cuda_policy_scope();
            if (!score_ok) {
                reject_policy(policy, "refine");
                continue;
            }
            policies.push_back(std::move(policy));
            if (note_skip_remaining("refine")) {
                break;
            }
        }
    }

    auto baseline_it = std::find_if(policies.begin(), policies.end(), [](const selector_policy & policy) {
        return policy.name == "baseline";
    });
    if (baseline_it != policies.end() &&
            (baseline_it->proxy_rejected || !std::isfinite(baseline_it->proxy_score))) {
        baseline_it = policies.end();
    }
    if (baseline_it == policies.end()) {
        baseline_it = std::find_if(policies.begin(), policies.end(), [](const selector_policy & policy) {
            return !policy.proxy_rejected && std::isfinite(policy.proxy_score);
        });
    }
    if (baseline_it != policies.end()) {
        const double soft_factor = SELECTOR_FIRST_TENSOR_SOFT_FACTOR;
        const double hard_factor = SELECTOR_FIRST_TENSOR_HARD_FACTOR;
        const double soft_penalty = SELECTOR_FIRST_TENSOR_PENALTY;
        const double base_rmse = std::max(1e-9, baseline_it->first_tensor_rmse);
        const double base_max = std::max(1e-9, baseline_it->first_tensor_max_abs);

        for (auto & policy : policies) {
            if (policy.name == baseline_it->name) {
                continue;
            }

            const double rmse_ratio = policy.first_tensor_rmse / base_rmse;
            const double max_ratio = policy.first_tensor_max_abs / base_max;
            const double worst_ratio = std::max(rmse_ratio, max_ratio);
            if (!std::isfinite(worst_ratio)) {
                continue;
            }

            if (worst_ratio > hard_factor) {
                policy.proxy_score = std::numeric_limits<double>::infinity();
                policy.proxy_rejected = true;
                fprintf(stderr,
                    "selector stage-a reject policy=%s first_tensor ratio=%.3f hard=%.3f\n",
                    policy.name.c_str(), worst_ratio, hard_factor);
            } else if (worst_ratio > soft_factor) {
                const double penalty = soft_penalty * (worst_ratio - soft_factor);
                policy.proxy_score += penalty;
                fprintf(stderr,
                    "selector stage-a penalize policy=%s first_tensor ratio=%.3f soft=%.3f penalty=%.3f\n",
                    policy.name.c_str(), worst_ratio, soft_factor, penalty);
            }
        }
    }

    std::sort(policies.begin(), policies.end(), selector_policy_proxy_less);

    const int survey_top = (int) std::max<int64_t>(1, selector_control_i64("SURVEY_TOP", 48));
    const int64_t survey_sample_blocks = std::max<int64_t>(0, selector_control_i64("SURVEY_SAMPLE_BLOCKS", 2048));
    const bool dedup_survey = SELECTOR_DEDUP_SURVEY_DEFAULT;
    std::vector<size_t> survey_policy_indices;
    survey_policy_indices.reserve((size_t) survey_top + 1);
    auto append_policy_index = [&](std::vector<size_t> & indices, size_t idx) {
        if (idx >= policies.size()) {
            return false;
        }
        if (policies[idx].proxy_rejected || !std::isfinite(policies[idx].proxy_score)) {
            return false;
        }
        if (selector_policy_skipped(policies[idx])) {
            return false;
        }
        if (std::find(indices.begin(), indices.end(), idx) != indices.end()) {
            return false;
        }
        indices.push_back(idx);
        return true;
    };
    auto find_rsf_companion_index = [&](const selector_policy & policy) -> size_t {
        const bool want_rsf = !selector_policy_is_rsf_family(policy);
        const std::string base = selector_rsf_base_policy_name(policy.name);
        for (size_t i = 0; i < policies.size(); ++i) {
            const bool is_rsf = selector_policy_is_rsf_family(policies[i]);
            if (is_rsf != want_rsf) {
                continue;
            }
            if (selector_rsf_base_policy_name(policies[i].name) == base) {
                return i;
            }
        }
        return SIZE_MAX;
    };
    auto append_rsf_companions = [&](std::vector<size_t> & indices, const char * phase) {
        const size_t original_size = indices.size();
        size_t added = 0;
        for (size_t pos = 0; pos < original_size; ++pos) {
            const size_t companion = find_rsf_companion_index(policies[indices[pos]]);
            if (companion == SIZE_MAX) {
                continue;
            }
            if (append_policy_index(indices, companion)) {
                ++added;
            }
        }
        if (added > 0) {
            fprintf(stderr,
                "%s: selector added %zu RSF/base companion policy candidate(s) for measured %s\n",
                __func__,
                added,
                phase != nullptr ? phase : "ranking");
        }
    };
    auto stable_sort_rsf_pairs = [&](std::vector<size_t> & indices) {
        std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            const std::string abase = selector_rsf_base_policy_name(policies[a].name);
            const std::string bbase = selector_rsf_base_policy_name(policies[b].name);
            if (abase == bbase) {
                return !selector_policy_is_rsf_family(policies[a]) &&
                       selector_policy_is_rsf_family(policies[b]);
            }
            return false;
        });
    };
    auto compact_policy_indices = [&](std::vector<size_t> & indices, bool prefer_survey, const char * phase) {
        std::vector<size_t> compact;
        compact.reserve(indices.size());
        size_t skipped = 0;
        for (size_t idx : indices) {
            bool duplicate = false;
            for (size_t selected : compact) {
                if (selector_cfg_equal(policies[idx].cfg, policies[selected].cfg) ||
                        selector_proxy_equivalent(policies[idx], policies[selected], prefer_survey)) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                ++skipped;
            } else {
                compact.push_back(idx);
            }
        }
        if (skipped > 0) {
            fprintf(stderr,
                "%s: selector removed %zu duplicate-equivalent policy candidate(s) before measured %s\n",
                __func__,
                skipped,
                phase != nullptr ? phase : "ranking");
        }
        indices.swap(compact);
    };
    size_t survey_dedup_skipped = 0;
    for (size_t i = 0; i < policies.size() && survey_policy_indices.size() < (size_t) survey_top; ++i) {
        if (policies[i].proxy_rejected || !std::isfinite(policies[i].proxy_score)) {
            continue;
        }
        if (selector_policy_skipped(policies[i])) {
            continue;
        }
        if (dedup_survey) {
            bool duplicate = false;
            for (size_t selected : survey_policy_indices) {
                if (selector_proxy_equivalent(policies[i], policies[selected], false)) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                ++survey_dedup_skipped;
                continue;
            }
        }
        survey_policy_indices.push_back(i);
    }
    for (size_t i = 0; i < policies.size(); ++i) {
        if (policies[i].name == "baseline") {
            if (!selector_policy_skipped(policies[i]) &&
                    std::find(survey_policy_indices.begin(), survey_policy_indices.end(), i) == survey_policy_indices.end()) {
                survey_policy_indices.push_back(i);
            }
            break;
        }
    }
    append_rsf_companions(survey_policy_indices, "survey");
    compact_policy_indices(survey_policy_indices, false, "survey");
    stable_sort_rsf_pairs(survey_policy_indices);
    if (dedup_survey && survey_dedup_skipped > 0) {
        fprintf(stderr,
            "%s: selector skipped %zu proxy-equivalent policies before full-tensor survey\n",
            __func__,
            survey_dedup_skipped);
    }
    if (!skip_remaining_tuning) {
        for (size_t survey_pos = 0; survey_pos < survey_policy_indices.size(); ++survey_pos) {
            const size_t policy_idx = survey_policy_indices[survey_pos];
            fprintf(stderr,
                "selector survey start [%zu/%zu] policy=%s tensors=%zu sample_blocks=%" PRId64 "\n",
                survey_pos + 1,
                survey_policy_indices.size(),
                policies[policy_idx].name.c_str(),
                all_binding_indices.size(),
                survey_sample_blocks);
            const bool score_ok = score_proxy_policy_extended(
                policies[policy_idx], all_binding_indices, survey_sample_blocks, true, "stage-a-survey");
            finish_selector_cuda_policy_scope();
            if (!score_ok) {
                reject_policy(policies[policy_idx], "survey");
                continue;
            }
            if (note_skip_remaining("survey")) {
                break;
            }
        }
    }

    std::sort(policies.begin(), policies.end(), selector_policy_proxy_less);

    fprintf(stderr,
        "%s: selector survey complete; shortlisted %zu policies for full-model proxy ranking, top policy=%s\n",
        __func__,
        survey_policy_indices.size(),
        policies.empty() ? "<none>" : policies.front().name.c_str());

    const bool dedup_eval = SELECTOR_DEDUP_EVAL_DEFAULT;
    const int64_t proxy_eval_top = std::max<int64_t>(0, eval_top);
    std::vector<size_t> eval_policy_indices;
    eval_policy_indices.reserve((size_t) std::max<int64_t>(0, eval_top) + 1);
    size_t eval_dedup_skipped = 0;
    if (proxy_eval_top > 0) {
        for (size_t i = 0; i < policies.size() && eval_policy_indices.size() < (size_t) proxy_eval_top; ++i) {
            if (policies[i].proxy_rejected || !std::isfinite(policies[i].proxy_score)) {
                continue;
            }
            if (selector_policy_skipped(policies[i])) {
                continue;
            }
            if (dedup_eval) {
                bool duplicate = false;
                for (size_t selected : eval_policy_indices) {
                    if (selector_proxy_equivalent(policies[i], policies[selected], true)) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) {
                    ++eval_dedup_skipped;
                    continue;
                }
            }
            eval_policy_indices.push_back(i);
        }
        for (size_t i = 0; i < policies.size(); ++i) {
            if (policies[i].name == "baseline") {
                append_policy_index(eval_policy_indices, i);
                break;
            }
        }
        append_rsf_companions(eval_policy_indices, "eval");
        compact_policy_indices(eval_policy_indices, true, "eval");
        stable_sort_rsf_pairs(eval_policy_indices);
    } else if (selector_fresh_tensor_plan) {
        fprintf(stderr,
            "%s: selector fresh tensor-plan mode skips whole-policy Stage-B sweep; exact KLD will score the tensor plan directly\n",
            __func__);
    }
    if (dedup_eval && eval_dedup_skipped > 0) {
        fprintf(stderr,
            "%s: selector skipped %zu proxy-equivalent policies before full PPL/KLD eval\n",
            __func__,
            eval_dedup_skipped);
    }
    note_skip_remaining("stage-b start");
    const bool run_stageb_eval = want_measured_eval && !skip_remaining_tuning;
    if (run_stageb_eval) {
        full_quant_eta_total = (int64_t) eval_policy_indices.size() + 2;
        full_quant_eta_done = 0;
        stageb_policy_total = (int64_t) eval_policy_indices.size();
        stageb_policy_done = 0;
        stageb_policy_done_seconds = 0.0;
        stageb_policy_eta_done = 0;
        stageb_policy_eta_done_seconds = 0.0;
        stageb_policy_active = false;
        stageb_policy_active_name.clear();
        stageb_policy_active_index = 0;
        update_full_quant_eta("selector-stage-b-runtime-load", true);
    } else {
        full_quant_eta_total = 1;
        full_quant_eta_done = 0;
        stageb_policy_total = 0;
        stageb_policy_done = 0;
        stageb_policy_done_seconds = 0.0;
        stageb_policy_eta_done = 0;
        stageb_policy_eta_done_seconds = 0.0;
        stageb_policy_active = false;
        stageb_policy_active_name.clear();
        stageb_policy_active_index = 0;
        update_full_quant_eta(skip_remaining_tuning ? "selector-skipping-to-final-materialization" : "selector-proxy-only", true);
    }
    const selector_kld_subset kld_budget = kld;
    const std::unique_ptr<selector_kld_subset> holdout_budget;
    bool selector_cuda_memory_known = false;
    double selector_cuda_free_gib = 0.0;
    double selector_cuda_total_gib = 0.0;
#if defined(GGML_USE_CUDA)
    if (run_stageb_eval && ggml_backend_cuda_get_device_count() > 0) {
        size_t cuda_free = 0;
        size_t cuda_total = 0;
        ggml_backend_cuda_get_device_memory(0, &cuda_free, &cuda_total);
        const bool reset_before_eval = true;
        if (reset_before_eval && cuda_total > 0 && cuda_free < cuda_total / 2) {
            fprintf(stderr,
                "%s: selector stage-b resetting CUDA device 0 before n_seq sizing to release encoder/proxy workspaces (free=%.2f GiB total=%.2f GiB)\n",
                __func__,
                (double) cuda_free / (1024.0 * 1024.0 * 1024.0),
                (double) cuda_total / (1024.0 * 1024.0 * 1024.0));
            if (selector_reset_cuda_device(0)) {
                ggml_backend_cuda_get_device_memory(0, &cuda_free, &cuda_total);
            }
        }
        if (cuda_total > 0) {
            selector_cuda_memory_known = true;
            selector_cuda_free_gib = (double) cuda_free / (1024.0 * 1024.0 * 1024.0);
            selector_cuda_total_gib = (double) cuda_total / (1024.0 * 1024.0 * 1024.0);
        }
    }
#endif
    const bool selector_n_seq_explicit = selector_control_has("N_SEQ");
    const selector_logits_budget logits_budget = selector_default_logits_budget();
    double max_logits_gib = std::max(SELECTOR_LOGITS_MIN_GIB, logits_budget.max_gib);
    double cuda_reserve_gib = 0.0;
    if (selector_cuda_memory_known && selector_cuda_free_gib > 0.0 && std::isfinite(selector_cuda_free_gib)) {
        cuda_reserve_gib = std::min(
            SELECTOR_N_SEQ_CUDA_RESERVE_MAX_GIB,
            std::max(
                SELECTOR_N_SEQ_CUDA_RESERVE_MIN_GIB,
                selector_cuda_total_gib * SELECTOR_N_SEQ_CUDA_RESERVE_FRACTION));
        const double cuda_logits_gib =
            std::max(SELECTOR_LOGITS_MIN_GIB, selector_cuda_free_gib - cuda_reserve_gib);
        max_logits_gib = std::min(max_logits_gib, cuda_logits_gib);
    }
    const double logits_gib_per_seq =
        (double) kld.n_ctx * (double) kld.n_vocab * (double) sizeof(float) / (1024.0 * 1024.0 * 1024.0);
    const int selector_n_seq_auto_cap = SELECTOR_N_SEQ_AUTO_MAX;
    int selector_n_seq = 1;
    if (logits_gib_per_seq > 0.0 && std::isfinite(logits_gib_per_seq)) {
        const int max_n_seq_by_logits = std::max(1, (int) std::floor(max_logits_gib / logits_gib_per_seq));
        if (selector_n_seq_explicit) {
            selector_n_seq = (int) std::max<int64_t>(1, selector_control_i64("N_SEQ", 1));
            if (selector_n_seq > max_n_seq_by_logits) {
                fprintf(stderr,
                    "%s: selector eval n_seq capped %d -> %d to keep logits workspace under %.2f GiB "
                    "(cuda_free=%.2f GiB cuda_reserve=%.2f GiB host_available=%.2f GiB host_reserve=%.2f GiB ctx=%d vocab=%d, %.2f GiB/seq)\n",
                    __func__,
                    selector_n_seq,
                    max_n_seq_by_logits,
                    max_logits_gib,
                    selector_cuda_free_gib,
                    cuda_reserve_gib,
                    logits_budget.available_gib,
                    logits_budget.reserve_gib,
                    kld.n_ctx,
                    kld.n_vocab,
                    logits_gib_per_seq);
                selector_n_seq = max_n_seq_by_logits;
            }
        } else {
            selector_n_seq = std::max(1, std::min({ max_n_seq_by_logits, selector_n_seq_auto_cap, kld.n_chunk }));
            fprintf(stderr,
                "%s: selector eval auto n_seq=%d from logits workspace %.2f GiB "
                "(cuda_free=%.2f GiB cuda_reserve=%.2f GiB host_available=%.2f GiB host_reserve=%.2f GiB ctx=%d vocab=%d, %.2f GiB/seq, auto_cap=%d)\n",
                __func__,
                selector_n_seq,
                max_logits_gib,
                selector_cuda_free_gib,
                cuda_reserve_gib,
                logits_budget.available_gib,
                logits_budget.reserve_gib,
                kld.n_ctx,
                kld.n_vocab,
                logits_gib_per_seq,
                selector_n_seq_auto_cap);
        }
    }
    common_params params;
    params.model.path = checkpoint_model_path;
    params.n_ctx = kld.n_ctx * selector_n_seq;
    params.n_batch = selector_eval_batch_override > 0
        ? std::min<int32_t>((int32_t) params.n_ctx, selector_eval_batch_override)
        : params.n_ctx;
    params.n_ubatch = params.n_batch;
    params.n_parallel = selector_n_seq;
    params.n_gpu_layers = (int32_t) selector_control_i64("N_GPU_LAYERS", 9999);
    params.cpuparams.n_threads = std::max(1, nthread);
    params.cpuparams_batch.n_threads = std::max(1, nthread);
    // Selector full PPL/KLD evaluation patches tensor buffers in a calibration-only context with
    // explicit n_ctx/n_batch/n_parallel. Let it load exactly that context instead of
    // running the general fit-to-VRAM probe, which can fail on freshly written local
    // NVFP4/MXFP6 GGUFs before the real selector load is attempted.
    params.fit_params = false;
    // Keep mmap enabled for selector full PPL/KLD evaluation. The non-mmap loader path
    // is fragile with derived/native NVFP4 GGUF tensors, while runtime tensor
    // patching still works through ggml_backend_tensor_set on loaded buffers.
    params.use_mmap = true;
    // Do not pre-fault the whole checkpoint mapping in one loader thread.
    // Stage-b only needs a calibration context; letting pages fault on demand
    // avoids a long CPU-only MAP_POPULATE phase before CUDA work can begin.
    params.use_mmap_prefetch = false;
    params.warmup = false;
    params.verbosity = 0;
    params.graph_reuse_disable = true;
    std::unique_ptr<selector_progress_heartbeat> load_heartbeat;
    if (run_stageb_eval) {
        load_heartbeat = std::make_unique<selector_progress_heartbeat>(
            "selector stage-b checkpoint load",
            10000);
    }
    params.load_progress_callback = [](float progress, void * user_data) -> bool {
        auto * heartbeat = static_cast<selector_progress_heartbeat *>(user_data);
        if (heartbeat != nullptr) {
            const int64_t done = (int64_t) std::llround(
                10000.0 * std::max(0.0f, std::min(1.0f, progress)));
            heartbeat->update(done, 10000, "loading runtime checkpoint", progress >= 1.0f);
        }
        return true;
    };
    params.load_progress_callback_user_data = load_heartbeat.get();

    auto stageb_binding_hash = [&](const std::vector<size_t> & indices) {
        selector_stageb_digest digest;
        for (size_t idx : indices) {
            if (idx < all_bindings.size()) {
                digest.add_string(all_bindings[idx].name);
            }
        }
        return digest.hex();
    };
    selector_stageb_digest mxfp6_binding_digest;
    for (const auto & b : mxfp6_bindings) {
        mxfp6_binding_digest.add_string(b.name);
    }
    const std::string all_binding_hash = stageb_binding_hash(all_binding_indices);
    const std::string stress_binding_hash = stageb_binding_hash(stress_binding_indices);
    const std::string mxfp6_binding_hash = mxfp6_binding_digest.hex();
    auto make_stageb_base_key = [&]() {
        nlohmann::ordered_json key = nlohmann::ordered_json::object();
        key["cache_version"] = SELECTOR_ENCODER_EVIDENCE_VERSION;
        key["source_model"] = selector_stageb_file_json(source_model_path);
        key["checkpoint_model"] = selector_stageb_file_json(checkpoint_model_path);
        key["imatrix"] = {
            {"entries", imatrix_data.size()},
            {"hash", imatrix_hash},
        };
        key["search_kld"] = selector_stageb_kld_json(kld_budget);
        key["validation_kld"] = holdout_budget ? selector_stageb_kld_json(*holdout_budget) : nlohmann::ordered_json(nullptr);
        key["eval"] = {
            {"chunks", kld_budget.n_chunk},
            {"n_seq", selector_n_seq},
            {"n_ctx", params.n_ctx},
            {"n_batch", params.n_batch},
            {"n_ubatch", params.n_ubatch},
            {"n_gpu_layers", params.n_gpu_layers},
            {"input_scale_policy", nvfp4_input_scale_policy},
        };
        key["encoder"] = {
            {"evidence_version", SELECTOR_ENCODER_EVIDENCE_VERSION},
            {"max_blocks", quantize_control_i64("LLAMA_NVFP4_ENCODER_MAX_BLOCKS", 0)},
            {"rsf_mode", nvfp4_rsf_mode_name(selector_rsf_mode)},
            {"rsf_depth", nvfp4_rsf_depth_name(selector_rsf_depth)},
        };
        key["rank"] = selector_stageb_rank_json(rank_cfg);
        key["bindings"] = {
            {"nvfp4_count", all_bindings.size()},
            {"nvfp4_all_hash", all_binding_hash},
            {"nvfp4_stress_hash", stress_binding_hash},
            {"mxfp6_count", mxfp6_bindings.size()},
            {"mxfp6_hash", mxfp6_binding_hash},
        };
        return key;
    };
    auto make_stageb_policy_key = [&](const selector_policy & policy, const std::vector<size_t> & binding_indices) {
        nlohmann::ordered_json key = make_stageb_base_key();
        key["kind"] = "policy";
        key["policy"] = {
            {"name", policy.name},
            {"cfg", selector_stageb_cfg_json(policy.cfg)},
            {"binding_mode", covers_all_nvfp4_bindings(binding_indices) ? "all" : "partial"},
            {"tensor_count", binding_indices.size()},
            {"tensor_hash", stageb_binding_hash(binding_indices)},
            {"has_survey", policy.has_survey},
        };
        return key;
    };
    auto find_current_baseline_policy = [&]() -> const selector_policy * {
        auto it = std::find_if(policies.begin(), policies.end(), [](const selector_policy & policy) {
            return policy.name == "baseline";
        });
        if (it != policies.end()) {
            return &*it;
        }
        it = std::find_if(policies.begin(), policies.end(), [](const selector_policy & policy) {
            return !selector_policy_is_rsf_family(policy);
        });
        return it == policies.end() ? nullptr : &*it;
    };
    auto make_stageb_baseline_key = [&]() {
        nlohmann::ordered_json key = make_stageb_base_key();
        const selector_policy * baseline_policy = find_current_baseline_policy();
        key["kind"] = "baseline";
        key["policy"] = {
            {"name", "seed_keep"},
            {"cfg", baseline_policy != nullptr ? selector_stageb_cfg_json(baseline_policy->cfg) : nlohmann::ordered_json(nullptr)},
            {"binding_mode", "runtime_checkpoint"},
            {"tensor_count", all_bindings.size()},
            {"tensor_hash", all_binding_hash},
        };
        return key;
    };
    auto make_tensor_rescue_key = [&](
            const selector_binding & b,
            const nvfp4_cuda_runtime_cfg & base_cfg,
            int64_t sample_blocks,
            int64_t refine_blocks,
            int64_t guard_blocks) {
        auto shape_json = [](const ggml_tensor * tensor) {
            nlohmann::ordered_json shape = nlohmann::ordered_json::array();
            if (tensor == nullptr) {
                return shape;
            }
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                shape.push_back(tensor->ne[i]);
            }
            return shape;
        };

        nlohmann::ordered_json key = nlohmann::ordered_json::object();
        key["cache_version"] = SELECTOR_ENCODER_EVIDENCE_VERSION;
        key["kind"] = "tensor_rescue";
        key["selector"] = make_stageb_base_key();
        key["tensor"] = {
            {"name", b.name},
            {"class", selector_tensor_class_name(b.cls)},
            {"bucket", b.bucket},
            {"layer", b.layer},
            {"source_type", b.source != nullptr ? ggml_type_name(b.source->type) : "unknown"},
            {"target_type", b.target != nullptr ? ggml_type_name(b.target->type) : "unknown"},
            {"source_nbytes", b.source_nbytes},
            {"target_nbytes", b.target_nbytes},
            {"source_shape", shape_json(b.source)},
            {"target_shape", shape_json(b.target)},
        };
        key["base_cfg"] = selector_stageb_cfg_json(base_cfg);
        key["rescue"] = {
            {"sample_blocks", sample_blocks},
            {"refine_blocks", refine_blocks},
            {"guard_blocks", guard_blocks},
            {"policy_budget", selector_control_i64("RESCUE_NVFP4_POLICY_BUDGET", SELECTOR_RESCUE_NVFP4_POLICY_BUDGET)},
            {"refine_top", selector_control_i64("RESCUE_NVFP4_REFINE_TOP", SELECTOR_RESCUE_NVFP4_REFINE_TOP)},
            {"guard_top", selector_control_i64("RESCUE_NVFP4_GUARD_TOP", SELECTOR_RESCUE_NVFP4_GUARD_TOP)},
            {"max_blocks", selector_control_i64("RESCUE_NVFP4_MAX_BLOCKS", 16384)},
            {"refine_max_blocks", selector_control_i64("RESCUE_NVFP4_REFINE_MAX_BLOCKS", 32768)},
            {"guard_max_blocks", selector_control_i64("RESCUE_NVFP4_GUARD_MAX_BLOCKS", 32768)},
        };
        return key;
    };
    selector_stageb_cache stageb_result_cache;
    stageb_result_cache.ledger = &ledger;
    if (run_stageb_eval) {
        stageb_result_cache.load(checkpoint_model_path + ".stageb-results.jsonl");
        if (ledger.enabled()) {
            stageb_result_cache.load_exact_eval_ledger(ledger.path());
        }
    }
    selector_tensor_rescue_cache tensor_rescue_cache;
    tensor_rescue_cache.load(checkpoint_model_path + ".tensor-rescue-results.jsonl");

    selector_derived_metrics baseline_eval;
    selector_derived_metrics baseline_holdout_eval;
    bool has_holdout_eval = false;
    bool stageb_cache_only_eval = false;
    bool stageb_runtime_decoded = false;

    auto try_stageb_cache_only_eval = [&]() -> bool {
        if (!run_stageb_eval) {
            return false;
        }
        if (!mxfp6_bindings.empty()) {
            return false;
        }
        if (!selector_control_string("SENSITIVITY_REPORT_FILE").empty()) {
            return false;
        }

        const nlohmann::ordered_json baseline_cache_key = make_stageb_baseline_key();
        selector_stageb_cache_entry cached_baseline;
        if (!stageb_result_cache.find(baseline_cache_key, "seed_keep", cached_baseline) ||
                !cached_baseline.search.ok) {
            return false;
        }

        struct cached_policy_eval {
            size_t policy_index = 0;
            selector_stageb_cache_entry entry;
        };
        std::vector<cached_policy_eval> cached_policies;
        cached_policies.reserve(eval_policy_indices.size());
        for (size_t policy_index : eval_policy_indices) {
            if (policy_index >= policies.size()) {
                return false;
            }
            const auto & policy = policies[policy_index];
            const std::vector<size_t> & eval_binding_indices =
                policy.has_survey ? all_binding_indices : stress_binding_indices;
            const nlohmann::ordered_json policy_cache_key =
                make_stageb_policy_key(policy, eval_binding_indices);
            selector_stageb_cache_entry cached_policy;
            if (!stageb_result_cache.find(policy_cache_key, policy.name, cached_policy) ||
                    !cached_policy.search.ok) {
                return false;
            }
            cached_policies.push_back({ policy_index, cached_policy });
        }

        baseline_eval = cached_baseline.search;
        baseline_holdout_eval = cached_baseline.validation;
        has_holdout_eval = cached_baseline.has_validation && baseline_holdout_eval.ok;

        for (const auto & cached : cached_policies) {
            auto & policy = policies[cached.policy_index];
            policy.measured = cached.entry.search;
            policy.measured_holdout = cached.entry.validation;
            policy.has_holdout = cached.entry.has_validation && policy.measured_holdout.ok;
            const selector_metric_rank policy_rank =
                selector_rank_policy_metrics(
                    policy.measured,
                    policy.has_holdout ? &policy.measured_holdout : nullptr,
                    baseline_eval,
                    has_holdout_eval ? &baseline_holdout_eval : nullptr,
                    rank_cfg);
            policy.measured_pass = policy_rank.pass;
            policy.measured_score = policy_rank.score;
        }

        stageb_policy_done = (int64_t) eval_policy_indices.size();
        stageb_policy_active = false;
        stageb_policy_active_name.clear();
        stageb_policy_active_index = 0;
        full_quant_eta_done = std::min<int64_t>(full_quant_eta_total, 2 + stageb_policy_done);
        fprintf(stderr,
            "%s: selector stage-b exact cache complete; skipping runtime checkpoint load policies=%zu search_chunks=%d validation_chunks=%d\n",
            __func__,
            cached_policies.size(),
            kld_budget.n_chunk,
            holdout_budget ? holdout_budget->n_chunk : 0);
        update_full_quant_eta("selector-stage-b-cache-complete", true);
        return true;
    };
    stageb_cache_only_eval = try_stageb_cache_only_eval();

    common_init_result_ptr init_res;
    llama_context * lctx = nullptr;
    if (run_stageb_eval && !stageb_cache_only_eval) {
#if defined(GGML_USE_CUDA)
        if (ggml_backend_cuda_get_device_count() > 0) {
            size_t cuda_free = 0;
            size_t cuda_total = 0;
            ggml_backend_cuda_get_device_memory(0, &cuda_free, &cuda_total);
            fprintf(stderr,
                "%s: selector stage-b CUDA memory before checkpoint load free=%.2f GiB total=%.2f GiB n_gpu_layers=%d n_seq=%d n_ctx=%d n_batch=%d n_ubatch=%d\n",
                __func__,
                (double) cuda_free / (1024.0 * 1024.0 * 1024.0),
                (double) cuda_total / (1024.0 * 1024.0 * 1024.0),
                params.n_gpu_layers,
                selector_n_seq,
                (int) params.n_ctx,
                (int) params.n_batch,
                (int) params.n_ubatch);
        }
#endif
        // The selector mutates already-loaded tensor buffers between KLD probes,
        // so this calibration context must not reuse graphs across evaluations.
        if (load_heartbeat) {
            load_heartbeat->detail("loading runtime checkpoint", true);
        }
        init_res = common_init_from_params(params);
        if (load_heartbeat) {
            load_heartbeat->finish(init_res ? "runtime checkpoint loaded" : "runtime checkpoint load failed");
        }
        lctx = init_res ? init_res->context() : nullptr;
    }
    bool have_runtime_eval = run_stageb_eval && lctx != nullptr;
    bool have_measured_eval = stageb_cache_only_eval || have_runtime_eval;
    if (run_stageb_eval && !have_measured_eval) {
        fprintf(stderr, "%s: selector stage-b unavailable for checkpoint %s; continuing with proxy-only selection and tensor plan\n", __func__, checkpoint_model_path.c_str());
        if (require_runtime_cache) {
            fprintf(stderr, "%s: selector runtime cache is required; aborting selector\n", __func__);
            return false;
        }
    }
    if (have_runtime_eval) {
        selector_progress_heartbeat runtime_bind_heartbeat(
            "selector stage-b runtime bind",
            (int64_t) all_bindings.size() + (int64_t) mxfp6_bindings.size());
        runtime_bind_heartbeat.detail("binding runtime tensors", true);
        std::unordered_map<std::string, ggml_tensor *> runtime_tensors;
        for (const auto & kv : llama_internal_get_tensor_map(init_res->model())) {
            if (kv.second != nullptr) {
                runtime_tensors.emplace(kv.first, kv.second);
            }
        }
        int64_t runtime_bind_done = 0;
        for (auto & b : all_bindings) {
            auto it = runtime_tensors.find(b.name);
            if (it == runtime_tensors.end() || it->second == nullptr || it->second->type != GGML_TYPE_NVFP4) {
                fprintf(stderr, "%s: selector stage-b missing runtime NVFP4 tensor %s; continuing with proxy-only selection\n",
                    __func__, b.name.c_str());
                if (require_runtime_cache) {
                    return false;
                }
                lctx = nullptr;
                break;
            }
            const size_t runtime_nbytes = ggml_nbytes(it->second);
            if (runtime_nbytes != b.target_nbytes) {
                fprintf(stderr, "%s: selector stage-b runtime tensor size mismatch for %s (%zu != %zu); continuing with proxy-only selection\n",
                    __func__, b.name.c_str(), runtime_nbytes, b.target_nbytes);
                if (require_runtime_cache) {
                    return false;
                }
                lctx = nullptr;
                break;
            }
            b.target = it->second;
            const int64_t scale_len = std::max<int64_t>(1, b.source ? b.source->ne[2] : 1);
            const std::string scale_name = llama_nvfp4_scale_tensor_name(b.name);
            auto sit = runtime_tensors.find(scale_name);
            if (sit != runtime_tensors.end() && sit->second != nullptr && sit->second->type == GGML_TYPE_F32 &&
                    ggml_nelements(sit->second) >= scale_len) {
                b.target_scale = sit->second;
                b.target_scale_nbytes = (size_t) scale_len * sizeof(float);
            }
            const std::string input_scale_name = llama_nvfp4_input_scale_tensor_name(b.name);
            auto iit = runtime_tensors.find(input_scale_name);
            if (iit != runtime_tensors.end() && iit->second != nullptr && iit->second->type == GGML_TYPE_F32 &&
                    ggml_nelements(iit->second) >= scale_len) {
                b.target_input_scale = iit->second;
                b.target_input_scale_nbytes = (size_t) scale_len * sizeof(float);
            }
            runtime_bind_heartbeat.update(++runtime_bind_done, "binding runtime tensors");
        }
        for (auto & b : mxfp6_bindings) {
            auto it = runtime_tensors.find(b.name);
            if (it == runtime_tensors.end() || it->second == nullptr || it->second->type != GGML_TYPE_MXFP6_E2M3) {
                fprintf(stderr, "%s: selector stage-b missing runtime MXFP6_E2M3 tensor %s; continuing without MXFP6_E2M3 scale measured tuning\n",
                    __func__, b.name.c_str());
                b.target = nullptr;
                continue;
            }
            const size_t runtime_nbytes = ggml_nbytes(it->second);
            if (runtime_nbytes != b.target_nbytes) {
                fprintf(stderr, "%s: selector stage-b runtime MXFP6_E2M3 tensor size mismatch for %s (%zu != %zu); continuing without that tensor\n",
                    __func__, b.name.c_str(), runtime_nbytes, b.target_nbytes);
                b.target = nullptr;
                continue;
            }
            b.target = it->second;

            const std::string scale_name = llama_nvfp4_scale_tensor_name(b.name);
            auto sit = runtime_tensors.find(scale_name);
            if (sit == runtime_tensors.end() || sit->second == nullptr || sit->second->type != GGML_TYPE_F32) {
                if (b.target_scale_nbytes == sizeof(float) && runtime_nbytes >= MXFP6_HEADER_OFFSET) {
                    b.target_scale = nullptr;
                    continue;
                }
                fprintf(stderr, "%s: selector stage-b missing runtime MXFP6_E2M3 scale tensor %s; continuing without %s\n",
                    __func__, scale_name.c_str(), b.name.c_str());
                b.target = nullptr;
                b.target_scale = nullptr;
                continue;
            }
            b.target_scale = sit->second;
            b.target_scale_nbytes = (size_t) std::max<int64_t>(1, b.source->ne[2]) * sizeof(float);
            runtime_bind_heartbeat.update(++runtime_bind_done, "binding runtime tensors");
        }
        runtime_bind_heartbeat.finish("runtime tensors bound");
    }
    have_runtime_eval = run_stageb_eval && lctx != nullptr;
    have_measured_eval = stageb_cache_only_eval || have_runtime_eval;
    auto apply_rsf_holdout_guard = [&](selector_policy & policy) {
        if (!selector_policy_is_rsf_family(policy)) {
            return;
        }
        const selector_policy * base_policy =
            selector_find_rsf_base_policy(policies, policy, true);
        if (base_policy == nullptr) {
            return;
        }
        std::string reason;
        if (!selector_rsf_holdout_guard(policy, *base_policy, rank_cfg, reason)) {
            policy.measured_pass = false;
            policy.measured_score = std::numeric_limits<double>::infinity();
            fprintf(stderr,
                "%s: selector RSF guard rejected policy=%s base=%s reason=%s\n",
                __func__,
                policy.name.c_str(),
                base_policy->name.c_str(),
                reason.c_str());
            return;
        }

        const selector_metric_rank accepted_rank =
            selector_rank_policy_metrics(
                policy.measured,
                policy.has_holdout ? &policy.measured_holdout : nullptr,
                baseline_eval,
                has_holdout_eval ? &baseline_holdout_eval : nullptr,
                rank_cfg);
        policy.measured_score = accepted_rank.score;
        if (!policy.measured_pass) {
            policy.measured_pass = true;
            fprintf(stderr,
                "%s: selector RSF guard accepted policy=%s base=%s despite stage-B hard gate; %s\n",
                __func__,
                policy.name.c_str(),
                base_policy->name.c_str(),
                reason.c_str());
        }
    };
    if (have_runtime_eval) {
        struct runtime_cache_task {
            bool mxfp6 = false;
            size_t index = 0;
        };
        std::vector<runtime_cache_task> runtime_cache_tasks;
        runtime_cache_tasks.reserve(all_bindings.size() + mxfp6_bindings.size());
        for (size_t i = 0; i < all_bindings.size(); ++i) {
            runtime_cache_tasks.push_back({false, i});
        }
        for (size_t i = 0; i < mxfp6_bindings.size(); ++i) {
            if (mxfp6_bindings[i].target != nullptr) {
                runtime_cache_tasks.push_back({true, i});
            }
        }
        selector_progress_heartbeat runtime_cache_heartbeat(
            "selector stage-b runtime cache",
            (int64_t) runtime_cache_tasks.size());
        runtime_cache_heartbeat.detail("caching original runtime tensors", true);
        auto runtime_cache_name = [&](size_t pos) -> const char * {
            const runtime_cache_task & task = runtime_cache_tasks[pos];
            return task.mxfp6 ? mxfp6_bindings[task.index].name.c_str() : all_bindings[task.index].name.c_str();
        };
        auto runtime_cache_one = [&](size_t pos) -> const char * {
            const runtime_cache_task & task = runtime_cache_tasks[pos];
            if (task.mxfp6) {
                auto & b = mxfp6_bindings[task.index];
                return (!quantize_binding_ensure_target_bytes(b) || !quantize_binding_ensure_scale_bytes(b)) ?
                    "MXFP6_E2M3 bytes" : nullptr;
            }
            auto & b = all_bindings[task.index];
            if (!quantize_binding_ensure_target_bytes(b)) {
                return "tensor bytes";
            }
            if (b.target_scale != nullptr && !quantize_binding_ensure_scale_bytes(b)) {
                return "NVFP4 scale bytes";
            }
            if (b.target_input_scale != nullptr && !quantize_binding_ensure_input_scale_bytes(b)) {
                return "NVFP4 input-scale bytes";
            }
            return nullptr;
        };
        auto runtime_cache_fail = [&](size_t pos, const char * what) {
            fprintf(stderr,
                "%s: selector failed to cache original runtime %s for %s\n",
                __func__,
                what != nullptr ? what : "tensor bytes",
                runtime_cache_name(pos));
            restore_all();
            return false;
        };
        const int runtime_cache_threads = (int) std::max<int64_t>(1, std::min<int64_t>(
            (int64_t) runtime_cache_tasks.size(),
            selector_control_i64("POLICY_THREADS", std::max(1, nthread))));
        if (runtime_cache_threads <= 1 || runtime_cache_tasks.size() <= 1) {
            int64_t runtime_cache_done = 0;
            for (size_t pos = 0; pos < runtime_cache_tasks.size(); ++pos) {
                const char * error = runtime_cache_one(pos);
                if (error != nullptr) {
                    return runtime_cache_fail(pos, error);
                }
                runtime_cache_heartbeat.update(++runtime_cache_done, "caching original runtime tensors");
            }
        } else {
            const size_t no_failed_cache = std::numeric_limits<size_t>::max();
            std::atomic<size_t> next_cache{0};
            std::atomic<size_t> failed_cache{no_failed_cache};
            std::atomic<int64_t> runtime_cache_done{0};
            std::vector<const char *> cache_errors(runtime_cache_tasks.size(), nullptr);
            std::mutex cache_progress_mutex;
            std::vector<std::thread> workers;
            workers.reserve((size_t) runtime_cache_threads);
            fprintf(stderr,
                "selector stage-b runtime cache threads=%d tensors=%zu\n",
                runtime_cache_threads,
                runtime_cache_tasks.size());
            for (int ti = 0; ti < runtime_cache_threads; ++ti) {
                workers.emplace_back([&]() {
                    while (true) {
                        const size_t pos = next_cache.fetch_add(1, std::memory_order_relaxed);
                        if (pos >= runtime_cache_tasks.size()) {
                            break;
                        }
                        const char * error = runtime_cache_one(pos);
                        if (error != nullptr) {
                            cache_errors[pos] = error;
                            size_t expected = no_failed_cache;
                            (void) failed_cache.compare_exchange_strong(
                                expected, pos, std::memory_order_acq_rel, std::memory_order_acquire);
                            continue;
                        }
                        const int64_t done = runtime_cache_done.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (done == (int64_t) runtime_cache_tasks.size() || (done % 8) == 0) {
                            std::lock_guard<std::mutex> lock(cache_progress_mutex);
                            runtime_cache_heartbeat.update(done, "caching original runtime tensors");
                        }
                    }
                });
            }
            for (auto & worker : workers) {
                worker.join();
            }
            const size_t failed_pos = failed_cache.load(std::memory_order_acquire);
            if (failed_pos != no_failed_cache) {
                return runtime_cache_fail(failed_pos, cache_errors[failed_pos]);
            }
            runtime_cache_heartbeat.update(
                (int64_t) runtime_cache_tasks.size(),
                "caching original runtime tensors");
        }
        runtime_cache_heartbeat.finish("original runtime tensors cached");
        int64_t kld_metric_threads = selector_kld_threads_override();
        if (kld_metric_threads <= 0) {
            kld_metric_threads = std::max<unsigned int>(1, std::thread::hardware_concurrency());
        }
        fprintf(stderr,
            "selector stage-b runtime cache active checkpoint=%s tensors=%zu mx6_tensors=%zu ctx=%d n_seq=%d n_batch=%d n_ubatch=%d kld_threads=%" PRId64 "; candidates patch loaded tensor buffers without GGUF reload\n",
            checkpoint_model_path.c_str(),
            all_bindings.size(),
            mxfp6_bindings.size(),
            kld.n_ctx,
            selector_n_seq,
            params.n_batch,
            params.n_ubatch,
            kld_metric_threads);
        full_quant_eta_done = 1;
        update_full_quant_eta("selector-stage-b-baseline-kld", true);
        const nlohmann::ordered_json baseline_cache_key = make_stageb_baseline_key();
        selector_stageb_cache_entry cached_baseline;
        if (stageb_result_cache.find(baseline_cache_key, "seed_keep", cached_baseline)) {
            baseline_eval = cached_baseline.search;
            baseline_holdout_eval = cached_baseline.validation;
            has_holdout_eval = cached_baseline.has_validation && baseline_holdout_eval.ok;
        } else {
            selector_kld_metrics baseline_km;
            if (!selector_eval_kld_subset(
                    lctx, kld_budget, params.n_batch, baseline_km, true,
                    "selector stage-b baseline search kld")) {
                restore_all();
                return false;
            }
            stageb_runtime_decoded = true;
            baseline_eval = selector_derive_metrics(baseline_km);

            if (holdout_budget && holdout_budget->n_chunk > 0) {
                selector_kld_metrics baseline_holdout_km;
                if (!selector_eval_kld_subset(
                        lctx, *holdout_budget, params.n_batch, baseline_holdout_km, true,
                        "selector stage-b baseline validation kld")) {
                    restore_all();
                    return false;
                }
                stageb_runtime_decoded = true;
                baseline_holdout_eval = selector_derive_metrics(baseline_holdout_km);
                has_holdout_eval = baseline_holdout_eval.ok;
            }
            const selector_metric_rank baseline_rank =
                selector_rank_policy_metrics(
                    baseline_eval,
                    has_holdout_eval ? &baseline_holdout_eval : nullptr,
                    baseline_eval,
                    has_holdout_eval ? &baseline_holdout_eval : nullptr,
                    rank_cfg);
            stageb_result_cache.append(
                "baseline",
                baseline_cache_key,
                "seed_keep",
                baseline_eval,
                has_holdout_eval ? &baseline_holdout_eval : nullptr,
                baseline_rank.pass,
                baseline_rank.score);
        }

        const std::string baseline_main_summary =
            selector_format_metrics("search", baseline_eval);
        fprintf(stderr,
            "selector stage-b baseline %s\n",
            baseline_main_summary.c_str());
        if (has_holdout_eval) {
            const std::string baseline_holdout_summary =
                selector_format_metrics("validation", baseline_holdout_eval);
            fprintf(stderr,
                "selector stage-b baseline %s\n",
                baseline_holdout_summary.c_str());
        }
        full_quant_eta_done = 2;
        update_full_quant_eta("selector-stage-b-policy-eval", true);

        for (size_t i = 0; i < eval_policy_indices.size(); ++i) {
            auto & policy = policies[eval_policy_indices[i]];
            stageb_policy_active = true;
            stageb_policy_start = std::chrono::steady_clock::now();
            stageb_policy_active_name = policy.name;
            stageb_policy_active_index = i + 1;
            stageb_policy_active_estimated_total_seconds = 0.0;
            stageb_policy_active_estimate_basis.clear();
            full_quant_eta_done = std::min<int64_t>(full_quant_eta_total, 2 + stageb_policy_done);
            update_full_quant_eta("selector-stage-b-policy-running", true);
            fprintf(stderr,
                "selector stage-b start [%zu/%zu] policy=%s tensors=%zu search_chunks=%d validation_chunks=%d\n",
                i + 1,
                eval_policy_indices.size(),
                policy.name.c_str(),
                policy.has_survey ? all_binding_indices.size() : stress_binding_indices.size(),
                kld_budget.n_chunk,
                holdout_budget ? holdout_budget->n_chunk : 0);
            const std::vector<size_t> & eval_binding_indices = policy.has_survey ? all_binding_indices : stress_binding_indices;
            const nlohmann::ordered_json policy_cache_key = make_stageb_policy_key(policy, eval_binding_indices);
            selector_stageb_cache_entry cached_policy;
            if (stageb_result_cache.find(policy_cache_key, policy.name, cached_policy)) {
                policy.measured = cached_policy.search;
                policy.measured_holdout = cached_policy.validation;
                policy.has_holdout = cached_policy.has_validation && policy.measured_holdout.ok;
                const selector_metric_rank policy_rank =
                    selector_rank_policy_metrics(
                        policy.measured,
                        policy.has_holdout ? &policy.measured_holdout : nullptr,
                        baseline_eval,
                        has_holdout_eval ? &baseline_holdout_eval : nullptr,
                        rank_cfg);
                policy.measured_pass = policy_rank.pass;
                policy.measured_score = policy_rank.score;
                apply_rsf_holdout_guard(policy);
                fprintf(stderr,
                    "selector stage-b cache hit policy=%s search_chunks=%d validation_chunks=%d\n",
                    policy.name.c_str(),
                    kld_budget.n_chunk,
                    policy.has_holdout ? (holdout_budget ? holdout_budget->n_chunk : 0) : 0);
                const std::string policy_main_summary =
                    selector_format_metrics("search", policy.measured);
                fprintf(stderr,
                    "selector stage-b policy=%s measured_score=%.6f pass=%s %s\n",
                    policy.name.c_str(),
                    policy.measured_score,
                    policy.measured_pass ? "yes" : "no",
                    policy_main_summary.c_str());
                if (policy.has_holdout) {
                    const std::string policy_holdout_summary =
                        selector_format_metrics("validation", policy.measured_holdout);
                    fprintf(stderr,
                        "selector stage-b policy=%s %s\n",
                        policy.name.c_str(),
                        policy_holdout_summary.c_str());
                }
                finish_selector_cuda_policy_scope();
                finish_stageb_policy_unit("selector-stage-b-policy-cached", true, false);
                if (note_skip_remaining("stage-b")) {
                    break;
                }
                continue;
            }
            if (!covers_all_nvfp4_bindings(eval_binding_indices)) {
                restore_all();
            }
            bool policy_patch_ok = true;
            const auto patch_t0 = std::chrono::steady_clock::now();
            for (size_t idx : eval_binding_indices) {
                auto & b = all_bindings[idx];
                if (!quantize_binding_ensure_target_bytes(b)) {
                    restore_all();
                    return false;
                }
            }
            const auto patch_ensure_t1 = std::chrono::steady_clock::now();
            struct stageb_patch_result {
                bool ok = true;
                bool direct_applied = false;
                double tensor_sq = 0.0;
                double tensor_abs = 0.0;
                double tensor_max = 0.0;
                int64_t tensor_n = 0;
            };
            // Direct stage-B patches keep the selector fast by updating the
            // loaded CUDA tensor in place. The CUDA helper must resolve the
            // actual active storage (including split-buffer data_device) and
            // write the runtime layout, not assume tensor->data is what matmul
            // will read.
            const bool stageb_direct_patch = true;
            const bool stageb_collect_patch_eval_metrics =
                selector_control_bool("STAGEB_PATCH_EVAL", false);
            const bool stageb_verify_direct_patch =
                selector_control_bool("STAGEB_DIRECT_VERIFY", !nvfp4_cfg_has_rsf(policy.cfg));
            std::vector<stageb_patch_result> patch_results(eval_binding_indices.size());
            const bool stageb_policy_threads_explicit = selector_control_has("POLICY_THREADS");
            const int requested_stageb_patch_threads = (int) std::max<int64_t>(1, std::min<int64_t>(
                (int64_t) eval_binding_indices.size(),
                selector_control_i64("POLICY_THREADS",
                    std::max<int>(1, nthread))));
            const int stageb_encoder_threads = nvfp4_encoder_threads > 0 ?
                std::min<int>(4, std::max<int>(1, nvfp4_encoder_threads)) : 4;
            const int default_stageb_patch_threads = std::max<int>(1, nthread / stageb_encoder_threads);
            const int stageb_patch_threads = stageb_policy_threads_explicit ?
                requested_stageb_patch_threads :
                std::max<int>(1, std::min<int>(requested_stageb_patch_threads, default_stageb_patch_threads));
            const int stageb_binding_nthread = std::max(1, nthread / std::max(1, stageb_patch_threads));
            if (stageb_patch_threads > 1) {
                fprintf(stderr,
                    "selector stage-b patch policy=%s threads=%d tensor_threads=%d encoder_threads=%d tensors=%zu patch_eval=%s direct_verify=%s\n",
                    policy.name.c_str(),
                    stageb_patch_threads,
                    stageb_binding_nthread,
                    stageb_encoder_threads,
                    eval_binding_indices.size(),
                    stageb_collect_patch_eval_metrics ? "yes" : "no",
                    stageb_verify_direct_patch ? "yes" : "skipped_rsf");
            }
            const size_t no_failed_patch = std::numeric_limits<size_t>::max();
            auto clear_stageb_retry_caches = [&]() {
                clear_selector_cuda_caches(true);
            };
            auto patch_policy_once = [&](int attempt, size_t & failed_pos) -> bool {
                failed_pos = no_failed_patch;
                for (auto & r : patch_results) {
                    r = stageb_patch_result{};
                }
                auto patch_detail = [&](const char * state, size_t pos) -> std::string {
                    char detail[512];
                    if (pos < eval_binding_indices.size()) {
                        const auto & b = all_bindings[eval_binding_indices[pos]];
                        snprintf(detail, sizeof(detail),
                            "policy=%s attempt=%d %s tensor=%s",
                            policy.name.c_str(),
                            attempt,
                            state,
                            b.name.c_str());
                    } else {
                        snprintf(detail, sizeof(detail),
                            "policy=%s attempt=%d %s",
                            policy.name.c_str(),
                            attempt,
                            state);
                    }
                    return detail;
                };
                selector_progress_heartbeat patch_heartbeat(
                    "selector stage-b patch",
                    (int64_t) eval_binding_indices.size());
                std::atomic<int64_t> patch_done { 0 };
                auto patch_update = [&](const char * state, size_t pos, bool print_now = false) {
                    note_stageb_policy_active_progress(
                        patch_done.load(std::memory_order_relaxed),
                        (int64_t) eval_binding_indices.size(),
                        "patch");
                    patch_heartbeat.update(
                        patch_done.load(std::memory_order_relaxed),
                        (int64_t) eval_binding_indices.size(),
                        patch_detail(state, pos),
                        print_now);
                    update_full_quant_eta("selector-stage-b-policy-patch", print_now);
                };
                auto patch_done_one = [&](const char * state, size_t pos) {
                    const int64_t done = patch_done.fetch_add(1, std::memory_order_relaxed) + 1;
                    note_stageb_policy_active_progress(done, (int64_t) eval_binding_indices.size(), "patch");
                    patch_heartbeat.update(
                        done,
                        (int64_t) eval_binding_indices.size(),
                        patch_detail(state, pos));
                    update_full_quant_eta("selector-stage-b-policy-patch");
                };
                patch_update("starting", no_failed_patch, true);
                if (stageb_patch_threads <= 1 || eval_binding_indices.size() <= 1) {
                    for (size_t pos = 0; pos < eval_binding_indices.size(); ++pos) {
                        auto & b = all_bindings[eval_binding_indices[pos]];
                        auto & r = patch_results[pos];
                        patch_update("encoding", pos);
                        if (stageb_direct_patch &&
                                nvfp4_selector_quantize_binding(b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                    r.tensor_sq, r.tensor_abs, r.tensor_max, r.tensor_n, 0, 0, true, nullptr,
                                    stageb_collect_patch_eval_metrics)) {
                            r.direct_applied = true;
                            patch_done_one("encoded direct", pos);
                            continue;
                        }
                        if (!nvfp4_selector_quantize_binding(b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                r.tensor_sq, r.tensor_abs, r.tensor_max, r.tensor_n, 0, 0, false, nullptr,
                                stageb_collect_patch_eval_metrics)) {
                            r.ok = false;
                            failed_pos = pos;
                            patch_update("failed", pos, true);
                            return false;
                        }
                        patch_done_one("encoded", pos);
                    }
                    patch_heartbeat.finish(patch_detail("complete", no_failed_patch));
                    return true;
                } else {
                    std::atomic<size_t> next_patch { 0 };
                    std::atomic<size_t> failed_patch { no_failed_patch };
                    std::atomic<bool> patch_failed { false };
                    std::vector<std::thread> workers;
                    workers.reserve((size_t) stageb_patch_threads);
                    for (int ti = 0; ti < stageb_patch_threads; ++ti) {
                        workers.emplace_back([&]() {
                            while (true) {
                                if (patch_failed.load(std::memory_order_acquire)) {
                                    break;
                                }
                                const size_t pos = next_patch.fetch_add(1, std::memory_order_relaxed);
                                if (pos >= eval_binding_indices.size()) {
                                    break;
                                }
                                auto & b = all_bindings[eval_binding_indices[pos]];
                                auto & r = patch_results[pos];
                                patch_update("encoding", pos);
                                if (stageb_direct_patch &&
                                        nvfp4_selector_quantize_binding(b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                            r.tensor_sq, r.tensor_abs, r.tensor_max, r.tensor_n, 0, 0, true, nullptr,
                                            stageb_collect_patch_eval_metrics)) {
                                    r.direct_applied = true;
                                    patch_done_one("encoded direct", pos);
                                    continue;
                                }
                                if (!nvfp4_selector_quantize_binding(b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                        r.tensor_sq, r.tensor_abs, r.tensor_max, r.tensor_n, 0, 0, false, nullptr,
                                        stageb_collect_patch_eval_metrics)) {
                                    r.ok = false;
                                    size_t expected = no_failed_patch;
                                    (void) failed_patch.compare_exchange_strong(
                                        expected, pos, std::memory_order_acq_rel, std::memory_order_acquire);
                                    patch_failed.store(true, std::memory_order_release);
                                    patch_update("failed", pos, true);
                                    continue;
                                }
                                patch_done_one("encoded", pos);
                            }
                        });
                    }
                    for (auto & worker : workers) {
                        worker.join();
                    }
                    failed_pos = failed_patch.load(std::memory_order_acquire);
                    if (failed_pos != no_failed_patch) {
                        auto & b = all_bindings[eval_binding_indices[failed_pos]];
                        fprintf(stderr,
                            "%s: selector stage-b patch attempt=%d failed policy=%s tensor=%s; stopped remaining workers\n",
                            __func__, attempt, policy.name.c_str(), b.name.c_str());
                        patch_update("failed", failed_pos, true);
                        return false;
                    }
                    patch_heartbeat.finish(patch_detail("complete", no_failed_patch));
                    return true;
                }
            };
            const std::string stageb_binding_threads_text = std::to_string(stageb_binding_nthread);
            // Stage-B has already split nthread across patch workers; keep a
            // recipe-level selector.threads from expanding each binding patch.
            const std::string stageb_selector_threads_key = selector_generic_control_key("THREADS");
            scoped_quantize_control stageb_selector_threads(
                stageb_selector_threads_key.c_str(),
                stageb_binding_threads_text.c_str());
            size_t failed_patch_pos = no_failed_patch;
            bool stageb_patch_quant_ok = patch_policy_once(1, failed_patch_pos);
            if (!stageb_patch_quant_ok) {
                const char * failed_name = failed_patch_pos != no_failed_patch
                    ? all_bindings[eval_binding_indices[failed_patch_pos]].name.c_str()
                    : "<unknown>";
                fprintf(stderr,
                    "%s: selector stage-b CUDA patch recovery policy=%s failed_tensor=%s; "
                    "clearing CUDA caches and retrying once with threads=%d\n",
                    __func__, policy.name.c_str(), failed_name, stageb_patch_threads);
                clear_stageb_retry_caches();
                restore_all();
                clear_stageb_retry_caches();
                failed_patch_pos = no_failed_patch;
                stageb_patch_quant_ok = patch_policy_once(2, failed_patch_pos);
                if (!stageb_patch_quant_ok) {
                    const char * retry_failed_name = failed_patch_pos != no_failed_patch
                        ? all_bindings[eval_binding_indices[failed_patch_pos]].name.c_str()
                        : "<unknown>";
                    fprintf(stderr,
                        "%s: selector stage-b CUDA patch recovery failed policy=%s tensor=%s after retry\n",
                        __func__, policy.name.c_str(), retry_failed_name);
                }
            }
            const auto patch_quant_t1 = std::chrono::steady_clock::now();
            size_t direct_patch_count = 0;
            size_t direct_fallback_count = 0;
            if (!stageb_patch_quant_ok) {
                policy_patch_ok = false;
            }
            for (size_t pos = 0; policy_patch_ok && pos < eval_binding_indices.size(); ++pos) {
                auto & b = all_bindings[eval_binding_indices[pos]];
                const auto & r = patch_results[pos];
                if (!r.ok) {
                    fprintf(stderr, "%s: selector rejected policy=%s during stage-b tensor=%s after quantize failure\n",
                        __func__, policy.name.c_str(), b.name.c_str());
                    policy_patch_ok = false;
                    break;
                }
                direct_patch_count += r.direct_applied ? 1 : 0;
                float header_weight_scale = 1.0f;
                float header_input_scale = 1.0f;
                if (!nvfp4_selector_prepare_nvfp4_runtime_scales(
                        b, nvfp4_input_scale_policy, &header_weight_scale, &header_input_scale)) {
                    restore_all();
                    return false;
                }
                if (b.target_scale != nullptr &&
                        !quantize_tensor_copy_in(b.target_scale, b.working_scale_bytes.data(), b.target_scale_nbytes)) {
                    restore_all();
                    return false;
                }
                if (b.target_input_scale != nullptr &&
                        !quantize_tensor_copy_in(
                            b.target_input_scale,
                            b.working_input_scale_bytes.data(),
                            b.target_input_scale_nbytes)) {
                    restore_all();
                    return false;
                }
                if (r.direct_applied &&
                        !ggml_cuda_nvfp4_tensor_set_header_scales(b.target, header_weight_scale, header_input_scale, nullptr)) {
                    restore_all();
                    return false;
                }
                if (!r.direct_applied && !quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                    restore_all();
                    return false;
                }
            }
            if (policy_patch_ok && direct_patch_count > 0 && stageb_verify_direct_patch) {
                bool direct_verified = false;
                bool direct_mismatch = false;
                std::string direct_mismatch_name;
                std::vector<uint8_t> verify_bytes;
                std::vector<uint8_t> reference_bytes;
                for (size_t pos = 0; pos < eval_binding_indices.size(); ++pos) {
                    auto & b = all_bindings[eval_binding_indices[pos]];
                    const auto & r = patch_results[pos];
                    if (!r.direct_applied) {
                        continue;
                    }
                    double ref_sq = 0.0;
                    double ref_abs = 0.0;
                    double ref_max = 0.0;
                    int64_t ref_n = 0;
                    if (!nvfp4_selector_quantize_binding(
                            b, policy.cfg, stageb_binding_nthread, reference_bytes,
                            ref_sq, ref_abs, ref_max, ref_n, 0, 0, false) || ref_n <= 0) {
                        direct_mismatch = true;
                        direct_mismatch_name = b.name;
                        direct_verified = true;
                        break;
                    }
                    verify_bytes.resize(b.target_nbytes);
                    if (!quantize_tensor_copy_out(b.target, verify_bytes.data(), b.target_nbytes) ||
                            reference_bytes.size() != b.target_nbytes ||
                            std::memcmp(verify_bytes.data(), reference_bytes.data(), b.target_nbytes) != 0) {
                        direct_mismatch = true;
                        direct_mismatch_name = b.name;
                    }
                    direct_verified = true;
                    break;
                }
                if (direct_verified && direct_mismatch) {
                    if (nvfp4_cfg_has_rsf(policy.cfg)) {
                        fprintf(stderr,
                            "%s: selector direct runtime patch readback mismatch policy=%s tensor=%s; keeping direct NVFP4 runtime layout for RSF policy\n",
                            __func__, policy.name.c_str(), direct_mismatch_name.c_str());
                    } else {
                        fprintf(stderr,
                            "%s: selector direct runtime patch readback mismatch policy=%s tensor=%s; falling back to backend tensor setter for this policy\n",
                            __func__, policy.name.c_str(), direct_mismatch_name.c_str());
                        for (size_t pos = 0; pos < eval_binding_indices.size(); ++pos) {
                            auto & b = all_bindings[eval_binding_indices[pos]];
                            const auto & r = patch_results[pos];
                            if (!r.direct_applied) {
                                continue;
                            }
                            double ref_sq = 0.0;
                            double ref_abs = 0.0;
                            double ref_max = 0.0;
                            int64_t ref_n = 0;
                            if (!nvfp4_selector_quantize_binding(
                                    b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                    ref_sq, ref_abs, ref_max, ref_n, 0, 0, false) || ref_n <= 0) {
                                restore_all();
                                return false;
                            }
                            if (!quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                                restore_all();
                                return false;
                            }
                            ++direct_fallback_count;
                        }
                    }
                }
            }
            const auto patch_apply_t1 = std::chrono::steady_clock::now();
            const double patch_ensure_s = std::chrono::duration<double>(patch_ensure_t1 - patch_t0).count();
            const double patch_quant_s = std::chrono::duration<double>(patch_quant_t1 - patch_ensure_t1).count();
            const double patch_apply_s = std::chrono::duration<double>(patch_apply_t1 - patch_quant_t1).count();
            double patch_sum_sq = 0.0;
            double patch_sum_abs = 0.0;
            double patch_max_abs = 0.0;
            int64_t patch_count = 0;
            if (stageb_collect_patch_eval_metrics) {
                for (const auto & r : patch_results) {
                    patch_sum_sq += r.tensor_sq;
                    patch_sum_abs += r.tensor_abs;
                    patch_max_abs = std::max(patch_max_abs, r.tensor_max);
                    patch_count += r.tensor_n;
                }
            }
            fprintf(stderr,
                "selector stage-b patch complete policy=%s tensors=%zu direct=%zu direct_fallback=%zu threads=%d patch_eval=%s direct_verify=%s patch_rmse=%.8f patch_abs=%.8f patch_max=%.8f patch_n=%" PRId64 " ensure=%.3fs encode=%.3fs apply=%.3fs total=%.3fs\n",
                policy.name.c_str(),
                eval_binding_indices.size(),
                direct_patch_count,
                direct_fallback_count,
                stageb_patch_threads,
                stageb_collect_patch_eval_metrics ? "yes" : "no",
                direct_patch_count == 0 ? "none" : (stageb_verify_direct_patch ? "checked" : "skipped_rsf"),
                patch_count > 0 ? std::sqrt(patch_sum_sq / (double) patch_count) : 0.0,
                patch_count > 0 ? patch_sum_abs / (double) patch_count : 0.0,
                patch_max_abs,
                patch_count,
                patch_ensure_s,
                patch_quant_s,
                patch_apply_s,
                patch_ensure_s + patch_quant_s + patch_apply_s);
            update_full_quant_eta("selector-stage-b-policy-kld", false);
            if (!policy_patch_ok) {
                restore_all();
                policy.proxy_rejected = true;
                policy.measured_pass = false;
                policy.measured_score = std::numeric_limits<double>::infinity();
                finish_selector_cuda_policy_scope();
                finish_stageb_policy_unit("selector-stage-b-policy-rejected", true);
                continue;
            }

            selector_kld_metrics km;
            const std::string policy_search_kld_label =
                "selector stage-b policy=" + policy.name + " search kld";
            if (!selector_eval_kld_subset(
                    lctx, kld_budget, params.n_batch, km, true,
                    policy_search_kld_label)) {
                restore_all();
                return false;
            }
            stageb_runtime_decoded = true;
            policy.measured = selector_derive_metrics(km);
            if (holdout_budget && has_holdout_eval) {
                selector_kld_metrics km_holdout;
                const std::string policy_validation_kld_label =
                    "selector stage-b policy=" + policy.name + " validation kld";
                if (!selector_eval_kld_subset(
                        lctx, *holdout_budget, params.n_batch, km_holdout, true,
                        policy_validation_kld_label)) {
                    restore_all();
                    return false;
                }
                stageb_runtime_decoded = true;
                policy.measured_holdout = selector_derive_metrics(km_holdout);
                policy.has_holdout = policy.measured_holdout.ok;
            }
            const selector_metric_rank policy_rank =
                selector_rank_policy_metrics(
                    policy.measured,
                    policy.has_holdout ? &policy.measured_holdout : nullptr,
                    baseline_eval,
                    has_holdout_eval ? &baseline_holdout_eval : nullptr,
                    rank_cfg);
            policy.measured_pass = policy_rank.pass;
            policy.measured_score = policy_rank.score;
            apply_rsf_holdout_guard(policy);
            const std::string policy_main_summary =
                selector_format_metrics("search", policy.measured);
            fprintf(stderr,
                "selector stage-b policy=%s measured_score=%.6f pass=%s %s\n",
                policy.name.c_str(),
                policy.measured_score,
                policy.measured_pass ? "yes" : "no",
                policy_main_summary.c_str());
            if (policy.has_holdout) {
                const std::string policy_holdout_summary =
                    selector_format_metrics("validation", policy.measured_holdout);
                fprintf(stderr,
                    "selector stage-b policy=%s %s\n",
                    policy.name.c_str(),
                    policy_holdout_summary.c_str());
            }
            stageb_result_cache.append(
                "policy_result",
                policy_cache_key,
                policy.name,
                policy.measured,
                policy.has_holdout ? &policy.measured_holdout : nullptr,
                policy.measured_pass,
                policy.measured_score);
            finish_selector_cuda_policy_scope();
            finish_stageb_policy_unit("selector-stage-b-policy-complete", true);
            if (note_skip_remaining("stage-b")) {
                break;
            }
        }
    } else if (!stageb_cache_only_eval) {
        for (auto & policy : policies) {
            policy.measured_pass = !policy.proxy_rejected;
            policy.measured_score = policy.proxy_score;
        }
    }

    std::optional<selector_policy> seed_keep_policy;
    if (have_measured_eval && baseline_eval.ok) {
        const selector_policy * baseline_policy = find_current_baseline_policy();
        selector_policy seed;
        seed.name = "seed_keep";
        seed.cfg = baseline_policy != nullptr ? baseline_policy->cfg : seed.cfg;
        seed.proxy_score = 0.0;
        seed.proxy_rmse = 0.0;
        seed.proxy_abs_mean = 0.0;
        seed.proxy_max_abs = 0.0;
        seed.measured = baseline_eval;
        seed.measured_holdout = baseline_holdout_eval;
        seed.has_holdout = has_holdout_eval;
        const selector_metric_rank seed_rank =
            selector_rank_policy_metrics(
                seed.measured,
                seed.has_holdout ? &seed.measured_holdout : nullptr,
                baseline_eval,
                seed.has_holdout ? &baseline_holdout_eval : nullptr,
                rank_cfg);
        seed.measured_pass = seed_rank.pass;
        seed.measured_score = seed_rank.score;
        seed_keep_policy = seed;
        const std::string seed_summary =
            selector_format_metrics("search", seed.measured);
        fprintf(stderr,
            "selector stage-b seed_keep measured_score=%.6f %s\n",
            seed.measured_score,
            seed_summary.c_str());
    }

    auto best_it = policies.begin();
    if (have_measured_eval) {
        for (size_t idx : eval_policy_indices) {
            if (idx < policies.size()) {
                apply_rsf_holdout_guard(policies[idx]);
            }
        }
        if (seed_keep_policy.has_value()) {
            policies.push_back(*seed_keep_policy);
        }

        std::vector<size_t> eval_candidates;
        eval_candidates.reserve(eval_policy_indices.size() + (seed_keep_policy.has_value() ? 1 : 0));
        for (size_t idx : eval_policy_indices) {
            eval_candidates.push_back(idx);
        }
        if (seed_keep_policy.has_value()) {
            eval_candidates.push_back(policies.size() - 1);
        }

        std::vector<size_t> gated_candidates;
        gated_candidates.reserve(eval_candidates.size());
        for (size_t idx : eval_candidates) {
            const auto & policy = policies[idx];
            if (!std::isfinite(policy.measured_score) || !policy.measured.ok) {
                continue;
            }
            if (policy.measured_pass) {
                gated_candidates.push_back(idx);
            }
        }

        const std::vector<size_t> & best_input = gated_candidates.empty() ? eval_candidates : gated_candidates;
        std::vector<size_t> best_set = selector_best_set(policies, best_input, rank_cfg);
        if (best_set.empty() && !gated_candidates.empty()) {
            best_set = selector_best_set(policies, eval_candidates, rank_cfg);
        }

        const selector_best_set_stats best_stats =
            selector_best_set_stats_for(policies, best_set, rank_cfg);
        bool have_best = false;
        size_t best_idx = 0;
        double best_utility = std::numeric_limits<double>::infinity();
        for (size_t idx : best_set) {
            const double utility = selector_best_set_utility(
                policies[idx],
                best_stats,
                rank_cfg);
            bool better = !have_best || utility < best_utility;
            if (!better && have_best && std::fabs(utility - best_utility) <= 1e-9) {
                better = selector_compare_policy_measured(
                        policies[idx],
                        policies[best_idx],
                        baseline_eval,
                        has_holdout_eval ? &baseline_holdout_eval : nullptr,
                        rank_cfg) < 0;
            }
            if (better) {
                best_idx = idx;
                best_utility = utility;
                have_best = true;
            }
        }
        if (have_best) {
            best_it = policies.begin() + (std::ptrdiff_t) best_idx;
            fprintf(stderr,
                "%s: selector best candidates=%zu gated=%zu non_dominated=%zu chosen=%s measured_score=%.6f utility=%.6f\n",
                __func__,
                eval_candidates.size(),
                gated_candidates.size(),
                best_set.size(),
                best_it->name.c_str(),
                best_it->measured_score,
                best_utility);
            for (size_t rank = 0; rank < best_set.size(); ++rank) {
                const auto & p = policies[best_set[rank]];
                const auto obj = selector_best_objectives_for(p, rank_cfg.holdout_weight);
                const double utility = selector_best_set_utility(p, best_stats, rank_cfg);
                fprintf(stderr,
                    "%s: selector Best[%zu] policy=%s pass=%s score=%.6f utility=%.6f ln=%.6f kld=%.6f p95=%.6f p99=%.6f p999=%.6f tail99=%.6f max=%.6f rms=%.6f top=%.4f flip_w=%.6f top_p=%.6f entropy=%.6f\n",
                    __func__,
                    rank + 1,
                    p.name.c_str(),
                    p.measured_pass ? "yes" : "no",
                    p.measured_score,
                    utility,
                    obj.ln_ratio,
                    obj.mean_kld,
                    obj.kld_p95,
                    obj.kld_p99,
                    obj.kld_p999,
                    obj.kld_tail_mean,
                    obj.max_kld,
                    obj.rms_dp,
                    obj.same_top,
                    obj.top_flip_weight,
                    obj.top_prob_rmse,
                    obj.entropy_rmse);
            }
        } else {
            best_it = policies.begin();
        }
    } else {
        best_it = std::min_element(policies.begin(), policies.end(), [](const auto & a, const auto & b) {
            if (a.proxy_score != b.proxy_score) return a.proxy_score < b.proxy_score;
            return a.name < b.name;
        });
    }

    const bool have_rsf_policies = std::any_of(policies.begin(), policies.end(), [](const selector_policy & policy) {
        return selector_policy_is_rsf_family(policy) && !policy.tensor_records.empty();
    });
    if (have_rsf_policies) {
        const std::string configured_rsf_report =
            selector_control_string("RSF_REPORT_FILE");
        const std::string rsf_report_file = configured_rsf_report.empty()
            ? checkpoint_model_path + ".rsf-report.txt"
            : configured_rsf_report;
        selector_write_rsf_report(rsf_report_file, policies);
    }

    const std::string sensitivity_report_file = selector_control_string("SENSITIVITY_REPORT_FILE");
    if (!skip_remaining_tuning &&
            have_runtime_eval &&
            !sensitivity_report_file.empty() &&
            best_it != policies.end()) {
        const std::string tensor_filter =
            selector_control_string("SENSITIVITY_TENSOR");
        const int32_t layer_filter =
            (int32_t) selector_control_i64("SENSITIVITY_LAYER", -999999);
        const int sensitivity_top = (int) std::max<int64_t>(
            0,
            selector_control_i64("SENSITIVITY_TOP",
                (!tensor_filter.empty() || layer_filter != -999999) ? 0 : 16));
        const int64_t sensitivity_sample_blocks = std::max<int64_t>(
            0,
            selector_control_i64("SENSITIVITY_SAMPLE_BLOCKS", 8192));

        std::ofstream sensitivity_report(sensitivity_report_file, std::ios::trunc);
        if (!sensitivity_report) {
            fprintf(stderr, "%s: failed to open selector sensitivity report %s\n", __func__, sensitivity_report_file.c_str());
        } else {
            fprintf(stderr,
                "%s: selector writing exact one-tensor sensitivity report %s top=%d layer_filter=%d tensor_filter=%s alt_search_sample_blocks=%" PRId64 " exact_patch_blocks=full\n",
                __func__,
                sensitivity_report_file.c_str(),
                sensitivity_top,
                layer_filter,
                tensor_filter.empty() ? "<none>" : tensor_filter.c_str(),
                sensitivity_sample_blocks);

            auto quantize_apply_sensitivity_patch = [&](
                    selector_binding & b,
                    const nvfp4_cuda_runtime_cfg & cfg,
                    double & sq,
                    double & ab,
                    double & mx,
                    int64_t & n) -> bool {
                if (!quantize_binding_ensure_target_bytes(b)) {
                    return false;
                }
                if (!nvfp4_selector_quantize_binding(b, cfg, nthread, b.working_target_bytes, sq, ab, mx, n, 0, 0, true) ||
                        n <= 0) {
                    return false;
                }

                float header_weight_scale = 1.0f;
                float header_input_scale = 1.0f;
                if (!nvfp4_selector_prepare_nvfp4_runtime_scales(
                        b, nvfp4_input_scale_policy, &header_weight_scale, &header_input_scale)) {
                    return false;
                }
                if (b.target_scale != nullptr &&
                        !quantize_tensor_copy_in(b.target_scale, b.working_scale_bytes.data(), b.target_scale_nbytes)) {
                    return false;
                }
                if (b.target_input_scale != nullptr &&
                        !quantize_tensor_copy_in(
                            b.target_input_scale,
                            b.working_input_scale_bytes.data(),
                            b.target_input_scale_nbytes)) {
                    return false;
                }
                return ggml_cuda_nvfp4_tensor_set_header_scales(
                    b.target, header_weight_scale, header_input_scale, nullptr);
            };

            bool best_patch_ok = true;
            restore_all();
            for (size_t i = 0; i < all_bindings.size(); ++i) {
                auto & b = all_bindings[i];
                double sq = 0.0;
                double ab = 0.0;
                double mx = 0.0;
                int64_t n = 0;
                if (!quantize_apply_sensitivity_patch(b, best_it->cfg, sq, ab, mx, n)) {
                    fprintf(stderr, "%s: selector sensitivity failed to patch best policy tensor=%s\n", __func__, b.name.c_str());
                    best_patch_ok = false;
                    break;
                }
            }

            selector_derived_metrics sensitivity_base = best_it->measured;
            if (best_patch_ok) {
                selector_kld_metrics base_km;
                if (!selector_eval_kld_subset(lctx, kld_budget, params.n_batch, base_km, true)) {
                    restore_all();
                    return false;
                }
                stageb_runtime_decoded = true;
                sensitivity_base = selector_derive_metrics(base_km);
            }

            sensitivity_report
                << "{\"schema\":\"blackwell-selector-sensitivity-v1\", \"event\":\"baseline\", \"policy\":\""
                << selector_json_escape(best_it->name)
                << "\", \"chunks\": " << kld_budget.n_chunk
                << ", \"ctx\": " << kld_budget.n_ctx
                << ", \"score_tokens\": " << kld_budget.n_score;
            selector_write_metric_json_fields(sensitivity_report, sensitivity_base, "");
            sensitivity_report << "}\n";

            std::vector<size_t> sensitivity_indices;
            sensitivity_indices.reserve(all_bindings.size());
            for (size_t i = 0; i < all_bindings.size(); ++i) {
                const auto & b = all_bindings[i];
                if (!tensor_filter.empty() && b.name.find(tensor_filter) == std::string::npos) {
                    continue;
                }
                if (layer_filter != -999999 && b.layer != layer_filter) {
                    continue;
                }
                sensitivity_indices.push_back(i);
            }
            std::sort(sensitivity_indices.begin(), sensitivity_indices.end(), [&](size_t ai, size_t bi) {
                const auto & a = all_bindings[ai];
                const auto & b = all_bindings[bi];
                const double as = selector_class_hotness(a.cls) * (double) std::max<size_t>(1, a.source_nbytes);
                const double bs = selector_class_hotness(b.cls) * (double) std::max<size_t>(1, b.source_nbytes);
                if (as != bs) return as > bs;
                if (a.layer != b.layer) return a.layer < b.layer;
                return a.name < b.name;
            });
            if (sensitivity_top > 0 && sensitivity_indices.size() > (size_t) sensitivity_top) {
                sensitivity_indices.resize((size_t) sensitivity_top);
            }

            int sensitivity_written = 0;
            if (best_patch_ok) {
                struct sensitivity_alt {
                    size_t index = 0;
                    nvfp4_cuda_runtime_cfg cfg = {};
                    std::string policy_name;
                    int64_t sample_blocks = 0;
                    double base_proxy_score = std::numeric_limits<double>::infinity();
                    double alt_proxy_score = std::numeric_limits<double>::infinity();
                };

                auto find_sensitivity_alt = [&](size_t idx, sensitivity_alt & out_alt) -> bool {
                    auto & b = all_bindings[idx];
                    out_alt.index = idx;
                    if (!selector_find_tensor_rescue_cfg(
                            b,
                            best_it->cfg,
                            nthread,
                            out_alt.cfg,
                            out_alt.policy_name,
                            out_alt.sample_blocks,
                            out_alt.base_proxy_score,
                            out_alt.alt_proxy_score) ||
                            selector_cfg_equal(out_alt.cfg, best_it->cfg)) {
                        return false;
                    }
                    return true;
                };

                if (layer_filter != -999999 && !sensitivity_indices.empty()) {
                    std::vector<sensitivity_alt> layer_alts;
                    layer_alts.reserve(sensitivity_indices.size());
                    for (const size_t idx : sensitivity_indices) {
                        sensitivity_alt alt;
                        if (find_sensitivity_alt(idx, alt)) {
                            layer_alts.push_back(std::move(alt));
                        }
                    }

                    if (!layer_alts.empty()) {
                        bool layer_patch_ok = true;
                        std::vector<std::string> tensor_names;
                        tensor_names.reserve(layer_alts.size());
                        double proxy_gain_sum = 0.0;
                        int64_t alt_sample_blocks_max = 0;

                        for (const auto & alt : layer_alts) {
                            auto & b = all_bindings[alt.index];
                            double sq = 0.0;
                            double ab = 0.0;
                            double mx = 0.0;
                            int64_t n = 0;
                            if (!quantize_apply_sensitivity_patch(b, best_it->cfg, sq, ab, mx, n)) {
                                fprintf(stderr, "%s: selector layer sensitivity failed tensor=%s alt=%s\n",
                                    __func__, b.name.c_str(), alt.policy_name.c_str());
                                layer_patch_ok = false;
                                break;
                            }
                            sq = 0.0;
                            ab = 0.0;
                            mx = 0.0;
                            n = 0;
                            if (!quantize_apply_sensitivity_patch(b, alt.cfg, sq, ab, mx, n)) {
                                fprintf(stderr, "%s: selector layer sensitivity failed tensor=%s alt=%s\n",
                                    __func__, b.name.c_str(), alt.policy_name.c_str());
                                layer_patch_ok = false;
                                break;
                            }
                            tensor_names.push_back(b.name + ":" + alt.policy_name);
                            if (std::isfinite(alt.base_proxy_score) && std::isfinite(alt.alt_proxy_score)) {
                                proxy_gain_sum += alt.base_proxy_score - alt.alt_proxy_score;
                            }
                            alt_sample_blocks_max = std::max<int64_t>(alt_sample_blocks_max, alt.sample_blocks);
                        }

                        if (layer_patch_ok) {
                            selector_kld_metrics km;
                            if (!selector_eval_kld_subset(lctx, kld_budget, params.n_batch, km, true)) {
                                restore_all();
                                return false;
                            }
                            stageb_runtime_decoded = true;
                            const selector_derived_metrics dm = selector_derive_metrics(km);
                            const double utility = dm.ok
                                ? selector_measured_score(dm, nullptr, rank_cfg)
                                : std::numeric_limits<double>::infinity();
                            const double base_utility = selector_measured_score(sensitivity_base, nullptr, rank_cfg);
                            const double delta_utility = utility - base_utility;

                            sensitivity_report
                                << "{\"schema\":\"blackwell-selector-sensitivity-v1\", \"event\":\"layer_delta\""
                                << ", \"layer\": " << layer_filter
                                << ", \"tensor_count\": " << layer_alts.size()
                                << ", \"base_policy\":\"" << selector_json_escape(best_it->name) << "\""
                                << ", \"alt_policy\":\"rescue_adaptive_layer\""
                                << ", \"alt_sample_blocks\": " << alt_sample_blocks_max
                                << ", \"measured_mode\":\"exact_layer\""
                                << ", \"tensors\":\"";
                            for (size_t ti = 0; ti < tensor_names.size(); ++ti) {
                                if (ti > 0) {
                                    sensitivity_report << ",";
                                }
                                sensitivity_report << selector_json_escape(tensor_names[ti]);
                            }
                            sensitivity_report
                                << "\"";
                            selector_write_json_number_field(sensitivity_report, "proxy_gain_sum", proxy_gain_sum);
                            selector_write_json_number_field(sensitivity_report, "utility", utility);
                            selector_write_json_number_field(sensitivity_report, "delta_utility", delta_utility);
                            selector_write_metric_json_fields(sensitivity_report, dm, "");
                            selector_write_delta_metric_json_fields(sensitivity_report, dm, sensitivity_base);
                            sensitivity_report << "}\n";

                            fprintf(stderr,
                                "%s: selector layer sensitivity layer=%d tensors=%zu delta_kld=%.6f delta_p99=%.6f delta_top_flip_w=%.6f delta_ln=%.6f\n",
                                __func__,
                                layer_filter,
                                layer_alts.size(),
                                dm.mean_kld - sensitivity_base.mean_kld,
                                dm.kld_p99 - sensitivity_base.kld_p99,
                                dm.top_flip_weight - sensitivity_base.top_flip_weight,
                                dm.ln_ratio - sensitivity_base.ln_ratio);
                        }

                        for (const auto & alt : layer_alts) {
                            auto & b = all_bindings[alt.index];
                            double sq = 0.0;
                            double ab = 0.0;
                            double mx = 0.0;
                            int64_t n = 0;
                            if (!quantize_apply_sensitivity_patch(b, best_it->cfg, sq, ab, mx, n)) {
                                restore_all();
                                return false;
                            }
                        }
                    }
                }

                for (size_t rank = 0; rank < sensitivity_indices.size(); ++rank) {
                    const size_t idx = sensitivity_indices[rank];
                    auto & b = all_bindings[idx];
                    sensitivity_alt alt;
                    if (!find_sensitivity_alt(idx, alt)) {
                        continue;
                    }

                    double sq = 0.0;
                    double ab = 0.0;
                    double mx = 0.0;
                    int64_t n = 0;
                    if (!quantize_apply_sensitivity_patch(b, best_it->cfg, sq, ab, mx, n)) {
                        fprintf(stderr, "%s: selector sensitivity failed tensor=%s alt=%s\n",
                            __func__, b.name.c_str(), alt.policy_name.c_str());
                        continue;
                    }
                    sq = 0.0;
                    ab = 0.0;
                    mx = 0.0;
                    n = 0;
                    if (!quantize_apply_sensitivity_patch(b, alt.cfg, sq, ab, mx, n)) {
                        fprintf(stderr, "%s: selector sensitivity failed tensor=%s alt=%s\n",
                            __func__, b.name.c_str(), alt.policy_name.c_str());
                        continue;
                    }

                    selector_kld_metrics km;
                    if (!selector_eval_kld_subset(lctx, kld_budget, params.n_batch, km, true)) {
                        restore_all();
                        return false;
                    }
                    stageb_runtime_decoded = true;
                    const selector_derived_metrics dm = selector_derive_metrics(km);
                    const double utility = dm.ok
                        ? selector_measured_score(dm, nullptr, rank_cfg)
                        : std::numeric_limits<double>::infinity();
                    const double base_utility = selector_measured_score(sensitivity_base, nullptr, rank_cfg);
                    const double delta_utility = utility - base_utility;

                    sensitivity_report
                        << "{\"schema\":\"blackwell-selector-sensitivity-v1\", \"event\":\"tensor_delta\", \"rank\": "
                        << (rank + 1)
                        << ", \"tensor\":\"" << selector_json_escape(b.name) << "\""
                        << ", \"class\":\"" << selector_tensor_class_name(b.cls) << "\""
                        << ", \"layer\": " << b.layer
                        << ", \"bucket\": " << b.bucket
                        << ", \"base_policy\":\"" << selector_json_escape(best_it->name) << "\""
                        << ", \"alt_policy\":\"" << selector_json_escape(alt.policy_name) << "\""
                        << ", \"alt_sample_blocks\": " << alt.sample_blocks
                        << ", \"measured_mode\":\"exact_single_tensor\"";
                    selector_write_json_number_field(sensitivity_report, "base_proxy_score", alt.base_proxy_score);
                    selector_write_json_number_field(sensitivity_report, "alt_proxy_score", alt.alt_proxy_score);
                    selector_write_json_number_field(sensitivity_report, "proxy_gain", alt.base_proxy_score - alt.alt_proxy_score);
                    selector_write_json_number_field(sensitivity_report, "utility", utility);
                    selector_write_json_number_field(sensitivity_report, "delta_utility", delta_utility);
                    selector_write_metric_json_fields(sensitivity_report, dm, "");
                    selector_write_delta_metric_json_fields(sensitivity_report, dm, sensitivity_base);
                    sensitivity_report << "}\n";
                    ++sensitivity_written;

                    fprintf(stderr,
                        "%s: selector sensitivity tensor=%s alt=%s delta_kld=%.6f delta_p99=%.6f delta_top_flip_w=%.6f delta_ln=%.6f\n",
                        __func__,
                        b.name.c_str(),
                        alt.policy_name.c_str(),
                        dm.mean_kld - sensitivity_base.mean_kld,
                        dm.kld_p99 - sensitivity_base.kld_p99,
                        dm.top_flip_weight - sensitivity_base.top_flip_weight,
                        dm.ln_ratio - sensitivity_base.ln_ratio);

                    sq = 0.0;
                    ab = 0.0;
                    mx = 0.0;
                    n = 0;
                    if (!quantize_apply_sensitivity_patch(b, best_it->cfg, sq, ab, mx, n)) {
                        restore_all();
                        return false;
                    }
                }
            }
            fprintf(stderr,
                "%s: wrote selector sensitivity report %s (%d tensor deltas)\n",
                __func__,
                sensitivity_report_file.c_str(),
                sensitivity_written);
            restore_all();
        }
    }

    std::unordered_map<std::string, const selector_policy *> tensor_policy_map;
    bool tensor_plan_materialization_state_ready = false;
    if (out_tensor_overrides &&
            best_it != policies.end() &&
            selector_control_i64("TENSOR_POLICY_MAP", 1) != 0) {
        const selector_policy * selected_global_policy = &*best_it;
        auto choose_tensor_plan_seed_policy = [&]() -> const selector_policy * {
            if (selected_global_policy != nullptr &&
                    !selected_global_policy->proxy_rejected &&
                    !selected_global_policy->tensor_records.empty()) {
                return selected_global_policy;
            }
            const selector_policy * baseline_policy = find_current_baseline_policy();
            if (baseline_policy != nullptr && !baseline_policy->proxy_rejected && !baseline_policy->tensor_records.empty()) {
                return baseline_policy;
            }
            if (recipe_cfg != nullptr) {
                auto recipe_it = std::find_if(policies.begin(), policies.end(), [](const selector_policy & policy) {
                    return selector_policy_is_recipe(policy.name) &&
                           !policy.proxy_rejected &&
                           !policy.tensor_records.empty();
                });
                if (recipe_it != policies.end()) {
                    return &*recipe_it;
                }
            }
            return best_it->tensor_records.empty() ? nullptr : &*best_it;
        };

        const selector_policy * tensor_plan_seed_policy = choose_tensor_plan_seed_policy();
        const auto base_records = tensor_plan_seed_policy != nullptr
            ? selector_rsf_record_map(*tensor_plan_seed_policy)
            : std::unordered_map<std::string, const selector_rsf_tensor_record *>();

        struct tensor_policy_choice {
            std::string tensor;
            const selector_policy * policy = nullptr;
            double base_score = std::numeric_limits<double>::infinity();
            double alt_score = std::numeric_limits<double>::infinity();
            double gain = 0.0;
        };
        std::unordered_map<std::string, tensor_policy_choice> best_by_tensor;

        if (tensor_plan_seed_policy != nullptr && !base_records.empty()) {
            double tensor_policy_abs_default = SELECTOR_TENSOR_POLICY_PROXY_DEFAULT_ABS_GAIN;
            const std::string work_mode = quantize_control_string("LLAMA_QUANTIZE_MODE");
            if (striequals(work_mode.c_str(), "fast")) {
                tensor_policy_abs_default = SELECTOR_TENSOR_POLICY_PROXY_FAST_ABS_GAIN;
            } else if (striequals(work_mode.c_str(), "deep")) {
                tensor_policy_abs_default = SELECTOR_TENSOR_POLICY_PROXY_DEEP_ABS_GAIN;
            }
            const double tensor_policy_abs_gain = std::max(0.0,
                quantize_control_f64("LLAMA_SELECTOR_TENSOR_POLICY_PROXY_ABS_GAIN",
                    tensor_policy_abs_default));
            const double tensor_policy_rel_gain = std::max(0.0,
                quantize_control_f64("LLAMA_SELECTOR_TENSOR_POLICY_PROXY_REL_GAIN",
                    SELECTOR_TENSOR_POLICY_PROXY_REL_GAIN));
            fprintf(stderr,
                "%s: selector tensor policy proxy gate abs=%.6g rel=%.6g\n",
                __func__,
                tensor_policy_abs_gain,
                tensor_policy_rel_gain);
            for (const auto & policy : policies) {
                if (&policy == tensor_plan_seed_policy ||
                        policy.name == "seed_keep" ||
                        policy.proxy_rejected ||
                        policy.tensor_records.empty() ||
                        selector_cfg_equal(policy.cfg, tensor_plan_seed_policy->cfg)) {
                    continue;
                }
                const auto alt_records = selector_rsf_record_map(policy);
                for (const auto & b : all_bindings) {
                    const auto bit = base_records.find(b.name);
                    const auto ait = alt_records.find(b.name);
                    if (bit == base_records.end() || ait == alt_records.end() ||
                            bit->second == nullptr || ait->second == nullptr ||
                            !std::isfinite(bit->second->proxy_score) ||
                            !std::isfinite(ait->second->proxy_score)) {
                        continue;
                    }
                    const double base_score = bit->second->proxy_score;
                    const double alt_score = ait->second->proxy_score;
                    const double gain = base_score - alt_score;
                    const double min_gain = std::max(
                        tensor_policy_abs_gain,
                        tensor_policy_rel_gain * std::max(std::fabs(base_score), std::fabs(alt_score)));
                    if (!(gain > min_gain)) {
                        continue;
                    }
                    auto cur = best_by_tensor.find(b.name);
                    if (cur == best_by_tensor.end() || gain > cur->second.gain) {
                        best_by_tensor[b.name] = { b.name, &policy, base_score, alt_score, gain };
                    }
                }
            }
        }

        std::vector<tensor_policy_choice> choices;
        choices.reserve(best_by_tensor.size());
        for (auto & kv : best_by_tensor) {
            choices.push_back(kv.second);
        }
        std::sort(choices.begin(), choices.end(), [](const tensor_policy_choice & a, const tensor_policy_choice & b) {
            if (a.gain != b.gain) return a.gain > b.gain;
            return a.tensor < b.tensor;
        });

        if (tensor_plan_seed_policy != nullptr && !base_records.empty()) {
            std::map<std::string, int> policy_counts;
            double total_gain = 0.0;
            for (const auto & choice : choices) {
                if (choice.policy == nullptr) {
                    continue;
                }
                tensor_policy_map[choice.tensor] = choice.policy;
                ++policy_counts[choice.policy->name];
                total_gain += choice.gain;
            }

            fprintf(stderr,
                "%s: selector tensor plan seed=%s tensor_switches=%zu/%zu proxy_gain_sum=%.6f exact_kld=%s\n",
                __func__,
                tensor_plan_seed_policy->name.c_str(),
                tensor_policy_map.size(),
                all_bindings.size(),
                total_gain,
                have_measured_eval ? "full-base" : "unavailable");
            for (const auto & kv : policy_counts) {
                fprintf(stderr,
                    "%s: selector tensor plan policy=%s tensors=%d\n",
                    __func__,
                    kv.first.c_str(),
                    kv.second);
            }

            auto tensor_plan_policy_for = [&](const selector_binding & b) -> const selector_policy * {
                const auto mit = tensor_policy_map.find(b.name);
                return mit != tensor_policy_map.end() && mit->second != nullptr
                    ? mit->second
                    : tensor_plan_seed_policy;
            };

            auto tensor_plan_digest_hex = [&]() {
                selector_stageb_digest digest;
                for (const auto & b : all_bindings) {
                    const selector_policy * p = tensor_plan_policy_for(b);
                    digest.add_string(b.name);
                    digest.add_string(p != nullptr ? p->name : "");
                    if (p != nullptr) {
                        digest.add_bytes(&p->cfg.choose46_mode, sizeof(p->cfg.choose46_mode));
                        digest.add_bytes(&p->cfg.refit_iters, sizeof(p->cfg.refit_iters));
                        digest.add_bytes(&p->cfg.use_compand_sat, sizeof(p->cfg.use_compand_sat));
                        digest.add_bytes(&p->cfg.reserved_i32, sizeof(p->cfg.reserved_i32));
                        digest.add_bytes(&p->cfg.cap_m6, sizeof(p->cfg.cap_m6));
                        digest.add_bytes(&p->cfg.cap_m4, sizeof(p->cfg.cap_m4));
                    }
                }
                return digest.hex();
            };

            auto make_tensor_plan_key = [&]() {
                nlohmann::ordered_json key = make_stageb_base_key();
                key["kind"] = "tensor_plan";
                key["plan"] = {
                    {"seed", tensor_plan_seed_policy->name},
	                    {"seed_cfg", selector_stageb_cfg_json(tensor_plan_seed_policy->cfg)},
	                    {"tensor_count", all_bindings.size()},
	                    {"switch_count", tensor_policy_map.size()},
	                    {"plan_hash", tensor_plan_digest_hex()},
	                    {"materialization", "exact_nvfp4_scales_v1"},
	                };
	                return key;
            };

	            bool tensor_plan_keep = true;
	            selector_derived_metrics tensor_plan_dm;
            double tensor_plan_score = std::numeric_limits<double>::infinity();
            auto tensor_plan_exact_guard = [&](
                    const selector_derived_metrics & plan_dm,
                    const selector_policy & global_policy,
                    std::string & reason) -> bool {
                if (!plan_dm.ok || !global_policy.measured.ok) {
                    reason = "missing exact tensor-plan or global policy metrics";
                    return false;
                }

                auto metric_margin = [](double a, double b, double abs_margin, double rel_margin) {
                    return std::max(std::max(0.0, abs_margin),
                        std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
                };
                auto lower_no_worse = [&](double plan_v, double global_v, double abs_margin, double rel_margin) {
                    if (!std::isfinite(plan_v) || !std::isfinite(global_v)) {
                        return false;
                    }
                    return plan_v <= global_v + metric_margin(plan_v, global_v, abs_margin, rel_margin);
                };
                auto higher_no_worse = [&](double plan_v, double global_v, double abs_margin, double rel_margin) {
                    if (!std::isfinite(plan_v) || !std::isfinite(global_v)) {
                        return false;
                    }
                    return plan_v + metric_margin(plan_v, global_v, abs_margin, rel_margin) >= global_v;
                };
                auto reject = [&](const char * metric, double plan_v, double global_v) {
                    char buf[192];
                    snprintf(buf, sizeof(buf), "%s plan=%.6f global=%.6f", metric, plan_v, global_v);
                    reason = buf;
                    return false;
                };

                const selector_derived_metrics & global_dm = global_policy.measured;
                const double plan_score_one = selector_measured_score_one(plan_dm, rank_cfg);
                const double global_score_one = selector_measured_score_one(global_dm, rank_cfg);
                if (selector_compare_measured_score(plan_score_one, global_score_one) > 0) {
                    return reject("score", plan_score_one, global_score_one);
                }
                if (!lower_no_worse(plan_dm.mean_kld, global_dm.mean_kld, rank_cfg.mean_kld_abs_floor, 0.0)) {
                    return reject("mean_kld", plan_dm.mean_kld, global_dm.mean_kld);
                }
                if (!lower_no_worse(plan_dm.kld_p99, global_dm.kld_p99, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin)) {
                    return reject("p99", plan_dm.kld_p99, global_dm.kld_p99);
                }
                if (!lower_no_worse(plan_dm.kld_p999, global_dm.kld_p999, rank_cfg.p999_abs_margin, rank_cfg.p999_rel_margin)) {
                    return reject("p999", plan_dm.kld_p999, global_dm.kld_p999);
                }
                if (!lower_no_worse(plan_dm.kld_tail_mean, global_dm.kld_tail_mean, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin)) {
                    return reject("tail99", plan_dm.kld_tail_mean, global_dm.kld_tail_mean);
                }
                if (!lower_no_worse(plan_dm.rms_dp, global_dm.rms_dp, rank_cfg.rms_dp_abs_floor, 0.0)) {
                    return reject("rms_dp", plan_dm.rms_dp, global_dm.rms_dp);
                }
                if (!lower_no_worse(plan_dm.top_flip_weight, global_dm.top_flip_weight, rank_cfg.top_flip_weight_abs_floor, 0.0)) {
                    return reject("top_flip_weight", plan_dm.top_flip_weight, global_dm.top_flip_weight);
                }
                if (!higher_no_worse(plan_dm.same_top, global_dm.same_top, rank_cfg.same_top_abs_floor, 0.0)) {
                    return reject("same_top", plan_dm.same_top, global_dm.same_top);
                }

                reason = "no worse than measured global policy";
                return true;
            };
            if (have_measured_eval) {
                const nlohmann::ordered_json tensor_plan_key = make_tensor_plan_key();
                selector_stageb_cache_entry cached_plan;
                if (stageb_result_cache.find(tensor_plan_key, "tensor_plan", cached_plan)) {
                    tensor_plan_dm = cached_plan.search;
                    const selector_metric_rank cached_rank =
                        selector_rank_policy_metrics(
                            tensor_plan_dm,
                            cached_plan.has_validation ? &cached_plan.validation : nullptr,
                            baseline_eval,
                            has_holdout_eval ? &baseline_holdout_eval : nullptr,
                            rank_cfg);
                    tensor_plan_keep = cached_rank.pass && tensor_plan_dm.ok;
                    tensor_plan_score = cached_rank.score;
                    fprintf(stderr,
                        "%s: selector tensor plan exact cache hit switches=%zu score=%.6f pass=%s\n",
                        __func__,
                        tensor_policy_map.size(),
                        tensor_plan_score,
                        tensor_plan_keep ? "yes" : "no");
                } else if (!have_runtime_eval) {
                    tensor_plan_keep = false;
                    fprintf(stderr,
                        "%s: selector tensor plan exact cache missing and runtime was skipped; falling back to measured global policy\n",
                        __func__);
                } else {
                    auto apply_tensor_plan_patch = [&](
                            selector_binding & b,
                            const selector_policy & policy) -> bool {
                        if (!quantize_binding_ensure_target_bytes(b)) {
                            return false;
                        }

                        double sum_sq = 0.0;
                        double sum_abs = 0.0;
                        double max_abs = 0.0;
                        int64_t count = 0;
                        bool direct_applied = false;
                        if (nvfp4_selector_quantize_binding(
                                b,
                                policy.cfg,
                                nthread,
                                b.working_target_bytes,
                                sum_sq,
                                sum_abs,
                                max_abs,
                                count,
                                0,
                                0,
                                true) && count > 0) {
                            direct_applied = true;
                        } else {
                            sum_sq = 0.0;
                            sum_abs = 0.0;
                            max_abs = 0.0;
                            count = 0;
                            if (!nvfp4_selector_quantize_binding(
                                    b,
                                    policy.cfg,
                                    nthread,
                                    b.working_target_bytes,
                                    sum_sq,
                                    sum_abs,
                                    max_abs,
                                    count,
                                    0,
                                    0,
                                    false) || count <= 0) {
                                return false;
                            }
                        }

                        float header_weight_scale = 1.0f;
                        float header_input_scale = 1.0f;
                        if (!nvfp4_selector_prepare_nvfp4_runtime_scales(
                                b, nvfp4_input_scale_policy, &header_weight_scale, &header_input_scale)) {
                            return false;
                        }
                        if (b.target_scale != nullptr &&
                                !quantize_tensor_copy_in(b.target_scale, b.working_scale_bytes.data(), b.target_scale_nbytes)) {
                            return false;
                        }
                        if (b.target_input_scale != nullptr &&
                                !quantize_tensor_copy_in(
                                    b.target_input_scale,
                                    b.working_input_scale_bytes.data(),
                                    b.target_input_scale_nbytes)) {
                            return false;
                        }
                        if (direct_applied) {
                            return ggml_cuda_nvfp4_tensor_set_header_scales(
                                b.target, header_weight_scale, header_input_scale, nullptr);
                        }
                        return quantize_tensor_copy_in(
                            b.target, b.working_target_bytes.data(), b.target_nbytes);
                    };

                    if (!stageb_runtime_decoded && kld_budget.n_chunk > 0) {
                        fprintf(stderr,
                            "%s: selector tensor plan patching runtime without partial KLD warmup; exact score uses the full KLD base\n",
                            __func__);
                    }

                    const auto patch_t0 = std::chrono::steady_clock::now();
                    bool plan_patch_ok = true;
                    for (auto & b : all_bindings) {
                        const selector_policy * materialize_policy = tensor_plan_policy_for(b);
                        if (materialize_policy == nullptr ||
                                !apply_tensor_plan_patch(b, *materialize_policy)) {
                            fprintf(stderr,
                                "%s: selector tensor plan failed to patch tensor=%s policy=%s\n",
                                __func__,
                                b.name.c_str(),
                                materialize_policy != nullptr ? materialize_policy->name.c_str() : "<none>");
                            plan_patch_ok = false;
                            break;
                        }
                    }
                    if (!plan_patch_ok) {
                        restore_all();
                        tensor_plan_keep = false;
                    } else {
                        const auto patch_t1 = std::chrono::steady_clock::now();
                        fprintf(stderr,
                            "%s: selector tensor plan patched tensors=%zu switches=%zu seconds=%.3f\n",
                            __func__,
                            all_bindings.size(),
                            tensor_policy_map.size(),
                            std::chrono::duration<double>(patch_t1 - patch_t0).count());
                        selector_kld_metrics plan_km;
                        if (!selector_eval_kld_subset(
                                lctx,
                                kld_budget,
                                params.n_batch,
                                plan_km,
                                true,
                                "selector tensor plan exact kld")) {
                            restore_all();
                            return false;
                        }
	                        stageb_runtime_decoded = true;
	                        tensor_plan_materialization_state_ready = true;
	                        tensor_plan_dm = selector_derive_metrics(plan_km);
                        const selector_metric_rank plan_rank =
                            selector_rank_policy_metrics(
                                tensor_plan_dm,
                                nullptr,
                                baseline_eval,
                                nullptr,
                                rank_cfg);
                        tensor_plan_keep = plan_rank.pass && tensor_plan_dm.ok;
                        tensor_plan_score = plan_rank.score;
                        stageb_result_cache.append(
                            "tensor_plan",
                            tensor_plan_key,
                            "tensor_plan",
                            tensor_plan_dm,
                            nullptr,
                            plan_rank.pass,
                            plan_rank.score);
                    }
                }
                fprintf(stderr,
                    "%s: selector tensor plan exact score=%.6f pass=%s switches=%zu %s\n",
                    __func__,
                    tensor_plan_score,
                    tensor_plan_keep ? "yes" : "no",
                    tensor_policy_map.size(),
                    selector_format_metrics("search", tensor_plan_dm).c_str());
                if (tensor_plan_keep &&
                        selected_global_policy != nullptr &&
                        selected_global_policy->measured.ok) {
                    std::string guard_reason;
                    if (!tensor_plan_exact_guard(tensor_plan_dm, *selected_global_policy, guard_reason)) {
                        tensor_plan_keep = false;
                        fprintf(stderr,
                            "%s: selector tensor plan rejected by measured-global guard global=%s reason=%s\n",
                            __func__,
                            selected_global_policy->name.c_str(),
                            guard_reason.c_str());
                    }
                }
            }

            if (tensor_plan_keep) {
                auto seed_it = std::find_if(policies.begin(), policies.end(), [&](const selector_policy & policy) {
                    return &policy == tensor_plan_seed_policy;
                });
                if (seed_it != policies.end()) {
                    best_it = seed_it;
                }
                fprintf(stderr,
                    "%s: selector tensor plan selected seed=%s switches=%zu/%zu\n",
                    __func__,
                    tensor_plan_seed_policy->name.c_str(),
                    tensor_policy_map.size(),
                    all_bindings.size());
	            } else {
	                tensor_policy_map.clear();
	                tensor_plan_materialization_state_ready = false;
	                fprintf(stderr,
                    "%s: selector tensor plan rejected by exact KLD guard; falling back to measured global policy=%s\n",
                    __func__,
                    best_it->name.c_str());
            }
        }
    }

    out_cfg = best_it->cfg;
    out_name = best_it->name.empty() ? "unnamed" : best_it->name;
    if (out_kept_seed != nullptr) {
        *out_kept_seed = out_name == "seed_keep";
    }

    std::vector<ggml_type> candidate_types = selector_type_list_from_control("CANDIDATE_TYPES");

    const std::string candidate_top_raw = selector_control_string("CANDIDATE_TOP");
    const std::string rescue_top_raw = selector_control_string("RESCUE_TOP");
    const int rescue_apply_top = (int) std::max<int64_t>(0, selector_control_i64("RESCUE_TOP", 0));
    int candidate_apply_top = 0;
    if (!skip_remaining_tuning && out_tensor_overrides && !candidate_types.empty()) {
        if (!candidate_top_raw.empty() || !rescue_top_raw.empty()) {
            candidate_apply_top = (int) std::max<int64_t>(
                0,
                selector_control_i64("CANDIDATE_TOP", rescue_apply_top));
        } else {
            const double candidate_fraction = std::clamp(
                selector_control_f64("CANDIDATE_FRACTION", 0.10),
                0.0,
                1.0);
            candidate_apply_top = (int) std::ceil(candidate_fraction * (double) all_bindings.size());
        }
    }
    const int candidate_report_top_default = candidate_apply_top > 0 ? 12 : 0;
    const int candidate_report_top = (int) std::max<int64_t>(
        0,
        selector_control_i64("CANDIDATE_REPORT_TOP",
            selector_control_i64("RESCUE_REPORT_TOP", candidate_report_top_default)));
    const size_t candidate_budget_bytes = (size_t) std::max<int64_t>(
        0,
        selector_control_i64("CANDIDATE_BUDGET_MB",
            selector_control_i64("RESCUE_BUDGET_MB", 0))) * 1024ull * 1024ull;
    const size_t rescue_bf16_budget_bytes = (size_t) std::max<int64_t>(0, selector_control_i64("RESCUE_BF16_BUDGET_MB", 0)) * 1024ull * 1024ull;
    const int candidate_class_limit = (int) std::max<int64_t>(
        0,
        selector_control_i64("CANDIDATE_CLASS_LIMIT",
            selector_control_i64("RESCUE_CLASS_LIMIT", 0)));
    const int candidate_encoder_refine_top = (int) std::max<int64_t>(
        0,
        selector_control_i64("CANDIDATE_NVFP4_TOP",
            selector_control_i64("RESCUE_NVFP4_TOP", 0)));
    const int64_t candidate_sens_sample_blocks = std::max<int64_t>(
        0,
        selector_control_i64("CANDIDATE_SAMPLE_BLOCKS",
            selector_control_i64("RESCUE_SENS_SAMPLE_BLOCKS", 8192)));
    const double rescue_nvfp4_min_gain = SELECTOR_RESCUE_NVFP4_MIN_GAIN;
    const double rescue_nvfp4_min_rel_gain = SELECTOR_RESCUE_NVFP4_MIN_REL_GAIN;
    const double rescue_nvfp4_prefer_gain = SELECTOR_RESCUE_NVFP4_PREFER_GAIN;
    const double rescue_nvfp4_prefer_rel_gain = SELECTOR_RESCUE_NVFP4_PREFER_REL_GAIN;
    const double rescue_speed_weight = SELECTOR_RESCUE_SPEED_WEIGHT;
    const std::string candidate_report_file = !selector_control_string("CANDIDATE_REPORT_FILE").empty()
        ? selector_control_string("CANDIDATE_REPORT_FILE")
        : selector_control_string("RESCUE_REPORT_FILE");
	    const std::string candidate_tensor_types_file = !selector_control_string("CANDIDATE_TENSOR_TYPES_FILE").empty()
	        ? selector_control_string("CANDIDATE_TENSOR_TYPES_FILE")
	        : selector_control_string("RESCUE_TENSOR_TYPES_FILE");
	    const bool selector_trace = quantize_control_i64("LLAMA_NVFP4_TRACE", 0) != 0;
	    auto tensor_plan_policy_for_binding = [&](const selector_binding & b) -> const selector_policy * {
	        const auto mit = tensor_policy_map.find(b.name);
	        if (mit != tensor_policy_map.end() && mit->second != nullptr) {
	            return mit->second;
	        }
	        return &*best_it;
	    };
		    bool skipped_materialization_replay = false;
		    auto replay_materialization_scales = [&]() -> bool {
		        if (!have_measured_eval || all_bindings.empty()) {
		            return true;
		        }
		        if (tensor_plan_materialization_state_ready) {
		            return true;
		        }
		        if (quantize_control_i64("LLAMA_SELECTOR_REPLAY_EXACT_SCALES", 0) == 0) {
		            skipped_materialization_replay = true;
		            return false;
		        }
		        const auto replay_t0 = std::chrono::steady_clock::now();
		        size_t replayed = 0;
	        std::vector<uint8_t> ignored_bytes;
	        for (auto & b : all_bindings) {
	            const selector_policy * policy = tensor_plan_policy_for_binding(b);
	            if (policy == nullptr || !quantize_binding_ensure_target_bytes(b)) {
	                return false;
	            }
	            double sum_sq = 0.0;
	            double sum_abs = 0.0;
	            double max_abs = 0.0;
	            int64_t count = 0;
	            if (!nvfp4_selector_quantize_binding(
	                    b,
	                    policy->cfg,
	                    nthread,
	                    ignored_bytes,
	                    sum_sq,
	                    sum_abs,
	                    max_abs,
	                    count,
	                    0,
	                    0,
	                    true,
	                    nullptr,
	                    false)) {
	                return false;
	            }
	            float header_weight_scale = 1.0f;
	            float header_input_scale = 1.0f;
	            if (!nvfp4_selector_prepare_nvfp4_runtime_scales(
	                    b, nvfp4_input_scale_policy, &header_weight_scale, &header_input_scale)) {
	                return false;
	            }
	            if (b.target_scale != nullptr &&
	                    !quantize_tensor_copy_in(b.target_scale, b.working_scale_bytes.data(), b.target_scale_nbytes)) {
	                return false;
	            }
	            if (b.target_input_scale != nullptr &&
	                    !quantize_tensor_copy_in(
	                        b.target_input_scale,
	                        b.working_input_scale_bytes.data(),
	                        b.target_input_scale_nbytes)) {
	                return false;
	            }
	            if (!ggml_cuda_nvfp4_tensor_set_header_scales(
	                    b.target, header_weight_scale, header_input_scale, nullptr)) {
	                return false;
	            }
	            ++replayed;
	        }
	        const auto replay_t1 = std::chrono::steady_clock::now();
	        fprintf(stderr,
	            "%s: selector materialization replayed exact NVFP4 scales tensors=%zu seconds=%.3f\n",
	            __func__,
	            replayed,
	            std::chrono::duration<double>(replay_t1 - replay_t0).count());
	        return true;
	    };
		    const bool exact_materialization_scales =
		        out_tensor_overrides != nullptr && replay_materialization_scales();
		    if (out_tensor_overrides != nullptr && !exact_materialization_scales) {
		        if (skipped_materialization_replay) {
		            fprintf(stderr,
		                "%s: selector materialization skipped cached exact-scale replay; final writer will recompute tensor scales once\n",
		                __func__);
		        } else {
		            fprintf(stderr,
		                "%s: selector materialization could not replay exact scales; final writer will recompute tensor scales once\n",
		                __func__);
		        }
		    }
	    if (out_tensor_overrides) {
	        out_tensor_overrides->clear();
	        if (out_name == "seed_keep" && !tensor_policy_map.empty()) {
            out_tensor_overrides->reserve(tensor_policy_map.size());
            for (const auto & b : all_bindings) {
                const auto mit = tensor_policy_map.find(b.name);
                if (mit == tensor_policy_map.end() || mit->second == nullptr) {
                    continue;
                }
                tensor_type_option opt;
                opt.name = selector_regex_escape(b.name);
                opt.type = GGML_TYPE_NVFP4;
	                opt.has_nvfp4_cfg = true;
	                opt.nvfp4_cfg = mit->second->cfg;
	                opt.nvfp4_policy_name = mit->second->name;
	                if (exact_materialization_scales) {
	                    selector_attach_exact_scales(opt, b);
	                }
	                out_tensor_overrides->push_back(std::move(opt));
	            }
            fprintf(stderr,
                "%s: selector materialization added %zu exact NVFP4 tensor plan entries for seed_keep tensor_policy_switches=%zu\n",
                __func__,
                out_tensor_overrides->size(),
                tensor_policy_map.size());
        } else if (out_name != "seed_keep" && !all_bindings.empty()) {
            out_tensor_overrides->reserve(out_tensor_overrides->size() + all_bindings.size());
            for (const auto & b : all_bindings) {
                const selector_policy * materialize_policy = &*best_it;
                const auto mit = tensor_policy_map.find(b.name);
                if (mit != tensor_policy_map.end() && mit->second != nullptr) {
                    materialize_policy = mit->second;
                }
                tensor_type_option opt;
                opt.name = selector_regex_escape(b.name);
                opt.type = GGML_TYPE_NVFP4;
	                opt.has_nvfp4_cfg = true;
	                opt.nvfp4_cfg = materialize_policy->cfg;
	                opt.nvfp4_policy_name = materialize_policy->name;
	                if (exact_materialization_scales) {
	                    selector_attach_exact_scales(opt, b);
	                }
	                out_tensor_overrides->push_back(std::move(opt));
	            }
            fprintf(stderr,
                "%s: selector materialization added %zu exact NVFP4 tensor plan entries for policy=%s%s tensor_policy_switches=%zu\n",
                __func__,
                all_bindings.size(),
                best_it->name.c_str(),
                selector_policy_is_rsf_family(*best_it) ? " (RSF)" : "",
                tensor_policy_map.size());
        }
    }
	    if (candidate_report_top > 0 || candidate_apply_top > 0) {
        auto format_candidate_types = [&]() {
            std::ostringstream out;
            for (size_t i = 0; i < candidate_types.size(); ++i) {
                if (i > 0) {
                    out << ',';
                }
                out << ggml_type_name(candidate_types[i]);
            }
            return out.str();
        };
        fprintf(stderr,
            "%s: selector tensor-type candidates scan=%zu apply_top=%d report_top=%d encoder_refine_top=%d types=%s\n",
            __func__,
            all_bindings.size(),
            candidate_apply_top,
            candidate_report_top,
            candidate_encoder_refine_top,
            format_candidate_types().c_str());
        std::vector<selector_tensor_sensitivity> sens;
        sens.reserve(all_bindings.size());
        for (size_t bind_i = 0; bind_i < all_bindings.size(); ++bind_i) {
            auto & b = all_bindings[bind_i];
            if (selector_trace) {
                fprintf(stderr,
                    "selector sensitivity [%zu/%zu] tensor=%s cls=%s bucket=%d layer=%d\n",
                    bind_i + 1,
                    all_bindings.size(),
                    b.name.c_str(),
                    selector_tensor_class_name(b.cls),
                    b.bucket,
                    b.layer);
            }
            double tensor_sq = 0.0;
            double tensor_abs = 0.0;
            double tensor_max = 0.0;
            int64_t tensor_n = 0;
            std::vector<uint8_t> sens_bytes;
            const selector_policy * tensor_base_policy = tensor_plan_policy_for_binding(b);
            if (!nvfp4_selector_quantize_binding(
                    b, tensor_base_policy->cfg, nthread, sens_bytes,
                    tensor_sq, tensor_abs, tensor_max, tensor_n,
                    candidate_sens_sample_blocks) || tensor_n <= 0) {
                continue;
            }
            selector_tensor_sensitivity s;
            s.name = b.name;
            s.role = selector_tensor_role_name(b.name, b.cls);
            s.cls = b.cls;
            s.bucket = b.bucket;
            s.layer = b.layer;
            s.binding_index = &b - all_bindings.data();
            s.target_nbytes = b.target_nbytes;
            s.best_candidate_type = candidate_types.empty() ? GGML_TYPE_Q8_0 : candidate_types.front();
            s.best_candidate_nbytes = quantize_tensor_nbytes_as_type(b.source, s.best_candidate_type);
            s.bf16_nbytes = quantize_tensor_nbytes_as_type(b.source, GGML_TYPE_BF16);
            const selector_proxy_metrics proxy_metrics =
                selector_proxy_score(tensor_sq, tensor_abs, tensor_max, tensor_n);
            if (!proxy_metrics.ok) {
                continue;
            }
            s.proxy_rmse = proxy_metrics.rmse;
            s.proxy_abs_mean = proxy_metrics.abs_mean;
            s.proxy_max_abs = proxy_metrics.max_abs;
            s.proxy_score = proxy_metrics.score;
            if (selector_trace) {
                fprintf(stderr,
                    "selector sensitivity result [%zu/%zu] tensor=%s baseline_error=%.6f rmse=%.6f abs=%.6f max=%.6f samples=%" PRId64 "\n",
                    bind_i + 1,
                    all_bindings.size(),
                    b.name.c_str(),
                    s.proxy_score,
                    s.proxy_rmse,
                    s.proxy_abs_mean,
                    s.proxy_max_abs,
                    tensor_n);
            }
            s.best_candidate_delta_bytes = s.best_candidate_nbytes > s.target_nbytes ? s.best_candidate_nbytes - s.target_nbytes : 0;
            s.bf16_delta_bytes = s.bf16_nbytes > s.target_nbytes ? s.bf16_nbytes - s.target_nbytes : 0;
            const double hotness = selector_class_hotness(b.cls);
            for (const ggml_type candidate_type : candidate_types) {
                if (!selector_type_candidate_allowed(candidate_type, s.cls)) {
                    continue;
                }
                selector_type_candidate cand;
                cand.type = candidate_type;
                cand.nbytes = quantize_tensor_nbytes_as_type(b.source, candidate_type);
                if (cand.nbytes == 0) {
                    continue;
                }
                cand.delta_bytes = cand.nbytes > s.target_nbytes ? cand.nbytes - s.target_nbytes : 0;
                cand.speed_penalty = hotness * selector_type_speed_penalty(candidate_type);
                cand.type_gain = selector_tensor_role_type_gain(b.name, b.cls, candidate_type);
                cand.roi = selector_type_candidate_roi(
                    s.proxy_score,
                    cand.delta_bytes,
                    cand.speed_penalty,
                    cand.type_gain,
                    rescue_speed_weight);
                s.type_candidates.push_back(cand);
            }
            std::sort(s.type_candidates.begin(), s.type_candidates.end(), [](const auto & a, const auto & b) {
                if (a.roi != b.roi) return a.roi > b.roi;
                if (a.speed_penalty != b.speed_penalty) return a.speed_penalty < b.speed_penalty;
                if (a.delta_bytes != b.delta_bytes) return a.delta_bytes < b.delta_bytes;
                return (int) a.type < (int) b.type;
            });
            if (!s.type_candidates.empty()) {
                const auto & best_candidate = s.type_candidates.front();
                s.best_candidate_type = best_candidate.type;
                s.best_candidate_nbytes = best_candidate.nbytes;
                s.best_candidate_delta_bytes = best_candidate.delta_bytes;
                s.best_candidate_speed_penalty = best_candidate.speed_penalty;
                s.best_candidate_roi = best_candidate.roi;
            }
            s.bf16_speed_penalty = hotness * selector_type_speed_penalty(GGML_TYPE_BF16);
            s.bf16_roi = selector_type_candidate_roi(
                s.proxy_score,
                s.bf16_delta_bytes,
                s.bf16_speed_penalty,
                selector_type_quality_gain(GGML_TYPE_BF16),
                rescue_speed_weight);
            sens.push_back(std::move(s));
        }

        std::sort(sens.begin(), sens.end(), [](const auto & a, const auto & b) {
            if (a.proxy_score != b.proxy_score) return a.proxy_score > b.proxy_score;
            if (a.proxy_max_abs != b.proxy_max_abs) return a.proxy_max_abs > b.proxy_max_abs;
            if (a.target_nbytes != b.target_nbytes) return a.target_nbytes > b.target_nbytes;
            return a.name < b.name;
        });

        const int candidate_scan_n = std::min<int>(candidate_encoder_refine_top, (int) sens.size());
        const int sensitivity_preview_n = std::min<int>(
            candidate_report_top > 0 ? candidate_report_top : candidate_scan_n,
            (int) sens.size());
        for (int i = 0; i < sensitivity_preview_n; ++i) {
            const auto & s = sens[(size_t) i];
            fprintf(stderr,
                "selector sensitivity rank=%d tensor=%s role=%s cls=%s bucket=%d layer=%d baseline_error=%.6f rmse=%.6f abs=%.6f max=%.6f\n",
                i + 1,
                s.name.c_str(),
                s.role.c_str(),
                selector_tensor_class_name(s.cls),
                s.bucket,
                s.layer,
                s.proxy_score,
                s.proxy_rmse,
                s.proxy_abs_mean,
                s.proxy_max_abs);
        }

        const auto candidate_scan_start = std::chrono::steady_clock::now();
        auto update_candidate_scan_eta = [&](int completed, bool print_now = false) {
            if (candidate_scan_n <= 0) {
                return;
            }
            const int clamped_done = std::min(std::max(completed, 0), candidate_scan_n);
            std::string eta_hint = "warming-up";
            if (clamped_done >= candidate_scan_n) {
                eta_hint = "done";
            } else if (clamped_done > 0) {
                const double elapsed_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - candidate_scan_start).count();
                const double remaining_s =
                    elapsed_s * (double) (candidate_scan_n - clamped_done) / (double) clamped_done;
                eta_hint = "~" + selector_format_duration(remaining_s) + " tensor_candidate_scan";
            }
            char detail[512];
            snprintf(detail, sizeof(detail),
                "phase=selector-tensor-type-candidate-scan scope=tensor_candidate_refine basis=proxy_error_and_candidate_type_score candidate_tensors=%d/%d report_top=%d apply_top=%d types=%s",
                clamped_done,
                candidate_scan_n,
                candidate_report_top,
                candidate_apply_top,
                format_candidate_types().c_str());
            full_quant_eta.eta_hint(eta_hint);
            full_quant_eta.update(clamped_done, candidate_scan_n, detail, print_now);
        };
        update_candidate_scan_eta(0, true);
        for (int i = 0; i < candidate_scan_n; ++i) {
            auto & s = sens[(size_t) i];
            auto & b = all_bindings[s.binding_index];
            const selector_policy * tensor_base_policy = tensor_plan_policy_for_binding(b);
            if (selector_trace) {
                fprintf(stderr,
                    "selector candidate scan [%d/%d] tensor=%s role=%s cls=%s bucket=%d layer=%d baseline_error=%.6f\n",
                    i + 1,
                    candidate_scan_n,
                    s.name.c_str(),
                    s.role.c_str(),
                    selector_tensor_class_name(s.cls),
                    s.bucket,
                    s.layer,
                    s.proxy_score);
            }
            nvfp4_cuda_runtime_cfg alt_cfg = tensor_base_policy->cfg;
            std::string alt_policy_name;
            int64_t alt_sample_blocks = 0;
            double base_score = std::numeric_limits<double>::infinity();
            double best_score = std::numeric_limits<double>::infinity();
            const int64_t rescue_nb_total =
                (b.source->ne[0] * b.source->ne[1]) / SELECTOR_WEIGHT_SAMPLE_BLOCK_SIZE;
            const int64_t rescue_sample_blocks =
                selector_rescue_sample_blocks(rescue_nb_total, b.cls);
            const int64_t rescue_refine_blocks =
                selector_rescue_refine_blocks(rescue_nb_total, b.cls, rescue_sample_blocks);
            const int64_t rescue_guard_blocks =
                selector_rescue_guard_blocks(rescue_nb_total, b.cls, rescue_refine_blocks);
            const nlohmann::ordered_json tensor_rescue_key = make_tensor_rescue_key(
                b,
                tensor_base_policy->cfg,
                rescue_sample_blocks,
                rescue_refine_blocks,
                rescue_guard_blocks);
            selector_tensor_rescue_cache_entry cached_rescue;
            bool rescue_ok = false;
            if (tensor_rescue_cache.find(tensor_rescue_key, s.name, cached_rescue)) {
                alt_cfg = cached_rescue.cfg;
                alt_policy_name = cached_rescue.policy;
                alt_sample_blocks = cached_rescue.sample_blocks;
                base_score = cached_rescue.base_score;
                best_score = cached_rescue.best_score;
                rescue_ok = true;
            } else {
                rescue_ok = selector_find_tensor_rescue_cfg(
                    b,
                    tensor_base_policy->cfg,
                    nthread,
                    alt_cfg,
                    alt_policy_name,
                    alt_sample_blocks,
                    base_score,
                    best_score);
                if (rescue_ok) {
                    selector_tensor_rescue_cache_entry rescue_entry;
                    rescue_entry.policy = alt_policy_name;
                    rescue_entry.cfg = alt_cfg;
                    rescue_entry.sample_blocks = alt_sample_blocks;
                    rescue_entry.base_score = base_score;
                    rescue_entry.best_score = best_score;
                    tensor_rescue_cache.append(tensor_rescue_key, s.name, rescue_entry);
                }
            }
            if (!rescue_ok) {
                continue;
            }

            s.alt_nvfp4_encoder_proxy_score = best_score;
            s.alt_nvfp4_encoder_gain = std::max(0.0, base_score - best_score);
            s.alt_nvfp4_encoder_gain_rel = s.alt_nvfp4_encoder_gain / std::max(1e-9, base_score);
            if ((s.alt_nvfp4_encoder_gain >= rescue_nvfp4_min_gain || s.alt_nvfp4_encoder_gain_rel >= rescue_nvfp4_min_rel_gain) &&
                !selector_cfg_equal(alt_cfg, tensor_base_policy->cfg)) {
                s.has_alt_nvfp4_encoder_cfg = true;
                s.alt_nvfp4_encoder_cfg = alt_cfg;
                s.alt_nvfp4_encoder_sample_blocks = alt_sample_blocks;
                s.alt_nvfp4_encoder_policy_name = alt_policy_name;
                if (selector_trace) {
                    fprintf(stderr,
                        "selector candidate alt tensor=%s policy=%s gain=%.6f rel=%.4f sample_blocks=%" PRId64 "\n",
                        s.name.c_str(),
                        s.alt_nvfp4_encoder_policy_name.c_str(),
                    s.alt_nvfp4_encoder_gain,
                    s.alt_nvfp4_encoder_gain_rel,
                    s.alt_nvfp4_encoder_sample_blocks);
                }
            }
            update_candidate_scan_eta(i + 1, i + 1 == candidate_scan_n);
        }

        for (auto & s : sens) {
            if (!s.type_candidates.empty()) {
                const auto & best_candidate = s.type_candidates.front();
                s.suggested_type = best_candidate.type;
                s.suggested_priority = best_candidate.roi;
                s.suggested_nbytes = best_candidate.nbytes;
                s.suggested_delta_bytes = best_candidate.delta_bytes;
                s.suggested_speed_penalty = best_candidate.speed_penalty;
                s.suggested_type_gain = best_candidate.type_gain;
            }
            if (s.has_alt_nvfp4_encoder_cfg) {
                const bool expert_nvfp4_first = selector_expert_class(s.cls);
                const double nvfp4_priority =
                    s.alt_nvfp4_encoder_gain +
                    (expert_nvfp4_first ? SELECTOR_TYPE_CANDIDATE_EXPERT_NVFP4_BONUS : SELECTOR_TYPE_CANDIDATE_NVFP4_BONUS) * s.proxy_score;
                const bool encoder_refine_is_clear_win =
                    (s.alt_nvfp4_encoder_gain >= rescue_nvfp4_prefer_gain ||
                     s.alt_nvfp4_encoder_gain_rel >= rescue_nvfp4_prefer_rel_gain) &&
                    nvfp4_priority > std::max(0.0, s.suggested_priority) * 1.25;
                if (s.suggested_priority <= 0.0 || encoder_refine_is_clear_win) {
                    s.suggested_type = GGML_TYPE_NVFP4;
                    s.suggested_priority = nvfp4_priority;
                    s.suggested_nbytes = s.target_nbytes;
                    s.suggested_delta_bytes = 0;
                    s.suggested_speed_penalty = 0.0;
                    s.suggested_type_gain = 1.0;
                }
            }
        }

	        std::sort(sens.begin(), sens.end(), [](const auto & a, const auto & b) {
	            if (a.suggested_priority != b.suggested_priority) return a.suggested_priority > b.suggested_priority;
	            if (a.proxy_score != b.proxy_score) return a.proxy_score > b.proxy_score;
	            if (a.suggested_speed_penalty != b.suggested_speed_penalty) return a.suggested_speed_penalty < b.suggested_speed_penalty;
	            if (a.suggested_delta_bytes != b.suggested_delta_bytes) return a.suggested_delta_bytes < b.suggested_delta_bytes;
	            return a.name < b.name;
	        });

	        double candidate_outlier_ratio_default = SELECTOR_TYPE_CANDIDATE_NORMAL_OUTLIER_RATIO;
	        double candidate_min_error_default = SELECTOR_TYPE_CANDIDATE_NORMAL_MIN_ERROR;
	        const std::string candidate_work_mode = quantize_control_string("LLAMA_QUANTIZE_MODE");
	        if (striequals(candidate_work_mode.c_str(), "fast")) {
	            candidate_outlier_ratio_default = SELECTOR_TYPE_CANDIDATE_FAST_OUTLIER_RATIO;
	            candidate_min_error_default = SELECTOR_TYPE_CANDIDATE_FAST_MIN_ERROR;
	        } else if (striequals(candidate_work_mode.c_str(), "deep")) {
	            candidate_outlier_ratio_default = SELECTOR_TYPE_CANDIDATE_DEEP_OUTLIER_RATIO;
	            candidate_min_error_default = SELECTOR_TYPE_CANDIDATE_DEEP_MIN_ERROR;
	        }
	        const double candidate_outlier_ratio = std::max(1.0,
	            quantize_control_f64("LLAMA_SELECTOR_TYPE_CANDIDATE_OUTLIER_RATIO",
	                candidate_outlier_ratio_default));
	        const double candidate_min_error = std::max(0.0,
	            quantize_control_f64("LLAMA_SELECTOR_TYPE_CANDIDATE_MIN_ERROR",
	                candidate_min_error_default));
	        auto proxy_quantile = [](std::vector<double> values, double q) {
	            values.erase(std::remove_if(values.begin(), values.end(), [](double v) {
	                return !std::isfinite(v) || v <= 0.0;
	            }), values.end());
	            if (values.empty()) {
	                return 0.0;
	            }
	            q = std::clamp(q, 0.0, 1.0);
	            const size_t idx = std::min(values.size() - 1,
	                (size_t) std::floor(q * (double) (values.size() - 1)));
	            std::nth_element(values.begin(), values.begin() + idx, values.end());
	            return values[idx];
	        };
	        std::vector<double> proxy_scores;
	        proxy_scores.reserve(sens.size());
	        for (const auto & s : sens) {
	            proxy_scores.push_back(s.proxy_score);
	        }
	        const double candidate_proxy_p50 = proxy_quantile(proxy_scores, 0.50);
	        const double candidate_proxy_p75 = proxy_quantile(proxy_scores, 0.75);
        const double candidate_proxy_floor = std::max(candidate_min_error, candidate_proxy_p50 * candidate_outlier_ratio);
        for (auto & s : sens) {
            const double role_gate = candidate_proxy_floor * selector_tensor_role_gate_mul(s.name, s.cls);
            s.suggested_gate_floor = std::max(candidate_min_error, role_gate);
            s.suggested_error_ratio =
                s.suggested_gate_floor > 0.0 ? s.proxy_score / s.suggested_gate_floor : 0.0;
            if (selector_type_is_k_rsf_capable(s.suggested_type)) {
                s.has_suggested_k_rsf_mode = true;
                s.suggested_k_rsf_mode =
                    s.suggested_error_ratio >= 1.25 ||
                    selector_tensor_role_importance(s.name, s.cls) >= 1.08 ? 2 : 1;
            }
        }
        fprintf(stderr,
            "%s: selector tensor-type outlier gate floor=%.6f p50=%.6f p75=%.6f ratio=%.3f min_error=%.6f apply_cap=%d\n",
            __func__,
	            candidate_proxy_floor,
	            candidate_proxy_p50,
	            candidate_proxy_p75,
	            candidate_outlier_ratio,
	            candidate_min_error,
	            candidate_apply_top);

	        auto format_candidate_list = [](const selector_tensor_sensitivity & s) {
	            std::ostringstream out;
	            for (size_t i = 0; i < s.type_candidates.size(); ++i) {
	                const auto & cand = s.type_candidates[i];
                if (i > 0) {
                    out << ';';
                }
                out << ggml_type_name(cand.type)
                    << ":roi=" << cand.roi
                    << ":delta=" << cand.delta_bytes
                    << ":speed=" << cand.speed_penalty;
            }
            return out.str();
        };

        if (!candidate_report_file.empty()) {
            std::ofstream report(candidate_report_file, std::ios::trunc);
            if (!report) {
                fprintf(stderr, "%s: failed to open selector candidate report file %s\n", __func__, candidate_report_file.c_str());
            } else {
                report << "rank,name,role,class,bucket,layer,baseline_error,proxy_rmse,proxy_abs_mean,proxy_max_abs,target_nbytes,alt_nvfp4_encoder_policy,alt_nvfp4_encoder_score,alt_nvfp4_encoder_gain,alt_nvfp4_encoder_gain_rel,alt_nvfp4_encoder_sample_blocks,alt_nvfp4_encoder_choose46,alt_nvfp4_encoder_refit,alt_nvfp4_encoder_compand,alt_nvfp4_encoder_cap6,alt_nvfp4_encoder_cap4,candidate_scores,suggested_type,suggested_nbytes,suggested_delta_bytes,suggested_speed_penalty,suggested_type_gain,suggested_priority,gate_floor,error_ratio\n";
                for (size_t i = 0; i < sens.size(); ++i) {
                    const auto & s = sens[i];
                    report
                        << (i + 1) << ','
                        << '"' << s.name << '"' << ','
                        << s.role << ','
                        << selector_tensor_class_name(s.cls) << ','
                        << s.bucket << ','
                        << s.layer << ','
                        << s.proxy_score << ','
                        << s.proxy_rmse << ','
                        << s.proxy_abs_mean << ','
                        << s.proxy_max_abs << ','
                        << s.target_nbytes << ','
                        << '"' << s.alt_nvfp4_encoder_policy_name << '"' << ','
                        << s.alt_nvfp4_encoder_proxy_score << ','
                        << s.alt_nvfp4_encoder_gain << ','
                        << s.alt_nvfp4_encoder_gain_rel << ','
                        << s.alt_nvfp4_encoder_sample_blocks << ','
                        << nvfp4_choose46_mode_name(s.alt_nvfp4_encoder_cfg.choose46_mode) << ','
                        << s.alt_nvfp4_encoder_cfg.refit_iters << ','
                        << s.alt_nvfp4_encoder_cfg.use_compand_sat << ','
                        << s.alt_nvfp4_encoder_cfg.cap_m6 << ','
                        << s.alt_nvfp4_encoder_cfg.cap_m4 << ','
                        << '"' << format_candidate_list(s) << '"' << ','
                        << format_selector_suggested_type(s) << ','
                        << s.suggested_nbytes << ','
                        << s.suggested_delta_bytes << ','
                        << s.suggested_speed_penalty << ','
                        << s.suggested_type_gain << ','
                        << s.suggested_priority << ','
                        << s.suggested_gate_floor << ','
                        << s.suggested_error_ratio
                        << '\n';
                }
                fprintf(stderr, "%s: wrote selector candidate report %s (%zu tensors)\n", __func__, candidate_report_file.c_str(), sens.size());
            }
        }

        for (int i = 0; i < candidate_report_top && i < (int) sens.size(); ++i) {
            const auto & s = sens[(size_t) i];
            fprintf(stderr,
                "selector candidate rank=%d tensor=%s role=%s cls=%s bucket=%d layer=%d baseline_error=%.6f rmse=%.6f abs=%.6f max=%.6f alt_nvfp4_encoder_gain=%.6f alt_nvfp4_encoder_rel=%.4f gate=%.6f ratio=%.3f speed_penalty=%.6f delta_bytes=%zu suggest=%s priority=%.6f\n",
                i + 1, s.name.c_str(), s.role.c_str(), selector_tensor_class_name(s.cls), s.bucket, s.layer,
                s.proxy_score, s.proxy_rmse, s.proxy_abs_mean, s.proxy_max_abs,
                s.alt_nvfp4_encoder_gain, s.alt_nvfp4_encoder_gain_rel,
                s.suggested_gate_floor, s.suggested_error_ratio,
                s.suggested_speed_penalty, s.suggested_delta_bytes,
                format_selector_suggested_type(s).c_str(),
                s.suggested_priority);
        }

	        if (out_tensor_overrides != nullptr && candidate_apply_top > 0) {
	            size_t budget_used = 0;
	            size_t bf16_budget_used = 0;
	            int applied = 0;
	            int skipped_outlier = 0;
	            std::unordered_map<int, int> class_used;
	            for (const auto & s : sens) {
	                if (applied >= candidate_apply_top) {
	                    break;
	                }
	                if (s.suggested_type == GGML_TYPE_COUNT || s.suggested_priority <= 0.0) {
	                    continue;
	                }
                const double effective_gate_floor =
                    s.suggested_gate_floor > 0.0 ? s.suggested_gate_floor : candidate_proxy_floor;
                if (s.proxy_score < effective_gate_floor) {
                    ++skipped_outlier;
                    continue;
                }
	                if (candidate_class_limit > 0 && class_used[(int) s.cls] >= candidate_class_limit) {
	                    continue;
	                }
                if (s.suggested_type == GGML_TYPE_NVFP4 && s.has_alt_nvfp4_encoder_cfg) {
                    tensor_type_option opt;
                    opt.name = selector_regex_escape(s.name);
                    opt.type = GGML_TYPE_NVFP4;
                    opt.has_nvfp4_cfg = true;
                    opt.nvfp4_cfg = s.alt_nvfp4_encoder_cfg;
                    opt.nvfp4_sample_blocks = s.alt_nvfp4_encoder_sample_blocks;
                    opt.nvfp4_policy_name = s.alt_nvfp4_encoder_policy_name;
                    out_tensor_overrides->push_back(std::move(opt));
                    class_used[(int) s.cls]++;
                    ++applied;
                    fprintf(stderr,
                        "selector candidate apply tensor=%s type=%s policy=%s sample_blocks=%" PRId64 "\n",
                        s.name.c_str(),
                        ggml_type_name(s.suggested_type),
                        s.alt_nvfp4_encoder_policy_name.c_str(),
                        s.alt_nvfp4_encoder_sample_blocks);
                    continue;
                }
                if (s.suggested_type == GGML_TYPE_NVFP4) {
                    continue;
                }
                const size_t rescue_nbytes = s.suggested_nbytes;
                if (rescue_nbytes == 0) {
                    continue;
                }
                const size_t delta = s.suggested_delta_bytes;
                if (candidate_budget_bytes > 0 && budget_used + delta > candidate_budget_bytes) {
                    continue;
                }
                if (s.suggested_type == GGML_TYPE_BF16 &&
                    rescue_bf16_budget_bytes > 0 &&
                    bf16_budget_used + delta > rescue_bf16_budget_bytes) {
                    continue;
                }
                tensor_type_option opt;
                opt.name = selector_regex_escape(s.name);
                opt.type = s.suggested_type;
                if (s.has_suggested_k_rsf_mode && s.suggested_k_rsf_mode > 0) {
                    opt.has_k_rsf_mode = true;
                    opt.k_rsf_mode = s.suggested_k_rsf_mode;
                }
                out_tensor_overrides->push_back(std::move(opt));
                budget_used += delta;
                if (s.suggested_type == GGML_TYPE_BF16) {
                    bf16_budget_used += delta;
                }
                class_used[(int) s.cls]++;
                ++applied;
	                fprintf(stderr,
	                    "selector candidate apply tensor=%s type=%s delta_bytes=%zu speed_penalty=%.6f gate=%.6f ratio=%.3f budget_used=%zu bf16_budget_used=%zu\n",
	                    s.name.c_str(),
	                    format_selector_suggested_type(s).c_str(),
	                    delta,
                    s.suggested_speed_penalty,
                    effective_gate_floor,
                    s.suggested_error_ratio,
                    budget_used,
	                    bf16_budget_used);
	            }
	            fprintf(stderr,
	                "%s: selector tensor-type applied=%d apply_cap=%d skipped_outlier=%d floor=%.6f budget_used=%zu bf16_budget_used=%zu\n",
	                __func__,
	                applied,
	                candidate_apply_top,
	                skipped_outlier,
	                candidate_proxy_floor,
	                budget_used,
	                bf16_budget_used);

	            if (!candidate_tensor_types_file.empty()) {
                std::ofstream patch_types(candidate_tensor_types_file, std::ios::trunc);
                if (!patch_types) {
                    fprintf(stderr, "%s: failed to open selector candidate tensor-type file %s\n", __func__, candidate_tensor_types_file.c_str());
                } else {
                    for (const auto & opt : *out_tensor_overrides) {
                        patch_types << opt.name << '=' << format_tensor_type_value(opt) << '\n';
                    }
                    fprintf(stderr, "%s: wrote selector candidate tensor-type file %s (%zu overrides)\n",
                        __func__, candidate_tensor_types_file.c_str(), out_tensor_overrides->size());
                }
            }
        }
    }

    const bool selector_only_control = selector_control_i64("ONLY", 0) != 0;
    const int mxfp6_e2m3_scale_top_default = selector_only_control
        ? 0
        : (candidate_apply_top > 0 ? candidate_apply_top : (mxfp6_bindings.empty() ? 0 : 96));
    const int mxfp6_e2m3_scale_top = out_tensor_overrides && have_runtime_eval
        ? (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_MXFP6_SELECTOR_SCALE_TOP", mxfp6_e2m3_scale_top_default))
        : 0;
    if (mxfp6_e2m3_scale_top > 0 && !mxfp6_bindings.empty()) {
        struct mxfp6_e2m3_scale_sens {
            size_t binding_index = 0;
            double proxy_score = 0.0;
            double rmse = 0.0;
            double abs_mean = 0.0;
            double max_abs = 0.0;
        };
        std::vector<mxfp6_e2m3_scale_sens> mx6_sens;
        mx6_sens.reserve(mxfp6_bindings.size());
        const bool scan_all_mxfp6 = mxfp6_e2m3_scale_top >= (int) mxfp6_bindings.size();
        if (scan_all_mxfp6) {
            auto class_priority = [](selector_tensor_class cls) {
                switch (cls) {
                    case selector_tensor_class::ATTN_QKV:       return 90.0;
                    case selector_tensor_class::ATTN_OUT:       return 88.0;
                    case selector_tensor_class::ROUTER:         return 76.0;
                    case selector_tensor_class::SSM:            return 72.0;
                    case selector_tensor_class::DENSE_MLP:      return 64.0;
                    case selector_tensor_class::EXPERT_DOWN:    return 62.0;
                    case selector_tensor_class::EXPERT_UP_GATE: return 60.0;
                    case selector_tensor_class::EMBEDDING_OUTPUT:return 40.0;
                    case selector_tensor_class::OTHER:          return 32.0;
                }
                return 32.0;
            };
            for (size_t i = 0; i < mxfp6_bindings.size(); ++i) {
                auto & b = mxfp6_bindings[i];
                if (b.target == nullptr || b.target_scale_nbytes == 0) {
                    continue;
                }
                mxfp6_e2m3_scale_sens s;
                s.binding_index = i;
                const double layer = b.layer >= 0 ? (double) b.layer : 9999.0;
                s.proxy_score = class_priority(b.cls) * 1000000.0 - layer * 1000.0 - (double) i;
                mx6_sens.push_back(s);
            }
        } else {
            for (size_t i = 0; i < mxfp6_bindings.size(); ++i) {
                auto & b = mxfp6_bindings[i];
                if (b.target == nullptr || b.target_scale_nbytes == 0) {
                    continue;
                }
                if (!quantize_binding_ensure_target_bytes(b) || !quantize_binding_ensure_scale_bytes(b)) {
                    continue;
                }
                double sq = 0.0;
                double ab = 0.0;
                double mx = 0.0;
                int64_t n = 0;
                if (!mxfp6_selector_quantize_binding(b, 1.0f, nthread, b.working_target_bytes, b.working_scale_bytes, sq, ab, mx, n) || n <= 0) {
                    continue;
                }
                mxfp6_e2m3_scale_sens s;
                s.binding_index = i;
                const selector_proxy_metrics proxy_metrics =
                    selector_proxy_score(sq, ab, mx, n);
                if (!proxy_metrics.ok) {
                    continue;
                }
                s.rmse = proxy_metrics.rmse;
                s.abs_mean = proxy_metrics.abs_mean;
                s.max_abs = proxy_metrics.max_abs;
                s.proxy_score = proxy_metrics.score;
                mx6_sens.push_back(s);
            }
        }
        std::sort(mx6_sens.begin(), mx6_sens.end(), [](const auto & a, const auto & b) {
            if (a.proxy_score != b.proxy_score) return a.proxy_score > b.proxy_score;
            return a.binding_index < b.binding_index;
        });

        const std::vector<float> scale_candidates = quantize_mxfp6_scale_candidates();
        const int scan_n = std::min<int>(mxfp6_e2m3_scale_top, (int) mx6_sens.size());
        update_full_quant_eta("selector-mxfp6-scale-refine", true);
        fprintf(stderr,
            "%s: MXFP6_E2M3 KLD-first scale refine scanning %d/%zu tensors with %zu scale candidates\n",
            __func__, scan_n, mx6_sens.size(), scale_candidates.size());

        struct mxfp6_e2m3_scale_accept {
            size_t binding_index = 0;
            std::vector<uint8_t> target_bytes;
            std::vector<uint8_t> scale_bytes;
            selector_device_snapshot target_device;
            selector_device_snapshot scale_device;
        };
        struct mxfp6_e2m3_scale_pending {
            size_t binding_index = 0;
            float scale_mul = 1.0f;
            double base_score = 0.0;
            double independent_score = 0.0;
            double replay_score = std::numeric_limits<double>::infinity();
            selector_derived_metrics base_metrics;
            selector_derived_metrics independent_metrics;
            selector_derived_metrics replay_metrics;
            bool active = true;
            int replay_count = 0;
        };

        std::vector<mxfp6_e2m3_scale_accept> accepted_scales;
        std::vector<mxfp6_e2m3_scale_pending> pending_scales;
        selector_derived_metrics current_scale_metrics = baseline_eval;
        int64_t mx6_runtime_direct_patches = 0;
        int64_t mx6_runtime_fallback_patches = 0;
        int64_t mx6_host_materializations = 0;
        int64_t mx6_accepted_snapshot_captures = 0;
        int64_t mx6_accepted_snapshot_failures = 0;
        selector_restore_counters mx6_accepted_restore_counters;
        const int pool_per_tensor = (int) std::max<int64_t>(1, MXFP6_SELECTOR_SCALE_POOL_PER_TENSOR);
        const int pool_top = (int) std::max<int64_t>(1, MXFP6_SELECTOR_SCALE_POOL_TOP);
        const int pool_rescore_top = (int) std::max<int64_t>(1, MXFP6_SELECTOR_SCALE_POOL_RESCORE_TOP);
        const int pool_apply_top = (int) std::max<int64_t>(0, MXFP6_SELECTOR_SCALE_POOL_APPLY_TOP);
        const double pool_score_slack = std::max(0.0, MXFP6_SELECTOR_SCALE_POOL_SCORE_SLACK);

        auto restore_accepted_scales = [&]() -> bool {
            restore_all();
            for (const auto & accepted : accepted_scales) {
                auto & ab = mxfp6_bindings[accepted.binding_index];
                if (!quantize_tensor_host_buffer(ab.target) &&
                        !quantize_restore_from_snapshot_or_host(
                        ab.target, accepted.target_device, accepted.target_bytes, ab.target_nbytes,
                        &mx6_accepted_restore_counters)) {
                    return false;
                }
                if (ab.target_scale != nullptr && !quantize_tensor_host_buffer(ab.target_scale) &&
                        !quantize_restore_from_snapshot_or_host(
                            ab.target_scale, accepted.scale_device, accepted.scale_bytes, ab.target_scale_nbytes,
                            &mx6_accepted_restore_counters)) {
                    return false;
                }
            }
            return true;
        };
        auto copy_mxfp6_scale_tensor = [&](selector_binding & b) -> bool {
            return b.target_scale == nullptr ||
                quantize_tensor_copy_in(b.target_scale, b.working_scale_bytes.data(), b.target_scale_nbytes);
        };
        auto quantize_mxfp6_runtime_patch = [&](
                selector_binding & b,
                float mul,
                bool materialize_host_bytes,
                double & sq,
                double & ab,
                double & mx,
                int64_t & n,
                bool & fatal_copy) -> bool {
            fatal_copy = false;
            bool direct_applied = false;
            if (mxfp6_selector_quantize_binding(
                    b, mul, nthread, b.working_target_bytes, b.working_scale_bytes, sq, ab, mx, n,
                    true, &direct_applied) && n > 0 && direct_applied) {
                if (!copy_mxfp6_scale_tensor(b)) {
                    return false;
                }
                ++mx6_runtime_direct_patches;
                if (!materialize_host_bytes) {
                    return true;
                }
                ++mx6_host_materializations;
                double host_sq = 0.0;
                double host_ab = 0.0;
                double host_mx = 0.0;
                int64_t host_n = 0;
                if (!mxfp6_selector_quantize_binding(
                        b, mul, nthread, b.working_target_bytes, b.working_scale_bytes,
                        host_sq, host_ab, host_mx, host_n) || host_n <= 0) {
                    return false;
                }
                return true;
            }
            if (!mxfp6_selector_quantize_binding(
                    b, mul, nthread, b.working_target_bytes, b.working_scale_bytes, sq, ab, mx, n) || n <= 0) {
                return false;
            }
            if (!quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                return false;
            }
            if (!copy_mxfp6_scale_tensor(b)) {
                return false;
            }
            ++mx6_runtime_fallback_patches;
            return true;
        };

        for (int si = 0; si < scan_n; ++si) {
            auto & b = mxfp6_bindings[mx6_sens[(size_t) si].binding_index];
            std::vector<mxfp6_e2m3_scale_pending> tensor_pending;
            selector_derived_metrics base_metrics = baseline_eval;
            if (!base_metrics.ok) {
                restore_all();
                selector_kld_metrics base_km;
                if (!selector_eval_kld_subset(lctx, kld_budget, params.n_batch, base_km, true)) {
                    restore_all();
                    return false;
                }
                base_metrics = selector_derive_metrics(base_km);
            }
            const selector_metric_score base_metric_score =
                selector_score_metrics(base_metrics, &base_metrics, rank_cfg, selector_metric_mode::MXFP6_SCALE);
            double base_score = base_metric_score.score;

            fprintf(stderr,
                "MXFP6_E2M3 scale refine tensor=%s rank=%d proxy=%.6f base_score=%.6f\n",
                b.name.c_str(), si + 1, mx6_sens[(size_t) si].proxy_score, base_score);

            for (float mul : scale_candidates) {
                if (std::fabs(mul - 1.0f) < 1e-6f) {
                    continue;
                }
                restore_all();
                double sq = 0.0;
                double ab = 0.0;
                double mx = 0.0;
                int64_t n = 0;
                bool fatal_copy = false;
                if (!quantize_mxfp6_runtime_patch(b, mul, false, sq, ab, mx, n, fatal_copy)) {
                    if (!fatal_copy) {
                        continue;
                    }
                    restore_all();
                    return false;
                }

                selector_kld_metrics km;
                if (!selector_eval_kld_subset(lctx, kld_budget, params.n_batch, km, true)) {
                    restore_all();
                    return false;
                }
                const selector_derived_metrics dm = selector_derive_metrics(km);
                const selector_metric_score metric_score =
                    selector_score_metrics(dm, &base_metrics, rank_cfg, selector_metric_mode::MXFP6_SCALE);
                const double score = metric_score.score;
                const bool pass = metric_score.pass;
                fprintf(stderr,
                    "MXFP6_E2M3 scale refine tensor=%s mul=%.8g pass=%d score=%.6f ln=%.6f kld=%.6f p99=%.6f p999=%.6f max=%.6f rms=%.6f top=%.4f\n",
                    b.name.c_str(), (double) mul, pass ? 1 : 0, score,
                    dm.ln_ratio, dm.mean_kld, dm.kld_p99, dm.kld_p999, dm.max_kld, dm.rms_dp, dm.same_top);
                if (pass && std::isfinite(score) && score < base_score + pool_score_slack) {
                    mxfp6_e2m3_scale_pending pending;
                    pending.binding_index = mx6_sens[(size_t) si].binding_index;
                    pending.scale_mul = mul;
                    pending.base_score = base_score;
                    pending.independent_score = score;
                    pending.base_metrics = base_metrics;
                    pending.independent_metrics = dm;
                    tensor_pending.push_back(std::move(pending));
                }
            }

            if (!restore_accepted_scales()) {
                return false;
            }

            std::sort(tensor_pending.begin(), tensor_pending.end(), [](const auto & a, const auto & b) {
                if (a.independent_score != b.independent_score) return a.independent_score < b.independent_score;
                if (a.independent_metrics.max_kld != b.independent_metrics.max_kld) return a.independent_metrics.max_kld < b.independent_metrics.max_kld;
                return a.scale_mul < b.scale_mul;
            });
            if ((int) tensor_pending.size() > pool_per_tensor) {
                tensor_pending.resize((size_t) pool_per_tensor);
            }
            for (auto & pending : tensor_pending) {
                auto & pb = mxfp6_bindings[pending.binding_index];
                fprintf(stderr,
                    "MXFP6_E2M3 scale refine candidate tensor=%s scale_mul=%.8g score %.6f -> %.6f kld %.6f -> %.6f p99 %.6f -> %.6f max %.6f -> %.6f\n",
                    pb.name.c_str(), (double) pending.scale_mul,
                    pending.base_score, pending.independent_score,
                    pending.base_metrics.mean_kld, pending.independent_metrics.mean_kld,
                    pending.base_metrics.kld_p99, pending.independent_metrics.kld_p99,
                    pending.base_metrics.max_kld, pending.independent_metrics.max_kld);
                pending_scales.push_back(std::move(pending));
            }
        }

        std::sort(pending_scales.begin(), pending_scales.end(), [](const auto & a, const auto & b) {
            if (a.independent_score != b.independent_score) return a.independent_score < b.independent_score;
            if (a.independent_metrics.max_kld != b.independent_metrics.max_kld) return a.independent_metrics.max_kld < b.independent_metrics.max_kld;
            return a.binding_index < b.binding_index;
        });
        if ((int) pending_scales.size() > pool_top) {
            pending_scales.resize((size_t) pool_top);
        }

            fprintf(stderr,
            "%s: MXFP6_E2M3 KLD-first scale refine pooled replay candidates=%zu per_tensor=%d rescore_top=%d apply_top=%d score_slack=%.6f\n",
            __func__, pending_scales.size(), pool_per_tensor, pool_rescore_top, pool_apply_top, pool_score_slack);

        std::vector<uint8_t> binding_accepted(mxfp6_bindings.size(), 0);
        int applied_pool = 0;
        for (int replay_round = 1; !pending_scales.empty(); ++replay_round) {
            std::sort(pending_scales.begin(), pending_scales.end(), [](const auto & a, const auto & b) {
                const double as = std::isfinite(a.replay_score) ? a.replay_score : a.independent_score;
                const double bs = std::isfinite(b.replay_score) ? b.replay_score : b.independent_score;
                if (as != bs) return as < bs;
                if (a.independent_score != b.independent_score) return a.independent_score < b.independent_score;
                if (a.replay_count != b.replay_count) return a.replay_count < b.replay_count;
                return a.binding_index < b.binding_index;
            });

            const selector_metric_score base_metric_score =
                selector_score_metrics(current_scale_metrics, &current_scale_metrics, rank_cfg, selector_metric_mode::MXFP6_SCALE);
            const double base_score = base_metric_score.score;
            int best_index = -1;
            double best_score = base_score;
            selector_derived_metrics best_metrics;
            std::vector<uint8_t> best_target_bytes;
            std::vector<uint8_t> best_scale_bytes;
            int rescored = 0;

            for (size_t pi = 0; pi < pending_scales.size() && rescored < pool_rescore_top; ++pi) {
                auto & pending = pending_scales[pi];
                if (!pending.active || pending.binding_index >= binding_accepted.size() || binding_accepted[pending.binding_index]) {
                    pending.active = false;
                    continue;
                }
                auto & b = mxfp6_bindings[pending.binding_index];
                if (!restore_accepted_scales()) {
                    return false;
                }
                double sq = 0.0;
                double ab = 0.0;
                double mx = 0.0;
                int64_t n = 0;
                bool fatal_copy = false;
                if (!quantize_mxfp6_runtime_patch(b, pending.scale_mul, false, sq, ab, mx, n, fatal_copy)) {
                    if (fatal_copy) {
                        restore_all();
                        return false;
                    }
                    pending.active = false;
                    continue;
                }

                selector_kld_metrics km;
                if (!selector_eval_kld_subset(lctx, kld_budget, params.n_batch, km, true)) {
                    restore_all();
                    return false;
                }
                const selector_derived_metrics dm = selector_derive_metrics(km);
                const selector_metric_score metric_score =
                    selector_score_metrics(dm, &current_scale_metrics, rank_cfg, selector_metric_mode::MXFP6_SCALE);
                const double score = metric_score.score;
                const bool pass = metric_score.pass;
                ++pending.replay_count;
                pending.replay_score = score;
                pending.replay_metrics = dm;
                ++rescored;
                fprintf(stderr,
                    "MXFP6_E2M3 scale refine pool round=%d tensor=%s mul=%.8g pass=%d score %.6f -> %.6f independent=%.6f ln=%.6f kld=%.6f p99=%.6f p999=%.6f max=%.6f rms=%.6f top=%.4f\n",
                    replay_round, b.name.c_str(), (double) pending.scale_mul, pass ? 1 : 0,
                    base_score, score, pending.independent_score,
                    dm.ln_ratio, dm.mean_kld, dm.kld_p99, dm.kld_p999, dm.max_kld, dm.rms_dp, dm.same_top);
                if (!pass || !std::isfinite(score) || score >= base_score) {
                    pending.active = false;
                    continue;
                }
                if (score < best_score) {
                    bool materialize_fatal = false;
                    double mat_sq = 0.0;
                    double mat_ab = 0.0;
                    double mat_mx = 0.0;
                    int64_t mat_n = 0;
                    if (!quantize_mxfp6_runtime_patch(
                            b, pending.scale_mul, true,
                            mat_sq, mat_ab, mat_mx, mat_n, materialize_fatal)) {
                        if (materialize_fatal) {
                            restore_all();
                            return false;
                        }
                        pending.active = false;
                        continue;
                    }
                    best_index = (int) pi;
                    best_score = score;
                    best_metrics = dm;
                    best_target_bytes = b.working_target_bytes;
                    best_scale_bytes = b.working_scale_bytes;
                }
            }

            if (best_index < 0) {
                fprintf(stderr,
                    "MXFP6_E2M3 scale refine pool stop round=%d rescored=%d base_score=%.6f active=%zu\n",
                    replay_round, rescored, base_score,
                    std::count_if(pending_scales.begin(), pending_scales.end(), [](const auto & p) { return p.active; }));
                break;
            }

            auto & accepted_pending = pending_scales[(size_t) best_index];
            auto & b = mxfp6_bindings[accepted_pending.binding_index];
            if (!restore_accepted_scales()) {
                return false;
            }
            if (!quantize_tensor_copy_in(b.target, best_target_bytes.data(), b.target_nbytes)) {
                restore_all();
                return false;
            }
            if (b.target_scale != nullptr &&
                    !quantize_tensor_copy_in(b.target_scale, best_scale_bytes.data(), b.target_scale_nbytes)) {
                restore_all();
                return false;
            }

            tensor_type_option opt;
            opt.name = selector_regex_escape(b.name);
            opt.type = GGML_TYPE_MXFP6_E2M3;
            opt.has_mxfp6_scale_mul = true;
            opt.mxfp6_e2m3_scale_mul = accepted_pending.scale_mul;
            opt.mxfp6_policy_name = "kld_scale";
            out_tensor_overrides->push_back(std::move(opt));
            fprintf(stderr,
                "MXFP6_E2M3 scale refine apply round=%d tensor=%s scale_mul=%.8g score %.6f -> %.6f kld %.6f -> %.6f p99 %.6f -> %.6f p999 %.6f -> %.6f max %.6f -> %.6f\n",
                replay_round, b.name.c_str(), (double) accepted_pending.scale_mul,
                base_score, best_score,
                current_scale_metrics.mean_kld, best_metrics.mean_kld,
                current_scale_metrics.kld_p99, best_metrics.kld_p99,
                current_scale_metrics.kld_p999, best_metrics.kld_p999,
                current_scale_metrics.max_kld, best_metrics.max_kld);
            mxfp6_e2m3_scale_accept accepted;
            accepted.binding_index = accepted_pending.binding_index;
            accepted.target_bytes = std::move(best_target_bytes);
            accepted.scale_bytes = std::move(best_scale_bytes);
            if (accepted.target_device.capture(b.target, b.target_nbytes)) {
                ++mx6_accepted_snapshot_captures;
            } else {
                ++mx6_accepted_snapshot_failures;
            }
            if (b.target_scale != nullptr) {
                if (accepted.scale_device.capture(b.target_scale, b.target_scale_nbytes)) {
                    ++mx6_accepted_snapshot_captures;
                } else {
                    ++mx6_accepted_snapshot_failures;
                }
            }
            accepted_scales.push_back(std::move(accepted));
            binding_accepted[accepted_pending.binding_index] = 1;
            current_scale_metrics = best_metrics;
            ++applied_pool;
            pending_scales.erase(
                std::remove_if(pending_scales.begin(), pending_scales.end(), [&](const auto & p) {
                    return !p.active || p.binding_index == accepted.binding_index;
                }),
                pending_scales.end());
            if (pool_apply_top > 0 && applied_pool >= pool_apply_top) {
                fprintf(stderr, "MXFP6_E2M3 scale refine pool reached apply_top=%d\n", pool_apply_top);
                break;
            }
        }
        fprintf(stderr,
            "%s: MXFP6_E2M3 scale refine runtime patches direct=%" PRId64 " fallback=%" PRId64 " materialize=%" PRId64
            " accepted_snapshots=%" PRId64 " accepted_snapshot_fail=%" PRId64
            " accepted_restore_device=%" PRId64 " accepted_restore_host=%" PRId64 " accepted_restore_fail=%" PRId64 "\n",
            __func__,
            mx6_runtime_direct_patches,
            mx6_runtime_fallback_patches,
            mx6_host_materializations,
            mx6_accepted_snapshot_captures,
            mx6_accepted_snapshot_failures,
            mx6_accepted_restore_counters.device,
            mx6_accepted_restore_counters.host,
            mx6_accepted_restore_counters.failed);
    }

    restore_all();
    if (runtime_restore_counters.device || runtime_restore_counters.host || runtime_restore_counters.failed) {
        fprintf(stderr,
            "%s: selector runtime original restores device=%" PRId64 " host=%" PRId64 " fail=%" PRId64 "\n",
            __func__,
            runtime_restore_counters.device,
            runtime_restore_counters.host,
            runtime_restore_counters.failed);
    }
    return true;
}

static bool parse_mixed_format_policy(const char * value, int32_t & out) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    std::string v = trim_copy(value);
    std::replace(v.begin(), v.end(), '-', '_');
    if (striequals(v.c_str(), "off") || striequals(v.c_str(), "none")) {
        out = LLAMA_NV4MX6_POLICY_OFF;
        return true;
    }
    if (striequals(v.c_str(), "auto")) {
        out = LLAMA_NV4MX6_POLICY_AUTO;
        return true;
    }
    if (striequals(v.c_str(), "nvfp4_quality_boost") ||
            striequals(v.c_str(), "nvfp4_primary") ||
            striequals(v.c_str(), "quality_boost") ||
            striequals(v.c_str(), "promote_mxfp6") ||
            striequals(v.c_str(), "nv4_promote_mx6")) {
        out = LLAMA_NV4MX6_POLICY_NV4_PROMOTE_MX6;
        return true;
    }
    if (striequals(v.c_str(), "mxfp6_primary") ||
            striequals(v.c_str(), "mxfp6_quality") ||
            striequals(v.c_str(), "quality_first") ||
            striequals(v.c_str(), "demote_nvfp4") ||
            striequals(v.c_str(), "mx6_demote_nv4")) {
        out = LLAMA_NV4MX6_POLICY_MX6_DEMOTE_NV4;
        return true;
    }
    if (striequals(v.c_str(), "mxfp6_slot") || striequals(v.c_str(), "mx6_slot")) {
        out = LLAMA_NV4MX6_POLICY_MX6_SLOT;
        return true;
    }
    if (striequals(v.c_str(), "bf16_mx6") || striequals(v.c_str(), "bf16_to_mxfp6")) {
        out = LLAMA_NV4MX6_POLICY_BF16_MX6;
        return true;
    }
    if (striequals(v.c_str(), "bf16_mx6_sse") || striequals(v.c_str(), "bf16_to_mxfp6_sse")) {
        out = LLAMA_NV4MX6_POLICY_BF16_MX6_SSE;
        return true;
    }
    return false;
}

static std::string canonical_quantize_cli_option(const char * raw) {
    if (raw == nullptr) {
        return {};
    }
    std::string arg(raw);
    static const std::string selector_prefix = "--selector-";
    if (arg.rfind(selector_prefix, 0) == 0) {
        if (arg == "--selector-rescue-encoder-top") {
            return "--nvfp4-selector-rescue-nvfp4-top";
        }
        return "--nvfp4-selector-" + arg.substr(selector_prefix.size());
    }
    return arg;
}

int llama_quantize(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 3) {
        usage(argv[0]);
    }

    quantize_control_clear();
    llama_model_quantize_params params = llama_model_quantize_default_params();

    int arg_idx = 1;
    std::string imatrix_file;
    std::string patch_base_model;
    std::vector<std::string> included_weights, excluded_weights;
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<tensor_type_option> tensor_type_opts;
    std::vector<int> prune_layers;
    std::vector<std::pair<std::string, std::string>> selector_controls;
    std::string assignment_jsonl;
    int32_t selector_eval_batch_override = 0;
    int32_t cli_nvfp4_encoder_threads = 0;
    int32_t selector_kld_threads_override = 0;
    quantize_work_mode cli_work_mode = quantize_work_mode::UNSET;
    bool selector_skip_remaining = false;
    std::string selector_skip_file;
    std::string selector_include_policies_cli;
    std::string selector_skip_policies_cli;
    std::string selector_candidate_types_cli;
    bool cli_nvfp4_cfg_valid = false;
    nvfp4_cuda_runtime_cfg cli_nvfp4_cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    std::string cli_nvfp4_policy_name;
    const auto add_selector_controls = [&](const char * key, const char * value) {
        selector_controls.emplace_back(key, value);
    };
    const auto add_selector_control = [&](const char * suffix, const char * value) {
        selector_controls.emplace_back(selector_legacy_control_key(suffix), value);
    };
    auto append_selector_skip_policy = [&](const char * value) {
        if (value == nullptr || value[0] == '\0') {
            return;
        }
        if (!selector_skip_policies_cli.empty()) {
            selector_skip_policies_cli.push_back(',');
        }
        selector_skip_policies_cli += value;
    };
    auto append_selector_include_policy = [&](const char * value) {
        if (value == nullptr || value[0] == '\0') {
            return;
        }
        if (!selector_include_policies_cli.empty()) {
            selector_include_policies_cli.push_back(',');
        }
        selector_include_policies_cli += value;
    };
    auto append_selector_candidate_type = [&](const char * value) {
        if (value == nullptr || value[0] == '\0') {
            return;
        }
        std::string normalized(value);
        for (char & ch : normalized) {
            if (ch == ';' || ch == '\n' || ch == '\r' || ch == '\t') {
                ch = ',';
            }
        }
        std::stringstream ss(normalized);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim_copy(token);
            if (token.empty()) {
                continue;
            }
            const ggml_type type = parse_ggml_type(token.c_str());
            if (type == GGML_TYPE_COUNT || type == GGML_TYPE_NVFP4) {
                usage(argv[0]);
            }
            if (!selector_candidate_types_cli.empty()) {
                selector_candidate_types_cli.push_back(',');
            }
            selector_candidate_types_cli += ggml_type_name(type);
        }
    };

    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        const std::string canonical_arg = canonical_quantize_cli_option(argv[arg_idx]);
        const char * arg_name = canonical_arg.c_str();
        if (strcmp(arg_name, "--leave-output-tensor") == 0) {
            params.quantize_output_tensor = false;
        } else if (strcmp(arg_name, "--output-tensor-type") == 0) {
            if (arg_idx < argc-1) {
                params.output_tensor_type = parse_ggml_type(argv[++arg_idx]);
                if (params.output_tensor_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--token-embedding-type") == 0) {
            if (arg_idx < argc-1) {
                params.token_embedding_type = parse_ggml_type(argv[++arg_idx]);
                if (params.token_embedding_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mtp-tensor-type") == 0) {
            if (arg_idx < argc-1) {
                params.mtp_tensor_type = parse_ggml_type(argv[++arg_idx]);
                if (params.mtp_tensor_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--tensor-type") == 0) {
            if (arg_idx == argc-1 || !parse_tensor_type(argv[++arg_idx], tensor_type_opts)) {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-cfg") == 0) {
            if (arg_idx < argc-1) {
                tensor_type_option opt;
                if (!parse_tensor_type_nvfp4_cfg(argv[++arg_idx], opt)) {
                    usage(argv[0]);
                }
                cli_nvfp4_cfg = opt.nvfp4_cfg;
                cli_nvfp4_policy_name = opt.nvfp4_policy_name.empty() ? "cli" : opt.nvfp4_policy_name;
                cli_nvfp4_cfg_valid = true;
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-preset") == 0) {
            if (arg_idx < argc-1) {
                if (!parse_nvfp4_preset(argv[++arg_idx], cli_nvfp4_cfg, &cli_nvfp4_policy_name)) {
                    usage(argv[0]);
                }
                cli_nvfp4_cfg_valid = true;
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-correction-denom") == 0) {
            if (arg_idx < argc-1) {
                const char * value = argv[++arg_idx];
                if (!parse_nvfp4_scale_denom(value, params.nvfp4_correction_denom)) {
                    usage(argv[0]);
                }
                add_selector_control("CORRECTION_DENOM", value);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-input-scale-policy") == 0) {
            if (arg_idx < argc-1) {
                if (!parse_nvfp4_input_scale_policy(argv[++arg_idx], params.nvfp4_input_scale_policy)) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-encoder-max-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_ENCODER_MAX_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-encoder-threads") == 0) {
            if (arg_idx < argc-1) {
                cli_nvfp4_encoder_threads = std::max<int32_t>(1, (int32_t) std::strtol(argv[++arg_idx], nullptr, 10));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mode") == 0) {
            if (arg_idx < argc-1) {
                quantize_work_mode mode = quantize_work_mode::UNSET;
                if (!parse_quantize_work_mode(argv[++arg_idx], mode)) {
                    usage(argv[0]);
                }
                cli_work_mode = mode;
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rsf-mode") == 0) {
            if (arg_idx < argc-1) {
                int rsf_mode = NVFP4_CUDA_RSF_MODE_TENSOR;
                const char * value = argv[++arg_idx];
                if (!nvfp4_parse_rsf_mode(value, rsf_mode)) {
                    usage(argv[0]);
                }
                add_selector_control("RSF_MODE", nvfp4_rsf_mode_name(rsf_mode));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rsf-depth") == 0) {
            if (arg_idx < argc-1) {
                int rsf_depth = NVFP4_CUDA_RSF_DEPTH_DEEPER;
                const char * value = argv[++arg_idx];
                if (!nvfp4_parse_rsf_depth(value, rsf_depth)) {
                    usage(argv[0]);
                }
                add_selector_control("RSF_DEPTH", nvfp4_rsf_depth_name(rsf_depth));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rsf-report") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RSF_REPORT_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--tensor-type-file") == 0) {
            if (arg_idx == argc-1 || !parse_tensor_type_file(argv[++arg_idx], tensor_type_opts)) {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--prune-layers") == 0) {
            if (arg_idx == argc-1 || !parse_layer_prune(argv[++arg_idx], prune_layers)) {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--override-kv") == 0) {
            if (arg_idx == argc-1 || !string_parse_kv_override(argv[++arg_idx], kv_overrides)) {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--patch-base") == 0) {
            if (arg_idx < argc-1) {
                patch_base_model = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mxfp6_e2m3-tensor-scale") == 0) {
            if (arg_idx < argc-1) {
                if (!parse_on_off_value(argv[++arg_idx], params.mxfp6_tensor_scale)) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mxfp6_e2m3-min-savings") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_min_savings_bytes = std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mxfp6_e2m3-input-scale-denom") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_input_scale_denom = std::max(0.0f, std::strtof(argv[++arg_idx], nullptr));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mxfp6_e2m3-input-scale-quantile") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_input_scale_quantile = std::clamp(std::strtof(argv[++arg_idx], nullptr), 0.0f, 1.0f);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mxfp6_e2m3-tensor-scale-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_tensor_scale_sample_blocks = std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mxfp6_e2m3-tensor-scale-steps") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_tensor_scale_steps = (int32_t) std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mxfp6_e2m3-selector-scale-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_MXFP6_SELECTOR_SCALE_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mxfp6_e2m3-selector-scale-candidates") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_MXFP6_SELECTOR_SCALE_CANDIDATES", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mixed-format-policy") == 0) {
            if (arg_idx < argc-1) {
                if (!parse_mixed_format_policy(argv[++arg_idx], params.nv4mx6_policy)) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mixed-mx6-penalty") == 0) {
            if (arg_idx < argc-1) {
                params.nv4mx6_mx6_penalty = std::max(0.0f, std::strtof(argv[++arg_idx], nullptr));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mixed-bf16-mx6-threshold") == 0) {
            if (arg_idx < argc-1) {
                params.nv4mx6_bf16_mx6_max_sse_ratio = std::max(0.0f, std::strtof(argv[++arg_idx], nullptr));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mixed-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_sample_blocks = std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mixed-sample-cap") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_sample_cap = std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mixed-imatrix-weight-blend") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_imatrix_blend = std::strtof(argv[++arg_idx], nullptr);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mixed-imatrix-weight-power") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_imatrix_power = std::strtof(argv[++arg_idx], nullptr);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mixed-imatrix-weight-min") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_imatrix_min = std::strtof(argv[++arg_idx], nullptr);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--mixed-imatrix-weight-max") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_imatrix_max = std::strtof(argv[++arg_idx], nullptr);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-kld") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("KLD_FILE", argv[++arg_idx]);
                add_selector_control("ENABLE_EVAL", "1");
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-ledger") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("LEDGER_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-checkpoint-model") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CHECKPOINT_MODEL", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-cache-dir") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CACHE_DIR", argv[++arg_idx]);
                add_selector_control("KEEP_CHECKPOINT", "1");
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-keep-checkpoint") == 0) {
            add_selector_control("KEEP_CHECKPOINT", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-require-runtime-cache") == 0) {
            add_selector_control("REQUIRE_RUNTIME_CACHE", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-skip-file") == 0) {
            if (arg_idx < argc-1) {
                selector_skip_file = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-skip-remaining") == 0) {
            selector_skip_remaining = true;
        } else if (strcmp(arg_name, "--nvfp4-selector-stagea-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("STAGEA_SAMPLE_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-stagea-max-policies") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("STAGEA_MAX_POLICIES", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-awq-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("AWQ_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-skip-policy") == 0) {
            if (arg_idx < argc-1) {
                append_selector_skip_policy(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-skip-policies") == 0) {
            if (arg_idx < argc-1) {
                append_selector_skip_policy(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-include-policy") == 0) {
            if (arg_idx < argc-1) {
                append_selector_include_policy(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-include-policies") == 0) {
            if (arg_idx < argc-1) {
                append_selector_include_policy(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-tensor-policy-map") == 0) {
            add_selector_control("TENSOR_POLICY_MAP", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-no-tensor-policy-map") == 0) {
            add_selector_control("TENSOR_POLICY_MAP", "0");
        } else if (strcmp(arg_name, "--nvfp4-selector-refine-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("REFINE_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-refine-budget") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("REFINE_BUDGET", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-survey-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("SURVEY_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-survey-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("SURVEY_SAMPLE_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-max-tensors") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("MAX_TENSORS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-trace") == 0) {
            add_selector_controls("LLAMA_NVFP4_TRACE", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-policy-threads") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("POLICY_THREADS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-stageb-patch-eval") == 0) {
            add_selector_control("STAGEB_PATCH_EVAL", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-stageb-direct-verify") == 0) {
            add_selector_control("STAGEB_DIRECT_VERIFY", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-no-stageb-direct-verify") == 0) {
            add_selector_control("STAGEB_DIRECT_VERIFY", "0");
        } else if (strcmp(arg_name, "--nvfp4-selector-threads") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("THREADS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-kld-threads") == 0) {
            if (arg_idx < argc-1) {
                selector_kld_threads_override = std::max<int32_t>(1, (int32_t) std::strtol(argv[++arg_idx], nullptr, 10));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-auto-rescue") == 0) {
            add_selector_controls("LLAMA_NVFP4_AUTO_RESCUE", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-only") == 0) {
            add_selector_control("ONLY", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-eval-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("EVAL_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-n-seq") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("N_SEQ", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-eval-batch") == 0) {
            if (arg_idx < argc-1) {
                selector_eval_batch_override = std::max<int32_t>(1, (int32_t) std::strtol(argv[++arg_idx], nullptr, 10));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-n-gpu-layers") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("N_GPU_LAYERS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-sensitivity-report") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("SENSITIVITY_REPORT_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-sensitivity-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("SENSITIVITY_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-sensitivity-layer") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("SENSITIVITY_LAYER", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-sensitivity-tensor") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("SENSITIVITY_TENSOR", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-sensitivity-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("SENSITIVITY_SAMPLE_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-type") == 0) {
            if (arg_idx < argc-1) {
                append_selector_candidate_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-types") == 0) {
            if (arg_idx < argc-1) {
                append_selector_candidate_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CANDIDATE_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-fraction") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CANDIDATE_FRACTION", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-budget-mb") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CANDIDATE_BUDGET_MB", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-class-limit") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CANDIDATE_CLASS_LIMIT", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-report-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CANDIDATE_REPORT_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CANDIDATE_SAMPLE_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-report") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CANDIDATE_REPORT_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-candidate-tensor-types") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("CANDIDATE_TENSOR_TYPES_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-report-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_REPORT_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-budget-mb") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_BUDGET_MB", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-bf16-budget-mb") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_BF16_BUDGET_MB", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-class-limit") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_CLASS_LIMIT", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-nvfp4-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_NVFP4_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_SENS_SAMPLE_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-coarse-max-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_NVFP4_MAX_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-refine-max-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_NVFP4_REFINE_MAX_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-guard-max-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_NVFP4_GUARD_MAX_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-report") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_REPORT_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rescue-tensor-types") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RESCUE_TENSOR_TYPES_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-kld-penalty") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RANK_KLD_PENALTY", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-p99-penalty") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RANK_P99_PENALTY", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-p999-penalty") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RANK_P999_PENALTY", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-max-kld-penalty") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RANK_MAX_KLD_PENALTY", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rank-kld-threshold") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RANK_KLD_THRESHOLD", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rank-kld-hard-gate") == 0) {
            add_selector_control("RANK_KLD_HARD_GATE", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-rank-p99-threshold") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RANK_P99_THRESHOLD", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rank-p99-hard-gate") == 0) {
            add_selector_control("RANK_P99_HARD_GATE", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-rank-p999-threshold") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RANK_P999_THRESHOLD", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rank-p999-hard-gate") == 0) {
            add_selector_control("RANK_P999_HARD_GATE", "1");
        } else if (strcmp(arg_name, "--nvfp4-selector-rank-max-kld-threshold") == 0) {
            if (arg_idx < argc-1) {
                add_selector_control("RANK_MAX_KLD_THRESHOLD", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--nvfp4-selector-rank-max-kld-hard-gate") == 0) {
            add_selector_control("RANK_MAX_KLD_HARD_GATE", "1");
        } else if (strcmp(arg_name, "--assignment-jsonl") == 0) {
            if (arg_idx < argc-1) {
                assignment_jsonl = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--dry-run") == 0) {
            params.dry_run = true;
        } else if (strcmp(arg_name, "--allow-requantize") == 0) {
            params.allow_requantize = true;
        } else if (strcmp(arg_name, "--pure") == 0) {
            params.pure = true;
        } else if (strcmp(arg_name, "--imatrix") == 0) {
            if (arg_idx < argc-1) {
                imatrix_file = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--include-weights") == 0) {
            if (arg_idx < argc-1) {
                included_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--exclude-weights") == 0) {
            if (arg_idx < argc-1) {
                excluded_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(arg_name, "--keep-split") == 0) {
            params.keep_split = true;
        } else {
            usage(argv[0]);
        }
    }

    if (argc - arg_idx < 2) {
        printf("%s: bad arguments\n", argv[0]);
        usage(argv[0]);
    }
    if (!included_weights.empty() && !excluded_weights.empty()) {
        usage(argv[0]);
    }

    auto selector_controls_has = [&](const char * key) {
        return std::any_of(selector_controls.begin(), selector_controls.end(), [&](const auto & entry) {
            return entry.first == key;
        });
    };
    auto selector_control_args_has = [&](const char * suffix) {
        const std::string key = selector_legacy_control_key(suffix);
        return selector_controls_has(key.c_str());
    };
    auto add_selector_controls_default = [&](const char * key, const char * value) {
        if (!selector_controls_has(key)) {
            selector_controls.emplace_back(key, value);
        }
    };
    auto add_selector_control_default = [&](const char * suffix, const char * value) {
        const std::string key = selector_legacy_control_key(suffix);
        if (!selector_controls_has(key.c_str())) {
            selector_controls.emplace_back(key, value);
        }
    };
    const bool selector_kld_requested_cli = selector_control_args_has("KLD_FILE");
    if (selector_kld_requested_cli && cli_work_mode == quantize_work_mode::UNSET) {
        fprintf(stderr, "llama_quantize: saved-logit selector runs require --mode fast|normal|deep\n");
        return 1;
    }
    auto add_work_mode_defaults = [&](quantize_work_mode mode) {
        if (mode == quantize_work_mode::UNSET) {
            return;
        }
        add_selector_controls_default("LLAMA_QUANTIZE_MODE", quantize_work_mode_name(mode));
        add_selector_controls_default("LLAMA_SELECTOR_TENSOR_POLICY_PROXY_REL_GAIN", "0.0005");
        switch (mode) {
            case quantize_work_mode::FAST:
                add_selector_controls_default("LLAMA_SELECTOR_TENSOR_POLICY_PROXY_ABS_GAIN", "0.0005");
                add_selector_control_default("RSF_DEPTH", "deeper");
                add_selector_controls_default("LLAMA_NVFP4_ENCODER_MAX_BLOCKS", "8192");
                add_selector_control_default("STAGEA_SAMPLE_BLOCKS", "2048");
                add_selector_control_default("STAGEA_MAX_POLICIES", "0");
                add_selector_control_default("AWQ_TOP", "4");
                add_selector_control_default("REFINE_TOP", "12");
                add_selector_control_default("REFINE_BUDGET", "96");
                add_selector_control_default("SURVEY_TOP", "48");
                add_selector_control_default("SURVEY_SAMPLE_BLOCKS", "2048");
                add_selector_control_default("EVAL_TOP", "4");
                add_selector_control_default("RANK_KLD_PENALTY", "4.0");
                add_selector_control_default("RANK_P99_PENALTY", "1.5");
                add_selector_control_default("RANK_P999_PENALTY", "0.75");
                add_selector_control_default("RANK_MAX_KLD_PENALTY", "0.10");
                add_selector_control_default("CANDIDATE_TOP", "17");
                add_selector_control_default("CANDIDATE_NVFP4_TOP", "6");
                add_selector_control_default("CANDIDATE_SAMPLE_BLOCKS", "2048");
                add_selector_controls_default("LLAMA_SELECTOR_TYPE_CANDIDATE_OUTLIER_RATIO", "1.35");
                add_selector_controls_default("LLAMA_SELECTOR_TYPE_CANDIDATE_MIN_ERROR", "0.030");
                add_selector_control_default("RESCUE_NVFP4_MAX_BLOCKS", "8192");
                add_selector_control_default("RESCUE_NVFP4_REFINE_MAX_BLOCKS", "8192");
                add_selector_control_default("RESCUE_NVFP4_GUARD_MAX_BLOCKS", "16384");
                add_selector_control_default("RESCUE_NVFP4_POLICY_BUDGET", "8");
                add_selector_control_default("RESCUE_NVFP4_REFINE_TOP", "1");
                add_selector_control_default("RESCUE_NVFP4_GUARD_TOP", "1");
                add_selector_control_default("CANDIDATE_REPORT_TOP", "17");
                break;
            case quantize_work_mode::NORMAL:
                add_selector_controls_default("LLAMA_SELECTOR_TENSOR_POLICY_PROXY_ABS_GAIN", "0.0002");
                add_selector_control_default("RSF_DEPTH", "deeper");
                add_selector_controls_default("LLAMA_NVFP4_ENCODER_MAX_BLOCKS", "32768");
                add_selector_control_default("STAGEA_SAMPLE_BLOCKS", "8192");
                add_selector_control_default("STAGEA_MAX_POLICIES", "0");
                add_selector_control_default("AWQ_TOP", "4");
                add_selector_control_default("REFINE_TOP", "24");
                add_selector_control_default("REFINE_BUDGET", "192");
                add_selector_control_default("SURVEY_TOP", "64");
                add_selector_control_default("SURVEY_SAMPLE_BLOCKS", "8192");
                add_selector_control_default("EVAL_TOP", "8");
                add_selector_control_default("RANK_KLD_PENALTY", "4.0");
                add_selector_control_default("RANK_P99_PENALTY", "3.0");
                add_selector_control_default("RANK_P999_PENALTY", "1.5");
                add_selector_control_default("RANK_MAX_KLD_PENALTY", "0.35");
                add_selector_control_default("CANDIDATE_TOP", "40");
                add_selector_control_default("CANDIDATE_NVFP4_TOP", "12");
                add_selector_control_default("CANDIDATE_SAMPLE_BLOCKS", "4096");
                add_selector_controls_default("LLAMA_SELECTOR_TYPE_CANDIDATE_OUTLIER_RATIO", "1.15");
                add_selector_controls_default("LLAMA_SELECTOR_TYPE_CANDIDATE_MIN_ERROR", "0.020");
                add_selector_control_default("RESCUE_NVFP4_MAX_BLOCKS", "16384");
                add_selector_control_default("RESCUE_NVFP4_REFINE_MAX_BLOCKS", "32768");
                add_selector_control_default("RESCUE_NVFP4_GUARD_MAX_BLOCKS", "32768");
                add_selector_control_default("RESCUE_NVFP4_POLICY_BUDGET", "12");
                add_selector_control_default("RESCUE_NVFP4_REFINE_TOP", "2");
                add_selector_control_default("RESCUE_NVFP4_GUARD_TOP", "1");
                add_selector_control_default("CANDIDATE_REPORT_TOP", "40");
                break;
            case quantize_work_mode::DEEP:
                add_selector_controls_default("LLAMA_SELECTOR_TENSOR_POLICY_PROXY_ABS_GAIN", "0.0001");
                add_selector_control_default("RSF_DEPTH", "exhaustive");
                add_selector_controls_default("LLAMA_NVFP4_ENCODER_MAX_BLOCKS", "65536");
                add_selector_control_default("STAGEA_SAMPLE_BLOCKS", "16384");
                add_selector_control_default("STAGEA_MAX_POLICIES", "0");
                add_selector_control_default("AWQ_TOP", "4");
                add_selector_control_default("REFINE_TOP", "24");
                add_selector_control_default("REFINE_BUDGET", "192");
                add_selector_control_default("SURVEY_TOP", "64");
                add_selector_control_default("SURVEY_SAMPLE_BLOCKS", "16384");
                add_selector_control_default("EVAL_TOP", "16");
                add_selector_control_default("RANK_KLD_PENALTY", "4.0");
                add_selector_control_default("RANK_P99_PENALTY", "3.0");
                add_selector_control_default("RANK_P999_PENALTY", "1.5");
                add_selector_control_default("RANK_MAX_KLD_PENALTY", "0.35");
                add_selector_control_default("CANDIDATE_TOP", "88");
                add_selector_control_default("CANDIDATE_NVFP4_TOP", "22");
                add_selector_control_default("CANDIDATE_SAMPLE_BLOCKS", "8192");
                add_selector_controls_default("LLAMA_SELECTOR_TYPE_CANDIDATE_OUTLIER_RATIO", "1.03");
                add_selector_controls_default("LLAMA_SELECTOR_TYPE_CANDIDATE_MIN_ERROR", "0.010");
                add_selector_control_default("RESCUE_NVFP4_MAX_BLOCKS", "32768");
                add_selector_control_default("RESCUE_NVFP4_REFINE_MAX_BLOCKS", "32768");
                add_selector_control_default("RESCUE_NVFP4_GUARD_MAX_BLOCKS", "65536");
                add_selector_control_default("RESCUE_NVFP4_POLICY_BUDGET", "12");
                add_selector_control_default("RESCUE_NVFP4_REFINE_TOP", "2");
                add_selector_control_default("RESCUE_NVFP4_GUARD_TOP", "1");
                add_selector_control_default("CANDIDATE_REPORT_TOP", "64");
                break;
            case quantize_work_mode::UNSET:
                break;
        }
        fprintf(stderr, "llama_quantize: quantizer mode=%s selector defaults\n", quantize_work_mode_name(mode));
    };
    add_work_mode_defaults(cli_work_mode);
    if (!selector_candidate_types_cli.empty()) {
        add_selector_control("CANDIDATE_TYPES", selector_candidate_types_cli.c_str());
    }
    if (!selector_include_policies_cli.empty()) {
        add_selector_control("INCLUDE_POLICIES", selector_include_policies_cli.c_str());
    }
    if (!selector_skip_policies_cli.empty()) {
        add_selector_control("SKIP_POLICIES", selector_skip_policies_cli.c_str());
    }
    selector_set_skip_state(selector_skip_file, selector_skip_remaining);
    std::vector<std::string> imatrix_datasets;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    std::vector<llama_model_imatrix_data> imatrix_entries;
    std::vector<int32_t> prune_layers_param;
    int m_last_call = prepare_imatrix(imatrix_file, imatrix_datasets, included_weights, excluded_weights, imatrix_data);

    std::vector<llama_model_imatrix_data> i_data;
    std::vector<llama_model_tensor_override> t_override;
    if (!imatrix_data.empty()) {
        imatrix_entries.reserve(imatrix_data.size() + 1);
        for (const auto & entry : imatrix_data) {
            imatrix_entries.push_back({
                entry.first.c_str(),
                entry.second.data(),
                entry.second.size(),
            });
        }
        imatrix_entries.push_back({ nullptr, nullptr, 0 });
        params.imatrix = imatrix_entries.data();
        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_FILE);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            strncpy(kvo.val_str, imatrix_file.c_str(), 127);
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }
        if (!imatrix_datasets.empty()) {
            llama_model_kv_override kvo;
            // TODO: list multiple datasets when there are more than one
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_DATASET);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            strncpy(kvo.val_str, imatrix_datasets[0].c_str(), 127);
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }
        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.val_i64 = imatrix_data.size();
            kv_overrides.emplace_back(std::move(kvo));
        }
        if (m_last_call > 0) {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.val_i64 = m_last_call;
            kv_overrides.emplace_back(std::move(kvo));
        }
    }
    if (!kv_overrides.empty()) {
        kv_overrides.emplace_back();
        kv_overrides.back().key[0] = 0;
        params.kv_overrides = kv_overrides.data();
    }
    if (!tensor_type_opts.empty()) {
        t_override.reserve(tensor_type_opts.size() + 1);
        for (const auto & tt : tensor_type_opts) {
            t_override.push_back({tt.name.c_str(), tt.type});
        }
        t_override.push_back({nullptr, GGML_TYPE_COUNT});  // array terminator
        params.tt_overrides = t_override.data();
        params.tensor_types = &tensor_type_opts;
    }
    if (!prune_layers.empty()) {
        prune_layers_param.assign(prune_layers.begin(), prune_layers.end());
        prune_layers_param.push_back(-1);
        params.prune_layers = prune_layers_param.data();
    }
    if (!patch_base_model.empty()) {
        params.patch_base_model = patch_base_model.c_str();
    }
    if (!assignment_jsonl.empty()) {
        params.assignment_jsonl = assignment_jsonl.c_str();
    }
    llama_backend_init();

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    std::string suffix = ".gguf";
    if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
        // argv[arg_idx] is the ftype directly: <input> <ftype>
        if (!params.dry_run) {
            std::string fpath;
            const size_t pos = fname_inp.find_last_of("/\\");
            if (pos != std::string::npos) {
                fpath = fname_inp.substr(0, pos + 1);
            }

            // export as [inp path]/ggml-model-[ftype]. Only add extension if there is no splitting
            fname_out = fpath + "ggml-model-" + ftype_str;
            if (!params.keep_split) {
                fname_out += suffix;
            }
        }
        arg_idx++;
        if (ftype_str == "COPY") {
            params.only_copy = true;
        }
    } else {
        // argv[arg_idx] is not a valid ftype, so treat it as output path: <input> <output> <ftype>
        fname_out = argv[arg_idx];
        if (params.keep_split && fname_out.find(suffix) != std::string::npos) {
            fname_out = fname_out.substr(0, fname_out.length() - suffix.length());
        }
        arg_idx++;

        if (argc <= arg_idx) {
            fprintf(stderr, "%s: missing ftype\n", __func__);
            return 1;
        }
        if (!try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
            fprintf(stderr, "%s: invalid ftype '%s'\n", __func__, argv[arg_idx]);
            return 1;
        }
        if (ftype_str == "COPY") {
           params.only_copy = true;
        }
        arg_idx++;
    }

    const bool ftype_mixed_nvfp4_mxfp6 = ftype_is_mixed_nvfp4_mxfp6(ftype_str);
    if (ftype_mixed_nvfp4_mxfp6) {
        params.ftype = LLAMA_FTYPE_MOSTLY_NVFP4;
        if (params.nv4mx6_policy == LLAMA_NV4MX6_POLICY_OFF) {
            params.nv4mx6_policy = LLAMA_NV4MX6_POLICY_AUTO;
        }
    }
    if (selector_control_args_has("KLD_FILE") &&
            params.ftype == LLAMA_FTYPE_MOSTLY_NVFP4 &&
            !selector_control_args_has("CANDIDATE_TYPES")) {
        add_selector_control_default(
            "CANDIDATE_TYPES",
            ftype_mixed_nvfp4_mxfp6 ? "MXFP6_E2M3,Q5_0,Q4_K,Q5_K,Q6_K,Q8_0" : "Q5_0,Q4_K,Q5_K,Q6_K,Q8_0");
    }
    if (ftype_str == "Q2_K_RSF" ||
            ftype_str == "Q3_K_RSF" || ftype_str == "Q3_K_M_RSF" ||
            ftype_str == "Q4_K_RSF" || ftype_str == "Q4_K_RSF_AWQ" || ftype_str == "Q4_K_RSF_SQ" ||
            ftype_str == "Q5_K_RSF" || ftype_str == "Q5_K_M_RSF" ||
            ftype_str == "Q6_K_RSF") {
        params.k_quant_rsf = true;
        params.k_quant_rsf_mode =
            ftype_str == "Q4_K_RSF_AWQ" ? 2 :
            ftype_str == "Q4_K_RSF_SQ"  ? 3 : 1;
    }

    // parse nthreads
    if (argc > arg_idx) {
        try {
            params.nthread = std::stoi(argv[arg_idx]);
        }
        catch (const std::exception & e) {
            fprintf(stderr, "%s: invalid nthread '%s' (%s)\n", __func__, argv[arg_idx], e.what());
            return 1;
        }
    }

    if (params.ftype == LLAMA_FTYPE_MOSTLY_NVFP4 && params.nv4mx6_policy == LLAMA_NV4MX6_POLICY_OFF) {
        if (params.token_embedding_type >= GGML_TYPE_COUNT) {
            params.token_embedding_type = GGML_TYPE_NVFP4;
        }
        if (params.output_tensor_type >= GGML_TYPE_COUNT) {
            params.output_tensor_type = GGML_TYPE_Q6_K;
        }
        if (params.mtp_tensor_type >= GGML_TYPE_COUNT) {
            params.mtp_tensor_type = GGML_TYPE_NVFP4;
        }
    }

    if (params.ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3 || ftype_mixed_nvfp4_mxfp6) {
        if (params.token_embedding_type >= GGML_TYPE_COUNT) {
            params.token_embedding_type = GGML_TYPE_MXFP6_E2M3;
        }
        if (params.output_tensor_type >= GGML_TYPE_COUNT) {
            params.output_tensor_type = GGML_TYPE_MXFP6_E2M3;
        }
        if (params.mtp_tensor_type >= GGML_TYPE_COUNT) {
            params.mtp_tensor_type = ftype_mixed_nvfp4_mxfp6 ? GGML_TYPE_NVFP4 : GGML_TYPE_MXFP6_E2M3;
        }
    }

    const bool has_selector_kld =
        selector_control_args_has("KLD_FILE");
    if (params.ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3 && has_selector_kld) {
        // Pure MXFP6 has no NVFP4 policy to choose. Keep the measured-eval selector
        // alive, but spend the KLD budget on MXFP6 tensor-scale refinement.
        if (!selector_control_args_has("EVAL_TOP")) {
            selector_controls.emplace_back(selector_legacy_control_key("EVAL_TOP"), "1");
        }
        if (!selector_control_args_has("STAGEA_MAX_POLICIES")) {
            selector_controls.emplace_back(selector_legacy_control_key("STAGEA_MAX_POLICIES"), "1");
        }
        if (!selector_control_args_has("SURVEY_TOP")) {
            selector_controls.emplace_back(selector_legacy_control_key("SURVEY_TOP"), "1");
        }
    }

    std::vector<std::unique_ptr<scoped_quantize_control>> selector_controls_scope;
    selector_controls_scope.reserve(selector_controls.size());
    for (const auto & entry : selector_controls) {
        selector_controls_scope.emplace_back(std::make_unique<scoped_quantize_control>(entry.first.c_str(), entry.second.c_str()));
    }

    params.nvfp4_encoder_max_blocks =
        std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_ENCODER_MAX_BLOCKS", params.nvfp4_encoder_max_blocks));
    if (cli_nvfp4_cfg_valid) {
        const int rsf_mode = selector_effective_rsf_mode(&cli_nvfp4_cfg);
        const int rsf_depth = selector_effective_rsf_depth(&cli_nvfp4_cfg);
        cli_nvfp4_cfg = selector_cfg_as_rsf(cli_nvfp4_cfg, rsf_mode, rsf_depth);
        cli_nvfp4_policy_name = selector_rsf_base_policy_name(cli_nvfp4_policy_name.empty() ? "cli" : cli_nvfp4_policy_name);
    }
    nvfp4_set_autotune_threads(cli_nvfp4_encoder_threads);
    selector_set_kld_threads_override(selector_kld_threads_override);

    if (!params.dry_run) {
        if (std::error_code ec; std::filesystem::equivalent(fname_inp, fname_out, ec)) {
            fprintf(stderr, "%s: error: input and output files are the same: '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }
    }

    if (selector_control_args_has("KLD_FILE") &&
            params.ftype == LLAMA_FTYPE_MOSTLY_NVFP4 &&
            !params.dry_run &&
            !params.keep_split &&
            !params.only_copy) {
        std::vector<tensor_type_option> stage1_priors =
            selector_stage1_type_priors(fname_inp, imatrix_data, cli_work_mode, tensor_type_opts);
        if (!stage1_priors.empty()) {
            tensor_type_opts.insert(tensor_type_opts.end(), stage1_priors.begin(), stage1_priors.end());
            params.tensor_types = &tensor_type_opts;
        }
    }

    const std::vector<tensor_type_option> tensor_type_opts_cli = tensor_type_opts;
    const std::string selector_kld_path = selector_control_string("KLD_FILE");
    const bool selector_bootstrap = selector_control_has("BOOTSTRAP");
    const bool selector_auto_rescue_running = quantize_control_has("LLAMA_NVFP4_AUTO_RESCUE_RUNNING");
    const bool selector_only = selector_control_i64("ONLY", 0) != 0;
    bool selector_cfg_valid = false;
    nvfp4_cuda_runtime_cfg selector_cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    std::string selector_policy_name;
    std::string selector_checkpoint_path;
    bool selector_checkpoint_created = false;
    bool selector_inputs_ready = false;
    selector_kld_subset selector_kld;
    std::unique_ptr<selector_kld_subset> selector_kld_holdout;
    selector_rank_config selector_rank_cfg;
    bool selector_applied_exact_overrides = false;
    size_t selector_applied_exact_override_count = 0;

    const bool selector_enabled =
        !selector_bootstrap &&
        !selector_auto_rescue_running &&
        !selector_kld_path.empty() &&
        (params.ftype == LLAMA_FTYPE_MOSTLY_NVFP4 || params.ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3) &&
        !params.dry_run &&
        !params.keep_split &&
        !params.only_copy;
    const bool selector_auto_rescue =
        selector_enabled &&
        !selector_only &&
        quantize_control_i64("LLAMA_NVFP4_AUTO_RESCUE", 0) != 0;

    llama_print_build_info();

    if (params.dry_run) {
        fprintf(stderr, "%s: calculating quantization size for '%s' as %s", __func__, fname_inp.c_str(), ftype_str.c_str());
    } else {
        fprintf(stderr, "%s: quantizing '%s' to '%s' as %s", __func__, fname_inp.c_str(), fname_out.c_str(), ftype_str.c_str());
    }
    if (!patch_base_model.empty()) {
        fprintf(stderr, " (patch-base=%s)", patch_base_model.c_str());
    }

    if (params.nthread > 0) {
        fprintf(stderr, " using %d threads", params.nthread);
    }
    fprintf(stderr, "\n");

	    auto run_selector_pass = [&](const std::string & seed_path,
	                                 nvfp4_cuda_runtime_cfg & out_cfg,
	                                 std::string & out_policy_name,
                                     bool * out_kept_seed,
	                                 std::vector<tensor_type_option> * out_overrides) -> bool {
        if (!selector_inputs_ready || seed_path.empty()) {
            return false;
        }
        if (out_overrides != nullptr) {
            out_overrides->clear();
        }
        const int selector_nthread = params.nthread > 0 ?
            params.nthread :
            std::max(1, (int) std::thread::hardware_concurrency());
        return selector_choose_policy(
            fname_inp,
            seed_path,
            imatrix_data,
            selector_nthread,
            selector_kld,
            selector_kld_holdout.get(),
            selector_rank_cfg,
            cli_nvfp4_cfg_valid ? &cli_nvfp4_cfg : nullptr,
            cli_nvfp4_policy_name,
            params.nvfp4_input_scale_policy,
            selector_eval_batch_override,
            cli_nvfp4_encoder_threads,
            out_cfg,
            out_policy_name,
            out_kept_seed,
            out_overrides);
	    };

    if (selector_enabled) {
        selector_rank_cfg.kld_threshold = selector_control_f64("RANK_KLD_THRESHOLD", -1.0);
        selector_rank_cfg.p99_threshold = selector_control_f64("RANK_P99_THRESHOLD", -1.0);
        selector_rank_cfg.p999_threshold = selector_control_f64("RANK_P999_THRESHOLD", -1.0);
        selector_rank_cfg.max_kld_threshold = selector_control_f64("RANK_MAX_KLD_THRESHOLD", -1.0);
        selector_rank_cfg.kld_penalty = selector_control_f64("RANK_KLD_PENALTY", 4.0);
        selector_rank_cfg.p99_penalty = selector_control_f64("RANK_P99_PENALTY", 1.5);
        selector_rank_cfg.p999_penalty = selector_control_f64("RANK_P999_PENALTY", 0.75);
        selector_rank_cfg.max_kld_penalty = selector_control_f64("RANK_MAX_KLD_PENALTY", 0.10);
        selector_rank_cfg.holdout_weight = SELECTOR_HOLDOUT_WEIGHT;
        selector_rank_cfg.kld_hard_gate = selector_control_i64("RANK_KLD_HARD_GATE", 0) != 0;
        selector_rank_cfg.p99_hard_gate = selector_control_i64("RANK_P99_HARD_GATE", 0) != 0;
        selector_rank_cfg.p999_hard_gate = selector_control_i64("RANK_P999_HARD_GATE", 0) != 0;
        selector_rank_cfg.max_kld_hard_gate = selector_control_i64("RANK_MAX_KLD_HARD_GATE", 0) != 0;
        selector_rank_cfg.ppl_sigma = SELECTOR_PPL_SIGMA;
        selector_rank_cfg.kld_sigma = SELECTOR_KLD_SIGMA;
        selector_rank_cfg.rms_dp_sigma = SELECTOR_RMS_DP_SIGMA;
        selector_rank_cfg.same_top_sigma = SELECTOR_SAME_TOP_SIGMA;
        selector_rank_cfg.ln_ratio_abs_floor = SELECTOR_LN_RATIO_ABS_FLOOR;
        selector_rank_cfg.mean_kld_abs_floor = SELECTOR_MEAN_KLD_ABS_FLOOR;
        selector_rank_cfg.rms_dp_abs_floor = SELECTOR_RMS_DP_ABS_FLOOR;
        selector_rank_cfg.same_top_abs_floor = SELECTOR_SAME_TOP_ABS_FLOOR;
        selector_rank_cfg.entropy_rmse_abs_floor = SELECTOR_ENTROPY_RMSE_ABS_FLOOR;
        selector_rank_cfg.top_prob_rmse_abs_floor = SELECTOR_TOP_PROB_RMSE_ABS_FLOOR;
        selector_rank_cfg.top_flip_weight_abs_floor = SELECTOR_TOP_FLIP_WEIGHT_ABS_FLOOR;
        selector_rank_cfg.p99_abs_margin = SELECTOR_P99_ABS_MARGIN;
        selector_rank_cfg.p99_rel_margin = SELECTOR_P99_REL_MARGIN;
        selector_rank_cfg.p999_abs_margin = SELECTOR_P999_ABS_MARGIN;
        selector_rank_cfg.p999_rel_margin = SELECTOR_P999_REL_MARGIN;
        selector_rank_cfg.max_kld_abs_margin = SELECTOR_MAX_KLD_ABS_MARGIN;
        selector_rank_cfg.max_kld_rel_margin = SELECTOR_MAX_KLD_REL_MARGIN;
        const bool selector_mxfp6_possible =
            params.ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3 ||
            params.nv4mx6_policy != LLAMA_NV4MX6_POLICY_OFF ||
            std::any_of(tensor_type_opts.begin(), tensor_type_opts.end(), [](const tensor_type_option & opt) {
                return opt.type == GGML_TYPE_MXFP6_E2M3;
            });
        if (selector_mxfp6_possible) {
            fprintf(stderr,
                "%s: selector rank penalties: mean=%.6g p99=%.6g p999=%.6g max=%.6g thresholds={mean=%.6g p99=%.6g p999=%.6g max=%.6g} mxfp6_tol={ppl_abs=%.6g ppl_rel=%.6g mean_kld_abs=%.6g mean_kld_rel=%.6g}\n",
                __func__,
                selector_rank_cfg.kld_penalty,
                selector_rank_cfg.p99_penalty,
                selector_rank_cfg.p999_penalty,
                selector_rank_cfg.max_kld_penalty,
                selector_rank_cfg.kld_threshold,
                selector_rank_cfg.p99_threshold,
                selector_rank_cfg.p999_threshold,
                selector_rank_cfg.max_kld_threshold,
                selector_rank_cfg.mxfp6.ppl_abs_tol,
                selector_rank_cfg.mxfp6.ppl_rel_tol,
                selector_rank_cfg.mxfp6.mean_kld_abs_tol,
                selector_rank_cfg.mxfp6.mean_kld_rel_tol);
        } else {
            fprintf(stderr,
                "%s: selector rank penalties: mean=%.6g p99=%.6g p999=%.6g max=%.6g thresholds={mean=%.6g p99=%.6g p999=%.6g max=%.6g}\n",
                __func__,
                selector_rank_cfg.kld_penalty,
                selector_rank_cfg.p99_penalty,
                selector_rank_cfg.p999_penalty,
                selector_rank_cfg.max_kld_penalty,
                selector_rank_cfg.kld_threshold,
                selector_rank_cfg.p99_threshold,
                selector_rank_cfg.p999_threshold,
                selector_rank_cfg.max_kld_threshold);
        }
        if (!selector_load_kld_subset(selector_kld_path.c_str(), selector_kld)) {
            fprintf(stderr, "%s: selector disabled because full KLD base could not be loaded from %s\n", __func__, selector_kld_path.c_str());
        } else {
            selector_inputs_ready = true;
            std::string selector_checkpoint_override = selector_control_string("CHECKPOINT_MODEL");
            const bool selector_keep_checkpoint = selector_control_i64("KEEP_CHECKPOINT", 0) != 0;
            const std::string selector_cache_dir = selector_control_string("CACHE_DIR");
            if (!selector_checkpoint_override.empty()) {
                selector_checkpoint_path = selector_checkpoint_override;
                fprintf(stderr, "%s: selector using existing candidate search checkpoint %s\n", __func__, selector_checkpoint_path.c_str());
            } else {
                if (!selector_cache_dir.empty()) {
                    std::string stem = std::filesystem::path(fname_inp).stem().string();
                    for (char & ch : stem) {
                        if (!std::isalnum((unsigned char) ch) && ch != '-' && ch != '_') {
                            ch = '_';
                        }
                    }
                    if (stem.empty()) {
                        stem = "model";
                    }
                    uint64_t h = 1469598103934665603ull;
                    const std::string hash_input = fname_inp + "|" + ftype_str;
                    for (unsigned char ch : hash_input) {
                        h ^= (uint64_t) ch;
                        h *= 1099511628211ull;
                    }
                    std::ostringstream hss;
                    hss << std::hex << h;
                    selector_checkpoint_path = (std::filesystem::path(selector_cache_dir) /
                        (stem + "-" + ftype_str + "-" + hss.str() + ".bwq-checkpoint.gguf")).string();
                    std::error_code ec;
                    std::filesystem::create_directories(selector_cache_dir, ec);
                    if (ec) {
                        fprintf(stderr, "%s: failed to create selector checkpoint cache dir %s: %s\n",
                            __func__, selector_cache_dir.c_str(), ec.message().c_str());
                    }
                } else {
                    selector_checkpoint_path = fname_out + ".bwq-checkpoint.gguf";
                }

                const std::string selector_checkpoint_partial_path = selector_checkpoint_path + ".partial";
                bool reuse_cached_checkpoint = false;
                if (selector_keep_checkpoint && !selector_checkpoint_path.empty()) {
                    std::error_code ec;
                    const bool cached_exists = std::filesystem::exists(selector_checkpoint_path, ec) && !ec;
                    const uintmax_t cached_size = cached_exists ? std::filesystem::file_size(selector_checkpoint_path, ec) : 0;
                    const bool cached_complete = cached_exists && !ec && cached_size > 0 &&
                        agq_file_has_gguf_magic(selector_checkpoint_path);
                    reuse_cached_checkpoint = cached_complete;
                    if (reuse_cached_checkpoint) {
                        fprintf(stderr, "%s: selector reusing cached candidate search checkpoint %s\n",
                            __func__, selector_checkpoint_path.c_str());
                    } else if (cached_exists) {
                        fprintf(stderr, "%s: selector ignoring incomplete candidate search checkpoint %s\n",
                            __func__, selector_checkpoint_path.c_str());
                        const std::string direct_resume_path = selector_checkpoint_path + ".resume";
                        std::error_code resume_ec;
                        const bool has_direct_resume =
                            std::filesystem::exists(direct_resume_path, resume_ec) && !resume_ec;
                        if (has_direct_resume) {
                            std::filesystem::remove(selector_checkpoint_partial_path, ec);
                            std::filesystem::remove(selector_checkpoint_partial_path + ".resume", ec);
                            ec.clear();
                            std::filesystem::rename(selector_checkpoint_path, selector_checkpoint_partial_path, ec);
                            if (!ec) {
                                std::filesystem::rename(direct_resume_path, selector_checkpoint_partial_path + ".resume", ec);
                            }
                            if (ec) {
                                fprintf(stderr, "%s: failed to migrate incomplete checkpoint for resume: %s\n",
                                    __func__, ec.message().c_str());
                                std::filesystem::remove(selector_checkpoint_path, ec);
                                std::filesystem::remove(direct_resume_path, ec);
                            } else {
                                fprintf(stderr, "%s: selector migrated incomplete checkpoint to resumable partial %s\n",
                                    __func__, selector_checkpoint_partial_path.c_str());
                            }
                        } else {
                            std::filesystem::remove(selector_checkpoint_path, ec);
                        }
                        std::filesystem::remove(selector_checkpoint_path + ".stageb-results.jsonl", ec);
                        std::filesystem::remove(selector_checkpoint_path + ".rsf-report.txt", ec);
                    }
                }

                if (!reuse_cached_checkpoint) {
                    std::error_code partial_ec;
                    const bool has_partial_resume =
                        std::filesystem::exists(selector_checkpoint_partial_path + ".resume", partial_ec) && !partial_ec;
                    if (has_partial_resume) {
                        fprintf(stderr, "%s: candidate search checkpoint will resume partial output %s\n",
                            __func__, selector_checkpoint_partial_path.c_str());
                    }
	                    fprintf(stderr, "%s: candidate search checkpoint quantization -> %s\n", __func__, selector_checkpoint_path.c_str());
	                    {
	                        const std::string bootstrap_key = selector_generic_control_key("BOOTSTRAP");
	                        scoped_quantize_control bootstrap(bootstrap_key.c_str(), "1");
	                        scoped_quantize_control resumable_output("LLAMA_QUANTIZE_RESUME_OUTPUT", "1");
	                        nvfp4_clear_runtime_cfg();
                        if (llama_model_quantize(fname_inp.c_str(), selector_checkpoint_partial_path.c_str(), &params)) {
                            fprintf(stderr, "%s: candidate search checkpoint build failed, continuing without selector\n", __func__);
                            selector_checkpoint_path.clear();
                            selector_checkpoint_created = false;
                        } else if (!agq_file_has_gguf_magic(selector_checkpoint_partial_path)) {
                            fprintf(stderr, "%s: candidate search checkpoint build produced an incomplete GGUF, continuing without selector\n", __func__);
                            std::error_code ec;
                            std::filesystem::remove(selector_checkpoint_partial_path, ec);
                            std::filesystem::remove(selector_checkpoint_partial_path + ".resume", ec);
                            selector_checkpoint_path.clear();
                            selector_checkpoint_created = false;
                        } else {
                            std::error_code ec;
                            std::filesystem::rename(selector_checkpoint_partial_path, selector_checkpoint_path, ec);
                            if (ec) {
                                fprintf(stderr, "%s: failed to publish candidate search checkpoint %s: %s\n",
                                    __func__, selector_checkpoint_path.c_str(), ec.message().c_str());
                                std::filesystem::remove(selector_checkpoint_partial_path, ec);
                                selector_checkpoint_path.clear();
                                selector_checkpoint_created = false;
                            } else {
                                selector_checkpoint_created = true;
                            }
                        }
                    }
                }
            }

	            std::vector<tensor_type_option> selector_overrides;
                bool selector_chose_seed_keep = false;
	            if (run_selector_pass(selector_checkpoint_path, selector_cfg, selector_policy_name, &selector_chose_seed_keep, &selector_overrides)) {
	                nvfp4_set_runtime_cfg(&selector_cfg);
	                selector_cfg_valid = true;
	                fprintf(stderr,
                    "%s: selector chose policy=%s cfg={choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f}\n",
                    __func__,
                    selector_policy_name.c_str(),
                    selector_cfg.choose46_mode,
                    selector_cfg.refit_iters,
                    selector_cfg.use_compand_sat,
                    (double) selector_cfg.cap_m6,
                    (double) selector_cfg.cap_m4);
                if (!selector_overrides.empty()) {
                    selector_applied_exact_overrides = true;
                    selector_applied_exact_override_count = selector_overrides.size();
                    tensor_type_opts = tensor_type_opts_cli;
                    tensor_type_opts.insert(tensor_type_opts.end(), selector_overrides.begin(), selector_overrides.end());
		                    params.tensor_types = &tensor_type_opts;
		                    fprintf(stderr, "%s: selector added %zu exact tensor plan entries\n", __func__, selector_overrides.size());
		                }
                    if (selector_chose_seed_keep && selector_overrides.empty() && !patch_base_model.empty()) {
                        std::error_code ec;
                        const bool seed_is_patch_base =
                            selector_checkpoint_path == patch_base_model ||
                            std::filesystem::equivalent(selector_checkpoint_path, patch_base_model, ec);
                        if (seed_is_patch_base) {
                            fprintf(stderr,
                                "%s: selector chose seed_keep; patch-base already contains the measured winner, leaving output unwritten: %s\n",
                                __func__,
                                patch_base_model.c_str());
                            if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
                                    selector_control_i64("KEEP_CHECKPOINT", 0) == 0) {
                                std::error_code remove_ec;
                                std::filesystem::remove(selector_checkpoint_path, remove_ec);
                            }
                            nvfp4_clear_runtime_cfg();
                            llama_backend_free();
                            return 0;
                        }
                    }
                    if (selector_chose_seed_keep && !selector_checkpoint_path.empty()) {
                        patch_base_model = selector_checkpoint_path;
                        params.patch_base_model = patch_base_model.c_str();
                        fprintf(stderr,
                            "%s: selector chose seed_keep; final output will patch unchanged tensors from measured checkpoint %s\n",
                            __func__,
                            patch_base_model.c_str());
                    }
	            } else if (!selector_checkpoint_path.empty()) {
                fprintf(stderr, "%s: selector failed, continuing with the built-in CUDA policy search\n", __func__);
                if (selector_control_i64("REQUIRE_RUNTIME_CACHE", 0) != 0) {
                    fprintf(stderr, "%s: selector runtime cache is required; aborting quantization\n", __func__);
                    if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
                            selector_control_i64("KEEP_CHECKPOINT", 0) == 0) {
                        std::error_code ec;
                        std::filesystem::remove(selector_checkpoint_path, ec);
                    }
                    llama_backend_free();
                    return 1;
                }
            }
        }
    }

    if (selector_only) {
        if (!selector_enabled) {
            fprintf(stderr, "%s: selector-only requested but selector is not enabled for this invocation\n", __func__);
            llama_backend_free();
            return 1;
        }
        if (!selector_cfg_valid) {
            fprintf(stderr, "%s: selector-only requested but no selector policy was chosen\n", __func__);
            if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
                    selector_control_i64("KEEP_CHECKPOINT", 0) == 0) {
                std::error_code ec;
                std::filesystem::remove(selector_checkpoint_path, ec);
            }
            llama_backend_free();
            return 1;
        }
        if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
                selector_control_i64("KEEP_CHECKPOINT", 0) == 0) {
            std::error_code ec;
            std::filesystem::remove(selector_checkpoint_path, ec);
        }
        nvfp4_clear_runtime_cfg();
        llama_backend_free();
        return 0;
    }

    if (!selector_cfg_valid && cli_nvfp4_cfg_valid) {
        nvfp4_set_runtime_cfg(&cli_nvfp4_cfg);
        selector_cfg_valid = true;
        fprintf(stderr,
            "%s: using CLI NVFP4 policy=%s cfg={choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f}\n",
            __func__,
            cli_nvfp4_policy_name.empty() ? "cli" : cli_nvfp4_policy_name.c_str(),
            cli_nvfp4_cfg.choose46_mode,
            cli_nvfp4_cfg.refit_iters,
            cli_nvfp4_cfg.use_compand_sat,
            (double) cli_nvfp4_cfg.cap_m6,
            (double) cli_nvfp4_cfg.cap_m4);
    }

    const int64_t t_main_start_us = llama_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = llama_time_us();

        {
            scoped_quantize_control resumable_output("LLAMA_QUANTIZE_RESUME_OUTPUT", "1");
            if (llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params)) {
                if (selector_cfg_valid) {
                    nvfp4_clear_runtime_cfg();
                }
                fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
                return 1;
            }
        }

        t_quantize_us = llama_time_us() - t_start_us;
    }

    if (selector_auto_rescue && selector_inputs_ready && selector_applied_exact_overrides) {
        fprintf(stderr,
            "%s: selector rescue already applied %zu exact tensor plan entries during final quantization; skipping post-write rescue analysis\n",
            __func__,
            selector_applied_exact_override_count);
    } else if (selector_auto_rescue && selector_inputs_ready) {
        nvfp4_cuda_runtime_cfg rescue_cfg = selector_cfg;
        std::string rescue_policy_name;
        std::vector<tensor_type_option> rescue_overrides;

        nvfp4_clear_runtime_cfg();
        llama_backend_free();
        llama_backend_init();

        fprintf(stderr, "%s: selector rescue pass inspecting completed model %s\n", __func__, fname_out.c_str());
        if (run_selector_pass(fname_out, rescue_cfg, rescue_policy_name, nullptr, &rescue_overrides)) {
            if (!rescue_overrides.empty()) {
                const std::string rescue_out = fname_out + ".rescue.tmp.gguf";
                std::vector<tensor_type_option> rescue_tensor_types = tensor_type_opts_cli;
                rescue_tensor_types.insert(rescue_tensor_types.end(), rescue_overrides.begin(), rescue_overrides.end());

                llama_model_quantize_params rescue_params = params;
                rescue_params.tensor_types = &rescue_tensor_types;
                rescue_params.patch_base_model = fname_out.c_str();

                fprintf(stderr,
                    "%s: selector rescue chose policy=%s and %zu exact tensor edit(s); patching via %s\n",
                    __func__,
                    rescue_policy_name.c_str(),
                    rescue_overrides.size(),
                    fname_out.c_str());

                {
                    std::error_code ec;
                    std::filesystem::remove(rescue_out, ec);
                }

                bool rescue_ok = false;
                {
                    scoped_quantize_control rescue_running("LLAMA_NVFP4_AUTO_RESCUE_RUNNING", "1");
                    scoped_quantize_control resumable_output("LLAMA_QUANTIZE_RESUME_OUTPUT", "1");
                    nvfp4_clear_runtime_cfg();
                    rescue_ok = llama_model_quantize(fname_inp.c_str(), rescue_out.c_str(), &rescue_params) == 0;
                }

                if (rescue_ok) {
                    std::error_code ec;
                    std::filesystem::rename(rescue_out, fname_out, ec);
                    if (ec) {
                        fprintf(stderr, "%s: selector rescue produced %s but failed to replace %s: %s\n",
                            __func__, rescue_out.c_str(), fname_out.c_str(), ec.message().c_str());
                    } else {
                        fprintf(stderr, "%s: selector rescue patched final output in-place\n", __func__);
                    }
                } else {
                    std::error_code ec;
                    std::filesystem::remove(rescue_out, ec);
                    fprintf(stderr, "%s: selector rescue patching failed; keeping original output\n", __func__);
                }

            } else {
                fprintf(stderr, "%s: selector rescue found no exact tensor overrides to patch\n", __func__);
            }
        } else {
            fprintf(stderr, "%s: selector rescue analysis failed; keeping original output\n", __func__);
        }

        if (selector_cfg_valid) {
            nvfp4_set_runtime_cfg(&selector_cfg);
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = llama_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    if (selector_cfg_valid) {
        nvfp4_clear_runtime_cfg();
    }
    if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
            selector_control_i64("KEEP_CHECKPOINT", 0) == 0) {
        std::error_code ec;
        std::filesystem::remove(selector_checkpoint_path, ec);
    } else if (selector_checkpoint_created && !selector_checkpoint_path.empty()) {
        fprintf(stderr, "%s: selector kept candidate search checkpoint %s\n", __func__, selector_checkpoint_path.c_str());
    }

    llama_backend_free();

    return 0;
}
