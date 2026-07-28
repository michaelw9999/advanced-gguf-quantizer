#include "llama-impl.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-quant.h"
#include "llama-quant-types.h"
#include "../ggml/include/ggml-cuda.h"
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cinttypes>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <thread>
#include <unordered_map>

static bool llama_nvfp4_ascii_iequals(const std::string & a, const char * b) {
    size_t i = 0;
    for (; i < a.size() && b[i] != '\0'; ++i) {
        if (std::tolower((unsigned char) a[i]) != std::tolower((unsigned char) b[i])) {
            return false;
        }
    }
    return i == a.size() && b[i] == '\0';
}

const std::vector<llama_nvfp4_named_preset> & llama_nvfp4_preset_catalog() {
    static const std::vector<llama_nvfp4_named_preset> presets = {
#define LLAMA_NVFP4_PRESET(name, choose46, refit, compand, cap6, cap4) \
        { name, { choose46, refit, compand, 0, cap6, cap4 } },
        GGML_NVFP4_CUDA_PRESET_LIST(LLAMA_NVFP4_PRESET)
#undef LLAMA_NVFP4_PRESET
    };
    return presets;
}

const llama_nvfp4_named_preset * llama_nvfp4_find_preset(const std::string & name) {
    const char * canonical = nullptr;
    if (llama_nvfp4_ascii_iequals(name, "awq_tail")) {
        canonical = "asym_tail";
    } else if (llama_nvfp4_ascii_iequals(name, "awq_tail_strict")) {
        canonical = "asym_tail_strict";
    } else if (llama_nvfp4_ascii_iequals(name, "awq_tail_relaxed")) {
        canonical = "asym_tail_relaxed";
    } else if (llama_nvfp4_ascii_iequals(name, "awq_tail_moe")) {
        canonical = "asym_tail_moe";
    } else if (llama_nvfp4_ascii_iequals(name, "awq_tail_moe_refit4")) {
        canonical = "asym_tail_moe_refit4";
    }

    for (const llama_nvfp4_named_preset & preset : llama_nvfp4_preset_catalog()) {
        if (llama_nvfp4_ascii_iequals(name, preset.name) ||
                (canonical != nullptr && std::strcmp(canonical, preset.name) == 0)) {
            return &preset;
        }
    }
    return nullptr;
}

std::string llama_nvfp4_scale_tensor_name(const std::string & weight_name) {
    static const std::string suffix = ".weight";
    if (weight_name.size() >= suffix.size() &&
            weight_name.compare(weight_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return weight_name.substr(0, weight_name.size() - suffix.size()) + ".scale";
    }
    return weight_name + ".scale";
}

std::string llama_nvfp4_input_scale_tensor_name(const std::string & weight_name) {
    static const std::string weight_suffix = ".weight";
    static const std::string canonical_suffix = ".input_scale";
    static const std::string legacy_suffix = ".weight.input_scale";

    if (weight_name.size() >= canonical_suffix.size() &&
            weight_name.compare(weight_name.size() - canonical_suffix.size(), canonical_suffix.size(), canonical_suffix) == 0) {
        if (weight_name.size() >= legacy_suffix.size() &&
                weight_name.compare(weight_name.size() - legacy_suffix.size(), legacy_suffix.size(), legacy_suffix) == 0) {
            return weight_name.substr(0, weight_name.size() - legacy_suffix.size()) + canonical_suffix;
        }
        return weight_name;
    }

    if (weight_name.size() >= weight_suffix.size() &&
            weight_name.compare(weight_name.size() - weight_suffix.size(), weight_suffix.size(), weight_suffix) == 0) {
        return weight_name.substr(0, weight_name.size() - weight_suffix.size()) + canonical_suffix;
    }

    return weight_name + canonical_suffix;
}

float llama_nvfp4_input_scale_from_imatrix(
        const float * imatrix,
        int64_t n_per_row,
        int32_t policy) {
    if (policy == LLAMA_NVFP4_INPUT_SCALE_IDENTITY) {
        return 1.0f;
    }
    if (imatrix == nullptr || n_per_row <= 0) {
        return 1.0f;
    }

    double sum = 0.0;
    size_t count = 0;
    double max_sqrt = 0.0;
    std::vector<float> sqrt_values;
    const bool need_quantile =
        policy == LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTP99 ||
        policy == LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTP999;
    if (need_quantile) {
        sqrt_values.reserve((size_t) n_per_row);
    }

    for (int64_t i = 0; i < n_per_row; ++i) {
        const float v = imatrix[i];
        if (!std::isfinite(v) || v <= 0.0f) {
            continue;
        }
        sum += (double) v;
        ++count;
        const double sv = std::sqrt((double) v);
        max_sqrt = std::max(max_sqrt, sv);
        if (need_quantile) {
            sqrt_values.push_back((float) sv);
        }
    }

    if (count == 0 || sum <= 0.0) {
        return 1.0f;
    }

    const auto file_input_scale_from_proxy = [](double proxy) {
        if (!(proxy > 0.0) || !std::isfinite(proxy)) {
            return 1.0f;
        }
        return (float) std::clamp(proxy, 1.0 / 32.0, 32.0);
    };

    if (policy == LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTMAX) {
        return file_input_scale_from_proxy(max_sqrt);
    }

    if (need_quantile && !sqrt_values.empty()) {
        const double q = policy == LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTP999 ? 0.999 : 0.99;
        const size_t idx = std::min(
            sqrt_values.size() - 1,
            (size_t) std::llround(q * (double) (sqrt_values.size() - 1)));
        std::nth_element(sqrt_values.begin(), sqrt_values.begin() + idx, sqrt_values.end());
        return file_input_scale_from_proxy((double) sqrt_values[idx]);
    }

    const double rms = std::sqrt(sum / (double) count);
    return file_input_scale_from_proxy(rms);
}

int64_t llama_nvfp4_sample_block_index(
        int64_t is,
        int64_t sample_nb,
        int64_t nb_total,
        int64_t row_blocks,
        int64_t sample_phase) {
    if (nb_total <= 1 || sample_nb <= 1) {
        return 0;
    }

    const int64_t phase_shift = sample_phase > 0 ? (sample_phase % nb_total) : 0;
    if (row_blocks > 1 && nb_total > row_blocks && sample_nb >= 8) {
        const int64_t global_nb = sample_nb / 2;
        if (is < global_nb) {
            const int64_t base = (is * (nb_total - 1)) / std::max<int64_t>(1, global_nb - 1);
            return (base + phase_shift) % nb_total;
        }

        const int64_t nrows = nb_total / row_blocks;
        const int64_t rowwise_nb = sample_nb - global_nb;
        const int64_t row_group = std::max<int64_t>(1, (rowwise_nb + 3) / 4);
        const int64_t js = is - global_nb;
        const int64_t row_slot = js / 4;
        const int64_t phase = (js + phase_shift) % 4;
        const int64_t row_base = row_group <= 1 ? 0 : (row_slot * (nrows - 1)) / (row_group - 1);
        const int64_t row = nrows > 0 ? (row_base + (phase_shift % std::max<int64_t>(1, nrows))) % nrows : 0;

        int64_t block_in_row = 0;
        switch (phase) {
            case 0: block_in_row = 0; break;
            case 1: block_in_row = row_blocks / 3; break;
            case 2: block_in_row = (2 * row_blocks) / 3; break;
            default: block_in_row = row_blocks - 1; break;
        }
        return std::min<int64_t>(nb_total - 1, row * row_blocks + std::min<int64_t>(row_blocks - 1, block_in_row));
    }

    return ((is * (nb_total - 1)) / (sample_nb - 1) + phase_shift) % nb_total;
}

#ifndef QK_MXFP6_E2M3
#define QK_MXFP6_E2M3 QK_MXFP6_SUB
#endif
#ifndef MXFP6_E2M3_TILE_ROWS
#define MXFP6_E2M3_TILE_ROWS MXFP6_TILE_ROWS
#endif

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};

static void zeros(std::ostream & file, size_t n) {
    static constexpr size_t zero_chunk_size = 64 * 1024;
    static const std::array<char, zero_chunk_size> zeroes{};
    while (n > 0) {
        const size_t chunk = std::min(n, zeroes.size());
        file.write(zeroes.data(), chunk);
        n -= chunk;
    }
}

static std::string llama_nvfp4_input_scale_tensor_name_legacy(const std::string & weight_name) {
    static const std::string canonical_suffix = ".input_scale";
    static const std::string legacy_suffix = ".weight.input_scale";

    if (weight_name.size() >= legacy_suffix.size() &&
        weight_name.compare(weight_name.size() - legacy_suffix.size(), legacy_suffix.size(), legacy_suffix) == 0) {
        return weight_name;
    }

    if (weight_name.size() >= canonical_suffix.size() &&
        weight_name.compare(weight_name.size() - canonical_suffix.size(), canonical_suffix.size(), canonical_suffix) == 0) {
        return weight_name.substr(0, weight_name.size() - canonical_suffix.size()) + legacy_suffix;
    }

    return weight_name + canonical_suffix;
}

static const llama_model_loader::llama_tensor_weight * llama_nvfp4_find_input_scale_weight(
        const llama_model_loader * ml, const std::string & weight_name) {
    if (!ml) {
        return nullptr;
    }

    const std::string canonical_name = llama_nvfp4_input_scale_tensor_name(weight_name);
    const auto * weight = ml->get_weight(canonical_name.c_str());
    if (weight != nullptr && weight->tensor != nullptr) {
        return weight;
    }

    const std::string legacy_name = llama_nvfp4_input_scale_tensor_name_legacy(weight_name);
    if (legacy_name != canonical_name) {
        weight = ml->get_weight(legacy_name.c_str());
    }

    return weight;
}

static constexpr size_t LLAMA_QUANT_MIN_SAVINGS_BYTES = 4u * 1024u * 1024u;
static constexpr size_t LLAMA_NVFP4_MIN_SAVINGS_BYTES = 2u * 1024u * 1024u;
static constexpr size_t LLAMA_MXFP6_MIN_SAVINGS_BYTES = LLAMA_QUANT_MIN_SAVINGS_BYTES;
static constexpr int64_t LLAMA_NVFP4_ENCODER_MAX_BLOCKS_DEFAULT = 0;
static constexpr int32_t NVFP4_CUDA_CHUNK_MIB_DEFAULT = 2;
static constexpr int64_t LLAMA_MXFP6_TENSOR_SCALE_SAMPLE_BLOCKS_DEFAULT = 4096;
static constexpr int32_t LLAMA_MXFP6_TENSOR_SCALE_STEPS_DEFAULT = 64;
static constexpr int64_t LLAMA_NV4MX6_SAMPLE_CAP_DEFAULT = 8192;
static constexpr float LLAMA_MXFP6_INPUT_SCALE_QUANTILE_DEFAULT = 0.99f;
static constexpr float LLAMA_NV4MX6_QW_BLEND_DEFAULT = 0.35f;
static constexpr float LLAMA_NV4MX6_QW_POWER_DEFAULT = 0.50f;
static constexpr float LLAMA_NV4MX6_QW_MIN_DEFAULT = 0.25f;
static constexpr float LLAMA_NV4MX6_QW_MAX_DEFAULT = 4.00f;
static constexpr int64_t LLAMA_NV4MX6_MX6_REFIT_RADIUS_DEFAULT = 8;
static constexpr int64_t LLAMA_NV4MX6_MX6_TOPK_DEFAULT = 8;
static constexpr int64_t LLAMA_NV4MX6_MX6_SLOT_RADIUS_DEFAULT = 4;
static constexpr int64_t LLAMA_NV4MX6_NV4_REFIT_RADIUS_DEFAULT = 8;
static constexpr int64_t LLAMA_NV4MX6_NV4_TOPK_DEFAULT = 6;
static constexpr float LLAMA_NV4MX6_NV4_CAP6_DEFAULT = 448.0f;
static constexpr float LLAMA_NV4MX6_NV4_CAP4_DEFAULT = 256.0f;
static constexpr bool LLAMA_NVFP4_TRACE_ENABLED = false;
static constexpr bool LLAMA_NV4MX6_TRACE_ENABLED = false;

static bool llama_nvfp4_trace_enabled() {
    return LLAMA_NVFP4_TRACE_ENABLED;
}

static bool llama_classic_quant_cuda_enabled() {
    return true;
}

static bool llama_classic_quant_cuda_trace_enabled() {
    return false;
}

static bool llama_classic_quant_cuda_supported_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

static int64_t llama_nvfp4_encoder_sample_cap_override(const llama_model_quantize_params * params) {
    if (params == nullptr || params->nvfp4_encoder_max_blocks <= 0) {
        return LLAMA_NVFP4_ENCODER_MAX_BLOCKS_DEFAULT;
    }
    return params->nvfp4_encoder_max_blocks;
}

static int llama_nvfp4_cuda_parallel_threads(int nthread, int64_t nchunk) {
    if (nthread < 2 || nchunk < 2) {
        return 1;
    }

    const int64_t capped = std::max<int64_t>(1, std::min<int64_t>(std::max(1, std::min(nthread, 16)), nchunk));
    return (int) std::max<int64_t>(1, std::min<int64_t>(capped, nthread));
}

static int64_t llama_nvfp4_cuda_chunk_rows(
        int64_t nrows,
        int64_t n_per_row,
        bool src_is_bf16,
        int64_t cpu_chunk_rows,
        int desired_parallel) {
    if (nrows <= 0 || n_per_row <= 0) {
        return 1;
    }

    const int64_t target_mb = NVFP4_CUDA_CHUNK_MIB_DEFAULT;
    const size_t target_bytes = (size_t) std::max<int64_t>(1, target_mb) * 1024u * 1024u;
    const size_t elem_bytes = src_is_bf16 ? sizeof(ggml_bf16_t) : sizeof(float);
    const size_t bytes_per_row = (size_t) n_per_row * elem_bytes;
    const int64_t target_rows = bytes_per_row > 0 ? (int64_t) std::max<size_t>(1, target_bytes / bytes_per_row) : 1;
    const int64_t desired_chunks = std::max<int64_t>(1, desired_parallel);
    const int64_t rows_for_parallel = std::max<int64_t>(1, (nrows + desired_chunks - 1) / desired_chunks);
    int64_t rows = std::max<int64_t>(1, std::min<int64_t>(target_rows, rows_for_parallel));
    if (cpu_chunk_rows > 0) {
        rows = std::max<int64_t>(1, std::min<int64_t>(rows, cpu_chunk_rows));
    }
    return std::min<int64_t>(nrows, rows);
}

static int64_t llama_mxfp6_e2m3_cuda_chunk_rows(int64_t nrows, int64_t rows) {
    if (nrows <= MXFP6_E2M3_TILE_ROWS) {
        return nrows;
    }
    rows = std::max<int64_t>(MXFP6_E2M3_TILE_ROWS, rows);
    rows = (rows / MXFP6_E2M3_TILE_ROWS) * MXFP6_E2M3_TILE_ROWS;
    return std::min<int64_t>(nrows, std::max<int64_t>(MXFP6_E2M3_TILE_ROWS, rows));
}

static int64_t llama_nvfp4_encoder_sample_blocks(const llama_model_quantize_params * params, const int64_t nb_total) {
    if (nb_total <= 0) {
        return 0;
    }
    const int64_t override_cap = llama_nvfp4_encoder_sample_cap_override(params);
    if (override_cap > 0) {
        return std::min(nb_total, override_cap);
    }
    const int64_t cap =
        nb_total >= 262144 ? 8192 :
        nb_total >= 65536  ? 6144 :
        nb_total >= 16384  ? 4096 :
        nb_total >= 4096   ? 2048 :
        nb_total >= 1024   ? 512  :
        nb_total >= 256    ? 256  : 64;
    return std::min(nb_total, cap);
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix(const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

static inline float llama_tensor_absmax(const float * data, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float v = std::fabs(data[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }
    return max_abs;
}

static float llama_tensor_absmax_mt(const float * data, size_t n, int nthread) {
    if (nthread < 2 || n < 1024) {
        return llama_tensor_absmax(data, n);
    }

    const int nthreads = std::max(1, nthread);
    std::vector<std::thread> threads;
    threads.reserve(nthreads - 1);
    std::vector<float> partials((size_t) nthreads, 0.0f);

    auto worker = [&](int tid, size_t start, size_t end) {
        float m = 0.0f;
        for (size_t i = start; i < end; ++i) {
            const float v = std::fabs(data[i]);
            if (v > m) {
                m = v;
            }
        }
        partials[(size_t) tid] = m;
    };

    const size_t chunk = (n + (size_t) nthreads - 1) / (size_t) nthreads;
    for (int t = 0; t < nthreads - 1; ++t) {
        const size_t start = (size_t) t * chunk;
        const size_t end = std::min(n, start + chunk);
        threads.emplace_back(worker, t, start, end);
    }
    {
        const size_t start = (size_t) (nthreads - 1) * chunk;
        const size_t end = std::min(n, start + chunk);
        worker(nthreads - 1, start, end);
    }

    for (auto & th : threads) {
        th.join();
    }

    float max_abs = 0.0f;
    for (int t = 0; t < nthreads; ++t) {
        if (partials[(size_t) t] > max_abs) {
            max_abs = partials[(size_t) t];
        }
    }
    return max_abs;
}

static inline float llama_tensor_absmax_bf16(const ggml_bf16_t * data, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float v = std::fabs(ggml_bf16_to_fp32(data[i]));
        if (v > max_abs) {
            max_abs = v;
        }
    }
    return max_abs;
}

static float llama_tensor_absmax_bf16_mt(const ggml_bf16_t * data, size_t n, int nthread) {
    if (nthread < 2 || n < 1024) {
        return llama_tensor_absmax_bf16(data, n);
    }

    const int nthreads = std::max(1, nthread);
    std::vector<std::thread> threads;
    threads.reserve(nthreads - 1);
    std::vector<float> partials((size_t) nthreads, 0.0f);

    auto worker = [&](int tid, size_t start, size_t end) {
        float m = 0.0f;
        for (size_t i = start; i < end; ++i) {
            const float v = std::fabs(ggml_bf16_to_fp32(data[i]));
            if (v > m) {
                m = v;
            }
        }
        partials[(size_t) tid] = m;
    };

    const size_t chunk = (n + (size_t) nthreads - 1) / (size_t) nthreads;
    for (int t = 0; t < nthreads - 1; ++t) {
        const size_t start = (size_t) t * chunk;
        const size_t end = std::min(n, start + chunk);
        threads.emplace_back(worker, t, start, end);
    }
    {
        const size_t start = (size_t) (nthreads - 1) * chunk;
        const size_t end = std::min(n, start + chunk);
        worker(nthreads - 1, start, end);
    }

    for (auto & th : threads) {
        th.join();
    }

    float max_abs = 0.0f;
    for (int t = 0; t < nthreads; ++t) {
        if (partials[(size_t) t] > max_abs) {
            max_abs = partials[(size_t) t];
        }
    }
    return max_abs;
}

static constexpr float LLAMA_NVFP4_MAX_FP4 = 6.0f;
static constexpr float LLAMA_NVFP4_TENSOR_CAP_FIXED = 448.0f;
// The NVFP4 GGUF compatibility contract stores tensor scale as
// amax / (E2M1_MAX * E4M3_MAX). In this code path the FP4 values are
// E2M1 with max 6 and the subblock scale is raw UE4M3 with max 448.
static constexpr float LLAMA_NVFP4_CORRECTION_DENOM_DEFAULT =
    LLAMA_NVFP4_MAX_FP4 * LLAMA_NVFP4_TENSOR_CAP_FIXED;
static constexpr int64_t LLAMA_NVFP4_BLOCK_SIZE = 64;
static constexpr float LLAMA_MXFP6_E2M3_MAX_FP6 = 7.5f;
// This keeps the MXFP6 activation-scale grid on the same correction denominator
// that was empirically strongest among the known Blackwell NVFP4 candidates for
// Qwen3.5, expressed as an E2M3 max-value phase instead of an identity scale.
static constexpr float LLAMA_MXFP6_INPUT_SCALE_PHASE_DEFAULT =
    LLAMA_NVFP4_CORRECTION_DENOM_DEFAULT / LLAMA_MXFP6_E2M3_MAX_FP6;
static constexpr float LLAMA_MXFP6_INPUT_SCALE_DENOM_DEFAULT =
    LLAMA_MXFP6_E2M3_MAX_FP6 * LLAMA_MXFP6_INPUT_SCALE_PHASE_DEFAULT;

static float llama_mxfp6_input_scale_denom(const llama_model_quantize_params * params) {
    if (params != nullptr &&
            params->mxfp6_input_scale_denom > 0.0f &&
            std::isfinite(params->mxfp6_input_scale_denom)) {
        return params->mxfp6_input_scale_denom;
    }
    return LLAMA_MXFP6_INPUT_SCALE_DENOM_DEFAULT;
}

static float llama_mxfp6_input_scale_quantile(const llama_model_quantize_params * params) {
    if (params != nullptr &&
            params->mxfp6_input_scale_quantile > 0.0f &&
            params->mxfp6_input_scale_quantile < 1.0f &&
            std::isfinite(params->mxfp6_input_scale_quantile)) {
        return params->mxfp6_input_scale_quantile;
    }
    return LLAMA_MXFP6_INPUT_SCALE_QUANTILE_DEFAULT;
}

static const char * llama_nvfp4_input_scale_policy_name(const int32_t policy) {
    switch (policy) {
        case LLAMA_NVFP4_INPUT_SCALE_IMATRIX_RMS:      return "imatrix-rms";
        case LLAMA_NVFP4_INPUT_SCALE_IDENTITY:         return "identity";
        case LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTMAX:  return "imatrix-sqrtmax";
        case LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTP99:  return "imatrix-sqrtp99";
        case LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTP999: return "imatrix-sqrtp999";
        default:                                       return "imatrix-rms";
    }
}

static int32_t llama_nvfp4_resolve_input_scale_policy(const llama_model_quantize_params * params) {
    const int32_t policy = params != nullptr ? params->nvfp4_input_scale_policy : LLAMA_NVFP4_INPUT_SCALE_IMATRIX_RMS;
    switch (policy) {
        case LLAMA_NVFP4_INPUT_SCALE_IMATRIX_RMS:
        case LLAMA_NVFP4_INPUT_SCALE_IDENTITY:
        case LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTMAX:
        case LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTP99:
        case LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTP999:
            return policy;
        default:
            return LLAMA_NVFP4_INPUT_SCALE_IMATRIX_RMS;
    }
}

static float llama_nvfp4_resolve_correction_denom(const llama_model_quantize_params * params) {
    if (params != nullptr && params->nvfp4_correction_denom > 0.0f && std::isfinite(params->nvfp4_correction_denom)) {
        return params->nvfp4_correction_denom;
    }
    return LLAMA_NVFP4_CORRECTION_DENOM_DEFAULT;
}

static float llama_nvfp4_correction_scale(
        const float * f32_data,
        const ggml_bf16_t * bf16_data,
        int64_t n,
        int nthread,
        float correction_denom) {
    if (n <= 0 || (f32_data == nullptr && bf16_data == nullptr)) {
        return 1.0f;
    }

    if (!(correction_denom > 0.0f) || !std::isfinite(correction_denom)) {
        return 1.0f;
    }

    const float amax = f32_data != nullptr
        ? llama_tensor_absmax_mt(f32_data, (size_t) n, nthread)
        : llama_tensor_absmax_bf16_mt(bf16_data, (size_t) n, nthread);

    if (!(amax > 0.0f) || !std::isfinite(amax)) {
        return 1.0f;
    }

    const double scale = (double) amax / (double) correction_denom;
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return 1.0f;
    }

    return (float) scale;
}

static float llama_mxfp6_e2m3_input_scale_from_imatrix(
    const float * imatrix,
    int64_t n_per_row,
    const llama_model_quantize_params * params) {
    if (imatrix == nullptr || n_per_row <= 0) {
        return 0.0f;
    }

    const float denom = llama_mxfp6_input_scale_denom(params);
    const double q = (double) llama_mxfp6_input_scale_quantile(params);

    std::vector<float> sqrt_values;
    sqrt_values.reserve((size_t) n_per_row);
    for (int64_t i = 0; i < n_per_row; ++i) {
        const float v = imatrix[i];
        if (std::isfinite(v) && v > 0.0f) {
            sqrt_values.push_back(std::sqrt(v));
        }
    }
    if (sqrt_values.empty()) {
        return 0.0f;
    }

    const size_t idx = std::min(
        sqrt_values.size() - 1,
        (size_t) std::llround(q * (double) (sqrt_values.size() - 1)));
    std::nth_element(sqrt_values.begin(), sqrt_values.begin() + idx, sqrt_values.end());

    const double input_scale = (double) sqrt_values[idx] / (double) denom;
    if (!(input_scale > 0.0) || !std::isfinite(input_scale)) {
        return 0.0f;
    }

    // This is an activation-domain phase shift for the MXFP6 UE8M0 block-scale
    // grid. It is intentionally not allowed to collapse to an identity scale.
    return (float) std::clamp(input_scale, std::ldexp(1.0, -24), std::ldexp(1.0, 8));
}

//
// helper functions for tensor name matching
//

static bool tensor_name_match_token_embd(const char * tensor_name) {
    return std::strcmp(tensor_name, "token_embd.weight") == 0 ||
           std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0;
}

static bool tensor_name_match_output_weight(const char * tensor_name) {
    return std::strcmp(tensor_name, "output.weight") == 0;
}

//
// tensor categorization for quantization
//
// (this is different from LLM_TN - we want broad categories, not specific tensor names per arch)
//

static tensor_category tensor_get_category(const std::string & tensor_name) {
    if (tensor_name_match_output_weight(tensor_name.c_str())) {
        return tensor_category::OUTPUT;
    }
    if (tensor_name_match_token_embd(tensor_name.c_str())) {
        return tensor_category::TOKEN_EMBD;
    }
    if (tensor_name.find("attn_qkv.weight") != std::string::npos) {
        return tensor_category::ATTENTION_QKV;
    }
    if (tensor_name.find("attn_kv_b.weight") != std::string::npos) {
        return tensor_category::ATTENTION_KV_B;
    }
    if (tensor_name.find("attn_v.weight") != std::string::npos) {
        return tensor_category::ATTENTION_V;
    }
    if (tensor_name.find("attn_k.weight") != std::string::npos) {
        return tensor_category::ATTENTION_K;
    }
    if (tensor_name.find("attn_q.weight") != std::string::npos) {
        return tensor_category::ATTENTION_Q;
    }
    if (tensor_name.find("attn_output.weight") != std::string::npos) {
        return tensor_category::ATTENTION_OUTPUT;
    }
    if (tensor_name.find("ffn_up") != std::string::npos) {
        return tensor_category::FFN_UP;
    }
    if (tensor_name.find("ffn_gate") != std::string::npos) {
        return tensor_category::FFN_GATE;
    }
    if (tensor_name.find("ffn_down") != std::string::npos) {
        return tensor_category::FFN_DOWN;
    }
    return tensor_category::OTHER;
}

static bool tensor_name_is_mtp(const llama_model & model, const std::string & tensor_name) {
    if (tensor_name.find(".nextn.") != std::string::npos ||
            tensor_name.find("mtp") != std::string::npos) {
        return true;
    }

    const uint32_t n_nextn = model.hparams.n_layer_nextn;
    if (n_nextn == 0 || model.hparams.n_layer_all <= n_nextn) {
        return false;
    }

    int il = -1;
    if (sscanf(tensor_name.c_str(), "blk.%d.", &il) != 1 || il < 0) {
        return false;
    }

    const uint32_t n_main = model.hparams.n_layer();
    return (uint32_t) il >= n_main && (uint32_t) il < model.hparams.n_layer_all;
}

static bool llama_tensor_has_aux_scale_slot(const std::string & tensor_name) {
    if (tensor_name_match_token_embd(tensor_name.c_str()) || tensor_name_match_output_weight(tensor_name.c_str())) {
        return false;
    }
    if (tensor_name.find(".weight") == std::string::npos) {
        return false;
    }
    return tensor_name.find(".attn_") != std::string::npos ||
        tensor_name.find(".ffn_") != std::string::npos ||
        tensor_name.find(".nextn.eh_proj.weight") != std::string::npos ||
        tensor_name.find(".nextn.shared_head_head.weight") != std::string::npos ||
        tensor_name.find(".ssm_in.weight") != std::string::npos ||
        tensor_name.find(".ssm_out.weight") != std::string::npos ||
        tensor_name.find(".ssm_alpha.weight") != std::string::npos ||
        tensor_name.find(".ssm_beta.weight") != std::string::npos;
}

// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool category_is_attn_v(tensor_category cat) {
    return cat == tensor_category::ATTENTION_V     ||
           cat == tensor_category::ATTENTION_QKV   ||
           cat == tensor_category::ATTENTION_KV_B;
}

//
// quantization state
//

struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
    bool has_tied_embeddings = true; // assume tied until we see output.weight

    // tensor type override patterns (compiled once, used twice)
    std::vector<std::pair<std::regex, tensor_type_option>> tensor_type_patterns;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params):
        model(model), params(params)
    {
        // compile regex patterns once - they are expensive
        if (params->tensor_types) {
            const auto & tensor_types = *static_cast<const std::vector<tensor_type_option> *>(params->tensor_types);
            for (const auto & opt : tensor_types) {
                tensor_type_patterns.emplace_back(std::regex(opt.name), opt);
            }
        }
    }
};

static const tensor_type_option * llama_tensor_find_manual_override(const quantize_state_impl & qs, const ggml_tensor * tensor) {
    if (tensor == nullptr || qs.tensor_type_patterns.empty()) {
        return nullptr;
    }

    const std::string tensor_name(tensor->name);
    for (auto it = qs.tensor_type_patterns.rbegin(); it != qs.tensor_type_patterns.rend(); ++it) {
        if (std::regex_search(tensor_name, it->first)) {
            return &it->second;
        }
    }

    return nullptr;
}

static ggml_type llama_tensor_find_manual_type(const quantize_state_impl & qs, const ggml_tensor * tensor) {
    const tensor_type_option * opt = llama_tensor_find_manual_override(qs, tensor);
    return opt ? opt->type : GGML_TYPE_COUNT;
}

static bool llama_tensor_type_has_k_rsf(ggml_type type) {
    return type == GGML_TYPE_Q2_K ||
           type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K;
}

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    std::string     name;
    ggml_type       target_type;
    tensor_category category;
    bool            is_mtp = false;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
    bool            copy_from_patch = false;
    bool            has_nvfp4_cfg_override = false;
    nvfp4_cuda_runtime_cfg nvfp4_cfg_override = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    int64_t         nvfp4_sample_blocks = 0;
    std::string     nvfp4_policy_name;
    bool            has_nvfp4_scale_overrides = false;
    std::vector<float> nvfp4_base_scale_overrides;
    std::vector<float> nvfp4_scale_overrides;
    bool            has_mxfp6_scale_mul = false;
    float           mxfp6_e2m3_scale_mul = 1.0f;
    std::string     mxfp6_policy_name;
    bool            has_k_rsf_mode_override = false;
    int32_t         k_rsf_mode_override = 0;
};

//
// dequantization
//

static void llama_tensor_dequantize_impl(
    ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype->to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype->to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

//
// do we allow this tensor to be quantized?
//

static bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor) {
    // trivial checks first -- no string ops needed
    if (params->only_copy)       return false;

    // quantize only 2D and 3D tensors (experts)
    if (ggml_n_dims(tensor) < 2) return false;

    const std::string name = ggml_get_name(tensor);

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

    // do not quantize norm tensors
    quantize &= name.find("_norm.weight") == std::string::npos;

    quantize &= params->quantize_output_tensor || name != "output.weight";

    // do not quantize expert gating tensors
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

    // do not quantize the i32 token-id -> expert-id routing table (DeepSeek-V4)
    quantize &= name.find("ffn_gate_tid2eid.weight") == std::string::npos;

    // these are very small (e.g. 4x4)
    quantize &= name.find("altup")  == std::string::npos;
    quantize &= name.find("laurel") == std::string::npos;

    // these are not too big so keep them as it is
    quantize &= name.find("per_layer_model_proj") == std::string::npos;

    // do not quantize positional embeddings and token types (BERT)
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_POS_EMBD,    "weight");
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

    // do not quantize Mamba/Kimi's small conv1d weights
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ssm_conv1d") == std::string::npos;
    quantize &= name.find("shortconv.conv.weight") == std::string::npos;

    // do not quantize MiniMax's tiny indexer projections
    quantize &= name.find("indexer.k_proj.weight") == std::string::npos;
    quantize &= name.find("indexer.q_proj.weight") == std::string::npos;

    // do not quantize RWKV's small yet 2D weights
    quantize &= name.find("time_mix_first.weight") == std::string::npos;
    quantize &= name.find("time_mix_w0.weight") == std::string::npos;
    quantize &= name.find("time_mix_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_v0.weight") == std::string::npos;
    quantize &= name.find("time_mix_v1.weight") == std::string::npos;
    quantize &= name.find("time_mix_v2.weight") == std::string::npos;
    quantize &= name.find("time_mix_a0.weight") == std::string::npos;
    quantize &= name.find("time_mix_a1.weight") == std::string::npos;
    quantize &= name.find("time_mix_a2.weight") == std::string::npos;
    quantize &= name.find("time_mix_g1.weight") == std::string::npos;
    quantize &= name.find("time_mix_g2.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

    // do not quantize relative position bias (T5)
    quantize &= name.find("attn_rel_b.weight") == std::string::npos;

    // do not quantize specific multimodal tensors
    quantize &= name.find(".position_embd") == std::string::npos;
    quantize &= name.find("sam.pos_embd")   == std::string::npos;
    quantize &= name.find("sam.neck.")      == std::string::npos;
    quantize &= name.find("sam.net_")       == std::string::npos;
    quantize &= name.find(".rel_pos")       == std::string::npos;
    quantize &= name.find(".patch_embd")    == std::string::npos;
    quantize &= name.find(".patch_merger")  == std::string::npos;

    return quantize;
}

//
// tensor type selection
//

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantize_state_impl & qs, const ggml_tensor * t, const ggml_type target_type) {
    ggml_type return_type = target_type;

    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);

    if (ncols % qk_k != 0) { // this tensor's shape is incompatible with this quant
        LLAMA_LOG_WARN("warning: %-36s - ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
                        t->name, ncols, qk_k, ggml_type_name(target_type));
        ++qs.n_fallback;

        switch (target_type) {
            // types on the left: block size 256
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:   // types on the right: block size 32
            case GGML_TYPE_IQ4_XS:  return_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:   return_type = GGML_TYPE_Q4_0;   break;
            case GGML_TYPE_MXFP4:
            case GGML_TYPE_MXFP6_E2M3:   return_type = GGML_TYPE_Q8_0;   break;
            case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;
            case GGML_TYPE_NVFP4:   return_type = GGML_TYPE_Q8_0;   break;
            default:
                throw std::runtime_error(format("no tensor type fallback is defined for type %s",
                                                ggml_type_name(target_type)));
        }
        if (ncols % ggml_blck_size(return_type) != 0) {
            //
            // the fallback return type is still not compatible for this tensor!
            //
            // most likely, this tensor's first dimension is not divisible by 32.
            // this is very rare. we can either abort the quantization, or
            // fallback to F16 / F32.
            //
            LLAMA_LOG_WARN("(WARNING: must use F16 due to unusual shape) ");
            return_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN("-> falling back to %7s\n", ggml_type_name(return_type));
    }
    return return_type;
}

static size_t llama_tensor_projected_nbytes(const ggml_tensor * tensor, ggml_type target_type) {
    size_t dst_nbytes =
        (size_t) ggml_row_size(target_type, tensor->ne[0]) *
        (size_t) tensor->ne[1] *
        (size_t) tensor->ne[2] *
        (size_t) tensor->ne[3];

    if (target_type == GGML_TYPE_NVFP4 && tensor->type != GGML_TYPE_NVFP4) {
        const size_t scale_len = (size_t) std::max<int64_t>(1, tensor->ne[2]);
        dst_nbytes += 2u * scale_len * sizeof(float);
    }

    return dst_nbytes;
}

static ggml_type tensor_type_avoid_nvfp4_token_embedding(
        quantize_state_impl & qs,
        const ggml_tensor * tensor,
        const tensor_metadata & tm,
        ggml_type target_type) {
    (void) qs;
    (void) tensor;
    (void) tm;
    return target_type;
}

// internal standard logic for selecting the target tensor type based on tensor category, ftype, and model arch
static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype, tensor_category category) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;
    const bool is_nemotron_arch =
        arch == LLM_ARCH_NEMOTRON ||
        arch == LLM_ARCH_NEMOTRON_H ||
        arch == LLM_ARCH_NEMOTRON_H_MOE;

    if (ftype == LLAMA_FTYPE_MOSTLY_NVFP4 && is_nemotron_arch) {
        // Nemotron is far more sensitive on the dense SSM/control path than on the large expert matrices.
        // Keep the dense recurrent projections in BF16, and push the remaining dense control-path weights
        // to Q8_0 so local NVFP4 focuses on the big expert tensors where it buys the most size.
        if (name.find("ssm_in.weight") != std::string::npos || name.find("ssm_out.weight") != std::string::npos) {
            return tensor->type;
        }
        if (name.find("attn_q.weight") != std::string::npos ||
            name.find("attn_k.weight") != std::string::npos ||
            name.find("attn_v.weight") != std::string::npos ||
            name.find("attn_qkv.weight") != std::string::npos ||
            name.find("attn_gate.weight") != std::string::npos ||
            name.find("attn_output.weight") != std::string::npos ||
            name.find("ffn_gate_inp.weight") != std::string::npos ||
            name.find("ffn_down_shexp.weight") != std::string::npos ||
            name.find("ffn_up_shexp.weight") != std::string::npos) {
            return GGML_TYPE_Q8_0;
        }
    }

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (category == tensor_category::OUTPUT || (qs.has_tied_embeddings && category == tensor_category::TOKEN_EMBD)) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_NVFP4 && category == tensor_category::TOKEN_EMBD) {
                new_type = GGML_TYPE_NVFP4;
                return tensor_type_fallback(qs, tensor, new_type);
            }
            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3) {
                new_type = GGML_TYPE_MXFP6_E2M3;
            }
            const int64_t nx = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);

            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0 && new_type != GGML_TYPE_MXFP6_E2M3) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (category == tensor_category::TOKEN_EMBD) {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_NVFP4) {
                new_type = GGML_TYPE_NVFP4;
            }
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (category_is_attn_v(category)) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && category == tensor_category::ATTENTION_K) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (category == tensor_category::FFN_DOWN) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (category == tensor_category::ATTENTION_OUTPUT) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (category_is_attn_v(category)) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (category == tensor_category::ATTENTION_K) {
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::ATTENTION_Q) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::FFN_DOWN) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (category == tensor_category::ATTENTION_OUTPUT) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (category == tensor_category::ATTENTION_QKV) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (category == tensor_category::FFN_GATE) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    }
    else if (category == tensor_category::FFN_UP) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    return new_type;
}

// outer wrapper: determine the ggml_type that this tensor should be quantized to
static ggml_type llama_tensor_get_type(quantize_state_impl & qs, const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type default_type, const tensor_metadata & tm) {
    if (!tensor_allows_quantization(params, qs.model.arch, tensor)) {
        return tensor->type;
    }
    if (params->token_embedding_type < GGML_TYPE_COUNT && tm.category == tensor_category::TOKEN_EMBD) {
        return params->token_embedding_type;
    }
    if (params->output_tensor_type < GGML_TYPE_COUNT && tm.category == tensor_category::OUTPUT) {
        return params->output_tensor_type;
    }

    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (ggml_is_quantized(default_type)) {
        // if the user provided tensor types - use those
        bool manual = false;
        const ggml_type manual_type = llama_tensor_find_manual_type(qs, tensor);
        if (manual_type != GGML_TYPE_COUNT) {
            if (manual_type != new_type) {
                LLAMA_LOG_WARN("%s: %-36s - applying manual override: %s -> %s\n",
                               __func__, tensor->name, ggml_type_name(new_type), ggml_type_name(manual_type));
            }
            new_type = manual_type;
            manual = true;
        }

        // NVFP4 is its own native Blackwell quantizer, not a k-quant mixture.
        // Do not inherit llama.cpp "mostly" rescue rules such as output -> Q6_K.
        if (!manual && !params->pure) {
            if (params->ftype == LLAMA_FTYPE_MOSTLY_NVFP4) {
                new_type = default_type;
            } else {
                new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category);
            }
        }

        // incompatible tensor shapes are handled here - fallback to a compatible type
        new_type = tensor_type_fallback(qs, tensor, new_type);
    }

    return new_type;
}

//
// quantization implementation
//

static size_t llama_tensor_quantize_impl(
        enum ggml_type new_type,
        const llama_model_quantize_params * params,
        const float * f32_data,
        const ggml_bf16_t * bf16_data,
        float tensor_scale,
        void * new_data,
        const int64_t chunk_size,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix,
        std::vector<std::thread> & workers,
	        const int nthread,
	        const char * tensor_name,
	        const nvfp4_cuda_runtime_cfg * nvfp4_cfg_hint = nullptr,
	        int64_t nvfp4_sample_blocks_override = 0,
	        float * actual_tensor_scale_out = nullptr,
	        float nvfp4_tune_tensor_scale_override = 0.0f,
	        bool nvfp4_fixed_tensor_scale = false,
            int32_t k_rsf_mode_override = 0) {
    const size_t row_size = ggml_row_size(new_type, n_per_row);
    if (actual_tensor_scale_out != nullptr) {
        *actual_tensor_scale_out = tensor_scale;
    }

#ifdef GGML_USE_CUDA
	    if (new_type == GGML_TYPE_NVFP4 &&
	            (f32_data != nullptr || bf16_data != nullptr)) {
        const bool nvfp4_trace = llama_nvfp4_trace_enabled();
        const auto cuda_begin = std::chrono::steady_clock::now();
        float nvfp4_a = NVFP4_A0;
        float nvfp4_b = NVFP4_B0;
        float nvfp4_scale_mul = 1.0f;
        float tensor_scale_eff =
            (std::isfinite(tensor_scale) && tensor_scale > 0.0f) ? tensor_scale : 1.0f;
        const float tune_tensor_scale_eff =
            (std::isfinite(nvfp4_tune_tensor_scale_override) && nvfp4_tune_tensor_scale_override > 0.0f)
            ? nvfp4_tune_tensor_scale_override
            : tensor_scale_eff;
        nvfp4_cuda_runtime_cfg nvfp4_cfg = {
            NVFP4_CUDA_CHOOSE46_ADAPTIVE,
            8,
            1,
            0,
            448.0f,
            256.0f,
        };
        bool nvfp4_cfg_valid = false;
        void * nvfp4_tune_stream = reinterpret_cast<void *>(2);
        const int64_t nb_total = (nrows * n_per_row) / LLAMA_NVFP4_BLOCK_SIZE;
        const int64_t sample_nb = nvfp4_sample_blocks_override > 0
            ? std::min<int64_t>(nb_total, nvfp4_sample_blocks_override)
            : llama_nvfp4_encoder_sample_blocks(params, nb_total);
        const int64_t sample_n = sample_nb * LLAMA_NVFP4_BLOCK_SIZE;
        double prep_ms = 0.0;
        double encoder_ms = 0.0;
        bool encoder_ok = false;

        if (nvfp4_get_runtime_cfg(&nvfp4_cfg, nvfp4_cfg_hint)) {
            nvfp4_cfg_valid = true;
        }

        if (sample_n > 0) {
            const auto prep_begin = std::chrono::steady_clock::now();
            const float inv_scale =
                (std::isfinite(tune_tensor_scale_eff) && tune_tensor_scale_eff > 0.0f) ? (1.0f / tune_tensor_scale_eff) : 1.0f;
            const int64_t nb_per_row = n_per_row / LLAMA_NVFP4_BLOCK_SIZE;
            const bool build_tune_qw = imatrix && nb_per_row > 0;
            const bool tune_unweighted_for_gate = tensor_name && strstr(tensor_name, "ffn_gate.weight") != nullptr;
            const bool use_tune_qw = build_tune_qw && !tune_unweighted_for_gate;
            const int64_t total_elems = nrows * n_per_row;
            const bool can_alias_tune_x = !bf16_data && inv_scale == 1.0f && sample_n == total_elems;

            std::vector<float> tune_x;
            const float * tune_x_ptr = nullptr;
            if (can_alias_tune_x) {
                tune_x_ptr = f32_data;
            } else {
                tune_x.resize((size_t) sample_n);
                tune_x_ptr = tune_x.data();
            }
            std::vector<float> tune_qw;
            const float * tune_qw_ptr = nullptr;
            if (use_tune_qw) {
                tune_qw.resize((size_t) sample_n);
                tune_qw_ptr = tune_qw.data();
            }

            int prep_threads = std::max(1, nthread);
            prep_threads = std::min<int>(prep_threads, (int) sample_nb);
            const bool needs_host_prep = !can_alias_tune_x || use_tune_qw;

            auto prep_worker = [&](int tid) {
                const int64_t ib0 = (sample_nb * tid) / prep_threads;
                const int64_t ib1 = (sample_nb * (tid + 1)) / prep_threads;
                for (int64_t ib = ib0; ib < ib1; ++ib) {
                    const int64_t src_block = llama_nvfp4_sample_block_index(ib, sample_nb, nb_total, nb_per_row);
                    const int64_t src_off = src_block * LLAMA_NVFP4_BLOCK_SIZE;
                    const int64_t dst_off = ib * LLAMA_NVFP4_BLOCK_SIZE;

                    if (!can_alias_tune_x) {
                        float * tune_x_dst = tune_x.data() + dst_off;
                        if (bf16_data) {
                            ggml_bf16_to_fp32_row(bf16_data + src_off, tune_x_dst, LLAMA_NVFP4_BLOCK_SIZE);
                        } else {
                            memcpy(tune_x_dst, f32_data + src_off, (size_t) LLAMA_NVFP4_BLOCK_SIZE * sizeof(float));
                        }

                        if (inv_scale != 1.0f) {
                            for (int j = 0; j < LLAMA_NVFP4_BLOCK_SIZE; ++j) {
                                tune_x_dst[j] *= inv_scale;
                            }
                        }
                    }

                    if (use_tune_qw) {
                        const int64_t ib_row = src_block % nb_per_row;
                        memcpy(tune_qw.data() + dst_off, imatrix + ib_row * LLAMA_NVFP4_BLOCK_SIZE, LLAMA_NVFP4_BLOCK_SIZE * sizeof(float));
                    }
                }
            };

            if (needs_host_prep && prep_threads > 1) {
                std::vector<std::thread> prep;
                prep.reserve((size_t) prep_threads - 1);
                for (int t = 1; t < prep_threads; ++t) {
                    prep.emplace_back(prep_worker, t);
                }
                prep_worker(0);
                for (auto & th : prep) {
                    th.join();
                }
            } else if (needs_host_prep) {
                prep_worker(0);
            }
            prep_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prep_begin).count();

            const auto encoder_begin = std::chrono::steady_clock::now();
            nvfp4_cuda_tune_result tune_result = {
                /* a       = */ NVFP4_A0,
                /* b       = */ NVFP4_B0,
                /* scale_mul = */ 1.0f,
                /* cfg     = */ nvfp4_cfg,
                /* has_cfg = */ 0,
            };
            encoder_ok = nvfp4_autotune_cuda_cfg(
                    tune_x_ptr,
                    tune_qw_ptr,
                    sample_n,
                    nvfp4_cfg_valid ? &nvfp4_cfg : nullptr,
                    &tune_result,
                    nvfp4_tune_stream);
            if (encoder_ok) {
                nvfp4_a = tune_result.a;
                nvfp4_b = tune_result.b;
                nvfp4_scale_mul =
                    (std::isfinite(tune_result.scale_mul) && tune_result.scale_mul > 0.0f)
                    ? tune_result.scale_mul
                    : 1.0f;
                if (!nvfp4_fixed_tensor_scale) {
                    tensor_scale_eff *= nvfp4_scale_mul;
                }
                if (tune_result.has_cfg) {
                    nvfp4_cfg = tune_result.cfg;
                    nvfp4_cfg_valid = true;
                }
            }
            encoder_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - encoder_begin).count();
        }

        const void * src = bf16_data != nullptr ? (const void *) bf16_data : (const void *) f32_data;
        const int64_t cpu_chunk_rows = std::max<int64_t>(1, chunk_size / n_per_row);
        const int desired_cuda_threads = std::max(1, (int) std::min<int64_t>(nrows, std::min<int>(nthread, 16)));
        const int64_t cuda_chunk_rows = llama_nvfp4_cuda_chunk_rows(nrows, n_per_row, bf16_data != nullptr, cpu_chunk_rows, desired_cuda_threads);
        const int64_t cuda_nchunk = (nrows + cuda_chunk_rows - 1) / cuda_chunk_rows;
        const int cuda_threads = llama_nvfp4_cuda_parallel_threads(nthread, cuda_nchunk);
        bool cuda_ok = false;

        auto quantize_cuda_once = [&](const void * src_chunk, void * dst_chunk, int64_t chunk_rows, void * stream_key) {
            return nvfp4_quantize_cuda_ab_cfg(
                src_chunk, bf16_data != nullptr, dst_chunk, chunk_rows, n_per_row, imatrix, tensor_scale_eff,
                nvfp4_a, nvfp4_b, nvfp4_cfg_valid ? &nvfp4_cfg : nullptr, stream_key);
        };

        if (cuda_threads > 1 && cuda_nchunk > 1) {
            std::atomic<int64_t> next_row{0};
            std::atomic<bool> chunks_ok{true};

            auto cuda_worker = [&](int worker_idx) {
                void * stream_key = reinterpret_cast<void *>((uintptr_t) (0x1000 + worker_idx));
                while (chunks_ok.load(std::memory_order_relaxed)) {
                    const int64_t first_row = next_row.fetch_add(cuda_chunk_rows, std::memory_order_relaxed);
                    if (first_row >= nrows) {
                        break;
                    }

                    const int64_t this_nrow = std::min(nrows - first_row, cuda_chunk_rows);
                    const void * src_chunk = bf16_data != nullptr
                        ? (const void *) (bf16_data + first_row * n_per_row)
                        : (const void *) (f32_data + first_row * n_per_row);
                    void * dst_chunk = (char *) new_data + first_row * row_size;

                    if (!quantize_cuda_once(src_chunk, dst_chunk, this_nrow, stream_key)) {
                        chunks_ok.store(false, std::memory_order_relaxed);
                        break;
                    }

                    const size_t chunk_bytes = (size_t) this_nrow * row_size;
                    if (!ggml_validate_row_data(new_type, dst_chunk, chunk_bytes)) {
                        LLAMA_LOG_WARN("%s: mxfp6_e2m3 cuda validation failed tensor=%s first_row=%" PRId64 " rows=%" PRId64 " bytes=%zu\n",
                                __func__, tensor_name ? tensor_name : "(unknown)", first_row, this_nrow, chunk_bytes);
                        chunks_ok.store(false, std::memory_order_relaxed);
                        break;
                    }
                }
            };

            workers.clear();
            workers.reserve((size_t) cuda_threads - 1);
            for (int t = 1; t < cuda_threads; ++t) {
                workers.emplace_back(cuda_worker, t);
            }
            cuda_worker(0);
            for (auto & w : workers) {
                w.join();
            }
            workers.clear();
            cuda_ok = chunks_ok.load(std::memory_order_relaxed);
        } else {
            cuda_ok = quantize_cuda_once(src, new_data, nrows, reinterpret_cast<void *>(2));
        }
        const double total_cuda_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_begin).count();
        if (cuda_ok) {
            if (actual_tensor_scale_out != nullptr) {
                *actual_tensor_scale_out = tensor_scale_eff;
            }
            if (nvfp4_trace) {
                LLAMA_LOG_INFO("%s: nvfp4 cuda ok tensor=%s rows=%" PRId64 " cols=%" PRId64 " sample_blocks=%" PRId64 " prep=%.2f ms encoder=%.2f ms total=%.2f ms tuned=%s cuda_threads=%d cuda_chunk_rows=%" PRId64 " scale_mul=%.8g final_scale=%.8g cfg={choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f}\n",
                        __func__,
                        tensor_name ? tensor_name : "(unknown)",
                        nrows, n_per_row, sample_nb, prep_ms, encoder_ms, total_cuda_ms,
                        encoder_ok ? "yes" : "no",
                        cuda_threads, cuda_chunk_rows,
                        (double) nvfp4_scale_mul,
                        (double) tensor_scale_eff,
                        nvfp4_cfg.choose46_mode,
                        nvfp4_cfg.refit_iters,
                        nvfp4_cfg.use_compand_sat,
                        (double) nvfp4_cfg.cap_m6,
                        (double) nvfp4_cfg.cap_m4);
            }
            const size_t new_size = (size_t) nrows * row_size;
            if (new_type != GGML_TYPE_MXFP6_E2M3 && !ggml_validate_row_data(new_type, new_data, new_size)) {
                throw std::runtime_error("quantized data validation failed");
            }
            return new_size;
        }
        LLAMA_LOG_WARN("%s: nvfp4 cuda fallback tensor=%s rows=%" PRId64 " cols=%" PRId64 " sample_blocks=%" PRId64 " prep=%.2f ms encoder=%.2f ms total=%.2f ms tuned=%s cuda_threads=%d cuda_chunk_rows=%" PRId64 " cfg={choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f}\n",
                __func__,
                tensor_name ? tensor_name : "(unknown)",
                nrows, n_per_row, sample_nb, prep_ms, encoder_ms, total_cuda_ms,
                encoder_ok ? "yes" : "no",
                cuda_threads, cuda_chunk_rows,
                nvfp4_cfg.choose46_mode,
                nvfp4_cfg.refit_iters,
                nvfp4_cfg.use_compand_sat,
                (double) nvfp4_cfg.cap_m6,
                (double) nvfp4_cfg.cap_m4);
    }

    if (new_type == GGML_TYPE_MXFP6_E2M3 &&
            (f32_data != nullptr || bf16_data != nullptr)) {
        const auto cuda_begin = std::chrono::steady_clock::now();
        const bool trace = llama_nvfp4_trace_enabled();
        const void * src = bf16_data != nullptr ? (const void *) bf16_data : (const void *) f32_data;
        const int64_t cpu_chunk_rows = std::max<int64_t>(1, chunk_size / n_per_row);
        const int desired_cuda_threads = std::max(1, (int) std::min<int64_t>(nrows, std::min<int>(nthread, 16)));
        const int64_t cuda_chunk_rows = llama_mxfp6_e2m3_cuda_chunk_rows(
                nrows, llama_nvfp4_cuda_chunk_rows(nrows, n_per_row, bf16_data != nullptr, cpu_chunk_rows, desired_cuda_threads));
        const int64_t cuda_nchunk = (nrows + cuda_chunk_rows - 1) / cuda_chunk_rows;
        const int cuda_threads = llama_nvfp4_cuda_parallel_threads(nthread, cuda_nchunk);
        bool cuda_ok = false;

        auto quantize_cuda_once = [&](const void * src_chunk, void * dst_chunk, int64_t chunk_rows, void * stream_key) {
            return mxfp6_e2m3_quantize_cuda(
                src_chunk, bf16_data != nullptr, dst_chunk, chunk_rows, n_per_row, imatrix, tensor_scale, stream_key);
        };

        if (cuda_threads > 1 && cuda_nchunk > 1) {
            std::atomic<int64_t> next_row{0};
            std::atomic<bool> chunks_ok{true};

            auto cuda_worker = [&](int worker_idx) {
                void * stream_key = reinterpret_cast<void *>((uintptr_t) (0x2000 + worker_idx));
                while (chunks_ok.load(std::memory_order_relaxed)) {
                    const int64_t first_row = next_row.fetch_add(cuda_chunk_rows, std::memory_order_relaxed);
                    if (first_row >= nrows) {
                        break;
                    }

                    const int64_t this_nrow = std::min(nrows - first_row, cuda_chunk_rows);
                    const void * src_chunk = bf16_data != nullptr
                        ? (const void *) (bf16_data + first_row * n_per_row)
                        : (const void *) (f32_data + first_row * n_per_row);
                    void * dst_chunk = (char *) new_data + first_row * row_size;

                    if (!quantize_cuda_once(src_chunk, dst_chunk, this_nrow, stream_key)) {
                        chunks_ok.store(false, std::memory_order_relaxed);
                        break;
                    }

                    const size_t chunk_bytes = (size_t) this_nrow * row_size;
                    if (new_type != GGML_TYPE_MXFP6_E2M3 && !ggml_validate_row_data(new_type, dst_chunk, chunk_bytes)) {
                        chunks_ok.store(false, std::memory_order_relaxed);
                        break;
                    }
                }
            };

            workers.clear();
            workers.reserve((size_t) cuda_threads - 1);
            for (int t = 1; t < cuda_threads; ++t) {
                workers.emplace_back(cuda_worker, t);
            }
            cuda_worker(0);
            for (auto & w : workers) {
                w.join();
            }
            workers.clear();
            cuda_ok = chunks_ok.load(std::memory_order_relaxed);
        } else {
            cuda_ok = quantize_cuda_once(src, new_data, nrows, reinterpret_cast<void *>(3));
        }

        const double total_cuda_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_begin).count();
        if (cuda_ok) {
            const size_t new_size = (size_t) nrows * row_size;
            if (new_type != GGML_TYPE_MXFP6_E2M3 && !ggml_validate_row_data(new_type, new_data, new_size)) {
                throw std::runtime_error("quantized data validation failed");
            }
            if (trace) {
                LLAMA_LOG_INFO("%s: mxfp6_e2m3 cuda ok tensor=%s rows=%" PRId64 " cols=%" PRId64 " total=%.2f ms cuda_threads=%d cuda_chunk_rows=%" PRId64 "\n",
                        __func__,
                        tensor_name ? tensor_name : "(unknown)",
                        nrows, n_per_row, total_cuda_ms, cuda_threads, cuda_chunk_rows);
            }
            return new_size;
        }

        LLAMA_LOG_WARN("%s: mxfp6_e2m3 cuda fallback tensor=%s rows=%" PRId64 " cols=%" PRId64 " total=%.2f ms cuda_threads=%d cuda_chunk_rows=%" PRId64 "\n",
                __func__,
                tensor_name ? tensor_name : "(unknown)",
                nrows, n_per_row, total_cuda_ms, cuda_threads, cuda_chunk_rows);
    }
#endif

    if (bf16_data != nullptr) {
        const int64_t nelements = nrows * n_per_row;
        std::vector<float> f32_tmp((size_t) nelements);

        const int convert_threads = nelements >= 65536 && nthread > 1
            ? std::min<int64_t>(nthread, (nelements + 65535) / 65536)
            : 1;
        if (convert_threads > 1) {
            std::atomic<int64_t> next{0};
            const int64_t chunk = std::max<int64_t>(65536, (nelements + convert_threads - 1) / convert_threads);
            auto convert_worker = [&]() {
                while (true) {
                    const int64_t first = next.fetch_add(chunk, std::memory_order_relaxed);
                    if (first >= nelements) {
                        break;
                    }
                    const int64_t count = std::min<int64_t>(chunk, nelements - first);
                    ggml_bf16_to_fp32_row(bf16_data + first, f32_tmp.data() + first, count);
                }
            };
            workers.clear();
            workers.reserve((size_t) convert_threads - 1);
            for (int it = 1; it < convert_threads; ++it) {
                workers.emplace_back(convert_worker);
            }
            convert_worker();
            for (auto & w : workers) {
                w.join();
            }
            workers.clear();
        } else {
            ggml_bf16_to_fp32_row(bf16_data, f32_tmp.data(), nelements);
        }
		        return llama_tensor_quantize_impl(new_type, params, f32_tmp.data(), nullptr, tensor_scale, new_data, chunk_size, nrows, n_per_row, imatrix, workers, nthread, tensor_name, nvfp4_cfg_hint, nvfp4_sample_blocks_override, actual_tensor_scale_out, nvfp4_tune_tensor_scale_override, nvfp4_fixed_tensor_scale, k_rsf_mode_override);
		    }

    const bool normalize_native_tensor_scale_cpu =
        (new_type == GGML_TYPE_NVFP4 || new_type == GGML_TYPE_MXFP4 || new_type == GGML_TYPE_MXFP6_E2M3) &&
        std::isfinite(tensor_scale) &&
        tensor_scale > 0.0f &&
        std::fabs(tensor_scale - 1.0f) > 1e-12f;
    const float inv_tensor_scale = normalize_native_tensor_scale_cpu ? (1.0f / tensor_scale) : 1.0f;
    int qk_rsf_mode = params != nullptr && params->k_quant_rsf ? std::max<int>(1, params->k_quant_rsf_mode) : 0;
    if (k_rsf_mode_override > 0) {
        qk_rsf_mode = std::max<int32_t>(1, std::min<int32_t>(3, k_rsf_mode_override));
    }
    const bool use_qk_rsf_cpu =
        qk_rsf_mode > 0 &&
        (new_type == GGML_TYPE_Q2_K || new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K ||
         new_type == GGML_TYPE_Q5_K || new_type == GGML_TYPE_Q6_K);
    auto quantize_k_rsf = [new_type, n_per_row, imatrix, qk_rsf_mode](const float * src, void * dst, int64_t rows) -> size_t {
        switch (new_type) {
            case GGML_TYPE_Q2_K: return quantize_q2_K_rsf(src, dst, rows, n_per_row, imatrix, qk_rsf_mode);
            case GGML_TYPE_Q3_K: return quantize_q3_K_rsf(src, dst, rows, n_per_row, imatrix, qk_rsf_mode);
            case GGML_TYPE_Q4_K: return quantize_q4_K_rsf(src, dst, rows, n_per_row, imatrix, qk_rsf_mode);
            case GGML_TYPE_Q5_K: return quantize_q5_K_rsf(src, dst, rows, n_per_row, imatrix, qk_rsf_mode);
            case GGML_TYPE_Q6_K: return quantize_q6_K_rsf(src, dst, rows, n_per_row, imatrix, qk_rsf_mode);
            default:             return 0;
        }
    };

#ifdef GGML_USE_CUDA
    if (llama_classic_quant_cuda_enabled() &&
            f32_data != nullptr &&
            !use_qk_rsf_cpu &&
            llama_classic_quant_cuda_supported_type(new_type)) {
        const auto cuda_begin = std::chrono::steady_clock::now();
        const bool trace = llama_classic_quant_cuda_trace_enabled();
        const int64_t cpu_chunk_rows = std::max<int64_t>(1, chunk_size / n_per_row);
        const int desired_cuda_threads = std::max(1, (int) std::min<int64_t>(nrows, std::min<int>(nthread, 16)));
        const int64_t cuda_chunk_rows = llama_nvfp4_cuda_chunk_rows(nrows, n_per_row, false, cpu_chunk_rows, desired_cuda_threads);
        const int64_t cuda_nchunk = (nrows + cuda_chunk_rows - 1) / cuda_chunk_rows;
        const int cuda_threads = llama_nvfp4_cuda_parallel_threads(nthread, cuda_nchunk);
        bool cuda_ok = false;

        auto quantize_cuda_once = [&](const float * src_chunk, void * dst_chunk, int64_t chunk_rows, void * stream_key) {
            return ggml_cuda_quantize_classic((int32_t) new_type, src_chunk, dst_chunk, chunk_rows, n_per_row, imatrix, 0, stream_key);
        };

        if (cuda_threads > 1 && cuda_nchunk > 1) {
            std::atomic<int64_t> next_row{0};
            std::atomic<bool> chunks_ok{true};

            auto cuda_worker = [&](int worker_idx) {
                void * stream_key = reinterpret_cast<void *>((uintptr_t) (0x3000 + worker_idx));
                while (chunks_ok.load(std::memory_order_relaxed)) {
                    const int64_t first_row = next_row.fetch_add(cuda_chunk_rows, std::memory_order_relaxed);
                    if (first_row >= nrows) {
                        break;
                    }

                    const int64_t this_nrow = std::min(nrows - first_row, cuda_chunk_rows);
                    const float * src_chunk = f32_data + first_row * n_per_row;
                    void * dst_chunk = (char *) new_data + first_row * row_size;
                    if (!quantize_cuda_once(src_chunk, dst_chunk, this_nrow, stream_key)) {
                        chunks_ok.store(false, std::memory_order_relaxed);
                        break;
                    }

                    const size_t chunk_bytes = (size_t) this_nrow * row_size;
                    if (!ggml_validate_row_data(new_type, dst_chunk, chunk_bytes)) {
                        LLAMA_LOG_WARN("%s: classic cuda validation failed tensor=%s type=%s first_row=%" PRId64 " rows=%" PRId64 " bytes=%zu\n",
                                __func__,
                                tensor_name ? tensor_name : "(unknown)",
                                ggml_type_name(new_type),
                                first_row,
                                this_nrow,
                                chunk_bytes);
                        chunks_ok.store(false, std::memory_order_relaxed);
                        break;
                    }
                }
            };

            workers.clear();
            workers.reserve((size_t) cuda_threads - 1);
            for (int t = 1; t < cuda_threads; ++t) {
                workers.emplace_back(cuda_worker, t);
            }
            cuda_worker(0);
            for (auto & w : workers) {
                w.join();
            }
            workers.clear();
            cuda_ok = chunks_ok.load(std::memory_order_relaxed);
        } else {
            cuda_ok = quantize_cuda_once(f32_data, new_data, nrows, reinterpret_cast<void *>(4));
            if (cuda_ok) {
                const size_t bytes = (size_t) nrows * row_size;
                cuda_ok = ggml_validate_row_data(new_type, new_data, bytes);
            }
        }

        const double total_cuda_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_begin).count();
        if (cuda_ok) {
            const size_t new_size = (size_t) nrows * row_size;
            if (trace) {
                LLAMA_LOG_INFO("%s: classic cuda ok tensor=%s type=%s rows=%" PRId64 " cols=%" PRId64 " total=%.2f ms cuda_threads=%d cuda_chunk_rows=%" PRId64 "\n",
                        __func__,
                        tensor_name ? tensor_name : "(unknown)",
                        ggml_type_name(new_type),
                        nrows,
                        n_per_row,
                        total_cuda_ms,
                        cuda_threads,
                        cuda_chunk_rows);
            }
            return new_size;
        }

        if (trace) {
            LLAMA_LOG_WARN("%s: classic cuda fallback tensor=%s type=%s rows=%" PRId64 " cols=%" PRId64 " total=%.2f ms cuda_threads=%d cuda_chunk_rows=%" PRId64 "\n",
                    __func__,
                    tensor_name ? tensor_name : "(unknown)",
                    ggml_type_name(new_type),
                    nrows,
                    n_per_row,
                    total_cuda_ms,
                    cuda_threads,
                    cuda_chunk_rows);
        }
    }
#endif

    if (nthread < 2 || new_type == GGML_TYPE_MXFP6_E2M3) {
        size_t new_size = 0;
        if (normalize_native_tensor_scale_cpu) {
            std::vector<float> scaled((size_t) nrows * (size_t) n_per_row);
            std::memcpy(scaled.data(), f32_data, scaled.size() * sizeof(float));
	            for (float & v : scaled) {
	                v *= inv_tensor_scale;
	            }
	            new_size = use_qk_rsf_cpu
                    ? quantize_k_rsf(scaled.data(), new_data, nrows)
                    : ggml_quantize_chunk(new_type, scaled.data(), new_data, 0, nrows, n_per_row, imatrix);
	        } else {
	            new_size = use_qk_rsf_cpu
                    ? quantize_k_rsf(f32_data, new_data, nrows)
                    : ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
	        }
        if (new_type != GGML_TYPE_MXFP6_E2M3 && !ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
	    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
	            nrows, n_per_row, imatrix, row_size, normalize_native_tensor_scale_cpu, inv_tensor_scale, use_qk_rsf_cpu, quantize_k_rsf]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        std::vector<float> scaled_chunk;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = 0;
            if (normalize_native_tensor_scale_cpu) {
                const size_t chunk_elems = (size_t) this_nrow * (size_t) n_per_row;
                if (scaled_chunk.size() < chunk_elems) {
                    scaled_chunk.resize(chunk_elems);
                }
                const float * src_chunk = f32_data + first_row * n_per_row;
                std::memcpy(scaled_chunk.data(), src_chunk, chunk_elems * sizeof(float));
                for (size_t i = 0; i < chunk_elems; ++i) {
                    scaled_chunk[i] *= inv_tensor_scale;
                }
	                void * dst_chunk = (char *) new_data + first_row * row_size;
                    this_size = use_qk_rsf_cpu
                        ? quantize_k_rsf(scaled_chunk.data(), dst_chunk, this_nrow)
                        : ggml_quantize_chunk(new_type, scaled_chunk.data(), dst_chunk, 0, this_nrow, n_per_row, imatrix);
	            } else {
	                void * dst_chunk = (char *) new_data + first_row * row_size;
                    this_size = use_qk_rsf_cpu
                        ? quantize_k_rsf(f32_data + first_row * n_per_row, dst_chunk, this_nrow)
                        : ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix);
	            }
            local_size += this_size;

            // validate the quantized data
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}

//
// imatrix requirement check
//

static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) {
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}

//
// given a file type, get the default tensor type
//

ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return GGML_TYPE_Q5_0;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return GGML_TYPE_Q5_1;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_F16:  return GGML_TYPE_F16;
        case LLAMA_FTYPE_MOSTLY_BF16: return GGML_TYPE_BF16;
        case LLAMA_FTYPE_ALL_F32:     return GGML_TYPE_F32;
        case LLAMA_FTYPE_MOSTLY_Q1_0: return GGML_TYPE_Q1_0;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return GGML_TYPE_MXFP4;
        case LLAMA_FTYPE_MOSTLY_MXFP6_E2M3:     return GGML_TYPE_MXFP6_E2M3;
        case LLAMA_FTYPE_MOSTLY_NVFP4:     return GGML_TYPE_NVFP4;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    return GGML_TYPE_Q2_K;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  return GGML_TYPE_IQ3_S;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  return GGML_TYPE_Q3_K;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  return GGML_TYPE_Q4_K;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  return GGML_TYPE_Q5_K;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    return GGML_TYPE_Q6_K;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   return GGML_TYPE_TQ1_0;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return GGML_TYPE_IQ2_S;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return GGML_TYPE_IQ1_S;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return GGML_TYPE_IQ3_S;

        default: return GGML_TYPE_COUNT;
    }
}

enum class nv4mx6_policy {
    OFF,
    MX6_SLOT,
    NV4_PROMOTE_MX6,
    MX6_DEMOTE_NV4,
    AUTO,
    BF16_MX6,
    BF16_MX6_SSE,
};

static const char * llama_nv4mx6_policy_name(nv4mx6_policy policy) {
    switch (policy) {
        case nv4mx6_policy::OFF:             return "off";
        case nv4mx6_policy::MX6_SLOT:        return "mx6_slot";
        case nv4mx6_policy::NV4_PROMOTE_MX6: return "nv4_promote_mx6";
        case nv4mx6_policy::MX6_DEMOTE_NV4:  return "mx6_demote_nv4";
        case nv4mx6_policy::AUTO:            return "auto";
        case nv4mx6_policy::BF16_MX6:        return "bf16_mx6";
        case nv4mx6_policy::BF16_MX6_SSE:    return "bf16_mx6_sse";
    }
    return "off";
}

static nv4mx6_policy llama_nv4mx6_policy_from_param(int32_t value) {
    switch (value) {
        case LLAMA_NV4MX6_POLICY_MX6_SLOT:        return nv4mx6_policy::MX6_SLOT;
        case LLAMA_NV4MX6_POLICY_NV4_PROMOTE_MX6: return nv4mx6_policy::NV4_PROMOTE_MX6;
        case LLAMA_NV4MX6_POLICY_MX6_DEMOTE_NV4:  return nv4mx6_policy::MX6_DEMOTE_NV4;
        case LLAMA_NV4MX6_POLICY_AUTO:            return nv4mx6_policy::AUTO;
        case LLAMA_NV4MX6_POLICY_BF16_MX6:        return nv4mx6_policy::BF16_MX6;
        case LLAMA_NV4MX6_POLICY_BF16_MX6_SSE:    return nv4mx6_policy::BF16_MX6_SSE;
        case LLAMA_NV4MX6_POLICY_OFF:
        default:                                  return nv4mx6_policy::OFF;
    }
}

static bool llama_nv4mx6_trace_enabled() {
    return LLAMA_NV4MX6_TRACE_ENABLED;
}

static float llama_nv4mx6_e2m3_to_fp32(uint8_t code) {
    code &= 0x3F;
    const int sign = code >> 5;
    const int exp  = (code >> 3) & 0x3;
    const int man  = code & 0x7;
    float v = exp == 0 ? (float) man * 0.125f : std::ldexp(1.0f + (float) man * 0.125f, exp - 1);
    return sign ? -v : v;
}

static uint8_t llama_nv4mx6_fp32_to_ue8m0(float x) {
    if (!(x > 0.0f) || !std::isfinite(x)) {
        return 127;
    }

    int exp2 = 0;
    const float m = std::frexp(x, &exp2);
    int unbiased = exp2 - 1;
    if (m >= 0.7071067811865475f) {
        ++unbiased;
    }

    return (uint8_t) std::clamp(unbiased + 127, 0, 254);
}

static float llama_nv4mx6_ue8m0_to_fp32(uint8_t code) {
    return code == 0xFF ? 0.0f : std::ldexp(1.0f, (int) code - 127);
}

static uint8_t llama_nv4mx6_e2m3_quant(float x, float inv_scale) {
    if (!std::isfinite(x) || x == 0.0f || !(inv_scale > 0.0f)) {
        return 0;
    }

    const int sign = x < 0.0f ? 0x20 : 0x00;
    const float ax = std::fabs(x) * inv_scale;
    static const float bounds[31] = {
        0.0625f, 0.1875f, 0.3125f, 0.4375f, 0.5625f, 0.6875f, 0.8125f, 0.9375f,
        1.0625f, 1.1875f, 1.3125f, 1.4375f, 1.5625f, 1.6875f, 1.8125f, 1.9375f,
        2.125f, 2.375f, 2.625f, 2.875f, 3.125f, 3.375f, 3.625f, 3.875f,
        4.25f, 4.75f, 5.25f, 5.75f, 6.25f, 6.75f, 7.25f,
    };

    for (int i = 0; i < 31; ++i) {
        if (ax < bounds[i] || (ax == bounds[i] && (i & 1) == 0)) {
            return (uint8_t) (sign | i);
        }
    }
    return (uint8_t) (sign | 31);
}

static float llama_nv4mx6_qw(const float * qw, int i) {
    if (qw == nullptr) {
        return 1.0f;
    }
    const float w = qw[i];
    return std::isfinite(w) && w > 0.0f ? w : 0.0f;
}

static const float * llama_nv4mx6_shape_qw32(
        const float * qw,
        const llama_model_quantize_params * params,
        std::array<float, 32> & shaped) {
    if (qw == nullptr) {
        return nullptr;
    }

    double sum = 0.0;
    int npos = 0;
    for (int i = 0; i < 32; ++i) {
        if (std::isfinite(qw[i]) && qw[i] > 0.0f) {
            sum += qw[i];
            ++npos;
        }
    }

    const float mean = npos > 0 && sum > 0.0 ? (float) (sum / npos) : 1.0f;
    const float blend_param = params != nullptr ? params->mixed_format_imatrix_blend : -1.0f;
    const float power_param = params != nullptr ? params->mixed_format_imatrix_power : -1.0f;
    const float min_param   = params != nullptr ? params->mixed_format_imatrix_min   : -1.0f;
    const float max_param   = params != nullptr ? params->mixed_format_imatrix_max   : -1.0f;
    const float blend = (float) std::clamp(
            std::isfinite(blend_param) && blend_param >= 0.0f ? (double) blend_param : (double) LLAMA_NV4MX6_QW_BLEND_DEFAULT,
            0.0, 1.0);
    const float power = (float) std::clamp(
            std::isfinite(power_param) && power_param >= 0.0f ? (double) power_param : (double) LLAMA_NV4MX6_QW_POWER_DEFAULT,
            0.0, 2.0);
    const float min_w = (float) std::clamp(
            std::isfinite(min_param) && min_param >= 0.0f ? (double) min_param : (double) LLAMA_NV4MX6_QW_MIN_DEFAULT,
            0.0, 1.0);
    const float max_w = (float) std::max<double>(
            min_w,
            std::isfinite(max_param) && max_param >= 0.0f ? (double) max_param : (double) LLAMA_NV4MX6_QW_MAX_DEFAULT);

    for (int i = 0; i < 32; ++i) {
        float w = std::isfinite(qw[i]) && qw[i] > 0.0f ? qw[i] : mean;
        w = mean > 0.0f ? w / mean : 1.0f;
        w = std::clamp(w, min_w, max_w);
        if (power != 1.0f) {
            w = std::pow(w, power);
        }
        shaped[i] = (1.0f - blend) + blend * w;
    }

    return shaped.data();
}

static float llama_nv4mx6_mx6_sse32(const float * x32, const float * qw32, uint8_t scale_code) {
    const float scale = llama_nv4mx6_ue8m0_to_fp32(scale_code);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float sse = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float w = llama_nv4mx6_qw(qw32, i);
        if (w == 0.0f) {
            continue;
        }
        const float xi = std::isfinite(x32[i]) ? x32[i] : 0.0f;
        const uint8_t q = llama_nv4mx6_e2m3_quant(xi, inv_scale);
        const float err = xi - scale * llama_nv4mx6_e2m3_to_fp32(q);
        sse += w * err * err;
    }
    return sse;
}

static void llama_nv4mx6_add_u8_candidate(std::vector<uint8_t> & candidates, uint8_t code) {
    if (code == 0xFF || std::find(candidates.begin(), candidates.end(), code) != candidates.end()) {
        return;
    }
    candidates.push_back(code);
}

static float llama_nv4mx6_best_mx6_sse32(const float * x32, const float * qw32) {
    std::array<float, 8> top{};
    for (int i = 0; i < 32; ++i) {
        if (!std::isfinite(x32[i])) {
            continue;
        }
        const float a = std::fabs(x32[i]);
        for (int j = 0; j < (int) top.size(); ++j) {
            if (a > top[j]) {
                for (int k = (int) top.size() - 1; k > j; --k) {
                    top[k] = top[k - 1];
                }
                top[j] = a;
                break;
            }
        }
    }
    if (!(top[0] > 0.0f)) {
        return 0.0f;
    }

    std::vector<uint8_t> candidates;
    candidates.reserve(256);
    const int radius = (int) std::clamp<int64_t>(LLAMA_NV4MX6_MX6_REFIT_RADIUS_DEFAULT, 0, 10);
    const uint8_t base = llama_nv4mx6_fp32_to_ue8m0(top[0] / 7.5f);
    for (int d = -radius; d <= radius; ++d) {
        llama_nv4mx6_add_u8_candidate(candidates, (uint8_t) std::clamp((int) base + d, 0, 254));
    }

    static const float slots[] = {
        7.5f, 7.0f, 6.5f, 6.0f, 5.5f, 5.0f, 4.5f, 4.0f,
        3.75f, 3.5f, 3.25f, 3.0f, 2.75f, 2.5f, 2.25f, 2.0f,
        1.875f, 1.75f, 1.625f, 1.5f, 1.375f, 1.25f, 1.125f, 1.0f,
        0.875f, 0.75f, 0.625f, 0.5f, 0.375f, 0.25f, 0.125f,
    };
    const int topk = (int) std::clamp<int64_t>(LLAMA_NV4MX6_MX6_TOPK_DEFAULT, 1, (int64_t) top.size());
    const int slot_radius = (int) std::clamp<int64_t>(LLAMA_NV4MX6_MX6_SLOT_RADIUS_DEFAULT, 0, 4);
    for (int t = 0; t < topk && top[t] > 0.0f; ++t) {
        for (float slot : slots) {
            const uint8_t c0 = llama_nv4mx6_fp32_to_ue8m0(top[t] / slot);
            for (int d = -slot_radius; d <= slot_radius; ++d) {
                llama_nv4mx6_add_u8_candidate(candidates, (uint8_t) std::clamp((int) c0 + d, 0, 254));
            }
        }
    }

    float best = std::numeric_limits<float>::infinity();
    for (uint8_t c : candidates) {
        best = std::min(best, llama_nv4mx6_mx6_sse32(x32, qw32, c));
    }
    return best;
}

static int64_t llama_mxfp6_sample_block_index(const int64_t is, const int64_t sample_nb, const int64_t nb_total) {
    if (sample_nb >= nb_total || sample_nb <= 1) {
        return is;
    }
    return (is * (nb_total - 1)) / (sample_nb - 1);
}

static float llama_mxfp6_tensor_scale(
        const llama_model_quantize_params * params,
        const float * f32_data,
        const ggml_bf16_t * bf16_data,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix,
        bool tensor_scale_enabled,
        int nthread) {
    if (!tensor_scale_enabled || nrows <= 0 || n_per_row <= 0 || n_per_row % QK_MXFP6_E2M3 != 0 ||
            (f32_data == nullptr && bf16_data == nullptr)) {
        return 1.0f;
    }

    const int64_t blocks_per_row = n_per_row / QK_MXFP6_E2M3;
    const int64_t nb_total = nrows * blocks_per_row;
    if (nb_total <= 0) {
        return 1.0f;
    }

    const int64_t sample_cap_config = params != nullptr && params->mxfp6_tensor_scale_sample_blocks > 0 ?
        params->mxfp6_tensor_scale_sample_blocks : LLAMA_MXFP6_TENSOR_SCALE_SAMPLE_BLOCKS_DEFAULT;
    const int64_t sample_cap = std::max<int64_t>(1, sample_cap_config);
    const int64_t sample_nb = std::min(nb_total, sample_cap);
    const int64_t steps_config = params != nullptr && params->mxfp6_tensor_scale_steps > 0 ?
        params->mxfp6_tensor_scale_steps : LLAMA_MXFP6_TENSOR_SCALE_STEPS_DEFAULT;
    const int steps = (int) std::clamp<int64_t>(steps_config, 4, 64);

    std::vector<float> candidates;
    candidates.reserve((size_t) steps + 1);
    candidates.push_back(1.0f);
    for (int i = 0; i < steps; ++i) {
        const float delta = ((float) i + 0.5f) / (float) steps - 0.5f;
        candidates.push_back(std::exp2(delta));
    }

    std::vector<double> candidate_sse(candidates.size(), std::numeric_limits<double>::infinity());
    const int worker_count = std::max<int>(1, std::min<int>((int) candidates.size(), nthread > 0 ? nthread : 1));
    std::atomic<size_t> next_candidate { 0 };

    auto eval_worker = [&]() {
        std::array<float, QK_MXFP6_E2M3> block{};
        std::array<float, QK_MXFP6_E2M3> scaled{};

        for (;;) {
            const size_t ci = next_candidate.fetch_add(1, std::memory_order_relaxed);
            if (ci >= candidates.size()) {
                break;
            }

            const float tensor_scale = candidates[ci];
        if (!(tensor_scale > 0.0f) || !std::isfinite(tensor_scale)) {
            continue;
        }

        const float inv_tensor_scale = 1.0f / tensor_scale;
        double sse = 0.0;

        for (int64_t is = 0; is < sample_nb; ++is) {
            const int64_t ib = llama_mxfp6_sample_block_index(is, sample_nb, nb_total);
            const int64_t row = ib / blocks_per_row;
            const int64_t block_in_row = ib - row * blocks_per_row;
            const int64_t off = row * n_per_row + block_in_row * QK_MXFP6_E2M3;

            if (f32_data != nullptr) {
                std::memcpy(block.data(), f32_data + off, sizeof(float) * QK_MXFP6_E2M3);
            } else {
                ggml_bf16_to_fp32_row(bf16_data + off, block.data(), QK_MXFP6_E2M3);
            }

            for (int i = 0; i < QK_MXFP6_E2M3; ++i) {
                scaled[i] = block[i] * inv_tensor_scale;
            }

            const float * qw = imatrix != nullptr ? imatrix + block_in_row * QK_MXFP6_E2M3 : nullptr;
            sse += (double) tensor_scale * (double) tensor_scale *
                (double) llama_nv4mx6_best_mx6_sse32(scaled.data(), qw);
        }

            candidate_sse[ci] = sse;
        }
    };

    std::vector<std::thread> workers;
    workers.reserve((size_t) std::max(0, worker_count - 1));
    for (int t = 1; t < worker_count; ++t) {
        workers.emplace_back(eval_worker);
    }
    eval_worker();
    for (auto & worker : workers) {
        worker.join();
    }

    double best_sse = std::numeric_limits<double>::infinity();
    float best_scale = 1.0f;
    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        if (candidate_sse[ci] < best_sse) {
            best_sse = candidate_sse[ci];
            best_scale = candidates[ci];
        }
    }

    if (!(best_scale > 0.0f) || !std::isfinite(best_scale)) {
        return 1.0f;
    }

    return best_scale;
}

static void llama_nv4mx6_consider_nv4_scale16(
        const float * x16,
        const float * qw16,
        int code,
        int cap_code,
        float & best_sse,
        float & best_scale) {
    static constexpr float kvalues[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };

    code = std::clamp(code, 0, std::min(cap_code, 0x7E));
    const float scale = 2.0f * ggml_ue4m3_to_fp32((uint8_t) code);
    float sse = 0.0f;
    for (int j = 0; j < QK_NVFP4_SUB; ++j) {
        const float w = qw16 == nullptr ? 1.0f : (!std::isfinite(qw16[j]) || qw16[j] <= 0.0f ? 0.0f : qw16[j]);
        if (w == 0.0f) {
            continue;
        }

        float q = 0.0f;
        if (scale > 0.0f && std::isfinite(scale) && std::isfinite(x16[j])) {
            float best_err = std::numeric_limits<float>::max();
            for (const float value : kvalues) {
                const float candidate = scale * value;
                const float err = (x16[j] - candidate) * (x16[j] - candidate);
                if (err < best_err) {
                    best_err = err;
                    q = candidate;
                }
            }
        }
        const float e = x16[j] - q;
        sse += w * e * e;
    }
    if (sse < best_sse) {
        best_sse = sse;
        best_scale = scale;
    }
}

static void llama_nv4mx6_consider_nv4_eff_scale16(
        const float * x16,
        const float * qw16,
        float eff_scale,
        int cap_code,
        float & best_sse,
        float & best_scale) {
    if (!(eff_scale > 0.0f) || !std::isfinite(eff_scale)) {
        return;
    }
    const int radius = (int) std::clamp<int64_t>(LLAMA_NV4MX6_NV4_REFIT_RADIUS_DEFAULT, 0, 16);
    const int base = (int) ggml_fp32_to_ue4m3(0.5f * eff_scale);
    for (int d = -radius; d <= radius; ++d) {
        llama_nv4mx6_consider_nv4_scale16(x16, qw16, base + d, cap_code, best_sse, best_scale);
    }
}

static float llama_nv4mx6_best_nv4_sse16(const float * x16, const float * qw16, float * best_scale_out) {
    std::array<float, 8> top{};
    for (int i = 0; i < 16; ++i) {
        if (!std::isfinite(x16[i])) {
            continue;
        }
        const float a = std::fabs(x16[i]);
        for (int j = 0; j < (int) top.size(); ++j) {
            if (a > top[j]) {
                for (int k = (int) top.size() - 1; k > j; --k) {
                    top[k] = top[k - 1];
                }
                top[j] = a;
                break;
            }
        }
    }
    if (!(top[0] > 0.0f)) {
        if (best_scale_out) {
            *best_scale_out = 0.0f;
        }
        return 0.0f;
    }

    const float cap6 = LLAMA_NV4MX6_NV4_CAP6_DEFAULT;
    const float cap4 = LLAMA_NV4MX6_NV4_CAP4_DEFAULT;
    const int cap6_code = (int) ggml_fp32_to_ue4m3(std::isfinite(cap6) && cap6 > 0.0f ? cap6 : 448.0f);
    const int cap4_code = (int) ggml_fp32_to_ue4m3(std::isfinite(cap4) && cap4 > 0.0f ? std::min(cap4, cap6) : 256.0f);

    float best_sse = std::numeric_limits<float>::infinity();
    float best_scale = 0.0f;

    static const float slots[] = { 6.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f };
    const int topk = (int) std::clamp<int64_t>(LLAMA_NV4MX6_NV4_TOPK_DEFAULT, 1, (int64_t) top.size());
    for (int t = 0; t < topk && top[t] > 0.0f; ++t) {
        for (float slot : slots) {
            llama_nv4mx6_consider_nv4_eff_scale16(
                    x16, qw16, top[t] / slot, slot <= 4.0f ? cap4_code : cap6_code, best_sse, best_scale);
        }
    }

    if (best_scale_out) {
        *best_scale_out = best_scale;
    }
    return best_sse;
}

static float llama_nv4mx6_fp4_gap_sse16(const float * x16, const float * qw16, float scale) {
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        return 0.0f;
    }
    float gap = 0.0f;
    for (int i = 0; i < 16; ++i) {
        const float w = llama_nv4mx6_qw(qw16, i);
        if (w == 0.0f) {
            continue;
        }
        const float u = std::fabs(x16[i]) / scale;
        if (u > 4.0f && u < 6.0f) {
            const float tri = std::max(0.0f, 1.0f - std::fabs(u - 5.0f));
            const float e = scale * tri;
            gap += w * e * e;
        }
    }
    return gap;
}

struct nv4mx6_tensor_eval {
    int64_t samples = 0;
    double sse_nv4 = 0.0;
    double sse_mx6 = 0.0;
    double gap = 0.0;
    double x2 = 0.0;
    int64_t mx6_wins = 0;
};

struct nv4mx6_actual_eval {
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double max_abs = 0.0;
    int64_t count = 0;
};

#ifdef GGML_USE_CUDA
static bool llama_nv4mx6_eval_quant_cuda_actual(
        ggml_type qtype,
        const llama_model_quantize_params * params,
        const float * f32_data,
        const ggml_bf16_t * bf16_data,
        float tensor_scale,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix,
        int nthread,
        const char * tensor_name,
        nv4mx6_actual_eval & out) {
    if ((qtype != GGML_TYPE_NVFP4 && qtype != GGML_TYPE_MXFP6_E2M3) ||
            (f32_data == nullptr && bf16_data == nullptr) ||
            nrows <= 0 || n_per_row <= 0) {
        return false;
    }

    const size_t row_size = ggml_row_size(qtype, n_per_row);
    if (row_size == 0) {
        return false;
    }
    if (!(tensor_scale > 0.0f) || !std::isfinite(tensor_scale)) {
        tensor_scale = 1.0f;
    }

    float nvfp4_a = NVFP4_A0;
    float nvfp4_b = NVFP4_B0;
    float nvfp4_scale_mul = 1.0f;
    nvfp4_cuda_runtime_cfg nvfp4_cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    bool nvfp4_cfg_valid = false;

    if (qtype == GGML_TYPE_NVFP4) {
        const int64_t nb_total = (nrows * n_per_row) / LLAMA_NVFP4_BLOCK_SIZE;
        const int64_t sample_nb = llama_nvfp4_encoder_sample_blocks(params, nb_total);
        const int64_t sample_n = sample_nb * LLAMA_NVFP4_BLOCK_SIZE;
        if (sample_n > 0) {
            const float inv_scale = tensor_scale > 0.0f && std::isfinite(tensor_scale) ? 1.0f / tensor_scale : 1.0f;
            const int64_t nb_per_row = n_per_row / LLAMA_NVFP4_BLOCK_SIZE;
            const bool build_qw = imatrix != nullptr && nb_per_row > 0;
            const bool tune_unweighted_for_gate = tensor_name && strstr(tensor_name, "ffn_gate.weight") != nullptr;
            const bool use_qw = build_qw && !tune_unweighted_for_gate;
            const int64_t total_elems = nrows * n_per_row;
            const bool can_alias_x = bf16_data == nullptr && inv_scale == 1.0f && sample_n == total_elems;

            std::vector<float> tune_x;
            const float * tune_x_ptr = nullptr;
            if (can_alias_x) {
                tune_x_ptr = f32_data;
            } else {
                tune_x.resize((size_t) sample_n);
                tune_x_ptr = tune_x.data();
            }
            std::vector<float> tune_qw;
            const float * tune_qw_ptr = nullptr;
            if (use_qw) {
                tune_qw.resize((size_t) sample_n);
                tune_qw_ptr = tune_qw.data();
            }

            const int prep_threads = std::max<int>(1, std::min<int>(nthread > 0 ? nthread : 1, (int) sample_nb));
            auto prep_worker = [&](int tid) {
                const int64_t ib0 = (sample_nb * tid) / prep_threads;
                const int64_t ib1 = (sample_nb * (tid + 1)) / prep_threads;
                for (int64_t ib = ib0; ib < ib1; ++ib) {
                    const int64_t src_block = llama_nvfp4_sample_block_index(ib, sample_nb, nb_total, nb_per_row);
                    const int64_t src_off = src_block * LLAMA_NVFP4_BLOCK_SIZE;
                    const int64_t dst_off = ib * LLAMA_NVFP4_BLOCK_SIZE;
                    if (!can_alias_x) {
                        float * dst = tune_x.data() + dst_off;
                        if (bf16_data != nullptr) {
                            ggml_bf16_to_fp32_row(bf16_data + src_off, dst, LLAMA_NVFP4_BLOCK_SIZE);
                        } else {
                            memcpy(dst, f32_data + src_off, (size_t) LLAMA_NVFP4_BLOCK_SIZE * sizeof(float));
                        }
                        if (inv_scale != 1.0f) {
                            for (int j = 0; j < LLAMA_NVFP4_BLOCK_SIZE; ++j) {
                                dst[j] *= inv_scale;
                            }
                        }
                    }
                    if (use_qw) {
                        const int64_t block_in_row = src_block % nb_per_row;
                        memcpy(tune_qw.data() + dst_off, imatrix + block_in_row * LLAMA_NVFP4_BLOCK_SIZE,
                                (size_t) LLAMA_NVFP4_BLOCK_SIZE * sizeof(float));
                    }
                }
            };
            if ((!can_alias_x || use_qw) && prep_threads > 1) {
                std::vector<std::thread> prep;
                prep.reserve((size_t) prep_threads - 1);
                for (int t = 1; t < prep_threads; ++t) {
                    prep.emplace_back(prep_worker, t);
                }
                prep_worker(0);
                for (auto & th : prep) {
                    th.join();
                }
            } else if (!can_alias_x || use_qw) {
                prep_worker(0);
            }

            nvfp4_cuda_tune_result tune = {
                NVFP4_A0,
                NVFP4_B0,
                1.0f,
                nvfp4_cfg,
                0,
            };
            if (nvfp4_autotune_cuda_cfg(tune_x_ptr, tune_qw_ptr, sample_n, nullptr, &tune, reinterpret_cast<void *>(0x5100))) {
                nvfp4_a = tune.a;
                nvfp4_b = tune.b;
                nvfp4_scale_mul =
                    (std::isfinite(tune.scale_mul) && tune.scale_mul > 0.0f)
                    ? tune.scale_mul
                    : 1.0f;
                if (tune.has_cfg) {
                    nvfp4_cfg = tune.cfg;
                    nvfp4_cfg_valid = true;
                }
            }
        }
    }

    const int64_t cpu_chunk_rows = std::max<int64_t>(1, (32 * 512) / n_per_row);
    const int desired_cuda_threads = std::max(1, (int) std::min<int64_t>(nrows, std::min<int>(nthread > 0 ? nthread : 1, 16)));
    int64_t cuda_chunk_rows = llama_nvfp4_cuda_chunk_rows(nrows, n_per_row, bf16_data != nullptr, cpu_chunk_rows, desired_cuda_threads);
    if (qtype == GGML_TYPE_MXFP6_E2M3) {
        cuda_chunk_rows = llama_mxfp6_e2m3_cuda_chunk_rows(nrows, cuda_chunk_rows);
    }
    const int64_t cuda_nchunk = (nrows + cuda_chunk_rows - 1) / cuda_chunk_rows;
    const int cuda_threads = llama_nvfp4_cuda_parallel_threads(nthread > 0 ? nthread : 1, cuda_nchunk);

    struct worker_accum {
        double sum_sq = 0.0;
        double sum_abs = 0.0;
        double max_abs = 0.0;
        int64_t count = 0;
        bool ok = true;
    };

    auto run_chunk = [&](int64_t first_row, int64_t this_nrow, void * stream_key, worker_accum & acc) {
        const void * src_chunk = bf16_data != nullptr
            ? (const void *) (bf16_data + first_row * n_per_row)
            : (const void *) (f32_data + first_row * n_per_row);
        std::vector<uint8_t> tmp(row_size * (size_t) this_nrow);
        nvfp4_cuda_eval_result eval{};
        bool ok = false;
        if (qtype == GGML_TYPE_NVFP4) {
            ok = nvfp4_quantize_cuda_ab_eval_cfg(
                    src_chunk, bf16_data != nullptr, tmp.data(),
                    this_nrow, n_per_row, imatrix, tensor_scale * nvfp4_scale_mul,
                    nvfp4_a, nvfp4_b,
                    nvfp4_cfg_valid ? &nvfp4_cfg : nullptr,
                    &eval, stream_key);
        } else {
            ok = mxfp6_e2m3_quantize_cuda_eval(
                    src_chunk, bf16_data != nullptr, tmp.data(),
                    this_nrow, n_per_row, imatrix, tensor_scale,
                    &eval, stream_key);
        }
        if (!ok || !ggml_validate_row_data(qtype, tmp.data(), row_size * (size_t) this_nrow)) {
            acc.ok = false;
            return;
        }
        acc.sum_sq += eval.sum_sq;
        acc.sum_abs += eval.sum_abs;
        acc.max_abs = std::max(acc.max_abs, eval.max_abs);
        acc.count += eval.count;
    };

    std::vector<worker_accum> accs((size_t) cuda_threads);
    if (cuda_threads > 1 && cuda_nchunk > 1) {
        std::atomic<int64_t> next_row{0};
        std::vector<std::thread> workers;
        workers.reserve((size_t) cuda_threads);
        for (int t = 0; t < cuda_threads; ++t) {
            workers.emplace_back([&, t]() {
                void * stream_key = reinterpret_cast<void *>((uintptr_t) (0x5200 + t));
                while (accs[(size_t) t].ok) {
                    const int64_t first_row = next_row.fetch_add(cuda_chunk_rows, std::memory_order_relaxed);
                    if (first_row >= nrows) {
                        break;
                    }
                    const int64_t this_nrow = std::min(nrows - first_row, cuda_chunk_rows);
                    run_chunk(first_row, this_nrow, stream_key, accs[(size_t) t]);
                }
            });
        }
        for (auto & th : workers) {
            th.join();
        }
    } else {
        run_chunk(0, nrows, reinterpret_cast<void *>(0x5200), accs[0]);
    }

    out = {};
    for (const auto & acc : accs) {
        if (!acc.ok) {
            return false;
        }
        out.sum_sq += acc.sum_sq;
        out.sum_abs += acc.sum_abs;
        out.max_abs = std::max(out.max_abs, acc.max_abs);
        out.count += acc.count;
    }
    return out.count > 0;
}
#endif

static bool llama_nv4mx6_eval_tensor_actual_demote(
        const ggml_tensor * tensor,
        const llama_model_quantize_params * params,
        const float * imatrix,
        int nthread,
        nv4mx6_tensor_eval & out) {
#ifndef GGML_USE_CUDA
    GGML_UNUSED(tensor);
    GGML_UNUSED(params);
    GGML_UNUSED(imatrix);
    GGML_UNUSED(nthread);
    GGML_UNUSED(out);
    return false;
#else
    if (tensor == nullptr || tensor->data == nullptr ||
            (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) ||
            tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
            tensor->ne[0] % QK_MXFP6_E2M3 != 0 ||
            tensor->ne[0] % LLAMA_NVFP4_BLOCK_SIZE != 0) {
        return false;
    }

    const int64_t n_per_row = tensor->ne[0];
    const int64_t nrows = tensor->ne[1];
    const int64_t n_slices = std::max<int64_t>(1, tensor->ne[2]);
    const int64_t slice_elems = nrows * n_per_row;
    const float correction_denom = llama_nvfp4_resolve_correction_denom(params);
    const bool mx6_tensor_scale = params == nullptr || params->mxfp6_tensor_scale;

    std::vector<float> f16_to_f32;
    if (tensor->type == GGML_TYPE_F16) {
        f16_to_f32.resize((size_t) slice_elems * (size_t) n_slices);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) tensor->data, f16_to_f32.data(), (int64_t) f16_to_f32.size());
    }

    struct slice_accum {
        double sse_nv4 = 0.0;
        double sse_mx6 = 0.0;
        int64_t count = 0;
        bool ok = true;
    };

    const int slice_threads = std::max<int>(1, std::min<int>((int) n_slices, nthread > 0 ? nthread : 1));
    const int slice_nthread = std::max(1, (nthread > 0 ? nthread : 1) / slice_threads);
    std::vector<slice_accum> accs((size_t) slice_threads);
    std::atomic<int64_t> next_slice{0};

    auto worker = [&](int tid) {
        while (accs[(size_t) tid].ok) {
            const int64_t i03 = next_slice.fetch_add(1, std::memory_order_relaxed);
            if (i03 >= n_slices) {
                break;
            }

            const float * f32_slice = nullptr;
            const ggml_bf16_t * bf16_slice = nullptr;
            if (tensor->type == GGML_TYPE_F32) {
                f32_slice = (const float *) tensor->data + i03 * slice_elems;
            } else if (tensor->type == GGML_TYPE_BF16) {
                bf16_slice = (const ggml_bf16_t *) tensor->data + i03 * slice_elems;
            } else {
                f32_slice = f16_to_f32.data() + i03 * slice_elems;
            }

            const float * imatrix_slice = imatrix != nullptr ? imatrix + i03 * n_per_row : nullptr;
            const float nv4_tensor_scale = llama_nvfp4_correction_scale(
                    f32_slice, bf16_slice, slice_elems, slice_nthread, correction_denom);
            const float mx6_tensor_scale_val = llama_mxfp6_tensor_scale(
                    params, f32_slice, bf16_slice, nrows, n_per_row, imatrix_slice, mx6_tensor_scale, slice_nthread);

            nv4mx6_actual_eval nv4{};
            nv4mx6_actual_eval mx6{};
            if (!llama_nv4mx6_eval_quant_cuda_actual(
                    GGML_TYPE_NVFP4, params, f32_slice, bf16_slice, nv4_tensor_scale,
                    nrows, n_per_row, imatrix_slice, slice_nthread, tensor->name, nv4) ||
                !llama_nv4mx6_eval_quant_cuda_actual(
                    GGML_TYPE_MXFP6_E2M3, params, f32_slice, bf16_slice, mx6_tensor_scale_val,
                    nrows, n_per_row, imatrix_slice, slice_nthread, tensor->name, mx6)) {
                accs[(size_t) tid].ok = false;
                break;
            }

            accs[(size_t) tid].sse_nv4 += nv4.sum_sq;
            accs[(size_t) tid].sse_mx6 += mx6.sum_sq;
            accs[(size_t) tid].count += std::min(nv4.count, mx6.count);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve((size_t) std::max(0, slice_threads - 1));
    for (int t = 1; t < slice_threads; ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto & th : workers) {
        th.join();
    }

    out = {};
    for (const auto & acc : accs) {
        if (!acc.ok) {
            return false;
        }
        out.sse_nv4 += acc.sse_nv4;
        out.sse_mx6 += acc.sse_mx6;
        out.samples += acc.count;
    }
    return out.samples > 0;
#endif
}

static const char * llama_nv4mx6_category_name(tensor_category category) {
    switch (category) {
        case tensor_category::TOKEN_EMBD:        return "token_embd";
        case tensor_category::ATTENTION_Q:       return "attn_q";
        case tensor_category::ATTENTION_V:       return "attn_v";
        case tensor_category::ATTENTION_K:       return "attn_k";
        case tensor_category::ATTENTION_QKV:     return "attn_qkv";
        case tensor_category::ATTENTION_KV_B:    return "attn_kv_b";
        case tensor_category::ATTENTION_OUTPUT:  return "attn_output";
        case tensor_category::FFN_UP:            return "ffn_up";
        case tensor_category::FFN_GATE:          return "ffn_gate";
        case tensor_category::FFN_DOWN:          return "ffn_down";
        case tensor_category::OUTPUT:            return "output";
        case tensor_category::OTHER:             return "other";
    }
    return "other";
}

static double llama_nv4mx6_category_mx6_multiplier(tensor_category category, nv4mx6_policy policy) {
    const bool auto_policy = policy == nv4mx6_policy::AUTO;

    double fallback = 1.0;
    switch (category) {
        case tensor_category::FFN_UP:
        case tensor_category::FFN_DOWN:
            fallback = 1.0;
            break;
        case tensor_category::FFN_GATE:
            // Gates help KL/same-top on Qwen, but are not a cheap PPL/speed win.
            fallback = auto_policy ? 1.30 : 1.0;
            break;
        case tensor_category::ATTENTION_Q:
        case tensor_category::ATTENTION_V:
        case tensor_category::ATTENTION_K:
        case tensor_category::ATTENTION_QKV:
        case tensor_category::ATTENTION_KV_B:
        case tensor_category::ATTENTION_OUTPUT:
            fallback = auto_policy ? 1.75 : 1.0;
            break;
        default:
            fallback = auto_policy ? 2.00 : 1.0;
            break;
    }

    return fallback;
}

static float llama_nv4mx6_get_tensor_f32(const ggml_tensor * tensor, int64_t idx) {
    switch (tensor->type) {
        case GGML_TYPE_F32:
            return ((const float *) tensor->data)[idx];
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(((const ggml_fp16_t *) tensor->data)[idx]);
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(((const ggml_bf16_t *) tensor->data)[idx]);
        default:
            return 0.0f;
    }
}

static bool llama_nv4mx6_eval_tensor_proxy(
        const ggml_tensor * tensor,
        const llama_model_quantize_params * params,
        const float * imatrix,
        nv4mx6_tensor_eval & out,
        int nthread) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
        return false;
    }
    if (tensor->ne[0] % 64 != 0) {
        return false;
    }

    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = tensor->ne[1];
    const int64_t ne2 = std::max<int64_t>(1, tensor->ne[2]);
    const int64_t ne3 = std::max<int64_t>(1, tensor->ne[3]);
    const int64_t row_blocks = ncols / 32;
    const int64_t nb_total = nrows * ne2 * ne3 * row_blocks;
    if (nb_total <= 0) {
        return false;
    }

    int64_t sample_nb = params != nullptr && params->mixed_format_sample_blocks > 0 ?
        params->mixed_format_sample_blocks : 0;
    if (sample_nb <= 0) {
        sample_nb = llama_nvfp4_encoder_sample_blocks(params, nb_total);
        const int64_t sample_cap = params != nullptr && params->mixed_format_sample_cap > 0 ?
            params->mixed_format_sample_cap : LLAMA_NV4MX6_SAMPLE_CAP_DEFAULT;
        if (sample_cap > 0) {
            sample_nb = std::min(sample_nb, sample_cap);
        }
    }
    sample_nb = std::max<int64_t>(1, std::min(sample_nb, nb_total));

    const int worker_count = std::max<int64_t>(1, std::min<int64_t>(sample_nb, nthread > 0 ? nthread : 1));
    std::vector<nv4mx6_tensor_eval> partial((size_t) worker_count);
    std::atomic<int64_t> next_sample { 0 };

    auto eval_worker = [&](int worker_idx) {
        auto & acc = partial[(size_t) worker_idx];
        std::array<float, 32> x{};
        for (;;) {
            const int64_t is = next_sample.fetch_add(1, std::memory_order_relaxed);
            if (is >= sample_nb) {
                break;
            }

            const int64_t src_block = llama_nvfp4_sample_block_index(is, sample_nb, nb_total, row_blocks);
            const int64_t block_col = src_block % row_blocks;
            const int64_t row_global = src_block / row_blocks;
            const int64_t i2 = (row_global / nrows) % ne2;
            const int64_t src_off = row_global * ncols + block_col * 32;

            for (int j = 0; j < 32; ++j) {
                const float v = llama_nv4mx6_get_tensor_f32(tensor, src_off + j);
                x[j] = std::isfinite(v) ? v : 0.0f;
            }

            const float * raw_qw = imatrix ? imatrix + i2 * ncols + block_col * 32 : nullptr;
            std::array<float, 32> shaped_qw{};
            const float * qw = llama_nv4mx6_shape_qw32(raw_qw, params, shaped_qw);
            float scale0 = 0.0f;
            float scale1 = 0.0f;
            const float nv0 = llama_nv4mx6_best_nv4_sse16(x.data() +  0, qw ? qw +  0 : nullptr, &scale0);
            const float nv1 = llama_nv4mx6_best_nv4_sse16(x.data() + 16, qw ? qw + 16 : nullptr, &scale1);
            const float mx6 = llama_nv4mx6_best_mx6_sse32(x.data(), qw);
            const float gap =
                llama_nv4mx6_fp4_gap_sse16(x.data() +  0, qw ? qw +  0 : nullptr, scale0) +
                llama_nv4mx6_fp4_gap_sse16(x.data() + 16, qw ? qw + 16 : nullptr, scale1);

            double x2 = 0.0;
            for (int j = 0; j < 32; ++j) {
                const float w = llama_nv4mx6_qw(qw, j);
                x2 += (double) w * x[j] * x[j];
            }

            acc.sse_nv4 += (double) nv0 + (double) nv1;
            acc.sse_mx6 += (double) mx6;
            acc.gap += (double) gap;
            acc.x2 += x2;
            acc.mx6_wins += mx6 < nv0 + nv1;
            acc.samples++;
        }
    };

    std::vector<std::thread> workers;
    workers.reserve((size_t) std::max<int64_t>(0, worker_count - 1));
    for (int t = 1; t < worker_count; ++t) {
        workers.emplace_back(eval_worker, t);
    }
    eval_worker(0);
    for (auto & worker : workers) {
        worker.join();
    }

    for (const auto & acc : partial) {
        out.sse_nv4 += acc.sse_nv4;
        out.sse_mx6 += acc.sse_mx6;
        out.gap += acc.gap;
        out.x2 += acc.x2;
        out.mx6_wins += acc.mx6_wins;
        out.samples += acc.samples;
    }

    return out.samples > 0;
}

static ggml_type llama_nv4mx6_select_tensor_type(
        const ggml_tensor * tensor,
        ggml_type current_type,
        nv4mx6_policy policy,
        const llama_model_quantize_params * params,
        const float * imatrix,
        float mx6_penalty_override,
        int nthread) {
    if (policy == nv4mx6_policy::OFF || policy == nv4mx6_policy::MX6_SLOT) {
        return current_type;
    }
    if (current_type != GGML_TYPE_NVFP4 && current_type != GGML_TYPE_MXFP6_E2M3) {
        return current_type;
    }

    const double promote_margin = 0.010;
    const double demote_margin  = 0.0025;
    const double gap_lambda     = 0.035;
    const double nv4_penalty    = 1.0;
    const tensor_category category = tensor_get_category(tensor->name);
    const double mx6_base_default = policy == nv4mx6_policy::AUTO ? 12.0 : 1.0;
    const double mx6_base_penalty = mx6_penalty_override > 0.0f ? (double) mx6_penalty_override : mx6_base_default;
    const double mx6_penalty =
        mx6_base_penalty * llama_nv4mx6_category_mx6_multiplier(category, policy);

    nv4mx6_tensor_eval ev;
    bool actual_eval = false;
    if (policy == nv4mx6_policy::MX6_DEMOTE_NV4 && current_type == GGML_TYPE_MXFP6_E2M3) {
        actual_eval = llama_nv4mx6_eval_tensor_actual_demote(tensor, params, imatrix, nthread, ev);
    }
    if (!actual_eval && !llama_nv4mx6_eval_tensor_proxy(tensor, params, imatrix, ev, nthread)) {
        return current_type;
    }

    const double obj_nv4 = nv4_penalty * ev.sse_nv4 + gap_lambda * ev.gap;
    const double obj_mx6 = mx6_penalty * ev.sse_mx6;
    const double tie_eps = 1e-24 * std::max({1.0, std::fabs(obj_nv4), std::fabs(obj_mx6)});

    ggml_type choice = current_type;
    switch (policy) {
        case nv4mx6_policy::NV4_PROMOTE_MX6:
            choice = obj_mx6 < obj_nv4 * (1.0 - promote_margin) ? GGML_TYPE_MXFP6_E2M3 : GGML_TYPE_NVFP4;
            break;
        case nv4mx6_policy::MX6_DEMOTE_NV4:
            choice = obj_nv4 <= obj_mx6 * (1.0 + demote_margin) + tie_eps ? GGML_TYPE_NVFP4 : GGML_TYPE_MXFP6_E2M3;
            break;
        case nv4mx6_policy::AUTO:
            choice = obj_mx6 < obj_nv4 * (1.0 - promote_margin) ? GGML_TYPE_MXFP6_E2M3 : GGML_TYPE_NVFP4;
            break;
        default:
            break;
    }

    if (choice != current_type || llama_nv4mx6_trace_enabled()) {
        LLAMA_LOG_INFO("%s: tensor=%s category=%s policy=%s eval=%s samples=%" PRId64 " choice=%s base=%s mx6_penalty=%.4f obj_nv4=%.6e obj_mx6=%.6e sse_nv4=%.6e sse_mx6=%.6e gap=%.6e mx6_win=%.1f%% x2=%.6e\n",
                __func__, tensor->name, llama_nv4mx6_category_name(category), llama_nv4mx6_policy_name(policy),
                actual_eval ? "actual_cuda" : "proxy", ev.samples,
                ggml_type_name(choice), ggml_type_name(current_type),
                mx6_penalty,
                obj_nv4, obj_mx6, ev.sse_nv4, ev.sse_mx6, ev.gap,
                ev.samples > 0 ? 100.0 * (double) ev.mx6_wins / (double) ev.samples : 0.0,
                ev.x2);
    }

    return choice;
}

static bool llama_nv4mx6_bf16_should_use_mx6(
        const ggml_tensor * tensor,
        const llama_model_quantize_params * params,
        const float * imatrix,
        nv4mx6_policy policy,
        float max_sse_ratio,
        int nthread) {
    if (policy == nv4mx6_policy::BF16_MX6) {
        LLAMA_LOG_INFO("%s: tensor=%s policy=%s choice=%s\n",
                __func__, tensor->name, llama_nv4mx6_policy_name(policy), ggml_type_name(GGML_TYPE_MXFP6_E2M3));
        return true;
    }

    if (policy != nv4mx6_policy::BF16_MX6_SSE) {
        return false;
    }

    nv4mx6_tensor_eval ev;
    if (!llama_nv4mx6_eval_tensor_proxy(tensor, params, imatrix, ev, nthread)) {
        return false;
    }

    const double ratio = ev.x2 > 0.0 ? ev.sse_mx6 / ev.x2 : 0.0;
    const bool use_mx6 = ratio <= (double) std::max(0.0f, max_sse_ratio);
    LLAMA_LOG_INFO("%s: tensor=%s policy=%s samples=%" PRId64 " choice=%s mx6_sse_ratio=%.8e threshold=%.8e sse_mx6=%.6e x2=%.6e\n",
            __func__, tensor->name, llama_nv4mx6_policy_name(policy), ev.samples,
            ggml_type_name(use_mx6 ? GGML_TYPE_MXFP6_E2M3 : tensor->type),
            ratio, (double) std::max(0.0f, max_sse_ratio), ev.sse_mx6, ev.x2);
    return use_mx6;
}

static std::string llama_quant_json_escape(const std::string & value) {
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

static bool llama_quantize_env_flag(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    if (std::strcmp(value, "0") == 0 ||
            llama_nvfp4_ascii_iequals(value, "false") ||
            llama_nvfp4_ascii_iequals(value, "no") ||
            llama_nvfp4_ascii_iequals(value, "off")) {
        return false;
    }
    return true;
}

static uint64_t llama_quantize_parse_u64(const std::string & value, uint64_t fallback) {
    if (value.empty()) {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const uint64_t parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || (end != nullptr && *end != '\0')) {
        return fallback;
    }
    return parsed;
}

static size_t llama_tensor_payload_nbytes_for_type(const ggml_tensor * tensor, ggml_type target_type) {
    ggml_tensor tmp = *tensor;
    const size_t  type_size = ggml_type_size(target_type);
    const int64_t blck_size = ggml_blck_size(target_type);

    tmp.type = target_type;
    GGML_ASSERT(tmp.ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tmp.nb[0] = type_size;
    tmp.nb[1] = tmp.nb[0] * (tmp.ne[0] / blck_size);
    for (int i = 2; i < GGML_MAX_DIMS; ++i) {
        tmp.nb[i] = tmp.nb[i - 1] * tmp.ne[i - 1];
    }
    return ggml_nbytes(&tmp);
}

struct llama_quantize_resume_state {
    bool enabled = false;
    bool active = false;
    std::string path;
    size_t next_tensor = 0;
    uint64_t file_size = 0;
    size_t total_size_org = 0;
    size_t total_size_new = 0;
};

static void llama_quantize_resume_remove(const llama_quantize_resume_state & resume) {
    if (!resume.enabled || resume.path.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::remove(resume.path, ec);
    std::filesystem::remove(resume.path + ".tmp", ec);
}

static bool llama_quantize_resume_load(
        const std::string & state_path,
        const std::string & fname_inp,
        const std::string & fname_out,
        llama_ftype ftype,
        size_t n_tensors,
        size_t meta_size,
        llama_quantize_resume_state & resume) {
    std::ifstream in(state_path);
    if (!in) {
        return false;
    }

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        kv.emplace(line.substr(0, eq), line.substr(eq + 1));
    }

    const uint64_t saved_meta_size = llama_quantize_parse_u64(kv["meta_size"], UINT64_MAX);
    if (kv["schema"] != "advanced-gguf-quantize-resume-v1" ||
            kv["input"] != fname_inp ||
            kv["output"] != fname_out ||
            llama_quantize_parse_u64(kv["ftype"], UINT64_MAX) != (uint64_t) ftype ||
            llama_quantize_parse_u64(kv["n_tensors"], UINT64_MAX) != (uint64_t) n_tensors ||
            (saved_meta_size != 0 && saved_meta_size != (uint64_t) meta_size)) {
        return false;
    }

    const uint64_t next_tensor = llama_quantize_parse_u64(kv["next_tensor"], UINT64_MAX);
    const uint64_t file_size = llama_quantize_parse_u64(kv["file_size"], 0);
    if (next_tensor == UINT64_MAX || next_tensor > n_tensors || file_size < meta_size) {
        return false;
    }

    std::error_code ec;
    if (!std::filesystem::exists(fname_out, ec) || ec) {
        return false;
    }
    const uint64_t actual_size = (uint64_t) std::filesystem::file_size(fname_out, ec);
    if (ec || actual_size < file_size) {
        return false;
    }
    if (actual_size > file_size) {
        std::filesystem::resize_file(fname_out, file_size, ec);
        if (ec) {
            return false;
        }
    }

    resume.path = state_path;
    resume.next_tensor = (size_t) next_tensor;
    resume.file_size = file_size;
    resume.total_size_org = (size_t) llama_quantize_parse_u64(kv["total_size_org"], 0);
    resume.total_size_new = (size_t) llama_quantize_parse_u64(kv["total_size_new"], 0);
    resume.active = resume.next_tensor > 0;
    return resume.active;
}

static void llama_quantize_resume_save(
        const llama_quantize_resume_state & resume,
        const std::string & fname_inp,
        const std::string & fname_out,
        llama_ftype ftype,
        size_t n_tensors,
        size_t meta_size,
        size_t next_tensor,
        uint64_t file_size,
        size_t total_size_org,
        size_t total_size_new) {
    if (!resume.enabled || resume.path.empty()) {
        return;
    }

    const std::string tmp_path = resume.path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::trunc);
        if (!out) {
            LLAMA_LOG_WARN("%s: failed to open resume sidecar %s\n", __func__, tmp_path.c_str());
            return;
        }
        out << "schema=advanced-gguf-quantize-resume-v1\n"
            << "input=" << fname_inp << "\n"
            << "output=" << fname_out << "\n"
            << "ftype=" << (int) ftype << "\n"
            << "n_tensors=" << n_tensors << "\n"
            << "meta_size=" << meta_size << "\n"
            << "next_tensor=" << next_tensor << "\n"
            << "file_size=" << file_size << "\n"
            << "total_size_org=" << total_size_org << "\n"
            << "total_size_new=" << total_size_new << "\n";
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, resume.path, ec);
    if (ec) {
        std::filesystem::remove(resume.path, ec);
        ec.clear();
        std::filesystem::rename(tmp_path, resume.path, ec);
        if (ec) {
            LLAMA_LOG_WARN("%s: failed to publish resume sidecar %s: %s\n",
                    __func__, resume.path.c_str(), ec.message().c_str());
        }
    }
}

//
// main quantization driver
//

static void init_quantize_state_counters(quantize_state_impl & qs, std::vector<tensor_metadata> & metadata) {
    for (auto & tm : metadata) {
        tensor_category cat = tensor_get_category(tm.name);
        tm.category = cat;
        tm.is_mtp = tensor_name_is_mtp(qs.model, tm.name);

        if (category_is_attn_v(cat)) {
            ++qs.n_attention_wv;
        }

        if (cat == tensor_category::OUTPUT) {
            qs.has_tied_embeddings = false;
        }
    }
    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer_all;
}

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    llama_ftype ftype = params->ftype;

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    ggml_type default_type = llama_ftype_get_default_type(ftype);
    if (default_type == GGML_TYPE_COUNT) {
        throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    const llama_model_kv_override * kv_overrides = params->kv_overrides;

    std::vector<std::string> splits = {};
    llama_model_loader ml(/*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr,
        fname_inp, splits, /*file*/ nullptr, use_mmap, /*use_direct_io*/ false, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    std::unique_ptr<llama_model_loader> patch_ml;
    std::vector<std::string> patch_splits = {};
    if (params->patch_base_model != nullptr && params->patch_base_model[0] != '\0') {
        patch_ml = std::make_unique<llama_model_loader>(
            /*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr,
            params->patch_base_model, patch_splits, /*file*/ nullptr, use_mmap, /*use_direct_io*/ false,
            /*check_tensors*/ false, /*no_alloc*/ false, /*param_overrides_p*/ nullptr, /*param_tensor_buft_overrides_p*/ nullptr);
        patch_ml->init_mappings(false);
        LLAMA_LOG_INFO("%s: patching unchanged tensors from %s\n", __func__, params->patch_base_model);
    }

    auto mparams = llama_model_default_params();
    std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, mparams));

    auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
    if (model == nullptr) {
        GGML_ABORT("fatal error: model does not implement llama_model_base");
    }

    model->load_hparams(ml);
    model->load_stats  (ml);

    quantize_state_impl qs(*model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    std::unordered_map<std::string, std::vector<float>> imatrix_storage;
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        for (const llama_model_imatrix_data * entry = params->imatrix; entry->name != nullptr; ++entry) {
            imatrix_storage.emplace(
                std::string(entry->name),
                std::vector<float>(entry->data, entry->data + entry->size));
        }
        imatrix_data = &imatrix_storage;
        LLAMA_LOG_INFO("\n%s: have importance matrix data with %d entries\n",
                       __func__, (int)imatrix_data->size());
        qs.has_imatrix = true;
        // check imatrix for nans or infs
        for (const auto & kv : *imatrix_data) {
            for (float f : kv.second) {
                if (!std::isfinite(f)) {
                    throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                }
            }
        }
    }
    const nv4mx6_policy nv4mx6_policy_mode = llama_nv4mx6_policy_from_param(params->nv4mx6_policy);
    if (nv4mx6_policy_mode != nv4mx6_policy::OFF) {
        LLAMA_LOG_INFO("%s: NVFP4/MXFP6_E2M3 quantizer policy: %s\n", __func__, llama_nv4mx6_policy_name(nv4mx6_policy_mode));
    }
    const float nvfp4_correction_denom = llama_nvfp4_resolve_correction_denom(params);
    const int32_t nvfp4_input_scale_policy = llama_nvfp4_resolve_input_scale_policy(params);
    if (params->ftype == LLAMA_FTYPE_MOSTLY_NVFP4) {
        LLAMA_LOG_INFO("%s: NVFP4 scales: correction_denom=%.6g input_policy=%s input_scale=file-domain\n",
                __func__,
                (double) nvfp4_correction_denom,
                llama_nvfp4_input_scale_policy_name(nvfp4_input_scale_policy));
    }
    if (params->ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3 && imatrix_data != nullptr) {
        const float mx6_denom = llama_mxfp6_input_scale_denom(params);
        const float mx6_q = llama_mxfp6_input_scale_quantile(params);
        LLAMA_LOG_INFO("%s: MXFP6_E2M3 input scales: imatrix-sqrtp%.6g / denom=%.6g\n",
                __func__,
                (double) mx6_q,
                (double) mx6_denom);
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        for (const int32_t * layer = params->prune_layers; *layer >= 0; ++layer) {
            prune_list.push_back(*layer);
        }
    }

    // For surgical patching, preserve the base quantized metadata contract by default.
    gguf_set_kv     (ctx_out.get(), patch_ml ? patch_ml->metadata : ml.metadata);
    if (patch_ml) {
        auto copy_source_u32 = [&](enum llm_kv kv) {
            const std::string key = ml.llm_kv(kv);
            const int kid = gguf_find_key(ml.metadata, key.c_str());
            if (kid >= 0) {
                gguf_set_val_u32(ctx_out.get(), key.c_str(), gguf_get_val_u32(ml.metadata, kid));
            }
        };
        copy_source_u32(LLM_KV_BLOCK_COUNT);
        copy_source_u32(LLM_KV_NEXTN_PREDICT_LAYERS);
    }
    gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_GENERAL_QUANTIZATION_VERSION).c_str(), GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_GENERAL_FILE_TYPE).c_str(), ftype);

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        for (const llama_model_kv_override * o = params->kv_overrides; o->key[0] != 0; ++o) {
            if (o->tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o->key, o->val_f64);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                // Setting type to UINT32. See https://github.com/ggml-org/llama.cpp/pull/14182 for context
                gguf_set_val_u32(ctx_out.get(), o->key, (uint32_t)std::abs(o->val_i64));
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o->key, o->val_bool);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o->key, o->val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o->key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }

        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n", __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }
        tensors.push_back(&it.second);
    }
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    // compute tensor metadata once and cache it
    std::vector<tensor_metadata> metadata(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        metadata[i].name = ggml_get_name(tensors[i]->tensor);
    }

    // initialize quantization state counters and metadata categories
    init_quantize_state_counters(qs, metadata);

    int idx = 0;
    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

	    struct nvfp4_aux_tensor_info {
	        std::string scale_name;
	        std::string input_scale_name;
	        uint16_t split = 0;
	        std::vector<float> scale_values;
	        std::vector<float> input_scale_values;
	    };
    std::unordered_map<std::string, nvfp4_aux_tensor_info> nvfp4_aux_tensors;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> nvfp4_scale_ctx(
        ggml_init({
	            /*mem_size   =*/ ggml_tensor_overhead() * (2 * tensors.size() + 1) + 1024,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ true,
        }),
        ggml_free);
    // flag for --dry-run
    bool will_require_imatrix = false;

    //
    // preliminary iteration over all weights
    //

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * it = tensors[i];
        const struct ggml_tensor * tensor = it->tensor;
        const std::string & name = metadata[i].name;

        uint16_t i_split = params->keep_split ? it->idx : 0;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);

        metadata[i].allows_quantization = tensor_allows_quantization(params, model->arch, tensor);
        const tensor_type_option * manual_opt = llama_tensor_find_manual_override(qs, tensor);
        const ggml_type manual_type = manual_opt ? manual_opt->type : GGML_TYPE_COUNT;
        const bool explicit_token_type =
            metadata[i].allows_quantization &&
            params->token_embedding_type < GGML_TYPE_COUNT &&
            metadata[i].category == tensor_category::TOKEN_EMBD;
        const bool explicit_output_type =
            metadata[i].allows_quantization &&
            params->output_tensor_type < GGML_TYPE_COUNT &&
            metadata[i].category == tensor_category::OUTPUT;
        const bool explicit_mtp_type =
            metadata[i].allows_quantization &&
            metadata[i].is_mtp &&
            params->mtp_tensor_type < GGML_TYPE_COUNT;
        const bool protected_mtp_type = metadata[i].is_mtp;
        const bool explicit_category_type = explicit_token_type || explicit_output_type || explicit_mtp_type || protected_mtp_type;

        const llama_model_loader::llama_tensor_weight * patch_weight =
            patch_ml ? patch_ml->get_weight(name.c_str()) : nullptr;
        const bool patch_shape_match =
            patch_weight != nullptr &&
            patch_weight->tensor != nullptr &&
            ggml_are_same_shape(patch_weight->tensor, tensor);

        if (explicit_mtp_type) {
            metadata[i].target_type = tensor_type_fallback(qs, tensor, params->mtp_tensor_type);
        } else if (metadata[i].is_mtp && manual_type == GGML_TYPE_COUNT) {
            metadata[i].target_type = tensor->ne[1] > 1 ? tensor_type_fallback(qs, tensor, GGML_TYPE_Q8_0) : tensor->type;
        } else if (manual_type != GGML_TYPE_COUNT) {
            metadata[i].target_type = tensor_type_fallback(qs, tensor, manual_type);
        } else if (explicit_token_type) {
            metadata[i].target_type = tensor_type_fallback(qs, tensor, params->token_embedding_type);
        } else if (explicit_output_type) {
            metadata[i].target_type = tensor_type_fallback(qs, tensor, params->output_tensor_type);
        } else if (patch_shape_match) {
            metadata[i].target_type = patch_weight->tensor->type;
        } else if (metadata[i].allows_quantization) {
            metadata[i].target_type = llama_tensor_get_type(qs, params, tensor, default_type, metadata[i]);
        } else {
            metadata[i].target_type = tensor->type;
        }
        metadata[i].target_type = tensor_type_avoid_nvfp4_token_embedding(qs, tensor, metadata[i], metadata[i].target_type);

        if (params->imatrix) {
            metadata[i].remapped_imatrix_name = remap_imatrix(tensor->name, mapped);
        }

        if (!params->dry_run &&
            ml.use_mmap &&
            manual_type == GGML_TYPE_COUNT &&
            !explicit_category_type &&
            !patch_shape_match &&
            (nv4mx6_policy_mode == nv4mx6_policy::NV4_PROMOTE_MX6 ||
             nv4mx6_policy_mode == nv4mx6_policy::MX6_DEMOTE_NV4 ||
             nv4mx6_policy_mode == nv4mx6_policy::AUTO) &&
            (metadata[i].target_type == GGML_TYPE_NVFP4 || metadata[i].target_type == GGML_TYPE_MXFP6_E2M3) &&
            (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16)) {
            const float * selector_imatrix = nullptr;
            if (imatrix_data) {
                const auto imi = imatrix_data->find(metadata[i].remapped_imatrix_name);
                if (imi != imatrix_data->end() && imi->second.size() == (size_t) tensor->ne[0] * (size_t) std::max<int64_t>(1, tensor->ne[2])) {
                    selector_imatrix = imi->second.data();
                }
            }
            ggml_tensor * tensor_mut = it->tensor;
            ml.load_data_for(tensor_mut);
            metadata[i].target_type = llama_nv4mx6_select_tensor_type(tensor_mut, metadata[i].target_type, nv4mx6_policy_mode, params, selector_imatrix, params->nv4mx6_mx6_penalty, nthread);
            metadata[i].target_type = tensor_type_fallback(qs, tensor, metadata[i].target_type);
            metadata[i].target_type = tensor_type_avoid_nvfp4_token_embedding(qs, tensor, metadata[i], metadata[i].target_type);
        }

        if (metadata[i].target_type != tensor->type &&
            (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) &&
            ggml_is_quantized(metadata[i].target_type)) {
            const size_t src_nbytes = ggml_nbytes(tensor);
            const size_t dst_nbytes = llama_tensor_projected_nbytes(tensor, metadata[i].target_type);
            size_t min_savings = 0;
            if (manual_type == GGML_TYPE_COUNT && !explicit_category_type) {
                if (metadata[i].target_type == GGML_TYPE_NVFP4) {
                    min_savings = LLAMA_NVFP4_MIN_SAVINGS_BYTES;
                } else if (metadata[i].target_type == GGML_TYPE_MXFP6_E2M3) {
                    min_savings = (size_t) std::max<int64_t>(0, params->mxfp6_min_savings_bytes);
                } else {
                    min_savings = LLAMA_QUANT_MIN_SAVINGS_BYTES;
                }
            }
            if (src_nbytes > dst_nbytes && (src_nbytes - dst_nbytes) < min_savings) {
                metadata[i].target_type = tensor->type;
            }
        }

        if (!params->dry_run &&
            ml.use_mmap &&
            manual_type == GGML_TYPE_COUNT &&
            !explicit_category_type &&
            tensor->type == GGML_TYPE_BF16 &&
            metadata[i].target_type == tensor->type &&
            metadata[i].allows_quantization &&
            (nv4mx6_policy_mode == nv4mx6_policy::BF16_MX6 ||
             nv4mx6_policy_mode == nv4mx6_policy::BF16_MX6_SSE)) {
            const float * selector_imatrix = nullptr;
            if (imatrix_data) {
                const auto imi = imatrix_data->find(metadata[i].remapped_imatrix_name);
                if (imi != imatrix_data->end() && imi->second.size() == (size_t) tensor->ne[0] * (size_t) std::max<int64_t>(1, tensor->ne[2])) {
                    selector_imatrix = imi->second.data();
                }
            }
            ggml_tensor * tensor_mut = it->tensor;
            ml.load_data_for(tensor_mut);
            if (llama_nv4mx6_bf16_should_use_mx6(tensor_mut, params, selector_imatrix, nv4mx6_policy_mode, params->nv4mx6_bf16_mx6_max_sse_ratio, nthread)) {
                metadata[i].target_type = tensor_type_fallback(qs, tensor, GGML_TYPE_MXFP6_E2M3);
            }
        }

        metadata[i].requires_imatrix = tensor_requires_imatrix(tensor->name, metadata[i].target_type, ftype);
        metadata[i].copy_from_patch = false;
        metadata[i].has_nvfp4_cfg_override =
            manual_opt != nullptr &&
            manual_opt->type == GGML_TYPE_NVFP4 &&
            manual_opt->has_nvfp4_cfg;
	        if (metadata[i].has_nvfp4_cfg_override) {
	            metadata[i].nvfp4_cfg_override = manual_opt->nvfp4_cfg;
	            metadata[i].nvfp4_sample_blocks = std::max<int64_t>(0, manual_opt->nvfp4_sample_blocks);
	            metadata[i].nvfp4_policy_name = manual_opt->nvfp4_policy_name;
	            const int64_t n_slices = std::max<int64_t>(1, tensor->ne[2]);
	            metadata[i].has_nvfp4_scale_overrides =
	                manual_opt->nvfp4_scale_overrides.size() == (size_t) n_slices &&
	                manual_opt->nvfp4_base_scale_overrides.size() == (size_t) n_slices;
	            if (metadata[i].has_nvfp4_scale_overrides) {
	                metadata[i].nvfp4_base_scale_overrides = manual_opt->nvfp4_base_scale_overrides;
	                metadata[i].nvfp4_scale_overrides = manual_opt->nvfp4_scale_overrides;
	            } else {
	                metadata[i].nvfp4_base_scale_overrides.clear();
	                metadata[i].nvfp4_scale_overrides.clear();
	            }
	        } else {
	            metadata[i].nvfp4_sample_blocks = 0;
	            metadata[i].nvfp4_policy_name.clear();
	            metadata[i].has_nvfp4_scale_overrides = false;
	            metadata[i].nvfp4_base_scale_overrides.clear();
	            metadata[i].nvfp4_scale_overrides.clear();
	        }
        metadata[i].has_mxfp6_scale_mul =
            manual_opt != nullptr &&
            manual_opt->type == GGML_TYPE_MXFP6_E2M3 &&
            manual_opt->has_mxfp6_scale_mul &&
            std::isfinite(manual_opt->mxfp6_e2m3_scale_mul) &&
            manual_opt->mxfp6_e2m3_scale_mul > 0.0f;
        if (metadata[i].has_mxfp6_scale_mul) {
            metadata[i].mxfp6_e2m3_scale_mul = manual_opt->mxfp6_e2m3_scale_mul;
            metadata[i].mxfp6_policy_name = manual_opt->mxfp6_policy_name;
        } else {
            metadata[i].mxfp6_e2m3_scale_mul = 1.0f;
            metadata[i].mxfp6_policy_name.clear();
        }
        metadata[i].has_k_rsf_mode_override =
            manual_opt != nullptr &&
            manual_opt->has_k_rsf_mode &&
            manual_opt->type == metadata[i].target_type &&
            llama_tensor_type_has_k_rsf(metadata[i].target_type);
        metadata[i].k_rsf_mode_override = metadata[i].has_k_rsf_mode_override
            ? std::max<int32_t>(1, std::min<int32_t>(3, manual_opt->k_rsf_mode))
            : 0;

        if (patch_ml) {
            if (patch_weight != nullptr && patch_weight->tensor != nullptr) {
                const ggml_tensor * patch_tensor = patch_weight->tensor;
                const bool type_match  = patch_tensor->type == metadata[i].target_type;
                const bool shape_match = ggml_are_same_shape(patch_tensor, tensor);
                bool aux_match = true;

                if (type_match && shape_match && llama_tensor_has_aux_scale_slot(name) &&
                        metadata[i].target_type == GGML_TYPE_NVFP4) {
                    const auto * scale_weight = patch_ml->get_weight(llama_nvfp4_scale_tensor_name(name).c_str());
                    aux_match = scale_weight != nullptr && scale_weight->tensor != nullptr;
                    if (metadata[i].target_type == GGML_TYPE_NVFP4) {
                        const auto * input_scale_weight = llama_nvfp4_find_input_scale_weight(patch_ml.get(), name);
                        aux_match =
                            aux_match &&
                            input_scale_weight != nullptr &&
                            input_scale_weight->tensor != nullptr;
                    }
	                }

                metadata[i].copy_from_patch =
                    type_match && shape_match && aux_match &&
                    !metadata[i].has_nvfp4_cfg_override &&
                    !metadata[i].has_mxfp6_scale_mul &&
                    !metadata[i].has_k_rsf_mode_override;
            }
        }

        const bool needs_native_scale_tensors =
            llama_tensor_has_aux_scale_slot(name) &&
            metadata[i].target_type == GGML_TYPE_NVFP4 &&
            tensor->type != metadata[i].target_type;
        if (needs_native_scale_tensors) {
            const int64_t scale_len = std::max<int64_t>(1, tensor->ne[2]);
            const std::string scale_name = llama_nvfp4_scale_tensor_name(name);
            ggml_tensor * scale_tensor = ggml_new_tensor_1d(nvfp4_scale_ctx.get(), GGML_TYPE_F32, scale_len);
            ggml_set_name(scale_tensor, scale_name.c_str());
            gguf_add_tensor(ctx_outs[i_split].get(), scale_tensor);
            const bool needs_input_scale =
                metadata[i].target_type == GGML_TYPE_NVFP4 ||
                (metadata[i].target_type == GGML_TYPE_MXFP6_E2M3 && imatrix_data != nullptr);
            const std::string input_scale_name = needs_input_scale ? llama_nvfp4_input_scale_tensor_name(name) : std::string();
            if (needs_input_scale) {
                ggml_tensor * input_scale_tensor = ggml_new_tensor_1d(nvfp4_scale_ctx.get(), GGML_TYPE_F32, scale_len);
                ggml_set_name(input_scale_tensor, input_scale_name.c_str());
                gguf_add_tensor(ctx_outs[i_split].get(), input_scale_tensor);
            }
            nvfp4_aux_tensors.emplace(name, nvfp4_aux_tensor_info{
                /*scale_name         =*/ scale_name,
                /*input_scale_name   =*/ input_scale_name,
                /*split              =*/ i_split,
                /*scale_values       =*/ std::vector<float>((size_t) scale_len, 1.0f),
                /*input_scale_values =*/ needs_input_scale ? std::vector<float>((size_t) scale_len, 0.0f) : std::vector<float>(),
                });
        }

        if (!params->imatrix && metadata[i].allows_quantization && metadata[i].requires_imatrix) {
            if (params->dry_run) {
                will_require_imatrix = true;
            } else {
                LLAMA_LOG_ERROR("\n============================================================================\n"
                                " ERROR: this quantization requires an importance matrix!\n"
                                "        - offending tensor: %s\n"
                                "        - target type: %s\n"
                                "============================================================================\n\n",
                                metadata[i].name.c_str(), ggml_type_name(metadata[i].target_type));
                throw std::runtime_error("this quantization requires an imatrix!");
            }
        }
    }

    // Set split info if needed
    if (n_split > 1) {
        size_t n_aux_tensors = 0;
        for (const auto & kv : nvfp4_aux_tensors) {
            n_aux_tensors += 1;
            if (!kv.second.input_scale_values.empty()) {
                n_aux_tensors += 1;
            }
        }
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)(tensors.size() + n_aux_tensors));
        }
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        const std::string name = ggml_get_name(tensors[i]->tensor);
        const uint16_t i_split = params->keep_split ? tensors[i]->idx : 0;
        gguf_remove_key(ctx_outs[i_split].get(), (name + ".weight_scale").c_str());
        gguf_remove_key(ctx_outs[i_split].get(), (name + ".weight_scale_2").c_str());
        gguf_remove_key(ctx_outs[i_split].get(), (name + ".tensor_scale").c_str());
        gguf_remove_key(ctx_outs[i_split].get(), (name + ".input_scale").c_str());
    }

    const char * assignment_jsonl = params->assignment_jsonl;
    if (assignment_jsonl != nullptr && assignment_jsonl[0] != '\0') {
        std::ofstream assignment_out(assignment_jsonl, std::ios::app);
        if (assignment_out) {
            assignment_out << "{\"schema\":\"advanced-gguf-quantizer-assignment-v1\",\"event\":\"writer_tensor_assignments_begin\""
                           << ",\"input\":\"" << llama_quant_json_escape(fname_inp)
                           << "\",\"output\":\"" << llama_quant_json_escape(fname_out) << "\"}\n";
            for (size_t i = 0; i < tensors.size(); ++i) {
                const ggml_tensor * tensor = tensors[i]->tensor;
                const auto & tm = metadata[i];
                const size_t src_bytes = ggml_nbytes(tensor);
                const size_t dst_bytes = llama_tensor_projected_nbytes(tensor, tm.target_type);
                assignment_out
                    << "{\"schema\":\"advanced-gguf-quantizer-assignment-v1\",\"event\":\"tensor_assignment\""
                    << ",\"name\":\"" << llama_quant_json_escape(tm.name) << "\""
                    << ",\"category\":\"" << llama_quant_json_escape(llama_nv4mx6_category_name(tm.category)) << "\""
                    << ",\"source_type\":\"" << ggml_type_name(tensor->type) << "\""
                    << ",\"target_type\":\"" << ggml_type_name(tm.target_type) << "\""
                    << ",\"source_bytes\":" << src_bytes
                    << ",\"target_bytes\":" << dst_bytes
                    << ",\"split\":" << (params->keep_split ? tensors[i]->idx : 0)
                    << ",\"allows_quantization\":" << (tm.allows_quantization ? "true" : "false")
                    << ",\"is_mtp\":" << (tm.is_mtp ? "true" : "false")
                    << ",\"copy_from_patch\":" << (tm.copy_from_patch ? "true" : "false")
                    << ",\"requires_imatrix\":" << (tm.requires_imatrix ? "true" : "false")
                    << ",\"has_nvfp4_aux\":" << (nvfp4_aux_tensors.find(tm.name) != nvfp4_aux_tensors.end() ? "true" : "false")
                    << ",\"nvfp4_policy\":\"" << llama_quant_json_escape(tm.nvfp4_policy_name) << "\""
                    << ",\"mxfp6_policy\":\"" << llama_quant_json_escape(tm.mxfp6_policy_name) << "\""
                    << "}\n";
            }
            assignment_out << "{\"schema\":\"advanced-gguf-quantizer-assignment-v1\",\"event\":\"writer_tensor_assignments_end\""
                           << ",\"tensors\":" << tensors.size() << "}\n";
        } else {
            LLAMA_LOG_WARN("%s: failed to write assignment JSONL %s\n", __func__, assignment_jsonl);
        }
    }


    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    if (!params->dry_run && patch_ml) {
        std::vector<std::string> patch_validate_names;
        std::unordered_set<std::string> patch_validate_seen;
        patch_validate_names.reserve(tensors.size());
        auto add_patch_validate_name = [&](const std::string & tensor_name) {
            if (!tensor_name.empty() && patch_validate_seen.insert(tensor_name).second) {
                patch_validate_names.push_back(tensor_name);
            }
        };

        for (size_t i = 0; i < tensors.size(); ++i) {
            if (!metadata[i].copy_from_patch) {
                continue;
            }
            add_patch_validate_name(metadata[i].name);
            auto aux_it = nvfp4_aux_tensors.find(metadata[i].name);
            if (aux_it != nvfp4_aux_tensors.end()) {
                add_patch_validate_name(aux_it->second.scale_name);
                add_patch_validate_name(aux_it->second.input_scale_name);
            }
        }

        if (!patch_validate_names.empty()) {
            const int validate_threads = std::max<int>(1, std::min<int>((int) patch_validate_names.size(), nthread));
            LLAMA_LOG_INFO("%s: patch-base prefetch/validate %zu tensor(s) using %d thread(s)\n",
                    __func__, patch_validate_names.size(), validate_threads);

            std::atomic<size_t> next_patch_validate{0};
            std::atomic<bool> patch_validate_ok{true};
            std::exception_ptr patch_validate_error;
            std::mutex patch_validate_error_mutex;

            auto validate_worker = [&]() {
                while (patch_validate_ok.load(std::memory_order_relaxed)) {
                    const size_t job = next_patch_validate.fetch_add(1, std::memory_order_relaxed);
                    if (job >= patch_validate_names.size()) {
                        break;
                    }

                    try {
                        const auto & patch_weight = patch_ml->require_weight(patch_validate_names[job].c_str());
                        ggml_tensor * patch_tensor = patch_weight.tensor;
                        patch_ml->load_data_for(patch_tensor);
                        if (!ggml_validate_row_data(patch_tensor->type, patch_tensor->data, ggml_nbytes(patch_tensor))) {
                            throw std::runtime_error(format("patch tensor '%s' has invalid data", patch_validate_names[job].c_str()));
                        }
                    } catch (...) {
                        patch_validate_ok.store(false, std::memory_order_relaxed);
                        std::lock_guard<std::mutex> lock(patch_validate_error_mutex);
                        if (!patch_validate_error) {
                            patch_validate_error = std::current_exception();
                        }
                        break;
                    }
                }
            };

            workers.clear();
            workers.reserve((size_t) std::max(0, validate_threads - 1));
            for (int it = 1; it < validate_threads; ++it) {
                workers.emplace_back(validate_worker);
            }
            validate_worker();
            for (auto & worker : workers) {
                worker.join();
            }
            workers.clear();

            if (patch_validate_error) {
                std::rethrow_exception(patch_validate_error);
            }
            if (!patch_validate_ok.load(std::memory_order_relaxed)) {
                throw std::runtime_error("patch-base prefetch/validate failed");
            }
        }
    }

    llama_quantize_resume_state resume;
    resume.enabled = !params->dry_run && llama_quantize_env_flag("LLAMA_QUANTIZE_RESUME_OUTPUT");
    if (resume.enabled && params->keep_split) {
        LLAMA_LOG_WARN("%s: resumable output is disabled for split GGUF writes\n", __func__);
        resume.enabled = false;
    }
    if (resume.enabled) {
        resume.path = fname_out + ".resume";
    }

    size_t writer_total_bytes = 0;
    if (!params->dry_run) {
        for (const auto & ctx : ctx_outs) {
            if (ctx) {
                writer_total_bytes += gguf_get_meta_size(ctx.get());
            }
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            writer_total_bytes += GGML_PAD(llama_tensor_payload_nbytes_for_type(tensors[i]->tensor, metadata[i].target_type), align);
            const auto aux_it = nvfp4_aux_tensors.find(metadata[i].name);
            if (aux_it != nvfp4_aux_tensors.end()) {
                const size_t scale_size = aux_it->second.scale_values.size() * sizeof(float);
                const size_t input_scale_size = aux_it->second.input_scale_values.size() * sizeof(float);
                writer_total_bytes += GGML_PAD(scale_size, align);
                if (input_scale_size > 0) {
                    writer_total_bytes += GGML_PAD(input_scale_size, align);
                }
            }
        }
    }
    size_t writer_bytes_done = 0;
    int writer_last_pct = -1;
    auto writer_last_log = std::chrono::steady_clock::now() - std::chrono::seconds(60);
    auto writer_bar = [](double progress) {
        static constexpr int width = 28;
        std::string bar((size_t) width, '-');
        const int filled = std::max(0, std::min(width, (int) std::floor(progress * width)));
        for (int i = 0; i < filled; ++i) {
            bar[(size_t) i] = '=';
        }
        if (filled < width) {
            bar[(size_t) filled] = '>';
        }
        return bar;
    };
    auto writer_progress = [&](const char * detail, bool force = false) {
        if (params->dry_run || writer_total_bytes == 0) {
            return;
        }
        const double progress = std::min(1.0, (double) writer_bytes_done / (double) writer_total_bytes);
        const int pct = std::max(0, std::min(100, (int) std::floor(progress * 100.0 + 0.5)));
        const auto now = std::chrono::steady_clock::now();
        const bool pct_step = writer_last_pct < 0 || pct >= writer_last_pct + 5 || pct == 100;
        if (!force && !pct_step && now - writer_last_log < std::chrono::seconds(30)) {
            return;
        }
        writer_last_pct = pct;
        writer_last_log = now;
        LLAMA_LOG_INFO(
                "%s: Writing GGUF to disk... %3d%% [%s] %.2f/%.2f GiB%s%s\n",
                __func__,
                pct,
                writer_bar(progress).c_str(),
                (double) writer_bytes_done / (1024.0 * 1024.0 * 1024.0),
                (double) writer_total_bytes / (1024.0 * 1024.0 * 1024.0),
                detail != nullptr && detail[0] != '\0' ? " " : "",
                detail != nullptr ? detail : "");
    };

    int cur_split = -1;
    std::fstream fout;
    std::vector<size_t> meta_placeholder_size(n_split, 0);
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            const size_t final_meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
            if (final_meta_size != meta_placeholder_size[cur_split]) {
                throw std::runtime_error(format(
                        "GGUF metadata size changed after data write: split %d reserved %zu bytes, final %zu bytes",
                        cur_split, meta_placeholder_size[cur_split], final_meta_size));
            }
            std::vector<uint8_t> data(final_meta_size);
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path.data());
        }

        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        meta_placeholder_size[cur_split] = meta_size;

        if (resume.enabled && cur_split == 0) {
            llama_quantize_resume_load(
                    resume.path, fname_inp, fname, ftype, tensors.size(), meta_size, resume);
        }

        if (resume.active && cur_split == 0) {
            LLAMA_LOG_INFO("%s: resuming output %s from tensor %zu/%zu at byte %" PRIu64 "\n",
                    __func__, fname.c_str(), resume.next_tensor, tensors.size(), resume.file_size);
            fout = std::fstream(fname, std::ios::binary | std::ios::in | std::ios::out);
            fout.exceptions(std::fstream::failbit | std::fstream::badbit); // fail fast on write errors
            fout.seekp((std::streamoff) resume.file_size);
            writer_bytes_done = std::max(writer_bytes_done, (size_t) resume.file_size);
        } else {
            fout = std::fstream(fname, std::ios::binary | std::ios::out | std::ios::trunc);
            fout.exceptions(std::fstream::failbit | std::fstream::badbit); // fail fast on write errors
            // placeholder for the meta data
            ::zeros(fout, meta_size);
            writer_bytes_done += meta_size;
        }
        writer_progress("opened output", true);
    };

    // no output file for --dry-run
    if (!params->dry_run) {
        new_ofstream(0);
    }

    //
    // main loop: iterate over all weights
    //

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & weight = *tensors[i];
        const auto & tm = metadata[i];
        ggml_tensor * tensor = weight.tensor;
        const std::string & name = tm.name;

        if (!params->dry_run && (weight.idx != cur_split && params->keep_split)) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const size_t tensor_size = ggml_nbytes(tensor);

        const ggml_type cur_type = tensor->type;
        const ggml_type new_type = tm.target_type;

        // If we've decided to quantize to the same type the tensor is already
        // in then there's nothing to do.
        bool quantize = cur_type != new_type;

        void * new_data;
        size_t new_size;
        size_t auxiliary_size = 0;
        auto nvfp4_aux_it = nvfp4_aux_tensors.find(name);
        nvfp4_aux_tensor_info * nvfp4_aux_info = nvfp4_aux_it != nvfp4_aux_tensors.end() ? &nvfp4_aux_it->second : nullptr;
        if (nvfp4_aux_info != nullptr) {
            auxiliary_size =
                nvfp4_aux_info->scale_values.size() * sizeof(float) +
                nvfp4_aux_info->input_scale_values.size() * sizeof(float);
        }

        if (!params->dry_run && resume.active && i < resume.next_tensor) {
            const size_t new_size = llama_tensor_payload_nbytes_for_type(tensor, new_type);

            LLAMA_LOG_INFO("[%4d/%4d] %-36s - [%s], type = %6s, resumed as %6s, size = %8.2f MiB -> %8.2f MiB\n",
                   ++idx, ml.n_tensors,
                   ggml_get_name(tensor),
                   llama_format_tensor_shape(tensor).c_str(),
                   ggml_type_name(tensor->type),
                   ggml_type_name(new_type),
                   tensor_size/1024.0/1024.0,
                   (new_size + auxiliary_size)/1024.0/1024.0);

            total_size_org += tensor_size;
            total_size_new += new_size + auxiliary_size;

            gguf_set_tensor_type(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), metadata[i].name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_outs[cur_split].get(), metadata[i].name.c_str(), nullptr);

            if (nvfp4_aux_info != nullptr) {
                const size_t scale_size = nvfp4_aux_info->scale_values.size() * sizeof(float);
                const size_t input_scale_size = nvfp4_aux_info->input_scale_values.size() * sizeof(float);

                GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), nvfp4_aux_info->scale_name.c_str())) == scale_size);
                gguf_set_tensor_data(ctx_outs[cur_split].get(), nvfp4_aux_info->scale_name.c_str(), nullptr);

                if (input_scale_size > 0) {
                    GGML_ASSERT(!nvfp4_aux_info->input_scale_name.empty());
                    GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), nvfp4_aux_info->input_scale_name.c_str())) == input_scale_size);
                    gguf_set_tensor_data(ctx_outs[cur_split].get(), nvfp4_aux_info->input_scale_name.c_str(), nullptr);
                }
            }
            continue;
        }

        if (!params->dry_run) {
            if (!tm.copy_from_patch && !ml.use_mmap) {
                if (read_data.size() < tensor_size) {
                    read_data.resize(tensor_size);
                }
                tensor->data = read_data.data();
            }
            if (!tm.copy_from_patch) {
                ml.load_data_for(tensor);
            }
        }

        LLAMA_LOG_INFO("[%4d/%4d] %-36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        if (params->dry_run) {
            // the --dry-run option calculates the final quantization size without quantizing
            if (quantize) {
                new_size = ggml_nrows(tensor) * ggml_row_size(new_type, tensor->ne[0]);
                if (new_type == GGML_TYPE_MXFP6_E2M3) {
                    new_size += MXFP6_HEADER_OFFSET;
                }
                if (nvfp4_aux_info != nullptr) {
                    new_size += auxiliary_size;
                }
                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB (%s)\n",
                               tensor_size/1024.0/1024.0,
                               new_size/1024.0/1024.0,
                               ggml_type_name(new_type));
                if (!will_require_imatrix && tm.requires_imatrix) {
                    will_require_imatrix = true;
                }
            } else {
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;
            continue;
        } else {
            // no --dry-run, perform quantization
            if (tm.copy_from_patch) {
                const auto & patch_weight = patch_ml->require_weight(name.c_str());
                ggml_tensor * patch_tensor = patch_weight.tensor;
                patch_ml->load_data_for(patch_tensor);
                new_data = patch_tensor->data;
                new_size = ggml_nbytes(patch_tensor);

                if (nvfp4_aux_info != nullptr) {
                    const auto & patch_scale_weight = patch_ml->require_weight(nvfp4_aux_info->scale_name.c_str());
                    patch_ml->load_data_for(patch_scale_weight.tensor);

	                    const size_t scale_size = nvfp4_aux_info->scale_values.size() * sizeof(float);
	                    const size_t input_scale_size = nvfp4_aux_info->input_scale_values.size() * sizeof(float);
	                    GGML_ASSERT(scale_size == ggml_nbytes(patch_scale_weight.tensor));
	                    std::memcpy(nvfp4_aux_info->scale_values.data(), patch_scale_weight.tensor->data, scale_size);
                    if (input_scale_size > 0) {
                        const auto * patch_input_scale_weight = llama_nvfp4_find_input_scale_weight(patch_ml.get(), name);
                        if (new_type == GGML_TYPE_NVFP4 && patch_input_scale_weight != nullptr && patch_input_scale_weight->tensor != nullptr) {
                            patch_ml->load_data_for(patch_input_scale_weight->tensor);
                            GGML_ASSERT(input_scale_size == ggml_nbytes(patch_input_scale_weight->tensor));
                            std::memcpy(nvfp4_aux_info->input_scale_values.data(), patch_input_scale_weight->tensor->data, input_scale_size);
                        } else {
                            GGML_ASSERT(new_type == GGML_TYPE_MXFP6_E2M3);
                            const float * imatrix = nullptr;
                            if (imatrix_data) {
                                const auto it = imatrix_data->find(tm.remapped_imatrix_name);
                                if (it != imatrix_data->end() &&
                                        it->second.size() == (size_t) tensor->ne[0] * (size_t) std::max<int64_t>(1, tensor->ne[2])) {
                                    imatrix = it->second.data();
                                }
                            }
                            GGML_ASSERT(imatrix != nullptr);
                            for (int64_t i03 = 0; i03 < std::max<int64_t>(1, tensor->ne[2]); ++i03) {
                                const float input_scale = llama_mxfp6_e2m3_input_scale_from_imatrix(
                                        imatrix + i03 * tensor->ne[0], tensor->ne[0], params);
                                GGML_ASSERT(input_scale > 0.0f && std::isfinite(input_scale));
                                nvfp4_aux_info->input_scale_values[(size_t) i03] = input_scale;
                            }
                        }
                    }
	                }

                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB (patch)\n",
                               tensor_size/1024.0/1024.0,
                               (new_size + auxiliary_size)/1024.0/1024.0);
            } else if (!quantize) {
                new_data = tensor->data;
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", tensor_size/1024.0/1024.0);
            } else {
                const int64_t nelements = ggml_nelements(tensor);

                const float * imatrix = nullptr;
                if (imatrix_data) {
                    auto it = imatrix_data->find(tm.remapped_imatrix_name);
                    if (it == imatrix_data->end()) {
                        LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                    } else {
                        if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                            imatrix = it->second.data();
                        } else {
                            LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                    int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                            // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                            // this is a significant error and it may be good idea to abort the process if this happens,
                            // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                            // tok_embd should be ignored in this case, since it always causes this warning
                            if (!tensor_name_match_token_embd(tensor->name)) {
                                throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                        int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                            }
                        }
                    }
                }
                if (!imatrix && tm.requires_imatrix) {
                    LLAMA_LOG_ERROR("\n\n============================================================\n");
                    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                    LLAMA_LOG_ERROR("============================================================\n\n");
                    throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
                }

                float * f32_data = nullptr;
                const ggml_bf16_t * bf16_data = nullptr;

                if (tensor->type == GGML_TYPE_F32) {
                    f32_data = (float *) tensor->data;
                } else if (tensor->type == GGML_TYPE_BF16 &&
                        (new_type == GGML_TYPE_NVFP4 || new_type == GGML_TYPE_MXFP6_E2M3)) {
                    bf16_data = (const ggml_bf16_t *) tensor->data;
                } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                    throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
                } else {
                    llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();
                }

	                if (nvfp4_aux_info != nullptr) {
	                    std::fill(nvfp4_aux_info->scale_values.begin(), nvfp4_aux_info->scale_values.end(), 1.0f);
	                    std::fill(nvfp4_aux_info->input_scale_values.begin(), nvfp4_aux_info->input_scale_values.end(), 0.0f);
	                }

                LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
                fflush(stdout);

                if (work.size() < (size_t)nelements * 4) {
                    work.resize(nelements * 4); // upper bound on size
                }
                new_data = work.data();

                const int64_t n_per_row = tensor->ne[0];
                const int64_t nrows = tensor->ne[1];
                const bool is_mxfp6 = new_type == GGML_TYPE_MXFP6_E2M3;
                const size_t row_size = ggml_row_size(new_type, n_per_row);
                const int64_t n_slices = std::max<int64_t>(1, tensor->ne[2]);
                const int64_t mxfp6_padded_nrows = is_mxfp6 ? GGML_PAD(nrows, MXFP6_TILE_ROWS) : nrows;
                const size_t mxfp6_payload_offset = is_mxfp6 ? MXFP6_HEADER_OFFSET : 0;
                const size_t slice_storage_size = row_size * (size_t) (is_mxfp6 ? mxfp6_padded_nrows : nrows);
                std::vector<float> mxfp6_scale_values;
                std::vector<float> mxfp6_input_scale_values;
                bool mxfp6_shared_tensor_scale = false;
                if (is_mxfp6) {
                    mxfp6_scale_values.assign((size_t) n_slices, 1.0f);
                    mxfp6_input_scale_values.assign((size_t) n_slices, 1.0f);
                    mxfp6_shared_tensor_scale = n_slices > 1 || tensor->ne[3] > 1;
                    if (mxfp6_shared_tensor_scale) {
                        float tensor_scale = llama_mxfp6_tensor_scale(
                                params, f32_data, bf16_data, nrows * n_slices * std::max<int64_t>(1, tensor->ne[3]),
                                n_per_row, nullptr, params->mxfp6_tensor_scale, nthread);
                        if (tm.has_mxfp6_scale_mul) {
                            tensor_scale *= tm.mxfp6_e2m3_scale_mul;
                        }
                        if (!(tensor_scale > 0.0f) || !std::isfinite(tensor_scale)) {
                            tensor_scale = 1.0f;
                        }
                        std::fill(mxfp6_scale_values.begin(), mxfp6_scale_values.end(), tensor_scale);
                    }
                    const size_t projected = mxfp6_payload_offset + slice_storage_size * (size_t) n_slices;
                    if (work.size() < projected) {
                        work.resize(projected);
                    }
                    std::memset(new_data, 0, projected);
                }

                static const int64_t min_chunk_size = 32 * 512;
                const int64_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row));

                const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
                const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
                const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;

                auto quantize_slice = [&](int64_t i03, int slice_nthread) -> size_t {
                    const float * f32_data_03 = f32_data ? (f32_data + i03 * nelements_matrix) : nullptr;
                    const ggml_bf16_t * bf16_data_03 = bf16_data ? (bf16_data + i03 * nelements_matrix) : nullptr;
                    void * new_data_03 = (char *) new_data + mxfp6_payload_offset + slice_storage_size * (size_t) i03;
	                    const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;
	                    float nvfp4_base_tensor_scale_03 = 1.0f;
	                    float nvfp4_tune_tensor_scale_03 = 0.0f;
	                    bool nvfp4_fixed_tensor_scale_03 = false;

	                    if (new_type == GGML_TYPE_NVFP4 && nvfp4_aux_info != nullptr) {
	                        const float tensor_scale = llama_nvfp4_correction_scale(
	                                f32_data_03, bf16_data_03, nelements_matrix, slice_nthread, nvfp4_correction_denom);
	                        nvfp4_base_tensor_scale_03 = tensor_scale;
	                        nvfp4_aux_info->scale_values[(size_t) i03] = tensor_scale;
	                        if (tm.has_nvfp4_scale_overrides &&
	                                (size_t) i03 < tm.nvfp4_scale_overrides.size() &&
	                                (size_t) i03 < tm.nvfp4_base_scale_overrides.size() &&
	                                std::isfinite(tm.nvfp4_scale_overrides[(size_t) i03]) &&
	                                tm.nvfp4_scale_overrides[(size_t) i03] > 0.0f &&
	                                std::isfinite(tm.nvfp4_base_scale_overrides[(size_t) i03]) &&
	                                tm.nvfp4_base_scale_overrides[(size_t) i03] > 0.0f) {
	                            nvfp4_base_tensor_scale_03 = tm.nvfp4_base_scale_overrides[(size_t) i03];
	                            nvfp4_aux_info->scale_values[(size_t) i03] = tm.nvfp4_scale_overrides[(size_t) i03];
	                            nvfp4_tune_tensor_scale_03 = nvfp4_base_tensor_scale_03;
	                            nvfp4_fixed_tensor_scale_03 = true;
	                        }

	                        const float input_scale = llama_nvfp4_input_scale_from_imatrix(
	                                imatrix_03, tensor->ne[0], nvfp4_input_scale_policy);
                        nvfp4_aux_info->input_scale_values[(size_t) i03] = input_scale;
                    } else if (is_mxfp6 && !mxfp6_shared_tensor_scale) {
                        float tensor_scale =
                            llama_mxfp6_tensor_scale(params, f32_data_03, bf16_data_03, nrows, n_per_row, imatrix_03, params->mxfp6_tensor_scale, slice_nthread);
                        if (tm.has_mxfp6_scale_mul) {
                            tensor_scale *= tm.mxfp6_e2m3_scale_mul;
                        }
                        if (!(tensor_scale > 0.0f) || !std::isfinite(tensor_scale)) {
                            tensor_scale = 1.0f;
                        }
                        mxfp6_scale_values[(size_t) i03] = tensor_scale;
                        if (nvfp4_aux_info != nullptr) {
                            nvfp4_aux_info->scale_values[(size_t) i03] = tensor_scale;
                        }
                        if (imatrix_03 != nullptr) {
                            const float input_scale = llama_mxfp6_e2m3_input_scale_from_imatrix(imatrix_03, tensor->ne[0], params);
                            GGML_ASSERT(input_scale > 0.0f && std::isfinite(input_scale));
                            mxfp6_input_scale_values[(size_t) i03] = input_scale;
                            if (nvfp4_aux_info != nullptr && !nvfp4_aux_info->input_scale_values.empty()) {
                                nvfp4_aux_info->input_scale_values[(size_t) i03] = input_scale;
                            }
                        }
                        if (tm.has_mxfp6_scale_mul) {
                            LLAMA_LOG_INFO("%s: tensor=%s slice=%" PRId64 " MXFP6_E2M3 scale_mul=%.8g policy=%s final_scale=%.8g\n",
                                    __func__, tensor->name, i03, (double) tm.mxfp6_e2m3_scale_mul,
                                    tm.mxfp6_policy_name.empty() ? "manual" : tm.mxfp6_policy_name.c_str(),
                                    (double) tensor_scale);
                        }
                    }

                    const float tensor_scale_03 =
                        is_mxfp6
                        ? mxfp6_scale_values[(size_t) i03]
                        : (nvfp4_aux_info != nullptr)
                        ? nvfp4_aux_info->scale_values[(size_t) i03]
                        : 1.0f;
                    std::vector<std::thread> local_workers;
                    local_workers.reserve((size_t) std::max(1, slice_nthread));
                    const tensor_metadata & md = tm;
                    const nvfp4_cuda_runtime_cfg * nvfp4_cfg_ptr =
                        md.has_nvfp4_cfg_override ? &md.nvfp4_cfg_override : nullptr;
                    const int64_t nvfp4_sample_blocks =
                        md.has_nvfp4_cfg_override ? md.nvfp4_sample_blocks : 0;
                    float actual_tensor_scale_03 = tensor_scale_03;
                    const size_t written = llama_tensor_quantize_impl(
                            new_type, params, f32_data_03, bf16_data_03, tensor_scale_03,
                            new_data_03, chunk_size, nrows, n_per_row, imatrix_03,
	                            local_workers, slice_nthread, tensor->name,
	                            nvfp4_cfg_ptr, nvfp4_sample_blocks,
		                            (new_type == GGML_TYPE_NVFP4 && nvfp4_aux_info != nullptr) ? &actual_tensor_scale_03 : nullptr,
		                            nvfp4_tune_tensor_scale_03,
		                            nvfp4_fixed_tensor_scale_03,
                                    md.has_k_rsf_mode_override ? md.k_rsf_mode_override : 0);
                    if (new_type == GGML_TYPE_NVFP4 && nvfp4_aux_info != nullptr) {
                        nvfp4_aux_info->scale_values[(size_t) i03] = actual_tensor_scale_03;
                    }
                    if (is_mxfp6) {
                        if (written > slice_storage_size) {
                            throw std::runtime_error(format("MXFP6_E2M3 tensor %s slice %" PRId64 " wrote %zu bytes into %zu-byte slice",
                                    tensor->name, i03, written, slice_storage_size));
                        }
                        if (written < slice_storage_size) {
                            std::memset((char *) new_data_03 + written, 0, slice_storage_size - written);
                        }
                        return slice_storage_size;
                    }
                    return written;
	                };

                // Quantize each expert separately since they have different importance matrices.
                // For Blackwell microscaling tensors, several slices can run in parallel and keep separate CUDA streams busy.
                new_size = 0;
                if ((new_type == GGML_TYPE_NVFP4 || new_type == GGML_TYPE_MXFP6_E2M3) && tensor->ne[2] > 1 && nthread_use > 1) {
                    const int slice_threads = llama_nvfp4_cuda_parallel_threads((int) nthread_use, tensor->ne[2]);
                    const int slice_nthread = std::max(1, (int) (nthread_use / slice_threads));
                    std::vector<size_t> slice_sizes((size_t) tensor->ne[2], 0);
                    std::atomic<int64_t> next_i03{0};
                    std::atomic<bool> slices_ok{true};
                    std::exception_ptr slice_error;
                    std::mutex slice_error_mutex;

                    auto slice_worker = [&]() {
                        while (slices_ok.load(std::memory_order_relaxed)) {
                            const int64_t i03 = next_i03.fetch_add(1, std::memory_order_relaxed);
                            if (i03 >= tensor->ne[2]) {
                                break;
                            }

                            try {
                                slice_sizes[(size_t) i03] = quantize_slice(i03, slice_nthread);
                            } catch (...) {
                                slices_ok.store(false, std::memory_order_relaxed);
                                std::lock_guard<std::mutex> lock(slice_error_mutex);
                                if (!slice_error) {
                                    slice_error = std::current_exception();
                                }
                                break;
                            }
                        }
                    };

                    workers.clear();
                    workers.reserve((size_t) std::max(0, slice_threads - 1));
                    for (int t = 1; t < slice_threads; ++t) {
                        workers.emplace_back(slice_worker);
                    }
                    slice_worker();
                    for (auto & w : workers) {
                        w.join();
                    }
                    workers.clear();

                    if (slice_error) {
                        std::rethrow_exception(slice_error);
                    }
                    if (!slices_ok.load(std::memory_order_relaxed)) {
                        throw std::runtime_error(format("parallel %s slice quantization failed", ggml_type_name(new_type)));
                    }
                    for (size_t sz : slice_sizes) {
                        new_size += sz;
                    }
                } else {
                    for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
                        new_size += quantize_slice(i03, (int) nthread_use);
                    }
                }
                if (is_mxfp6) {
                    auto * header = (tensor_mxfp6 *) new_data;
                    header->weight_scale  = mxfp6_scale_values.empty() ? 1.0f : mxfp6_scale_values[0];
                    header->input_scale   = mxfp6_input_scale_values.empty() ? 1.0f : mxfp6_input_scale_values[0];
                    header->weight_scales = nullptr;
                    header->input_scales  = nullptr;
                    new_size += MXFP6_HEADER_OFFSET;
                    if (!ggml_validate_row_data(new_type, new_data, new_size)) {
                        throw std::runtime_error("MXFP6_E2M3 tensor/header validation failed");
                    }
                }
                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", tensor_size/1024.0/1024.0, (new_size + auxiliary_size)/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size + auxiliary_size;

            // update the gguf meta data as we go
            gguf_set_tensor_type(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), metadata[i].name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_data);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
            writer_bytes_done += GGML_PAD(new_size, align);
            if (nvfp4_aux_info != nullptr) {
                const size_t scale_size = nvfp4_aux_info->scale_values.size() * sizeof(float);
                const size_t input_scale_size = nvfp4_aux_info->input_scale_values.size() * sizeof(float);

                GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), nvfp4_aux_info->scale_name.c_str())) == scale_size);
                gguf_set_tensor_data(ctx_outs[cur_split].get(), nvfp4_aux_info->scale_name.c_str(), nvfp4_aux_info->scale_values.data());
                fout.write((const char *) nvfp4_aux_info->scale_values.data(), scale_size);
                zeros(fout, GGML_PAD(scale_size, align) - scale_size);
                writer_bytes_done += GGML_PAD(scale_size, align);

                if (input_scale_size > 0) {
                    GGML_ASSERT(!nvfp4_aux_info->input_scale_name.empty());
                    GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), nvfp4_aux_info->input_scale_name.c_str())) == input_scale_size);
                    gguf_set_tensor_data(ctx_outs[cur_split].get(), nvfp4_aux_info->input_scale_name.c_str(), nvfp4_aux_info->input_scale_values.data());
                    fout.write((const char *) nvfp4_aux_info->input_scale_values.data(), input_scale_size);
                    zeros(fout, GGML_PAD(input_scale_size, align) - input_scale_size);
                    writer_bytes_done += GGML_PAD(input_scale_size, align);
                }
		    }
            writer_progress(name.c_str());

            if (resume.enabled && !params->keep_split) {
                const std::streampos pos = fout.tellp();
                if (pos >= std::streampos(0)) {
                    llama_quantize_resume_save(
                            resume, fname_inp, fname_out, ftype, tensors.size(),
                            meta_placeholder_size[cur_split], i + 1, (uint64_t) pos,
                            total_size_org, total_size_new);
                }
            }
        } // no --dry-run
    } // main loop

    if (!params->dry_run) {
        close_ofstream();
        writer_bytes_done = writer_total_bytes;
        writer_progress("complete", true);
        llama_quantize_resume_remove(resume);
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (!params->imatrix && params->dry_run && will_require_imatrix) {
        LLAMA_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n",
                       __func__
        );
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, ml.n_tensors);
    }
}

//
// interface implementation
//

llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q8_0,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.mtp_tensor_type             =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.dry_run                     =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tt_overrides                =*/ nullptr,
        /*.tensor_types                =*/ nullptr,
        /*.prune_layers                =*/ nullptr,
        /*.patch_base_model            =*/ nullptr,
        /*.nv4mx6_policy               =*/ LLAMA_NV4MX6_POLICY_OFF,
        /*.nv4mx6_mx6_penalty          =*/ 0.0f,
        /*.nv4mx6_bf16_mx6_max_sse_ratio =*/ 0.0f,
        /*.mxfp6_tensor_scale          =*/ true,
        /*.mxfp6_min_savings_bytes     =*/ LLAMA_MXFP6_MIN_SAVINGS_BYTES,
        /*.nvfp4_correction_denom      =*/ 0.0f,
        /*.nvfp4_input_scale_policy    =*/ LLAMA_NVFP4_INPUT_SCALE_IMATRIX_RMS,
        /*.assignment_jsonl            =*/ nullptr,
        /*.nvfp4_encoder_max_blocks    =*/ 0,
        /*.mxfp6_input_scale_denom     =*/ 0.0f,
        /*.mxfp6_input_scale_quantile  =*/ 0.0f,
        /*.mxfp6_tensor_scale_sample_blocks =*/ 0,
        /*.mxfp6_tensor_scale_steps    =*/ 0,
        /*.mixed_format_sample_blocks  =*/ 0,
        /*.mixed_format_sample_cap     =*/ 0,
        /*.mixed_format_imatrix_blend  =*/ -1.0f,
        /*.mixed_format_imatrix_power  =*/ -1.0f,
        /*.mixed_format_imatrix_min    =*/ -1.0f,
        /*.mixed_format_imatrix_max    =*/ -1.0f,
        /*.k_quant_rsf                 =*/ true,
        /*.k_quant_rsf_mode            =*/ 1,
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}

//
// Helper functions for external tools exposed in llama-ext.h
//

quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params) {
    return new quantize_state_impl(*model, params);
}

void llama_quant_free(quantize_state_impl * qs) {
    delete qs;
}

llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc) {
    struct llama_model_params mparams = llama_model_default_params();
    auto arch = llm_arch_from_string(desc->architecture);
    auto * model = llama_model_create(arch, mparams);
    model->arch = arch;

    // infer llm_type: only LLM_TYPE_70B matters for quantization logic
    if (model->arch == LLM_ARCH_LLAMA && desc->n_layer == 80 && desc->n_head != desc->n_head_kv) {
        model->type = LLM_TYPE_70B;
    }

    model->hparams.n_embd             = desc->n_embd;
    model->hparams.n_embd_head_k_full = desc->n_embd_head_k;
    model->hparams.n_embd_head_v_full = desc->n_embd_head_v;
    model->hparams.n_layer_all        = desc->n_layer;
    GGML_ASSERT(desc->n_layer > 0 && desc->n_layer <= LLAMA_MAX_LAYERS);
    model->hparams.n_expert           = desc->n_expert;

    for (uint32_t i = 0; i < desc->n_layer; i++) {
        model->hparams.n_head_arr[i]    = desc->n_head;
        model->hparams.n_head_kv_arr[i] = desc->n_head_kv;
        model->hparams.n_ff_arr[i]      = desc->n_ff;
    }

    return model;
}

bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor) {
    return tensor_allows_quantization(qs->params, qs->model.arch, tensor);
}

void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors) {
    // reset per-computation state
    qs->n_attention_wv      = 0;
    qs->n_ffn_down          = 0;
    qs->n_ffn_gate          = 0;
    qs->n_ffn_up            = 0;
    qs->i_attention_wv      = 0;
    qs->i_ffn_down          = 0;
    qs->i_ffn_gate          = 0;
    qs->i_ffn_up            = 0;
    qs->n_fallback          = 0;
    qs->has_imatrix         = false;
    qs->has_tied_embeddings = true;

    // build metadata from tensor names
    std::vector<tensor_metadata> metadata(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) {
        metadata[i].name = ggml_get_name(tensors[i]);
    }

    // initialize counters and categories
    init_quantize_state_counters(*qs, metadata);

    // use a local copy of params with the requested ftype
    llama_model_quantize_params local_params = *qs->params;
    local_params.ftype = ftype;

    ggml_type default_type = llama_ftype_get_default_type(ftype);

    // compute types
    for (size_t i = 0; i < n_tensors; i++) {
        result_types[i] = llama_tensor_get_type(*qs, &local_params, tensors[i], default_type, metadata[i]);
    }
}
