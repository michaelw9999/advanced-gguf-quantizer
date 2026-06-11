#include "ggml-impl.h"
#include "ggml-cuda.h"

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#include <atomic>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <unordered_map>
#endif
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

#ifndef __cuda_cuda_h__
typedef void * cudaStream_t;
#endif

using ggml_backend_cuda_get_device_count_t = int (*)(void);
using ggml_cuda_nvfp4_register_autotune_t = void (*)(void);
using ggml_cuda_nvfp4_autotune_t = void (*)(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, cudaStream_t stream);
using ggml_cuda_nvfp4_autotune_ex_t = bool (*)(
    const float * x, const float * qw, int64_t n,
    const nvfp4_cuda_runtime_cfg * cfg_hint,
    nvfp4_cuda_tune_result * result,
    cudaStream_t stream);
using ggml_cuda_nvfp4_sample_cache_create_t = bool (*)(
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
    cudaStream_t stream);
using ggml_cuda_nvfp4_sample_cache_free_t = void (*)(void * cache);
using ggml_cuda_nvfp4_set_autotune_threads_t = void (*)(int32_t n_threads);
using ggml_cuda_nvfp4_clear_thread_cache_t = void (*)(void);
using ggml_cuda_nvfp4_quantize_t = bool (*)(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, cudaStream_t stream);
using ggml_cuda_nvfp4_quantize_cfg_t = bool (*)(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, const nvfp4_cuda_runtime_cfg * cfg, cudaStream_t stream);
using ggml_cuda_nvfp4_quantize_eval_cfg_t = bool (*)(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, const nvfp4_cuda_runtime_cfg * cfg,
    nvfp4_cuda_eval_result * eval, cudaStream_t stream);
using ggml_cuda_nvfp4_quantize_eval_to_tensor_cfg_t = bool (*)(
    const void * x, bool x_bf16, float x_scale,
    ggml_tensor * tensor, int64_t plane_index,
    int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, const nvfp4_cuda_runtime_cfg * cfg,
    nvfp4_cuda_eval_result * eval, cudaStream_t stream);
using ggml_cuda_mxfp6_e2m3_quantize_eval_t = bool (*)(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    nvfp4_cuda_eval_result * eval, cudaStream_t stream);
using ggml_cuda_mxfp6_e2m3_quantize_eval_to_tensor_t = bool (*)(
    const void * x, bool x_bf16, float x_scale,
    ggml_tensor * tensor, int64_t plane_index,
    int64_t nrow, int64_t n_per_row, const float * qw,
    float header_weight_scale, float header_input_scale,
    nvfp4_cuda_eval_result * eval, cudaStream_t stream);
using ggml_cuda_quantize_classic_impl_t = bool (*)(
    int32_t type,
    const float * x,
    void * vy,
    int64_t nrow,
    int64_t n_per_row,
    const float * qw,
    int32_t rsf_mode,
    cudaStream_t stream);
using ggml_cuda_tensor_set_host_t = bool (*)(
    ggml_tensor * tensor, const void * src, size_t nbytes, cudaStream_t stream);
using ggml_cuda_nvfp4_tensor_set_header_scales_t = bool (*)(
    ggml_tensor * tensor, float weight_scale, float input_scale, cudaStream_t stream);
using ggml_cuda_tensor_get_host_t = bool (*)(
    const ggml_tensor * tensor, void * dst, size_t nbytes, cudaStream_t stream);
using ggml_cuda_tensor_snapshot_t = bool (*)(
    const ggml_tensor * tensor, size_t nbytes, void ** snapshot, cudaStream_t stream);
using ggml_cuda_tensor_restore_t = bool (*)(
    ggml_tensor * tensor, const void * snapshot, size_t nbytes, cudaStream_t stream);
using ggml_cuda_tensor_snapshot_free_t = void (*)(void * snapshot);
using ggml_cuda_nvfp4_kld_reduce_tensor_t = bool (*)(
    const ggml_tensor * logits,
    const uint16_t * base_logp_u16,
    const int32_t * token_ids,
    int32_t logits_row_offset,
    int32_t n_eval,
    int32_t n_vocab,
    int32_t nv,
    nvfp4_cuda_kld_result * result,
    double * kld_values,
    cudaStream_t stream);

static std::atomic<int> g_nvfp4_ab_valid{0};
static float g_nvfp4_a = NVFP4_A0;
static float g_nvfp4_b = NVFP4_B0;
static std::mutex g_nvfp4_cfg_mutex;
static std::mutex g_nvfp4_stream_mutex;
static std::unordered_map<void *, cudaStream_t> g_nvfp4_streams;
static nvfp4_cuda_runtime_cfg g_nvfp4_cfg = {
    NVFP4_CUDA_CHOOSE46_ADAPTIVE,
    8,
    1,
    0,
    448.0f,
    256.0f,
};
static bool g_nvfp4_cfg_valid = false;

#if defined(_WIN32)
static int g_nvfp4_cuda_module_anchor;

static HMODULE ggml_cuda_load_library(const std::wstring & path) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
    HMODULE handle = LoadLibraryW(path.c_str());
    SetErrorMode(old_mode);
    return handle;
}

template<typename T>
static inline T ggml_cuda_sym(const char * name) {
    static HMODULE cuda_handle = []() -> HMODULE {
        HMODULE self = nullptr;
        if (!GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&g_nvfp4_cuda_module_anchor), &self)) {
            return nullptr;
        }

        wchar_t self_path[32768];
        const DWORD self_path_cap = static_cast<DWORD>(sizeof(self_path) / sizeof(self_path[0]));
        const DWORD len = GetModuleFileNameW(self, self_path, self_path_cap);
        if (len == 0 || len >= self_path_cap) {
            return nullptr;
        }

        std::wstring cuda_path(self_path, len);
        const size_t slash = cuda_path.find_last_of(L"/\\");
        if (slash == std::wstring::npos) {
            return nullptr;
        }
        cuda_path.resize(slash + 1);
        cuda_path += L"ggml-cuda.dll";
        return ggml_cuda_load_library(cuda_path);
    }();

    if (cuda_handle) {
        if (FARPROC sym = GetProcAddress(cuda_handle, name)) {
            return reinterpret_cast<T>(sym);
        }
    }
    return nullptr;
}
#else
template<typename T>
static inline T ggml_cuda_sym(const char * name) {
    static void * cuda_handle = []() -> void * {
        Dl_info info{};
        std::string sibling_cuda;
        if (dladdr((void *) &ggml_cuda_sym<T>, &info) != 0 && info.dli_fname) {
            std::string self_path(info.dli_fname);
            const size_t slash = self_path.find_last_of('/');
            if (slash != std::string::npos) {
                sibling_cuda = self_path.substr(0, slash + 1) + "libggml-cuda.so.0";
            }
        }

        void * handle = nullptr;
        if (!sibling_cuda.empty()) {
            handle = dlopen(sibling_cuda.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
            if (!handle) {
                handle = dlopen(sibling_cuda.c_str(), RTLD_NOW | RTLD_GLOBAL);
            }
        }
        if (!handle) {
            handle = dlopen("libggml-cuda.so.0", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
        }
        if (!handle) {
            handle = dlopen("libggml-cuda.so.0", RTLD_NOW | RTLD_GLOBAL);
        }
        return handle;
    }();

    if (cuda_handle) {
        if (void * sym = dlsym(cuda_handle, name)) {
            return reinterpret_cast<T>(sym);
        }
    }
    return reinterpret_cast<T>(dlsym(RTLD_DEFAULT, name));
}
#endif

static inline bool nvfp4_cuda_available() {
    const auto ggml_backend_cuda_get_device_count = ggml_cuda_sym<ggml_backend_cuda_get_device_count_t>("ggml_backend_cuda_get_device_count");
    return ggml_backend_cuda_get_device_count && ggml_backend_cuda_get_device_count() > 0;
}

static inline cudaStream_t nvfp4_resolve_stream(void * stream_key) {
    if (stream_key == nullptr) {
        return (cudaStream_t) 0;
    }

    std::lock_guard<std::mutex> lock(g_nvfp4_stream_mutex);
    auto it = g_nvfp4_streams.find(stream_key);
    if (it != g_nvfp4_streams.end()) {
        return it->second;
    }

    cudaStream_t stream = nullptr;
    using cudaStreamCreateWithFlags_t = int (*)(cudaStream_t *, unsigned int);
    const auto cudaStreamCreateWithFlags_fn = ggml_cuda_sym<cudaStreamCreateWithFlags_t>("cudaStreamCreateWithFlags");
    if (cudaStreamCreateWithFlags_fn) {
        constexpr unsigned int cudaStreamNonBlocking = 1U;
        if (cudaStreamCreateWithFlags_fn(&stream, cudaStreamNonBlocking) != 0) {
            stream = (cudaStream_t) 0;
        }
    }

    g_nvfp4_streams.emplace(stream_key, stream);
    return stream;
}

static inline void nvfp4_register_autotune() {
    const auto ggml_cuda_nvfp4_register_autotune = ggml_cuda_sym<ggml_cuda_nvfp4_register_autotune_t>("ggml_cuda_nvfp4_register_autotune");
    if (ggml_cuda_nvfp4_register_autotune) {
        ggml_cuda_nvfp4_register_autotune();
    }
}

static inline void nvfp4_get_ab(float * a_out, float * b_out) {
    float a = NVFP4_A0;
    float b = NVFP4_B0;
    if (g_nvfp4_ab_valid.load(std::memory_order_acquire) != 0) {
        a = g_nvfp4_a;
        b = g_nvfp4_b;
    }
    *a_out = a;
    *b_out = b;
}

extern "C" bool nvfp4_get_runtime_cfg(nvfp4_cuda_runtime_cfg * cfg_out, const nvfp4_cuda_runtime_cfg * cfg_hint) {
    if (cfg_out == nullptr) {
        return false;
    }
    if (cfg_hint != nullptr) {
        *cfg_out = *cfg_hint;
        return true;
    }
    std::lock_guard<std::mutex> lock(g_nvfp4_cfg_mutex);
    if (!g_nvfp4_cfg_valid) {
        return false;
    }
    *cfg_out = g_nvfp4_cfg;
    return true;
}

extern "C" bool nvfp4_autotune(const float * x, const float * qw, int64_t n, float * best_a, float * best_b) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    nvfp4_register_autotune();
    const auto ggml_cuda_nvfp4_autotune = ggml_cuda_sym<ggml_cuda_nvfp4_autotune_t>("ggml_cuda_nvfp4_autotune");
    if (!ggml_cuda_nvfp4_autotune) {
        return false;
    }
    ggml_cuda_nvfp4_autotune(x, qw, n, best_a, best_b, nullptr);
    return true;
}

extern "C" bool nvfp4_autotune_cuda(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, void * stream) {
    nvfp4_cuda_tune_result result = {
        /* a       = */ NVFP4_A0,
        /* b       = */ NVFP4_B0,
        /* scale_mul = */ 1.0f,
        /* cfg     = */ { NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8, 1, 0, 448.0f, 256.0f },
        /* has_cfg = */ 0,
    };
    if (!nvfp4_autotune_cuda_cfg(x, qw, n, nullptr, &result, stream)) {
        return false;
    }
    if (best_a) {
        *best_a = result.a;
    }
    if (best_b) {
        *best_b = result.b;
    }
    return true;
}

extern "C" bool nvfp4_autotune_cuda_cfg(
    const float * x, const float * qw, int64_t n,
    const nvfp4_cuda_runtime_cfg * cfg_hint,
    nvfp4_cuda_tune_result * result,
    void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    nvfp4_register_autotune();
    nvfp4_cuda_tune_result local = {
        /* a       = */ NVFP4_A0,
        /* b       = */ NVFP4_B0,
        /* scale_mul = */ 1.0f,
        /* cfg     = */ { NVFP4_CUDA_CHOOSE46_ADAPTIVE, 8, 1, 0, 448.0f, 256.0f },
        /* has_cfg = */ 0,
    };
    nvfp4_cuda_runtime_cfg cfg_local{};
    const nvfp4_cuda_runtime_cfg * cfg_ptr = nvfp4_get_runtime_cfg(&cfg_local, cfg_hint) ? &cfg_local : nullptr;

    const auto ggml_cuda_nvfp4_autotune_ex = ggml_cuda_sym<ggml_cuda_nvfp4_autotune_ex_t>("ggml_cuda_nvfp4_autotune_ex");
    if (ggml_cuda_nvfp4_autotune_ex) {
        const bool ok = ggml_cuda_nvfp4_autotune_ex(x, qw, n, cfg_ptr, &local, nvfp4_resolve_stream(stream));
        if (!ok) {
            return false;
        }
    } else {
        const auto ggml_cuda_nvfp4_autotune = ggml_cuda_sym<ggml_cuda_nvfp4_autotune_t>("ggml_cuda_nvfp4_autotune");
        if (!ggml_cuda_nvfp4_autotune) {
            return false;
        }
        ggml_cuda_nvfp4_autotune(x, qw, n, &local.a, &local.b, nvfp4_resolve_stream(stream));
        if (cfg_ptr != nullptr) {
            local.cfg = *cfg_ptr;
            local.has_cfg = 1;
        }
    }
    if (result) {
        *result = local;
    }
    return true;
}

extern "C" bool nvfp4_sample_cache_cuda_create(
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
    void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto ggml_cuda_nvfp4_sample_cache_create =
        ggml_cuda_sym<ggml_cuda_nvfp4_sample_cache_create_t>("ggml_cuda_nvfp4_sample_cache_create");
    if (!ggml_cuda_nvfp4_sample_cache_create) {
        return false;
    }
    return ggml_cuda_nvfp4_sample_cache_create(
        x, x_type, nrow, n_per_row, qw, sample_nb, sample_phase, tune_x_mul,
        cache, x_device, tune_x_device, qw_device, n_device, nvfp4_resolve_stream(stream));
}

extern "C" void nvfp4_sample_cache_cuda_free(void * cache) {
    if (cache == nullptr || !nvfp4_cuda_available()) {
        return;
    }
    const auto ggml_cuda_nvfp4_sample_cache_free =
        ggml_cuda_sym<ggml_cuda_nvfp4_sample_cache_free_t>("ggml_cuda_nvfp4_sample_cache_free");
    if (ggml_cuda_nvfp4_sample_cache_free) {
        ggml_cuda_nvfp4_sample_cache_free(cache);
    }
}

extern "C" void nvfp4_set_ab(float a, float b) {
    g_nvfp4_a = a;
    g_nvfp4_b = b;
    g_nvfp4_ab_valid.store(1, std::memory_order_release);
}

extern "C" void nvfp4_clear_ab(void) {
    g_nvfp4_ab_valid.store(0, std::memory_order_release);
}

extern "C" void nvfp4_set_runtime_cfg(const nvfp4_cuda_runtime_cfg * cfg) {
    if (cfg == nullptr) {
        nvfp4_clear_runtime_cfg();
        return;
    }
    std::lock_guard<std::mutex> lock(g_nvfp4_cfg_mutex);
    g_nvfp4_cfg = *cfg;
    g_nvfp4_cfg_valid = true;
}

extern "C" void nvfp4_clear_runtime_cfg(void) {
    std::lock_guard<std::mutex> lock(g_nvfp4_cfg_mutex);
    g_nvfp4_cfg_valid = false;
}

extern "C" void nvfp4_set_autotune_threads(int32_t n_threads) {
    if (nvfp4_cuda_available()) {
        const auto ggml_cuda_nvfp4_set_autotune_threads =
            ggml_cuda_sym<ggml_cuda_nvfp4_set_autotune_threads_t>("ggml_cuda_nvfp4_set_autotune_threads");
        if (ggml_cuda_nvfp4_set_autotune_threads) {
            ggml_cuda_nvfp4_set_autotune_threads(n_threads);
        }
    }
}

extern "C" void nvfp4_clear_cuda_stream_cache(void) {
    std::lock_guard<std::mutex> lock(g_nvfp4_stream_mutex);
    g_nvfp4_streams.clear();
    if (nvfp4_cuda_available()) {
        const auto ggml_cuda_nvfp4_clear_thread_cache =
            ggml_cuda_sym<ggml_cuda_nvfp4_clear_thread_cache_t>("ggml_cuda_nvfp4_clear_thread_cache");
        if (ggml_cuda_nvfp4_clear_thread_cache) {
            ggml_cuda_nvfp4_clear_thread_cache();
        }
    }
}

extern "C" bool nvfp4_quantize_cuda_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    const nvfp4_cuda_runtime_cfg * cfg, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }

    float a = NVFP4_A0;
    float b = NVFP4_B0;
    nvfp4_get_ab(&a, &b);
    nvfp4_cuda_runtime_cfg cfg_local{};
    const nvfp4_cuda_runtime_cfg * cfg_ptr = nvfp4_get_runtime_cfg(&cfg_local, cfg) ? &cfg_local : nullptr;

    const auto ggml_cuda_nvfp4_quantize_cfg = ggml_cuda_sym<ggml_cuda_nvfp4_quantize_cfg_t>("ggml_cuda_nvfp4_quantize_cfg");
    if (ggml_cuda_nvfp4_quantize_cfg) {
        return ggml_cuda_nvfp4_quantize_cfg(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, cfg_ptr, nvfp4_resolve_stream(stream));
    }
    const auto ggml_cuda_nvfp4_quantize = ggml_cuda_sym<ggml_cuda_nvfp4_quantize_t>("ggml_cuda_nvfp4_quantize");
    if (!ggml_cuda_nvfp4_quantize) {
        return false;
    }
    return ggml_cuda_nvfp4_quantize(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, nvfp4_resolve_stream(stream));
}

extern "C" bool nvfp4_quantize_cuda(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, void * stream) {
    return nvfp4_quantize_cuda_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, float a, float b, void * stream) {
    return nvfp4_quantize_cuda_ab_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, a, b, nullptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    float a, float b,
    const nvfp4_cuda_runtime_cfg * cfg, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    nvfp4_cuda_runtime_cfg cfg_local{};
    const nvfp4_cuda_runtime_cfg * cfg_ptr = nvfp4_get_runtime_cfg(&cfg_local, cfg) ? &cfg_local : nullptr;
    const auto ggml_cuda_nvfp4_quantize_cfg = ggml_cuda_sym<ggml_cuda_nvfp4_quantize_cfg_t>("ggml_cuda_nvfp4_quantize_cfg");
    if (ggml_cuda_nvfp4_quantize_cfg) {
        return ggml_cuda_nvfp4_quantize_cfg(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, cfg_ptr, nvfp4_resolve_stream(stream));
    }
    const auto ggml_cuda_nvfp4_quantize = ggml_cuda_sym<ggml_cuda_nvfp4_quantize_t>("ggml_cuda_nvfp4_quantize");
    if (!ggml_cuda_nvfp4_quantize) {
        return false;
    }
    return ggml_cuda_nvfp4_quantize(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, nvfp4_resolve_stream(stream));
}

extern "C" bool nvfp4_quantize_cuda_ab_eval_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    float a, float b,
    const nvfp4_cuda_runtime_cfg * cfg,
    nvfp4_cuda_eval_result * eval,
    void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    nvfp4_cuda_runtime_cfg cfg_local{};
    const nvfp4_cuda_runtime_cfg * cfg_ptr = nvfp4_get_runtime_cfg(&cfg_local, cfg) ? &cfg_local : nullptr;
    const auto ggml_cuda_nvfp4_quantize_eval_cfg = ggml_cuda_sym<ggml_cuda_nvfp4_quantize_eval_cfg_t>("ggml_cuda_nvfp4_quantize_eval_cfg");
    if (ggml_cuda_nvfp4_quantize_eval_cfg) {
        return ggml_cuda_nvfp4_quantize_eval_cfg(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, cfg_ptr, eval, nvfp4_resolve_stream(stream));
    }
    if (eval != nullptr) {
        eval->sum_sq = 0.0;
        eval->sum_abs = 0.0;
        eval->max_abs = 0.0;
        eval->count = 0;
    }
    return nvfp4_quantize_cuda_ab_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, a, b, cfg_ptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab_eval_to_tensor_cfg(
    const void * x, bool x_bf16, ggml_tensor * tensor,
    int64_t plane_index,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    float a, float b,
    const nvfp4_cuda_runtime_cfg * cfg,
    nvfp4_cuda_eval_result * eval,
    void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    nvfp4_cuda_runtime_cfg cfg_local{};
    const nvfp4_cuda_runtime_cfg * cfg_ptr = nvfp4_get_runtime_cfg(&cfg_local, cfg) ? &cfg_local : nullptr;
    const auto ggml_cuda_nvfp4_quantize_eval_to_tensor_cfg =
        ggml_cuda_sym<ggml_cuda_nvfp4_quantize_eval_to_tensor_cfg_t>("ggml_cuda_nvfp4_quantize_eval_to_tensor_cfg");
    if (!ggml_cuda_nvfp4_quantize_eval_to_tensor_cfg) {
        return false;
    }
    return ggml_cuda_nvfp4_quantize_eval_to_tensor_cfg(
        x, x_bf16, x_scale, tensor, plane_index, nrow, n_per_row, qw, a, b, cfg_ptr, eval, nvfp4_resolve_stream(stream));
}

extern "C" bool mxfp6_e2m3_quantize_cuda_eval(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    nvfp4_cuda_eval_result * eval, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto ggml_cuda_mxfp6_e2m3_quantize_eval =
        ggml_cuda_sym<ggml_cuda_mxfp6_e2m3_quantize_eval_t>("ggml_cuda_mxfp6_e2m3_quantize_eval");
    if (!ggml_cuda_mxfp6_e2m3_quantize_eval) {
        return false;
    }
    return ggml_cuda_mxfp6_e2m3_quantize_eval(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, eval, nvfp4_resolve_stream(stream));
}

extern "C" bool mxfp6_e2m3_quantize_cuda(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale, void * stream) {
    return mxfp6_e2m3_quantize_cuda_eval(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool mxfp6_e2m3_quantize_cuda_eval_to_tensor(
    const void * x, bool x_bf16, ggml_tensor * tensor,
    int64_t plane_index,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    float header_weight_scale, float header_input_scale,
    nvfp4_cuda_eval_result * eval, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto ggml_cuda_mxfp6_e2m3_quantize_eval_to_tensor =
        ggml_cuda_sym<ggml_cuda_mxfp6_e2m3_quantize_eval_to_tensor_t>("ggml_cuda_mxfp6_e2m3_quantize_eval_to_tensor");
    if (!ggml_cuda_mxfp6_e2m3_quantize_eval_to_tensor) {
        return false;
    }
    return ggml_cuda_mxfp6_e2m3_quantize_eval_to_tensor(
        x, x_bf16, x_scale, tensor, plane_index, nrow, n_per_row, qw,
        header_weight_scale, header_input_scale, eval, nvfp4_resolve_stream(stream));
}

extern "C" bool ggml_cuda_quantize_classic(
    int32_t type,
    const float * x,
    void * vy,
    int64_t nrow,
    int64_t n_per_row,
    const float * qw,
    int32_t rsf_mode,
    void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto fn = ggml_cuda_sym<ggml_cuda_quantize_classic_impl_t>("ggml_cuda_quantize_classic_impl");
    if (!fn) {
        return false;
    }
    return fn(type, x, vy, nrow, n_per_row, qw, rsf_mode, nvfp4_resolve_stream(stream));
}

extern "C" bool ggml_cuda_tensor_set_host(
    ggml_tensor * tensor, const void * src, size_t nbytes, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto fn = ggml_cuda_sym<ggml_cuda_tensor_set_host_t>("ggml_cuda_tensor_set_host_impl");
    if (!fn) {
        return false;
    }
    return fn(tensor, src, nbytes, nvfp4_resolve_stream(stream));
}

extern "C" bool ggml_cuda_nvfp4_tensor_set_header_scales(
    ggml_tensor * tensor, float weight_scale, float input_scale, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto fn = ggml_cuda_sym<ggml_cuda_nvfp4_tensor_set_header_scales_t>("ggml_cuda_nvfp4_tensor_set_header_scales_impl");
    if (!fn) {
        return false;
    }
    return fn(tensor, weight_scale, input_scale, nvfp4_resolve_stream(stream));
}

extern "C" bool ggml_cuda_tensor_get_host(
    const ggml_tensor * tensor, void * dst, size_t nbytes, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto fn = ggml_cuda_sym<ggml_cuda_tensor_get_host_t>("ggml_cuda_tensor_get_host_impl");
    if (!fn) {
        return false;
    }
    return fn(tensor, dst, nbytes, nvfp4_resolve_stream(stream));
}

extern "C" bool ggml_cuda_tensor_snapshot(
    const ggml_tensor * tensor, size_t nbytes, void ** snapshot, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto fn = ggml_cuda_sym<ggml_cuda_tensor_snapshot_t>("ggml_cuda_tensor_snapshot_impl");
    if (!fn) {
        return false;
    }
    return fn(tensor, nbytes, snapshot, nvfp4_resolve_stream(stream));
}

extern "C" bool ggml_cuda_tensor_restore(
    ggml_tensor * tensor, const void * snapshot, size_t nbytes, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto fn = ggml_cuda_sym<ggml_cuda_tensor_restore_t>("ggml_cuda_tensor_restore_impl");
    if (!fn) {
        return false;
    }
    return fn(tensor, snapshot, nbytes, nvfp4_resolve_stream(stream));
}

extern "C" void ggml_cuda_tensor_snapshot_free(void * snapshot) {
    if (!nvfp4_cuda_available()) {
        return;
    }
    const auto fn = ggml_cuda_sym<ggml_cuda_tensor_snapshot_free_t>("ggml_cuda_tensor_snapshot_free_impl");
    if (!fn) {
        return;
    }
    fn(snapshot);
}

extern "C" bool nvfp4_kld_reduce_cuda_tensor(
    const ggml_tensor * logits,
    const uint16_t * base_logp_u16,
    const int32_t * token_ids,
    int32_t logits_row_offset,
    int32_t n_eval,
    int32_t n_vocab,
    int32_t nv,
    nvfp4_cuda_kld_result * result,
    double * kld_values,
    void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    const auto ggml_cuda_nvfp4_kld_reduce_tensor =
        ggml_cuda_sym<ggml_cuda_nvfp4_kld_reduce_tensor_t>("ggml_cuda_nvfp4_kld_reduce_tensor");
    if (!ggml_cuda_nvfp4_kld_reduce_tensor) {
        return false;
    }
    return ggml_cuda_nvfp4_kld_reduce_tensor(
        logits, base_logp_u16, token_ids, logits_row_offset, n_eval, n_vocab, nv, result, kld_values, nvfp4_resolve_stream(stream));
}

#else

extern "C" bool nvfp4_autotune(const float * x, const float * qw, int64_t n, float * best_a, float * best_b) {
    (void) x; (void) qw; (void) n; (void) best_a; (void) best_b;
    return false;
}

extern "C" bool nvfp4_autotune_cuda(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, void * stream) {
    (void) x; (void) qw; (void) n; (void) best_a; (void) best_b; (void) stream;
    return false;
}

extern "C" bool nvfp4_autotune_cuda_cfg(
    const float * x, const float * qw, int64_t n,
    const nvfp4_cuda_runtime_cfg * cfg_hint,
    nvfp4_cuda_tune_result * result,
    void * stream) {
    (void) x; (void) qw; (void) n; (void) cfg_hint; (void) result; (void) stream;
    return false;
}

extern "C" bool nvfp4_sample_cache_cuda_create(
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
    void * stream) {
    (void) x; (void) x_type; (void) nrow; (void) n_per_row; (void) qw;
    (void) sample_nb; (void) sample_phase; (void) tune_x_mul; (void) cache;
    (void) x_device; (void) tune_x_device; (void) qw_device; (void) n_device; (void) stream;
    return false;
}

extern "C" void nvfp4_sample_cache_cuda_free(void * cache) {
    (void) cache;
}

extern "C" void nvfp4_set_ab(float a, float b) {
    (void) a; (void) b;
}

extern "C" void nvfp4_clear_ab(void) {
}

extern "C" void nvfp4_set_runtime_cfg(const nvfp4_cuda_runtime_cfg * cfg) {
    (void) cfg;
}

extern "C" void nvfp4_clear_runtime_cfg(void) {
}

extern "C" void nvfp4_set_autotune_threads(int32_t n_threads) {
    (void) n_threads;
}

extern "C" void nvfp4_clear_cuda_stream_cache(void) {
}

extern "C" bool nvfp4_quantize_cuda_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    const nvfp4_cuda_runtime_cfg * cfg, void * stream) {
    (void) x; (void) x_bf16; (void) vy; (void) nrow; (void) n_per_row;
    (void) qw; (void) x_scale; (void) cfg; (void) stream;
    return false;
}

extern "C" bool nvfp4_quantize_cuda(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, void * stream) {
    return nvfp4_quantize_cuda_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, float a, float b, void * stream) {
    (void) a; (void) b;
    return nvfp4_quantize_cuda_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    float a, float b,
    const nvfp4_cuda_runtime_cfg * cfg, void * stream) {
    (void) a; (void) b; (void) cfg;
    return nvfp4_quantize_cuda_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab_eval_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    float a, float b,
    const nvfp4_cuda_runtime_cfg * cfg,
    nvfp4_cuda_eval_result * eval,
    void * stream) {
    if (eval != nullptr) {
        eval->sum_sq = 0.0;
        eval->sum_abs = 0.0;
        eval->max_abs = 0.0;
        eval->count = 0;
    }
    (void) a; (void) b; (void) cfg;
    return nvfp4_quantize_cuda_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab_eval_to_tensor_cfg(
    const void * x, bool x_bf16, ggml_tensor * tensor,
    int64_t plane_index,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    float a, float b,
    const nvfp4_cuda_runtime_cfg * cfg,
    nvfp4_cuda_eval_result * eval,
    void * stream) {
    (void) x;
    (void) x_bf16;
    (void) tensor;
    (void) plane_index;
    (void) nrow;
    (void) n_per_row;
    (void) qw;
    (void) x_scale;
    (void) a;
    (void) b;
    (void) cfg;
    (void) eval;
    (void) stream;
    return false;
}

extern "C" bool mxfp6_e2m3_quantize_cuda_eval(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    nvfp4_cuda_eval_result * eval, void * stream) {
    if (eval != nullptr) {
        eval->sum_sq = 0.0;
        eval->sum_abs = 0.0;
        eval->max_abs = 0.0;
        eval->count = 0;
    }
    (void) x; (void) x_bf16; (void) vy; (void) nrow; (void) n_per_row;
    (void) qw; (void) x_scale; (void) stream;
    return false;
}

extern "C" bool mxfp6_e2m3_quantize_cuda(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale, void * stream) {
    return mxfp6_e2m3_quantize_cuda_eval(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool mxfp6_e2m3_quantize_cuda_eval_to_tensor(
    const void * x, bool x_bf16, ggml_tensor * tensor,
    int64_t plane_index,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    float header_weight_scale, float header_input_scale,
    nvfp4_cuda_eval_result * eval, void * stream) {
    if (eval != nullptr) {
        eval->sum_sq = 0.0;
        eval->sum_abs = 0.0;
        eval->max_abs = 0.0;
        eval->count = 0;
    }
    (void) x; (void) x_bf16; (void) tensor; (void) plane_index;
    (void) nrow; (void) n_per_row; (void) qw; (void) x_scale;
    (void) header_weight_scale; (void) header_input_scale; (void) stream;
    return false;
}

extern "C" bool ggml_cuda_quantize_classic(
    int32_t type,
    const float * x,
    void * vy,
    int64_t nrow,
    int64_t n_per_row,
    const float * qw,
    int32_t rsf_mode,
    void * stream) {
    (void) type;
    (void) x;
    (void) vy;
    (void) nrow;
    (void) n_per_row;
    (void) qw;
    (void) rsf_mode;
    (void) stream;
    return false;
}

extern "C" bool ggml_cuda_tensor_set_host(
    ggml_tensor * tensor, const void * src, size_t nbytes, void * stream) {
    (void) tensor;
    (void) src;
    (void) nbytes;
    (void) stream;
    return false;
}

extern "C" bool ggml_cuda_nvfp4_tensor_set_header_scales(
    ggml_tensor * tensor, float weight_scale, float input_scale, void * stream) {
    (void) tensor;
    (void) weight_scale;
    (void) input_scale;
    (void) stream;
    return false;
}

extern "C" bool ggml_cuda_tensor_get_host(
    const ggml_tensor * tensor, void * dst, size_t nbytes, void * stream) {
    (void) tensor;
    (void) dst;
    (void) nbytes;
    (void) stream;
    return false;
}

extern "C" bool ggml_cuda_tensor_snapshot(
    const ggml_tensor * tensor, size_t nbytes, void ** snapshot, void * stream) {
    (void) tensor;
    (void) nbytes;
    (void) snapshot;
    (void) stream;
    return false;
}

extern "C" bool ggml_cuda_tensor_restore(
    ggml_tensor * tensor, const void * snapshot, size_t nbytes, void * stream) {
    (void) tensor;
    (void) snapshot;
    (void) nbytes;
    (void) stream;
    return false;
}

extern "C" void ggml_cuda_tensor_snapshot_free(void * snapshot) {
    (void) snapshot;
}

extern "C" bool nvfp4_kld_reduce_cuda_tensor(
    const ggml_tensor * logits,
    const uint16_t * base_logp_u16,
    const int32_t * token_ids,
    int32_t logits_row_offset,
    int32_t n_eval,
    int32_t n_vocab,
    int32_t nv,
    nvfp4_cuda_kld_result * result,
    double * kld_values,
    void * stream) {
    (void) logits;
    (void) base_logp_u16;
    (void) token_ids;
    (void) logits_row_offset;
    (void) n_eval;
    (void) n_vocab;
    (void) nv;
    (void) result;
    (void) kld_values;
    (void) stream;
    return false;
}

#endif
