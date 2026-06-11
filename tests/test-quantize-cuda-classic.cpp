#include "ggml.h"
#include "ggml-cuda.h"
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void fill_synthetic(std::vector<float> & data, int64_t nrow, int64_t n_per_row) {
    for (int64_t r = 0; r < nrow; ++r) {
        for (int64_t c = 0; c < n_per_row; ++c) {
            const float a = 1.7f * std::sinf(0.013f * (float) (17 * r + c));
            const float b = 0.3f * std::cosf(0.071f * (float) (3 * c + r));
            const float spike = (c % 97 == 0) ? (0.5f + 0.01f * (float) r) : 0.0f;
            data[(size_t) r * (size_t) n_per_row + (size_t) c] = a + b + spike;
        }
    }
}

static void fill_weights(std::vector<float> & weights) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = 0.35f + 1.4f * (0.5f + 0.5f * std::sinf(0.019f * (float) i + 0.3f));
    }
}

static bool parity_one(
        ggml_type type,
        int64_t nrow,
        int64_t n_per_row,
        const std::vector<float> & data,
        const float * weights,
        const char * label,
        bool expect_supported) {
    const size_t bytes = (size_t) nrow * ggml_row_size(type, n_per_row);
    std::vector<uint8_t> cpu(bytes);
    std::vector<uint8_t> gpu(bytes);

    const size_t cpu_bytes = ggml_quantize_chunk(type, data.data(), cpu.data(), 0, nrow, n_per_row, weights);
    if (cpu_bytes != bytes) {
        std::fprintf(stderr, "CPU writer for %s/%s wrote %zu bytes, expected %zu\n",
                ggml_type_name(type), label, cpu_bytes, bytes);
        return false;
    }

    const bool supported = ggml_cuda_quantize_classic((int32_t) type, data.data(), gpu.data(), nrow, n_per_row, weights, 0, nullptr);
    if (!supported) {
        if (expect_supported) {
            std::fprintf(stderr, "CUDA classic writer for %s/%s is not enabled\n", ggml_type_name(type), label);
            return false;
        }
        std::printf("TODO: CUDA classic writer byte parity pending for %s/%s\n", ggml_type_name(type), label);
        return true;
    }
    if (!expect_supported) {
        std::fprintf(stderr, "CUDA classic writer unexpectedly enabled for %s/%s\n", ggml_type_name(type), label);
        return false;
    }

    if (!ggml_validate_row_data(type, gpu.data(), bytes)) {
        std::fprintf(stderr, "CUDA classic writer produced invalid row data for %s/%s\n", ggml_type_name(type), label);
        return false;
    }
    if (std::memcmp(cpu.data(), gpu.data(), bytes) != 0) {
        for (size_t i = 0; i < bytes; ++i) {
            if (cpu[i] != gpu[i]) {
                std::fprintf(stderr, "%s/%s byte mismatch at %zu: cpu=%u cuda=%u\n",
                        ggml_type_name(type), label, i, (unsigned) cpu[i], (unsigned) gpu[i]);
                break;
            }
        }
        return false;
    }

    std::printf("CUDA classic writer byte parity OK for %s/%s (%" PRId64 "x%" PRId64 ")\n",
            ggml_type_name(type), label, nrow, n_per_row);
    return true;
}

int main() {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        std::printf("skip: CUDA backend is not available\n");
        return 0;
    }

    const int64_t nrow = 3;
    const int64_t n_per_row = QK_K * 2;
    std::vector<float> data((size_t) nrow * (size_t) n_per_row);
    fill_synthetic(data, nrow, n_per_row);
    std::vector<float> weights((size_t) n_per_row);
    fill_weights(weights);

    bool ok = true;
    const ggml_type types[] = {
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
    };
    for (ggml_type type : types) {
        const bool plain_supported = type != GGML_TYPE_Q2_K && type != GGML_TYPE_Q5_K;
        const bool weighted_supported = type != GGML_TYPE_Q2_K;
        ok = parity_one(type, nrow, n_per_row, data, nullptr, "plain", plain_supported) && ok;
        ok = parity_one(type, nrow, n_per_row, data, weights.data(), "weighted", weighted_supported) && ok;
    }

    std::vector<uint8_t> tmp((size_t) nrow * ggml_row_size(GGML_TYPE_Q4_K, n_per_row));
    if (ggml_cuda_quantize_classic((int32_t) GGML_TYPE_Q4_K, data.data(), tmp.data(), nrow, n_per_row, weights.data(), 1, nullptr)) {
        std::fprintf(stderr, "expected RSF Q4_K CUDA classic writer to fall back until RSF kernels are ported\n");
        ok = false;
    }

    return ok ? 0 : 1;
}
