#pragma once

#include "llama.h"
#include "../ggml/include/ggml-cuda.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_nvfp4_named_preset {
    const char * name;
    nvfp4_cuda_runtime_cfg cfg;
};

LLAMA_API const std::vector<llama_nvfp4_named_preset> & llama_nvfp4_preset_catalog();
LLAMA_API const llama_nvfp4_named_preset * llama_nvfp4_find_preset(const std::string & name);

LLAMA_API std::string llama_nvfp4_scale_tensor_name(const std::string & weight_name);
LLAMA_API std::string llama_nvfp4_input_scale_tensor_name(const std::string & weight_name);

LLAMA_API float llama_nvfp4_input_scale_from_imatrix(
        const float * imatrix,
        int64_t n_per_row,
        int32_t policy);

LLAMA_API int64_t llama_nvfp4_sample_block_index(
        int64_t is,
        int64_t sample_nb,
        int64_t nb_total,
        int64_t row_blocks,
        int64_t sample_phase = 0);
