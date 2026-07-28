#include "quantize-options.h"

#include "../../src/llama-quant.h"

#include "common.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

bool striequals(const char * a, const char * b) {
    while (*a && *b) {
        if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
            return false;
        }
        a++;
        b++;
    }
    return *a == *b;
}

std::string trim_copy(std::string value) {
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

static bool parse_nvfp4_bool_value(const std::string & value) {
    if (striequals(value.c_str(), "true") || striequals(value.c_str(), "yes") ||
            striequals(value.c_str(), "on")) {
        return true;
    }
    if (striequals(value.c_str(), "false") || striequals(value.c_str(), "no") ||
            striequals(value.c_str(), "off")) {
        return false;
    }
    return std::stoi(value) != 0;
}

static bool tensor_type_is_k_rsf_capable(ggml_type type) {
    return type == GGML_TYPE_Q2_K ||
           type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K;
}

static bool parse_k_rsf_mode_value(const std::string & value, int32_t & out) {
    if (striequals(value.c_str(), "1") ||
            striequals(value.c_str(), "scale") ||
            striequals(value.c_str(), "normal") ||
            striequals(value.c_str(), "default") ||
            striequals(value.c_str(), "rsf")) {
        out = 1;
        return true;
    }
    if (striequals(value.c_str(), "2") || striequals(value.c_str(), "awq")) {
        out = 2;
        return true;
    }
    if (striequals(value.c_str(), "3") ||
            striequals(value.c_str(), "sq") ||
            striequals(value.c_str(), "smooth")) {
        out = 3;
        return true;
    }
    return false;
}

static const char * k_rsf_mode_name(int32_t mode) {
    switch (mode) {
        case 2: return "awq";
        case 3: return "sq";
        default: return "scale";
    }
}

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<quant_option> QUANT_OPTIONS = {
    { "Q1_0",     LLAMA_FTYPE_MOSTLY_Q1_0,     " 1.125 bpw quantization",           },
    { "Q2_0",     LLAMA_FTYPE_MOSTLY_Q2_0,     " 2.25 bpw quantization (group 64)",  },
    { "Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0,     " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1,     " 4.78G, +0.4511 ppl @ Llama-3-8B",  },
    { "MXFP4_MOE",LLAMA_FTYPE_MOSTLY_MXFP4_MOE," MXFP4 MoE",  },
    { "NVFP4",    LLAMA_FTYPE_MOSTLY_NVFP4,    " NVFP4 (64-value 16x4 block layout)",  },
    { "NVFP4_MXFP6", LLAMA_FTYPE_MOSTLY_NVFP4, " NVFP4 with measured MXFP6 tensor promotion", },
    { "MXFP6",    LLAMA_FTYPE_MOSTLY_MXFP6_E2M3, " MXFP6 E2M3",  },
    { "MXFP6_E2M3", LLAMA_FTYPE_MOSTLY_MXFP6_E2M3, "alias for MXFP6",  },
    { "MXFP8",    LLAMA_FTYPE_MOSTLY_MXFP8,    " 8.25 bpw OCP MX E4M3 quantization", },
    { "Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0,     " 5.21G, +0.1316 ppl @ Llama-3-8B",  },
    { "Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1,     " 5.65G, +0.1062 ppl @ Llama-3-8B",  },
    { "IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS,  " 2.06 bpw quantization",            },
    { "IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS,   " 2.31 bpw quantization",            },
    { "IQ2_S",    LLAMA_FTYPE_MOSTLY_IQ2_S,    " 2.5  bpw quantization",            },
    { "IQ2_M",    LLAMA_FTYPE_MOSTLY_IQ2_M,    " 2.7  bpw quantization",            },
    { "IQ1_S",    LLAMA_FTYPE_MOSTLY_IQ1_S,    " 1.56 bpw quantization",            },
    { "IQ1_M",    LLAMA_FTYPE_MOSTLY_IQ1_M,    " 1.75 bpw quantization",            },
    { "TQ1_0",    LLAMA_FTYPE_MOSTLY_TQ1_0,    " 1.69 bpw ternarization",           },
    { "TQ2_0",    LLAMA_FTYPE_MOSTLY_TQ2_0,    " 2.06 bpw ternarization",           },
    { "Q2_K",     LLAMA_FTYPE_MOSTLY_Q2_K,     " 2.96G, +3.5199 ppl @ Llama-3-8B",  },
    { "Q2_K_RSF", LLAMA_FTYPE_MOSTLY_Q2_K,     " Q2_K with refined scale fit",      },
    { "Q2_K_S",   LLAMA_FTYPE_MOSTLY_Q2_K_S,   " 2.96G, +3.1836 ppl @ Llama-3-8B",  },
    { "IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS,  " 3.06 bpw quantization",            },
    { "IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S,    " 3.44 bpw quantization",            },
    { "IQ3_M",    LLAMA_FTYPE_MOSTLY_IQ3_M,    " 3.66 bpw quantization mix",        },
    { "Q3_K",     LLAMA_FTYPE_MOSTLY_Q3_K_M,   "alias for Q3_K_M"                   },
    { "Q3_K_RSF", LLAMA_FTYPE_MOSTLY_Q3_K_M,   "alias for Q3_K_M_RSF"               },
    { "IQ3_XS",   LLAMA_FTYPE_MOSTLY_IQ3_XS,   " 3.3 bpw quantization",             },
    { "Q3_K_S",   LLAMA_FTYPE_MOSTLY_Q3_K_S,   " 3.41G, +1.6321 ppl @ Llama-3-8B",  },
    { "Q3_K_M",   LLAMA_FTYPE_MOSTLY_Q3_K_M,   " 3.74G, +0.6569 ppl @ Llama-3-8B",  },
    { "Q3_K_M_RSF", LLAMA_FTYPE_MOSTLY_Q3_K_M, " Q3_K_M with refined scale fit",    },
    { "Q3_K_L",   LLAMA_FTYPE_MOSTLY_Q3_K_L,   " 4.03G, +0.5562 ppl @ Llama-3-8B",  },
    { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization", },
    { "IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS,   " 4.25 bpw non-linear quantization", },
    { "Q4_K",     LLAMA_FTYPE_MOSTLY_Q4_K_M,   "alias for Q4_K_M",                  },
    { "Q4_K_S",   LLAMA_FTYPE_MOSTLY_Q4_K_S,   " 4.37G, +0.2689 ppl @ Llama-3-8B",  },
    { "Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 4.58G, +0.1754 ppl @ Llama-3-8B",  },
    { "Q4_K_RSF", LLAMA_FTYPE_MOSTLY_Q4_K_M,   " Q4_K_M with refined scale fit",    },
    { "Q4_K_RSF_AWQ", LLAMA_FTYPE_MOSTLY_Q4_K_M, " Q4_K_M with AWQ-style RSF",      },
    { "Q4_K_RSF_SQ",  LLAMA_FTYPE_MOSTLY_Q4_K_M, " Q4_K_M with SQ-style RSF",       },
    { "Q5_K",     LLAMA_FTYPE_MOSTLY_Q5_K_M,   "alias for Q5_K_M",                  },
    { "Q5_K_S",   LLAMA_FTYPE_MOSTLY_Q5_K_S,   " 5.21G, +0.1049 ppl @ Llama-3-8B",  },
    { "Q5_K_M",   LLAMA_FTYPE_MOSTLY_Q5_K_M,   " 5.33G, +0.0569 ppl @ Llama-3-8B",  },
    { "Q5_K_RSF", LLAMA_FTYPE_MOSTLY_Q5_K_M,   " Q5_K_M with refined scale fit",    },
    { "Q5_K_M_RSF", LLAMA_FTYPE_MOSTLY_Q5_K_M, " Q5_K_M with refined scale fit",    },
    { "Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K,     " 6.14G, +0.0217 ppl @ Llama-3-8B",  },
    { "Q6_K_RSF", LLAMA_FTYPE_MOSTLY_Q6_K,     " Q6_K with refined scale fit",      },
    { "Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0,     " 7.96G, +0.0026 ppl @ Llama-3-8B",  },
    { "F16",      LLAMA_FTYPE_MOSTLY_F16,      "14.00G, +0.0020 ppl @ Mistral-7B",  },
    { "BF16",     LLAMA_FTYPE_MOSTLY_BF16,     "14.00G, -0.0050 ppl @ Mistral-7B",  },
    { "F32",      LLAMA_FTYPE_ALL_F32,         "26.00G              @ 7B",          },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    { "COPY",     LLAMA_FTYPE_ALL_F32,         "only copy tensors, no quantizing",  },
};

bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    std::string ftype_str;

    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::toupper(ch));
    }
    for (const auto & it : QUANT_OPTIONS) {
        if (striequals(it.name.c_str(), ftype_str.c_str())) {
            ftype = it.ftype;
            ftype_str_out = it.name;
            return true;
        }
    }
    try {
        int ftype_int = std::stoi(ftype_str);
        for (const auto & it : QUANT_OPTIONS) {
            if (it.ftype == ftype_int) {
                ftype = it.ftype;
                ftype_str_out = it.name;
                return true;
            }
        }
    }
    catch (...) {
        // stoi failed
    }
    return false;
}

bool ftype_is_mixed_nvfp4_mxfp6(const std::string & ftype_str) {
    return ftype_str == "NVFP4_MXFP6";
}

[[noreturn]] void usage(const char * executable) {
    printf("usage: %s [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]\n", executable);
    printf("       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--mtp-tensor-type] [--tensor-type] [--tensor-type-file]\n");
    printf("       [--prune-layers] [--keep-split] [--override-kv] [--patch-base]\n");
    printf("       [--mode fast|normal|deep] [--selector-kld] [--selector-auto-rescue] [--dry-run]\n");
    printf("       model-f32.gguf [model-quant.gguf] type [nthreads]\n\n");
    printf("  --allow-requantize\n");
    printf("                                      allow requantizing tensors that have already been quantized\n");
    printf("                                      WARNING: this can severely reduce quality compared to quantizing\n");
    printf("                                               from 16bit or 32bit!\n");
    printf("  --leave-output-tensor\n");
    printf("                                      leave output.weight un(re)quantized\n");
    printf("                                      increases model size but may also increase quality, especially when requantizing\n");
    printf("  --pure\n");
    printf("                                      disable k-quant mixtures and quantize all tensors to the same type\n");
    printf("  --imatrix file_name\n");
    printf("                                      use data in file_name as importance matrix for quant optimizations\n");
    printf("  --include-weights tensor_name\n");
    printf("                                      use importance matrix for this/these tensor(s)\n");
    printf("  --exclude-weights tensor_name\n");
    printf("                                      do not use importance matrix for this/these tensor(s)\n");
    printf("  --output-tensor-type ggml_type\n");
    printf("                                      use this ggml_type for the output.weight tensor\n");
    printf("  --token-embedding-type ggml_type\n");
    printf("                                      use this ggml_type for the token embeddings tensor\n");
    printf("  --mtp-tensor-type ggml_type\n");
    printf("                                      use this ggml_type for MTP/NextN matrix tensors; blank follows the ftype/profile default.\n");
    printf("  --tensor-type tensor_name=ggml_type\n");
    printf("                                      quantize this tensor to this ggml_type\n");
    printf("                                      this is an advanced option to selectively quantize tensors. may be specified multiple times.\n");
    printf("                                      example: --tensor-type attn_q=q8_0\n");
    printf("  --tensor-type-file tensor_types.txt\n");
    printf("                                      list of tensors to quantize to a specific ggml_type\n");
    printf("                                      this is an advanced option to selectively quantize a long list of tensors.\n");
    printf("                                      the file should use the same format as above, separated by spaces or newlines.\n");
    printf("  --prune-layers L0,L1,L2...\n");
    printf("                                      comma-separated list of layer numbers to prune from the model\n");
    printf("                                      WARNING: this is an advanced option, use with care.\n");
    printf("  --keep-split\n");
    printf("                                      generate quantized model in the same shards as input\n");
    printf("  --override-kv KEY=TYPE:VALUE\n");
    printf("                                      override model metadata by key in the quantized model. may be specified multiple times.\n");
    printf("                                      WARNING: this is an advanced option, use with care.\n");
    printf("  --patch-base model.gguf\n");
    printf("                                      reuse unchanged tensor bytes from an existing quantized GGUF.\n");
    printf("                                      useful for surgical tensor rescues without a full requantization pass.\n");
    printf("  --nvfp4-cfg NVFP4{choose46=...,refit=...,compand=...,cap6=...,cap4=...,rsf_depth=...}\n");
    printf("                                      set the global NVFP4 CUDA encoder policy directly.\n");
    printf("  --nvfp4-preset name\n");
    printf("                                      set a named global NVFP4 encoder policy, e.g. baseline or asym_tail.\n");
    printf("  --nvfp4-correction-denom value\n");
    printf("                                      set NVFP4 weight correction denominator, e.g. 105, 1536, 1728, 2304, 2688.\n");
    printf("  --nvfp4-input-scale-policy name\n");
    printf("                                      set NVFP4 input scale policy: imatrix-rms, identity, imatrix-sqrtmax,\n");
    printf("                                      imatrix-sqrtp99, imatrix-sqrtp999.\n");
    printf("  --nvfp4-encoder-max-blocks N / --nvfp4-encoder-threads N\n");
    printf("                                      set CUDA NVFP4 encoder sample cap and worker count.\n");
    printf("  --mode fast|normal|deep\n");
    printf("                                      choose quantizer work mode; required for saved-logit selector runs.\n");
    printf("  --selector-rsf-mode tensor|slice|expert|group / --selector-rsf-depth normal|deep|deeper|exhaustive\n");
    printf("                                      set refined scale fit (RSF) granularity/search depth.\n");
    printf("  --selector-rsf-report file.txt\n");
    printf("                                      write the RSF policy summary, tensor rows, and scale-multiplier histograms.\n");
    printf("  --selector-tensor-policy-map / --selector-no-tensor-policy-map\n");
    printf("                                      allow first-pass per-tensor policy planning; enabled by default.\n");
    printf("  --mxfp6_e2m3-tensor-scale on|off\n");
    printf("                                      write MXFP6_E2M3 tensor correction scales. default: on.\n");
    printf("                                      WARNING: MXFP6_E2M3 is experimental and unsupported by NVIDIA/llama.cpp.\n");
    printf("                                      Future official formats may not read GGUFs created here; feedback requested.\n");
    printf("                                      Branch: https://github.com/michaelw9999/llama.cpp/tree/mxfp6-cuda\n");
    printf("  --mxfp6_e2m3-min-savings bytes\n");
    printf("                                      minimum automatic per-tensor space saving before MXFP6_E2M3 fallback. default: 4 MiB.\n");
    printf("  --mxfp6_e2m3-input-scale-denom value / --mxfp6_e2m3-input-scale-quantile value\n");
    printf("                                      override MXFP6 imatrix input-scale calibration.\n");
    printf("  --mxfp6_e2m3-tensor-scale-sample-blocks N / --mxfp6_e2m3-tensor-scale-steps N\n");
    printf("                                      set MXFP6 tensor-scale search effort for expert reproduction runs.\n");
    printf("  --mxfp6_e2m3-selector-scale-top N\n");
    printf("                                      KLD/PPL-refine tensor-scale multipliers for the top N existing MXFP6_E2M3 tensors.\n");
    printf("  --mxfp6_e2m3-selector-scale-candidates c1,c2,...\n");
    printf("                                      MXFP6_E2M3 tensor-scale multipliers tested by the selector.\n");
    printf("  --mixed-format-policy name\n");
    printf("                                      mixed NVFP4/MXFP6 allocation policy: nvfp4-quality-boost,\n");
    printf("                                      mxfp6-primary, auto, or off.\n");
    printf("  --mixed-mx6-penalty value\n");
    printf("                                      quality/size pressure for mixed allocation; higher values prefer NVFP4.\n");
    printf("  --mixed-bf16-mx6-threshold value\n");
    printf("                                      BF16-to-MXFP6 SSE threshold for repair/edit workflows.\n");
    printf("  --mixed-sample-blocks N / --mixed-sample-cap N\n");
    printf("                                      set mixed NVFP4/MXFP6 proxy sample breadth for expert runs.\n");
    printf("  --mixed-imatrix-weight-blend value / --mixed-imatrix-weight-power value\n");
    printf("  --mixed-imatrix-weight-min value / --mixed-imatrix-weight-max value\n");
    printf("                                      shape imatrix weights for mixed-format proxy scoring.\n");
    printf("  --selector-kld file.kld\n");
    printf("                                      use saved BF16 logits as the policy/rescue selector oracle.\n");
    printf("                                      enables true KLD/PPL ranking for the selected KLD subset.\n");
    printf("  --selector-ledger file.jsonl\n");
    printf("                                      append raw selector evidence rows to a JSONL ledger.\n");
    printf("  --selector-checkpoint-model model.gguf\n");
    printf("                                      existing quantized checkpoint used by candidate search instead of creating one.\n");
    printf("  --selector-cache-dir dir / --selector-keep-checkpoint\n");
    printf("                                      cache auto-created candidate-search checkpoints for future runs.\n");
    printf("  --selector-require-runtime-cache\n");
    printf("                                      fail instead of falling back when Stage-B resident tensor patching is unavailable.\n");
    printf("  --selector-skip-file file\n");
    printf("                                      if file appears during selector tuning, use the best available policy.\n");
    printf("  --selector-skip-remaining\n");
    printf("                                      skip optional selector tuning now and use the best available policy.\n");
    printf("  --selector-stagea-sample-blocks N\n");
    printf("                                      weight blocks sampled per tensor during coarse selector policy ranking.\n");
    printf("  --selector-stagea-max-policies N\n");
    printf("                                      limit coarse selector policies before refinement; 0 keeps all.\n");
    printf("  --selector-awq-top N\n");
    printf("                                      keep up to N asym/AWQ-tail policies when the coarse policy limit is active.\n");
    printf("  --selector-skip-policy name / --selector-skip-policies a,b\n");
    printf("                                      skip named policies during survey and measured eval.\n");
    printf("  --selector-include-policy name / --selector-include-policies a,b\n");
    printf("                                      evaluate only named policies; seed_keep remains allowed internally.\n");
    printf("  --selector-refine-top N / --selector-refine-budget N\n");
    printf("                                      selector refinement breadth and exact tensor override budget.\n");
    printf("  --selector-survey-top N / --selector-survey-sample-blocks N\n");
    printf("                                      selector survey breadth/sample depth before KLD ranking.\n");
    printf("  --selector-max-tensors N / --selector-trace\n");
    printf("                                      limit selector stress tensors and enable verbose selector tensor tracing.\n");
    printf("  --selector-policy-threads N / --selector-threads N\n");
    printf("                                      selector policy/tensor worker counts; final quantization still uses nthreads.\n");
    printf("  --selector-stageb-patch-eval\n");
    printf("                                      also collect tensor reconstruction stats while materializing Stage-B policies.\n");
    printf("  --selector-stageb-direct-verify / --selector-no-stageb-direct-verify\n");
    printf("                                      force-enable or disable Stage-B direct-patch readback verification.\n");
    printf("  --selector-kld-threads N\n");
    printf("                                      host worker count for full-vocab KLD/PPL metric reduction during selector full PPL/KLD eval.\n");
    printf("  --selector-auto-rescue\n");
    printf("                                      after the selected pass, patch high-risk exact tensors using the rescue settings.\n");
    printf("  --selector-only\n");
    printf("                                      run selector analysis and exit without writing a final model.\n");
    printf("  --selector-candidate-type T / --selector-candidate-types A,B\n");
    printf("                                      main-path tensor type candidates for high-error tensors.\n");
    printf("                                      defaults with saved-logit KLD selector: Q5_0,Q4_K,Q5_K,Q6_K,Q8_0; mixed adds MXFP6_E2M3 first.\n");
    printf("  --selector-candidate-fraction F / --selector-candidate-top N\n");
    printf("                                      cap type candidates to the worst baseline-error tensor fraction or exact count.\n");
    printf("  --selector-candidate-budget-mb N / --selector-candidate-class-limit N\n");
    printf("                                      cap extra size and per-class use for main-path type candidates.\n");
    printf("  --selector-candidate-report file.csv / --selector-candidate-tensor-types file.txt\n");
    printf("                                      write main-path candidate ranking and final exact tensor-type map.\n");
    printf("  --selector-eval-top N\n");
    printf("                                      number of policies used for full-base KLD/PPL evaluation. default: 16.\n");
    printf("  --selector-n-seq N\n");
    printf("                                      override selector KLD sequence count. default: auto from free CUDA memory.\n");
    printf("  --selector-eval-batch N\n");
    printf("                                      token batch size used during selector KLD eval; values below ctx exercise multi-batch KLD.\n");
    printf("  --selector-n-gpu-layers N\n");
    printf("                                      GPU layers used during selector KLD full PPL/KLD evaluation.\n");
    printf("  --selector-sensitivity-report file.jsonl\n");
    printf("                                      write exact one-tensor KLD/PPL what-if deltas after selecting a policy.\n");
    printf("  --selector-sensitivity-top N / --selector-sensitivity-layer N\n");
    printf("                                      limit exact what-if sensitivity probes by count or layer.\n");
    printf("  --selector-sensitivity-tensor text\n");
    printf("                                      only probe tensors whose name contains text.\n");
    printf("  --selector-rescue-top N\n");
    printf("                                      maximum exact tensor rescue overrides to apply.\n");
    printf("  --selector-rescue-encoder-top N / --selector-rescue-report-top N\n");
    printf("                                      tensors scanned/reported for per-tensor policy rescue.\n");
    printf("  --selector-rescue-sample-blocks N\n");
    printf("                                      weight blocks sampled per tensor for rescue sensitivity ranking.\n");
    printf("  --selector-rescue-coarse-max-blocks N\n");
    printf("  --selector-rescue-refine-max-blocks N\n");
    printf("  --selector-rescue-guard-max-blocks N\n");
    printf("                                      cap sampled blocks for per-tensor rescue policy search.\n");
    printf("  --selector-rescue-budget-mb N\n");
    printf("                                      total extra-size budget for non-BF16 rescue overrides.\n");
    printf("  --selector-rescue-bf16-budget-mb N\n");
    printf("                                      extra-size budget for BF16 rescue overrides.\n");
    printf("  --selector-rescue-class-limit N\n");
    printf("                                      maximum rescues per tensor class; 0 disables the class limit.\n");
    printf("  --selector-rescue-report file.csv\n");
    printf("                                      write selector rescue ranking/report CSV.\n");
    printf("  --selector-rescue-tensor-types file.txt\n");
    printf("                                      write exact tensor-type overrides selected by rescue.\n");
    printf("  --selector-kld-penalty value / --selector-p99-penalty value\n");
    printf("                                      increase selector pressure on mean KLD / p99 KLD.\n");
    printf("  --selector-p999-penalty value / --selector-max-kld-penalty value\n");
    printf("                                      increase selector pressure on tail KLD outliers.\n");
    printf("  --selector-rank-kld-threshold value / --selector-rank-p99-threshold value\n");
    printf("                                      apply thresholded selector pressure on mean/p99 KLD.\n");
    printf("  --selector-rank-p999-threshold value / --selector-rank-max-kld-threshold value\n");
    printf("                                      apply thresholded selector pressure on p999/max KLD.\n");
    printf("  --selector-rank-kld-hard-gate / --selector-rank-p99-hard-gate\n");
    printf("                                      reject selector candidates above the corresponding mean/p99 threshold.\n");
    printf("  --selector-rank-p999-hard-gate / --selector-rank-max-kld-hard-gate\n");
    printf("                                      reject selector candidates above the corresponding p999/max KLD threshold.\n");
    printf("  --dry-run\n");
    printf("                                      calculate and show the final quantization size without performing quantization\n");
    printf("                                      example: llama-quantize --dry-run model-f32.gguf Q4_K\n\n");
    printf("note: --include-weights and --exclude-weights cannot be used together\n\n");
    printf("-----------------------------------------------------------------------------\n");
    printf(" allowed quantization types\n");
    printf("-----------------------------------------------------------------------------\n\n");
    for (const auto & it : QUANT_OPTIONS) {
        if (it.name != "COPY") {
            printf("  %2d  or  ", it.ftype);
        } else {
            printf("          ");
        }
        printf("%-7s : %s\n", it.name.c_str(), it.desc.c_str());
    }
    exit(1);
}

ggml_type parse_ggml_type(const char * arg) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        auto type = (ggml_type) i;
        const auto * name = ggml_type_name(type);
        if (name && striequals(name, arg)) {
            return type;
        }
    }
    fprintf(stderr, "\n%s: invalid ggml_type '%s'\n\n", __func__, arg);
    return GGML_TYPE_COUNT;
}

const char * nvfp4_choose46_mode_name(int32_t mode) {
    switch (mode) {
        case NVFP4_CUDA_CHOOSE46_ADAPTIVE: return "adaptive";
        case NVFP4_CUDA_CHOOSE46_FORCE_M6: return "force_m6";
        case NVFP4_CUDA_CHOOSE46_FORCE_M4: return "force_m4";
        default: return "adaptive";
    }
}

static bool parse_nvfp4_choose46_mode(const std::string & value, int32_t & out_mode) {
    if (striequals(value.c_str(), "adaptive") || striequals(value.c_str(), "auto")) {
        out_mode = NVFP4_CUDA_CHOOSE46_ADAPTIVE;
        return true;
    }
    if (striequals(value.c_str(), "force_m6") || striequals(value.c_str(), "m6")) {
        out_mode = NVFP4_CUDA_CHOOSE46_FORCE_M6;
        return true;
    }
    if (striequals(value.c_str(), "force_m4") || striequals(value.c_str(), "m4")) {
        out_mode = NVFP4_CUDA_CHOOSE46_FORCE_M4;
        return true;
    }
    return false;
}

bool parse_nvfp4_preset(const std::string & value, nvfp4_cuda_runtime_cfg & out_cfg, std::string * out_name) {
    const llama_nvfp4_named_preset * preset = llama_nvfp4_find_preset(value);
    if (preset == nullptr) {
        return false;
    }
    out_cfg = preset->cfg;
    if (out_name != nullptr) {
        *out_name = preset->name;
    }
    return true;
}

bool parse_tensor_type_nvfp4_cfg(const std::string & spec, tensor_type_option & out) {
    if (spec.size() < 7 || !striequals(spec.substr(0, 6).c_str(), "NVFP4{") || spec.back() != '}') {
        return false;
    }

    out.type = GGML_TYPE_NVFP4;
    out.has_nvfp4_cfg = true;
    out.nvfp4_cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    out.nvfp4_sample_blocks = 0;
    out.nvfp4_policy_name.clear();

    const std::string body = spec.substr(6, spec.size() - 7);
    std::stringstream ss(body);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) {
            continue;
        }
        const size_t eq = token.find('=');
        if (eq == std::string::npos) {
            fprintf(stderr, "%s: malformed NVFP4 tensor override token '%s'\n", __func__, token.c_str());
            return false;
        }
        const std::string key = trim_copy(token.substr(0, eq));
        const std::string value = trim_copy(token.substr(eq + 1));
        if (key.empty() || value.empty()) {
            fprintf(stderr, "%s: malformed NVFP4 tensor override token '%s'\n", __func__, token.c_str());
            return false;
        }

        if (striequals(key.c_str(), "choose46") || striequals(key.c_str(), "mode")) {
            if (!parse_nvfp4_choose46_mode(value, out.nvfp4_cfg.choose46_mode)) {
                fprintf(stderr, "%s: invalid choose46 mode '%s'\n", __func__, value.c_str());
                return false;
            }
        } else if (striequals(key.c_str(), "preset") || striequals(key.c_str(), "profile")) {
            if (!parse_nvfp4_preset(value, out.nvfp4_cfg, &out.nvfp4_policy_name)) {
                fprintf(stderr, "%s: invalid NVFP4 preset '%s'\n", __func__, value.c_str());
                return false;
            }
        } else if (striequals(key.c_str(), "refit") || striequals(key.c_str(), "refit_iters")) {
            out.nvfp4_cfg.refit_iters = std::max(1, std::stoi(value));
        } else if (striequals(key.c_str(), "compand") || striequals(key.c_str(), "use_compand_sat")) {
            out.nvfp4_cfg.use_compand_sat =
                (striequals(value.c_str(), "true") || striequals(value.c_str(), "yes")) ? 1 :
                (striequals(value.c_str(), "false") || striequals(value.c_str(), "no")) ? 0 :
                std::stoi(value);
        } else if (striequals(key.c_str(), "cap6") || striequals(key.c_str(), "cap_m6")) {
            out.nvfp4_cfg.cap_m6 = (float) std::stof(value);
        } else if (striequals(key.c_str(), "cap4") || striequals(key.c_str(), "cap_m4")) {
            out.nvfp4_cfg.cap_m4 = (float) std::stof(value);
        } else if (striequals(key.c_str(), "rsf") || striequals(key.c_str(), "nvfp4_rsf")) {
            if (parse_nvfp4_bool_value(value)) {
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_FLAG_RSF;
                out.nvfp4_cfg.reserved_i32 &= ~NVFP4_CUDA_RSF_DEPTH_MASK;
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_RSF_DEPTH_DEEPER << NVFP4_CUDA_RSF_DEPTH_SHIFT;
            } else {
                out.nvfp4_cfg.reserved_i32 &= ~(NVFP4_CUDA_FLAG_RSF | NVFP4_CUDA_RSF_MODE_MASK | NVFP4_CUDA_RSF_DEPTH_MASK);
            }
        } else if (striequals(key.c_str(), "rsf_mode")) {
            out.nvfp4_cfg.reserved_i32 &= ~NVFP4_CUDA_RSF_MODE_MASK;
            if (striequals(value.c_str(), "tensor")) {
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_RSF_MODE_TENSOR << NVFP4_CUDA_RSF_MODE_SHIFT;
            } else if (striequals(value.c_str(), "slice")) {
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_RSF_MODE_SLICE << NVFP4_CUDA_RSF_MODE_SHIFT;
            } else if (striequals(value.c_str(), "expert")) {
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_RSF_MODE_EXPERT << NVFP4_CUDA_RSF_MODE_SHIFT;
            } else if (striequals(value.c_str(), "group")) {
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_RSF_MODE_GROUP << NVFP4_CUDA_RSF_MODE_SHIFT;
            } else {
                fprintf(stderr, "%s: invalid RSF mode '%s'\n", __func__, value.c_str());
                return false;
            }
        } else if (striequals(key.c_str(), "rsf_depth")) {
            out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_FLAG_RSF;
            out.nvfp4_cfg.reserved_i32 &= ~NVFP4_CUDA_RSF_DEPTH_MASK;
            if (striequals(value.c_str(), "normal") || striequals(value.c_str(), "default")) {
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_RSF_DEPTH_NORMAL << NVFP4_CUDA_RSF_DEPTH_SHIFT;
            } else if (striequals(value.c_str(), "deep")) {
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_RSF_DEPTH_DEEP << NVFP4_CUDA_RSF_DEPTH_SHIFT;
            } else if (striequals(value.c_str(), "deeper")) {
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_RSF_DEPTH_DEEPER << NVFP4_CUDA_RSF_DEPTH_SHIFT;
            } else if (striequals(value.c_str(), "exhaustive") || striequals(value.c_str(), "max")) {
                out.nvfp4_cfg.reserved_i32 |= NVFP4_CUDA_RSF_DEPTH_EXHAUSTIVE << NVFP4_CUDA_RSF_DEPTH_SHIFT;
            } else {
                fprintf(stderr, "%s: invalid RSF depth '%s'\n", __func__, value.c_str());
                return false;
            }
        } else if (striequals(key.c_str(), "sample") || striequals(key.c_str(), "sample_blocks")) {
            out.nvfp4_sample_blocks = std::max<int64_t>(0, std::stoll(value));
        } else if (striequals(key.c_str(), "policy") || striequals(key.c_str(), "name")) {
            out.nvfp4_policy_name = value;
        } else {
            fprintf(stderr, "%s: unknown NVFP4 tensor override key '%s'\n", __func__, key.c_str());
            return false;
        }
    }

    return true;
}

static bool parse_tensor_type_k_rsf_cfg(const std::string & spec, tensor_type_option & out) {
    const size_t open = spec.find('{');
    if (open == std::string::npos || spec.empty() || spec.back() != '}') {
        return false;
    }

    const std::string type_name = trim_copy(spec.substr(0, open));
    const ggml_type type = parse_ggml_type(type_name.c_str());
    if (!tensor_type_is_k_rsf_capable(type)) {
        return false;
    }

    out.type = type;
    out.has_k_rsf_mode = true;
    out.k_rsf_mode = 1;

    const std::string body = spec.substr(open + 1, spec.size() - open - 2);
    std::stringstream ss(body);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) {
            continue;
        }
        const size_t eq = token.find('=');
        if (eq == std::string::npos) {
            fprintf(stderr, "%s: malformed K-RSF tensor override token '%s'\n", __func__, token.c_str());
            return false;
        }
        const std::string key = trim_copy(token.substr(0, eq));
        const std::string value = trim_copy(token.substr(eq + 1));
        if (key.empty() || value.empty()) {
            fprintf(stderr, "%s: malformed K-RSF tensor override token '%s'\n", __func__, token.c_str());
            return false;
        }
        if (striequals(key.c_str(), "rsf") ||
                striequals(key.c_str(), "k_rsf") ||
                striequals(key.c_str(), "k-rsf") ||
                striequals(key.c_str(), "mode") ||
                striequals(key.c_str(), "scale_fit")) {
            if (!parse_k_rsf_mode_value(value, out.k_rsf_mode)) {
                fprintf(stderr, "%s: invalid K-RSF mode '%s'\n", __func__, value.c_str());
                return false;
            }
        } else {
            fprintf(stderr, "%s: unknown K-RSF tensor override key '%s'\n", __func__, key.c_str());
            return false;
        }
    }

    return true;
}

static bool parse_tensor_type_mxfp6_cfg(const std::string & spec, tensor_type_option & out) {
    static constexpr const char * prefix = "MXFP6_E2M3{";
    static constexpr size_t prefix_len = 11;
    if (spec.size() < prefix_len + 1 || !striequals(spec.substr(0, prefix_len).c_str(), prefix) || spec.back() != '}') {
        return false;
    }

    out.type = GGML_TYPE_MXFP6_E2M3;
    out.has_mxfp6_scale_mul = false;
    out.mxfp6_e2m3_scale_mul = 1.0f;
    out.mxfp6_policy_name.clear();

    const std::string body = spec.substr(prefix_len, spec.size() - prefix_len - 1);
    std::stringstream ss(body);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) {
            continue;
        }
        const size_t eq = token.find('=');
        if (eq == std::string::npos) {
            fprintf(stderr, "%s: malformed MXFP6_E2M3 tensor override token '%s'\n", __func__, token.c_str());
            return false;
        }
        const std::string key = trim_copy(token.substr(0, eq));
        const std::string value = trim_copy(token.substr(eq + 1));
        if (key.empty() || value.empty()) {
            fprintf(stderr, "%s: malformed MXFP6_E2M3 tensor override token '%s'\n", __func__, token.c_str());
            return false;
        }

        if (striequals(key.c_str(), "scale_mul") || striequals(key.c_str(), "scale-mul") ||
                striequals(key.c_str(), "mul") || striequals(key.c_str(), "tensor_scale_mul")) {
            out.mxfp6_e2m3_scale_mul = (float) std::stof(value);
            if (!(out.mxfp6_e2m3_scale_mul > 0.0f) || !std::isfinite(out.mxfp6_e2m3_scale_mul)) {
                fprintf(stderr, "%s: invalid MXFP6_E2M3 scale_mul '%s'\n", __func__, value.c_str());
                return false;
            }
            out.has_mxfp6_scale_mul = true;
        } else if (striequals(key.c_str(), "policy") || striequals(key.c_str(), "name")) {
            out.mxfp6_policy_name = value;
        } else {
            fprintf(stderr, "%s: unknown MXFP6_E2M3 tensor override key '%s'\n", __func__, key.c_str());
            return false;
        }
    }

    return true;
}

bool parse_on_off_value(const std::string & value, bool & out) {
    if (striequals(value.c_str(), "1") || striequals(value.c_str(), "on") ||
            striequals(value.c_str(), "true") || striequals(value.c_str(), "yes")) {
        out = true;
        return true;
    }
    if (striequals(value.c_str(), "0") || striequals(value.c_str(), "off") ||
            striequals(value.c_str(), "false") || striequals(value.c_str(), "no")) {
        out = false;
        return true;
    }
    return false;
}

bool parse_nvfp4_input_scale_policy(const std::string & value, int32_t & out) {
    if (striequals(value.c_str(), "imatrix-rms") || striequals(value.c_str(), "imatrix_rms") ||
            striequals(value.c_str(), "rms") || striequals(value.c_str(), "current")) {
        out = LLAMA_NVFP4_INPUT_SCALE_IMATRIX_RMS;
        return true;
    }
    if (striequals(value.c_str(), "identity") || striequals(value.c_str(), "none") ||
            striequals(value.c_str(), "1")) {
        out = LLAMA_NVFP4_INPUT_SCALE_IDENTITY;
        return true;
    }
    if (striequals(value.c_str(), "imatrix-sqrtmax") || striequals(value.c_str(), "imatrix_sqrtmax") ||
            striequals(value.c_str(), "sqrtmax")) {
        out = LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTMAX;
        return true;
    }
    if (striequals(value.c_str(), "imatrix-sqrtp99") || striequals(value.c_str(), "imatrix_sqrtp99") ||
            striequals(value.c_str(), "sqrtp99")) {
        out = LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTP99;
        return true;
    }
    if (striequals(value.c_str(), "imatrix-sqrtp999") || striequals(value.c_str(), "imatrix_sqrtp999") ||
            striequals(value.c_str(), "sqrtp999")) {
        out = LLAMA_NVFP4_INPUT_SCALE_IMATRIX_SQRTP999;
        return true;
    }
    return false;
}

bool parse_nvfp4_scale_denom(const std::string & value, float & out) {
    if (striequals(value.c_str(), "current") || striequals(value.c_str(), "default") ||
            striequals(value.c_str(), "2688")) {
        out = 2688.0f;
        return true;
    }
    if (striequals(value.c_str(), "105")) {
        out = 105.0f;
        return true;
    }
    if (striequals(value.c_str(), "1536") || striequals(value.c_str(), "6x256")) {
        out = 1536.0f;
        return true;
    }
    if (striequals(value.c_str(), "1728") || striequals(value.c_str(), "6x288")) {
        out = 1728.0f;
        return true;
    }
    if (striequals(value.c_str(), "2304") || striequals(value.c_str(), "6x384")) {
        out = 2304.0f;
        return true;
    }

    try {
        out = std::stof(value);
    } catch (const std::exception &) {
        return false;
    }
    return out > 0.0f && std::isfinite(out);
}

std::string format_tensor_type_value(const tensor_type_option & opt) {
    if (opt.type == GGML_TYPE_NVFP4 && opt.has_nvfp4_cfg) {
        std::ostringstream os;
        os << "NVFP4{"
           << "choose46=" << nvfp4_choose46_mode_name(opt.nvfp4_cfg.choose46_mode)
           << ",refit=" << opt.nvfp4_cfg.refit_iters
           << ",compand=" << opt.nvfp4_cfg.use_compand_sat
           << ",cap6=" << opt.nvfp4_cfg.cap_m6
           << ",cap4=" << opt.nvfp4_cfg.cap_m4;
        if ((opt.nvfp4_cfg.reserved_i32 & NVFP4_CUDA_FLAG_RSF) != 0) {
            os << ",rsf=1";
            const int rsf_mode = (opt.nvfp4_cfg.reserved_i32 & NVFP4_CUDA_RSF_MODE_MASK) >> NVFP4_CUDA_RSF_MODE_SHIFT;
            switch (rsf_mode) {
                case NVFP4_CUDA_RSF_MODE_SLICE:  os << ",rsf_mode=slice"; break;
                case NVFP4_CUDA_RSF_MODE_EXPERT: os << ",rsf_mode=expert"; break;
                case NVFP4_CUDA_RSF_MODE_GROUP:  os << ",rsf_mode=group"; break;
                default:                         os << ",rsf_mode=tensor"; break;
            }
            const int rsf_depth = (opt.nvfp4_cfg.reserved_i32 & NVFP4_CUDA_RSF_DEPTH_MASK) >> NVFP4_CUDA_RSF_DEPTH_SHIFT;
            switch (rsf_depth) {
                case NVFP4_CUDA_RSF_DEPTH_DEEP:       os << ",rsf_depth=deep"; break;
                case NVFP4_CUDA_RSF_DEPTH_DEEPER:     os << ",rsf_depth=deeper"; break;
                case NVFP4_CUDA_RSF_DEPTH_EXHAUSTIVE: os << ",rsf_depth=exhaustive"; break;
                default:                              os << ",rsf_depth=normal"; break;
            }
        }
        if (opt.nvfp4_sample_blocks > 0) {
            os << ",sample=" << opt.nvfp4_sample_blocks;
        }
        if (!opt.nvfp4_policy_name.empty()) {
            os << ",policy=" << opt.nvfp4_policy_name;
        }
        os << "}";
        return os.str();
    }
    if (opt.type == GGML_TYPE_MXFP6_E2M3 && opt.has_mxfp6_scale_mul) {
        std::ostringstream os;
        os << "MXFP6_E2M3{scale_mul=" << opt.mxfp6_e2m3_scale_mul;
        if (!opt.mxfp6_policy_name.empty()) {
            os << ",policy=" << opt.mxfp6_policy_name;
        }
        os << "}";
        return os.str();
    }
    if (tensor_type_is_k_rsf_capable(opt.type) && opt.has_k_rsf_mode && opt.k_rsf_mode > 0) {
        std::ostringstream os;
        os << ggml_type_name(opt.type) << "{rsf=" << k_rsf_mode_name(opt.k_rsf_mode) << "}";
        return os.str();
    }
    return ggml_type_name(opt.type);
}

bool parse_tensor_type(const char * data, std::vector<tensor_type_option> & tensor_type) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr) {
        printf("\n%s: malformed tensor type '%s'\n\n", __func__, data);
        return false;
    }

    const size_t tn_len = sep - data;
    if (tn_len == 0) {
        printf("\n%s: missing tensor name\n\n", __func__);
        return false;
    }
    if (const size_t qt_len = strlen(sep); qt_len == 1) {
        printf("\n%s: missing quantization type\n\n", __func__);
        return false;
    }

    std::string tn(data, tn_len);
    std::transform(tn.begin(), tn.end(), tn.begin(), [](unsigned char ch) {
        return (char) std::tolower(ch);
    });
    sep++;
    tensor_type_option tensor_type_opt;
    tensor_type_opt.name = tn;
    if (!parse_tensor_type_nvfp4_cfg(sep, tensor_type_opt) &&
        !parse_tensor_type_k_rsf_cfg(sep, tensor_type_opt) &&
        !parse_tensor_type_mxfp6_cfg(sep, tensor_type_opt)) {
        tensor_type_opt.type = parse_ggml_type(sep);
    }
    tensor_type.emplace_back(std::move(tensor_type_opt));
    if (tensor_type_opt.type == GGML_TYPE_COUNT) {
        printf("\n%s: invalid quantization type '%s'\n\n", __func__, sep);
        return false;
    }

    return true;
}

bool parse_tensor_type_file(const char * filename, std::vector<tensor_type_option> & tensor_type) {
    std::ifstream file(filename);
    if (!file) {
        printf("\n%s: failed to open file '%s': %s\n\n", __func__, filename, std::strerror(errno));
        return false;
    }

    std::string arg;
    while (file >> arg) {
        if (!parse_tensor_type(arg.c_str(), tensor_type)) {
            return false;
        }
    }

    return true;
}

bool parse_layer_prune(const char * data, std::vector<int> & prune_layers) {
    if (!data) {
        printf("\n%s: no layer pruning ids provided\n\n", __func__);
        return false;
    }

    const auto block_ids = string_split<std::string>(data, ',');
    for (const auto & block_id : block_ids) {
        int id;
        try {
            id = std::stoi(block_id);
        } catch (...) {
            id = -1;
        }
        if (id < 0) {
            printf("\n%s: invalid layer id '%s'\n\n", __func__, block_id.c_str());
            return false;
        }
        prune_layers.emplace_back(id);
    }

    std::sort(prune_layers.begin(), prune_layers.end());
    prune_layers.erase(std::unique(prune_layers.begin(), prune_layers.end()), prune_layers.end());
    return true;
}
