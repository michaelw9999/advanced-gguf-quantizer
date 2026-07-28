#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace bq {

struct Recipe {
    struct Target {
        std::string precision_mode = "NVFP4";
        double model_params_b = 0.0;
        int vram_gb = 0;
        int ram_gb = 0;
        double target_bpw = 0.0;
        double weight_budget_gib = 0.0;
        double kv_cache_gib = 0.0;
        double activation_headroom_gib = 0.0;
        bool fit_to_vram = false;
        std::string sizing_note;
    } target;

    struct Quantizer {
        bool enabled = true;
        std::string mode = "normal";
        std::string objective = "kld-first";
        std::string evidence = "real-ppl-kld";
        std::string policy_set = "native-full";
        bool require_kld = true;
        bool require_corpus = true;
        bool require_imatrix = true;
        bool allow_diagnostic = false;
    } quantizer;

    struct StockFtype {
        std::string source = "llama.cpp";
        std::string mostly_type;
        bool preserve_embeddings = true;
        std::string output_policy = "stock-or-explicit";
        bool sweep_tensor_policy = true;
        bool sweep_sensitive_tensors = true;
        std::vector<std::string> token_embedding_candidates = { "NVFP4" };
        std::vector<std::string> output_tensor_candidates = { "Q6_K" };
        double min_quant_savings_mib = 2.0;
        std::vector<std::string> technique_candidates = { "ptq", "kld-best" };
        std::string rationale;
    } stock_ftype;

    struct Io {
        std::string input;
        std::string output;
        std::string patch_base;
        bool keep_split = false;
    } io;

    struct Base {
        std::string ftype = "NVFP4";
        int threads = 0;
        std::string output_tensor_type;
        std::string token_embedding_type;
        std::string mtp_tensor_type;
        bool dry_run = false;
        bool allow_requantize = false;
        bool leave_output_tensor = false;
        bool pure = false;
        bool copy_only = false;
    } base;

    struct Model {
        std::string prune_layers;
    } model;

    struct Metadata {
        std::vector<std::string> overrides;
    } metadata;

    struct Calibration {
        std::string imatrix;
        std::string corpus;
        std::string imatrix_bin = "llama-imatrix";
        std::string ctx_size;
        std::string batch_size = "2048";
        std::string ubatch_size = "1024";
        std::string n_gpu_layers;
        std::string threads;
        std::string threads_batch;
        std::string extra_args;
        std::vector<std::string> include_weights;
        std::vector<std::string> exclude_weights;
    } calibration;

    struct Evaluation {
        std::string kld_mode;
        std::string bf16_reference;
        std::string corpus;
        std::string kld_base;
        std::string bundle;
        std::string perplexity_bin = "llama-perplexity";
    } evaluation;

    struct TensorOverrides {
        std::vector<std::string> files;
        std::vector<std::string> entries;
    } tensor_overrides;

    struct Nvfp4 {
        std::string preset = "baseline";
        std::string cfg;
        std::string correction_denom = "2688";
        std::string input_scale_policy = "imatrix-rms";
        std::vector<std::string> calibration_families = { "max", "kld_best" };
        std::string scale_tie = "none";

        struct Rsf {
            std::string mode = "tensor";
            std::string depth;
        } rsf;

        struct Encoder {
            std::string max_blocks;
            std::string threads;
        } encoder;

        struct FourSix {
            std::string choose46;
            std::string refit_iters;
            std::string compand;
            std::string cap6;
            std::string cap4;
        } four_six;
    } nvfp4;

    struct Mxfp6 {
        std::string tensor_scale = "on";
        std::string min_savings_bytes = "2097152";
        std::string input_scale_denom;
        std::string input_scale_quantile;
        std::string tensor_scale_sample_blocks;
        std::string tensor_scale_steps;
        std::string selector_scale_top;
        std::string selector_scale_candidates;
    } mxfp6;

    struct Nv4Mx6 {
        std::string policy;
        std::string mx6_penalty;
        std::string bf16_mx6_threshold;
        std::string sample_blocks;
        std::string sample_cap;
        std::string imatrix_weight_blend;
        std::string imatrix_weight_power;
        std::string imatrix_weight_min;
        std::string imatrix_weight_max;
    } nv4mx6;

    struct Selector {
        std::string effort = "deep";
        std::string kld;
        std::string checkpoint_model;
        std::string cache_dir;
        std::string skip_file;
        std::string ledger;
        std::string search;
        std::string local_top_k;
        std::string group_units;
        std::string beam_width;
        std::string exact_budget;
        std::string delta_mode;
        bool keep_checkpoint = false;
        bool require_runtime_cache = false;
        std::string stagea_sample_blocks;
        std::string stagea_max_policies;
        std::string refine_top;
        std::string refine_budget;
        std::string survey_top;
        std::string survey_sample_blocks;
        std::string max_tensors;
        bool trace = false;
        std::string policy_threads;
        std::string threads;
        std::string kld_threads;
        bool only = false;
        std::string eval_top;
        std::string n_seq;
        std::string sensitivity_report;
        std::string sensitivity_top;
        std::string sensitivity_layer;
        std::string sensitivity_tensor;
        std::string sensitivity_sample_blocks;
        std::string rsf_report;

        struct Ranking {
            std::string kld_penalty;
            std::string p99_penalty;
            std::string p999_penalty;
            std::string max_kld_penalty;
            std::string kld_threshold;
            std::string p99_threshold;
            std::string p999_threshold;
            std::string max_kld_threshold;
            bool kld_hard_gate = false;
            bool p99_hard_gate = false;
            bool p999_hard_gate = false;
            bool max_kld_hard_gate = false;
        } ranking;
    } selector;

    struct Rescue {
        bool enabled = false;
        std::string type = "Q8_0";
        std::string top;
        std::string report_top;
        std::string budget_mb;
        std::string bf16_budget_mb;
        std::string class_limit;
        std::string encoder_top;
        std::string sample_blocks;
        std::string coarse_max_blocks;
        std::string refine_max_blocks;
        std::string guard_max_blocks;
        std::string report;
        std::string tensor_types;
    } rescue;

    struct Artifacts {
        std::string run_dir;
    } artifacts;
};

struct LoadedRecipe {
    Recipe recipe;
    std::map<std::string, std::string> raw_values;
};

LoadedRecipe load_recipe_file(const std::string & path);
void apply_override(LoadedRecipe & loaded, const std::string & assignment);
std::vector<std::string> validate_recipe(const Recipe & recipe, bool require_io);
std::string dump_recipe_toml(const Recipe & recipe);
std::string canonical_quant_type(std::string value);
std::string sanitize_tensor_type_token(std::string value);
bool quant_type_uses_nvfp4(const std::string & value);
bool quant_type_uses_mxfp6(const std::string & value);
bool quant_type_uses_mxfp8(const std::string & value);
std::vector<std::string> quant_type_choices();
Recipe default_recipe();
Recipe default_recipe(const std::string & profile);
Recipe default_recipe_for_quant_type(const std::string & precision_mode);
std::string default_recipe_toml(const std::string & profile);
void apply_quantizer_mode(Recipe & recipe);
std::vector<std::string> build_quantize_args(const Recipe & recipe, bool force_dry_run);

} // namespace bq
