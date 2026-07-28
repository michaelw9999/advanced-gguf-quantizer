#include "recipe.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace bq {
namespace {

static std::string trim(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

static std::string strip_comment(const std::string & line) {
    bool in_string = false;
    bool escape = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escape = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (c == '#' && !in_string) {
            return line.substr(0, i);
        }
    }
    return line;
}

static std::string unquote(std::string value) {
    value = trim(value);
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        std::string out;
        out.reserve(value.size() - 2);
        bool escape = false;
        for (size_t i = 1; i + 1 < value.size(); ++i) {
            const char c = value[i];
            if (escape) {
                switch (c) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    default:  out.push_back(c);    break;
                }
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else {
                out.push_back(c);
            }
        }
        return out;
    }
    return value;
}

static bool parse_bool_value(const std::string & value) {
    const std::string v = trim(value);
    if (v == "true" || v == "on" || v == "1" || v == "yes") {
        return true;
    }
    if (v == "false" || v == "off" || v == "0" || v == "no") {
        return false;
    }
    throw std::runtime_error("invalid boolean value: " + value);
}

static double parse_double_value(const std::string & value) {
    size_t pos = 0;
    const std::string v = trim(value);
    const double out = std::stod(v, &pos);
    if (pos != v.size()) {
        throw std::runtime_error("invalid floating point value: " + value);
    }
    return out;
}

static int parse_int_value(const std::string & value) {
    size_t pos = 0;
    const int out = std::stoi(trim(value), &pos);
    if (pos != trim(value).size()) {
        throw std::runtime_error("invalid integer value: " + value);
    }
    return out;
}

static std::string canonical_mixed_policy(std::string value);

struct QuantTypeProfile {
    const char * type;
    const char * profile;
};

static constexpr std::array<QuantTypeProfile, 4> QUANT_TYPE_PROFILES = {{
    { "NVFP4", "nvfp4" },
    { "MXFP6", "mxfp6" },
    { "MXFP8", "mxfp8" },
    { "NVFP4_MXFP6", "nvfp4_mxfp6" },
}};

static std::vector<std::string> parse_string_list(std::string value) {
    value = trim(value);
    if (value.empty()) {
        return {};
    }
    if (value.front() != '[' || value.back() != ']') {
        return { unquote(value) };
    }
    value = value.substr(1, value.size() - 2);
    std::vector<std::string> out;
    bool in_string = false;
    bool escape = false;
    size_t start = 0;
    for (size_t i = 0; i <= value.size(); ++i) {
        const char c = i < value.size() ? value[i] : ',';
        if (escape) {
            escape = false;
        } else if (c == '\\' && in_string) {
            escape = true;
        } else if (c == '"') {
            in_string = !in_string;
        } else if (c == ',' && !in_string) {
            const std::string item = trim(value.substr(start, i - start));
            if (!item.empty()) {
                out.push_back(unquote(item));
            }
            start = i + 1;
        }
    }
    return out;
}

static void set_value(LoadedRecipe & loaded, const std::string & path, const std::string & raw_value) {
    Recipe & r = loaded.recipe;
    const std::string value = unquote(raw_value);
    loaded.raw_values[path] = value;

    if (path == "version") { return; }

    if (path == "target.precision_mode") { r.target.precision_mode = canonical_quant_type(value); return; }
    if (path == "target.model_params_b") { r.target.model_params_b = parse_double_value(value); return; }
    if (path == "target.vram_gb") { r.target.vram_gb = parse_int_value(value); return; }
    if (path == "target.ram_gb") { r.target.ram_gb = parse_int_value(value); return; }
    if (path == "target.target_bpw") { r.target.target_bpw = parse_double_value(value); return; }
    if (path == "target.weight_budget_gib") { r.target.weight_budget_gib = parse_double_value(value); return; }
    if (path == "target.kv_cache_gib") { r.target.kv_cache_gib = parse_double_value(value); return; }
    if (path == "target.activation_headroom_gib") { r.target.activation_headroom_gib = parse_double_value(value); return; }
    if (path == "target.fit_to_vram") { r.target.fit_to_vram = parse_bool_value(value); return; }
    if (path == "target.sizing_note") { r.target.sizing_note = value; return; }

    if (path == "quantizer.enabled") { r.quantizer.enabled = parse_bool_value(value); return; }
    if (path == "quantizer.mode") { r.quantizer.mode = value; return; }
    if (path == "quantizer.objective") { r.quantizer.objective = value; return; }
    if (path == "quantizer.evidence") { r.quantizer.evidence = value; return; }
    if (path == "quantizer.policy_set") { r.quantizer.policy_set = value; return; }
    if (path == "quantizer.require_kld") { r.quantizer.require_kld = parse_bool_value(value); return; }
    if (path == "quantizer.require_corpus") { r.quantizer.require_corpus = parse_bool_value(value); return; }
    if (path == "quantizer.require_imatrix") { r.quantizer.require_imatrix = parse_bool_value(value); return; }
    if (path == "quantizer.allow_diagnostic") { r.quantizer.allow_diagnostic = parse_bool_value(value); return; }

    if (path == "stock_ftype.source") { r.stock_ftype.source = value; return; }
    if (path == "stock_ftype.mostly_type") { r.stock_ftype.mostly_type = value; return; }
    if (path == "stock_ftype.preserve_embeddings") { r.stock_ftype.preserve_embeddings = parse_bool_value(value); return; }
    if (path == "stock_ftype.output_policy") { r.stock_ftype.output_policy = value; return; }
    if (path == "stock_ftype.sweep_tensor_policy") { r.stock_ftype.sweep_tensor_policy = parse_bool_value(value); r.stock_ftype.sweep_sensitive_tensors = r.stock_ftype.sweep_tensor_policy; return; }
    if (path == "stock_ftype.sweep_sensitive_tensors") { r.stock_ftype.sweep_sensitive_tensors = parse_bool_value(value); r.stock_ftype.sweep_tensor_policy = r.stock_ftype.sweep_sensitive_tensors; return; }
    if (path == "stock_ftype.token_embedding_candidates") { r.stock_ftype.token_embedding_candidates = parse_string_list(raw_value); return; }
    if (path == "stock_ftype.output_tensor_candidates") { r.stock_ftype.output_tensor_candidates = parse_string_list(raw_value); return; }
    if (path == "stock_ftype.min_quant_savings_mib") { r.stock_ftype.min_quant_savings_mib = parse_double_value(value); return; }
    if (path == "stock_ftype.technique_candidates") { r.stock_ftype.technique_candidates = parse_string_list(raw_value); return; }
    if (path == "stock_ftype.rationale") { r.stock_ftype.rationale = value; return; }

    if (path == "io.input") { r.io.input = value; return; }
    if (path == "io.output") { r.io.output = value; return; }
    if (path == "io.patch_base") { r.io.patch_base = value; return; }
    if (path == "io.keep_split") { r.io.keep_split = parse_bool_value(value); return; }

    if (path == "base.ftype") { r.base.ftype = canonical_quant_type(value); return; }
    if (path == "base.threads") { r.base.threads = parse_int_value(value); return; }
    if (path == "base.output_tensor_type") { r.base.output_tensor_type = sanitize_tensor_type_token(value); return; }
    if (path == "base.token_embedding_type") { r.base.token_embedding_type = sanitize_tensor_type_token(value); return; }
    if (path == "base.mtp_tensor_type") { r.base.mtp_tensor_type = sanitize_tensor_type_token(value); return; }
    if (path == "base.dry_run") { r.base.dry_run = parse_bool_value(value); return; }
    if (path == "base.allow_requantize") { r.base.allow_requantize = parse_bool_value(value); return; }
    if (path == "base.leave_output_tensor") { r.base.leave_output_tensor = parse_bool_value(value); return; }
    if (path == "base.pure") { r.base.pure = parse_bool_value(value); return; }
    if (path == "base.copy_only") { r.base.copy_only = parse_bool_value(value); return; }

    if (path == "model.prune_layers") { r.model.prune_layers = value; return; }

    if (path == "metadata.overrides") { r.metadata.overrides = parse_string_list(raw_value); return; }

    if (path == "calibration.imatrix") { r.calibration.imatrix = value; return; }
    if (path == "calibration.corpus") { r.calibration.corpus = value; return; }
    if (path == "calibration.imatrix_bin") { r.calibration.imatrix_bin = value; return; }
    if (path == "calibration.ctx_size") { r.calibration.ctx_size = value; return; }
    if (path == "calibration.batch_size") { r.calibration.batch_size = value; return; }
    if (path == "calibration.ubatch_size") { r.calibration.ubatch_size = value; return; }
    if (path == "calibration.n_gpu_layers") { r.calibration.n_gpu_layers = value; return; }
    if (path == "calibration.threads") { r.calibration.threads = value; return; }
    if (path == "calibration.threads_batch") { r.calibration.threads_batch = value; return; }
    if (path == "calibration.extra_args") { r.calibration.extra_args = value; return; }
    if (path == "calibration.include_weights") { r.calibration.include_weights = parse_string_list(raw_value); return; }
    if (path == "calibration.exclude_weights") { r.calibration.exclude_weights = parse_string_list(raw_value); return; }

    if (path == "evaluation.kld_mode") { r.evaluation.kld_mode = value; return; }
    if (path == "evaluation.bf16_reference") { r.evaluation.bf16_reference = value; return; }
    if (path == "evaluation.corpus") { r.evaluation.corpus = value; return; }
    if (path == "evaluation.kld_base") { r.evaluation.kld_base = value; return; }
    if (path == "evaluation.bundle") { r.evaluation.bundle = value; return; }
    if (path == "evaluation.perplexity_bin") { r.evaluation.perplexity_bin = value; return; }

    if (path == "tensor_overrides.files") { r.tensor_overrides.files = parse_string_list(raw_value); return; }
    if (path == "tensor_overrides.entries") { r.tensor_overrides.entries = parse_string_list(raw_value); return; }

    if (path == "nvfp4.preset") { r.nvfp4.preset = value; return; }
    if (path == "nvfp4.cfg") { r.nvfp4.cfg = value; return; }
    if (path == "nvfp4.correction_denom") { r.nvfp4.correction_denom = value; return; }
    if (path == "nvfp4.input_scale_policy") { r.nvfp4.input_scale_policy = value; return; }
    if (path == "nvfp4.calibration_families") { r.nvfp4.calibration_families = parse_string_list(raw_value); return; }
    if (path == "nvfp4.scale_tie") { r.nvfp4.scale_tie = value; return; }
    if (path == "nvfp4.rsf.mode") { r.nvfp4.rsf.mode = value; return; }
    if (path == "nvfp4.rsf.depth") { r.nvfp4.rsf.depth = value; return; }
    if (path == "nvfp4.encoder.max_blocks") { r.nvfp4.encoder.max_blocks = value; return; }
    if (path == "nvfp4.encoder.threads") { r.nvfp4.encoder.threads = value; return; }

    if (path == "nvfp4.four_six.choose46") { r.nvfp4.four_six.choose46 = value; return; }
    if (path == "nvfp4.four_six.refit_iters") { r.nvfp4.four_six.refit_iters = value; return; }
    if (path == "nvfp4.four_six.compand") { r.nvfp4.four_six.compand = value; return; }
    if (path == "nvfp4.four_six.cap6") { r.nvfp4.four_six.cap6 = value; return; }
    if (path == "nvfp4.four_six.cap4") { r.nvfp4.four_six.cap4 = value; return; }

    if (path == "mxfp6.tensor_scale") { r.mxfp6.tensor_scale = value; return; }
    if (path == "mxfp6.min_savings_bytes") { r.mxfp6.min_savings_bytes = value; return; }
    if (path == "mxfp6.input_scale_denom") { r.mxfp6.input_scale_denom = value; return; }
    if (path == "mxfp6.input_scale_quantile") { r.mxfp6.input_scale_quantile = value; return; }
    if (path == "mxfp6.tensor_scale_sample_blocks") { r.mxfp6.tensor_scale_sample_blocks = value; return; }
    if (path == "mxfp6.tensor_scale_steps") { r.mxfp6.tensor_scale_steps = value; return; }
    if (path == "mxfp6.selector_scale_top") { r.mxfp6.selector_scale_top = value; return; }
    if (path == "mxfp6.selector_scale_candidates") { r.mxfp6.selector_scale_candidates = value; return; }

    if (path == "mixed.policy") { r.nv4mx6.policy = canonical_mixed_policy(value); return; }
    if (path == "mixed.mx6_penalty") { r.nv4mx6.mx6_penalty = value; return; }
    if (path == "mixed.bf16_mx6_threshold") { r.nv4mx6.bf16_mx6_threshold = value; return; }
    if (path == "mixed.sample_blocks") { r.nv4mx6.sample_blocks = value; return; }
    if (path == "mixed.sample_cap") { r.nv4mx6.sample_cap = value; return; }
    if (path == "mixed.imatrix_weight_blend") { r.nv4mx6.imatrix_weight_blend = value; return; }
    if (path == "mixed.imatrix_weight_power") { r.nv4mx6.imatrix_weight_power = value; return; }
    if (path == "mixed.imatrix_weight_min") { r.nv4mx6.imatrix_weight_min = value; return; }
    if (path == "mixed.imatrix_weight_max") { r.nv4mx6.imatrix_weight_max = value; return; }

    if (path == "selector.effort") { r.selector.effort = value; return; }
    if (path == "selector.kld") { r.selector.kld = value; return; }
    if (path == "selector.checkpoint_model") { r.selector.checkpoint_model = value; return; }
    if (path == "selector.cache_dir") { r.selector.cache_dir = value; return; }
    if (path == "selector.skip_file") { r.selector.skip_file = value; return; }
    if (path == "selector.ledger") { r.selector.ledger = value; return; }
    if (path == "selector.search") { r.selector.search = value; return; }
    if (path == "selector.local_top_k") { r.selector.local_top_k = value; return; }
    if (path == "selector.group_units") { r.selector.group_units = value; return; }
    if (path == "selector.beam_width") { r.selector.beam_width = value; return; }
    if (path == "selector.exact_budget") { r.selector.exact_budget = value; return; }
    if (path == "selector.delta_mode") { r.selector.delta_mode = value; return; }
    if (path == "selector.keep_checkpoint") { r.selector.keep_checkpoint = parse_bool_value(value); return; }
    if (path == "selector.require_runtime_cache") { r.selector.require_runtime_cache = parse_bool_value(value); return; }
    if (path == "selector.stagea_sample_blocks") { r.selector.stagea_sample_blocks = value; return; }
    if (path == "selector.stagea_max_policies") { r.selector.stagea_max_policies = value; return; }
    if (path == "selector.refine_top") { r.selector.refine_top = value; return; }
    if (path == "selector.refine_budget") { r.selector.refine_budget = value; return; }
    if (path == "selector.survey_top") { r.selector.survey_top = value; return; }
    if (path == "selector.survey_sample_blocks") { r.selector.survey_sample_blocks = value; return; }
    if (path == "selector.max_tensors") { r.selector.max_tensors = value; return; }
    if (path == "selector.trace") { r.selector.trace = parse_bool_value(value); return; }
    if (path == "selector.policy_threads") { r.selector.policy_threads = value; return; }
    if (path == "selector.threads") { r.selector.threads = value; return; }
    if (path == "selector.kld_threads") { r.selector.kld_threads = value; return; }
    if (path == "selector.only") { r.selector.only = parse_bool_value(value); return; }
    if (path == "selector.eval_top") { r.selector.eval_top = value; return; }
    if (path == "selector.n_seq") { r.selector.n_seq = value; return; }
    if (path == "selector.sensitivity_report") { r.selector.sensitivity_report = value; return; }
    if (path == "selector.sensitivity_top") { r.selector.sensitivity_top = value; return; }
    if (path == "selector.sensitivity_layer") { r.selector.sensitivity_layer = value; return; }
    if (path == "selector.sensitivity_tensor") { r.selector.sensitivity_tensor = value; return; }
    if (path == "selector.sensitivity_sample_blocks") { r.selector.sensitivity_sample_blocks = value; return; }
    if (path == "selector.rsf_report") { r.selector.rsf_report = value; return; }

    if (path == "selector.ranking.kld_penalty") { r.selector.ranking.kld_penalty = value; return; }
    if (path == "selector.ranking.p99_penalty") { r.selector.ranking.p99_penalty = value; return; }
    if (path == "selector.ranking.p999_penalty") { r.selector.ranking.p999_penalty = value; return; }
    if (path == "selector.ranking.max_kld_penalty") { r.selector.ranking.max_kld_penalty = value; return; }
    if (path == "selector.ranking.kld_threshold") { r.selector.ranking.kld_threshold = value; return; }
    if (path == "selector.ranking.p99_threshold") { r.selector.ranking.p99_threshold = value; return; }
    if (path == "selector.ranking.p999_threshold") { r.selector.ranking.p999_threshold = value; return; }
    if (path == "selector.ranking.max_kld_threshold") { r.selector.ranking.max_kld_threshold = value; return; }
    if (path == "selector.ranking.kld_hard_gate") { r.selector.ranking.kld_hard_gate = parse_bool_value(value); return; }
    if (path == "selector.ranking.p99_hard_gate") { r.selector.ranking.p99_hard_gate = parse_bool_value(value); return; }
    if (path == "selector.ranking.p999_hard_gate") { r.selector.ranking.p999_hard_gate = parse_bool_value(value); return; }
    if (path == "selector.ranking.max_kld_hard_gate") { r.selector.ranking.max_kld_hard_gate = parse_bool_value(value); return; }

    if (path == "rescue.enabled") { r.rescue.enabled = parse_bool_value(value); return; }
    if (path == "rescue.type") { r.rescue.type = value; return; }
    if (path == "rescue.top") { r.rescue.top = value; return; }
    if (path == "rescue.report_top") { r.rescue.report_top = value; return; }
    if (path == "rescue.budget_mb") { r.rescue.budget_mb = value; return; }
    if (path == "rescue.bf16_budget_mb") { r.rescue.bf16_budget_mb = value; return; }
    if (path == "rescue.class_limit") { r.rescue.class_limit = value; return; }
    if (path == "rescue.encoder_top") { r.rescue.encoder_top = value; return; }
    if (path == "rescue.sample_blocks") { r.rescue.sample_blocks = value; return; }
    if (path == "rescue.coarse_max_blocks") { r.rescue.coarse_max_blocks = value; return; }
    if (path == "rescue.refine_max_blocks") { r.rescue.refine_max_blocks = value; return; }
    if (path == "rescue.guard_max_blocks") { r.rescue.guard_max_blocks = value; return; }
    if (path == "rescue.report") { r.rescue.report = value; return; }
    if (path == "rescue.tensor_types") { r.rescue.tensor_types = value; return; }

    if (path == "artifacts.run_dir") { r.artifacts.run_dir = value; return; }

    throw std::runtime_error("unknown recipe field: " + path);
}

static std::string nvfp4_four_six_cfg(const Recipe::Nvfp4::FourSix & fs) {
    std::vector<std::pair<std::string, std::string>> tokens;
    if (!fs.choose46.empty()) {
        tokens.emplace_back("choose46", fs.choose46);
    }
    if (!fs.refit_iters.empty()) {
        tokens.emplace_back("refit", fs.refit_iters);
    }
    if (!fs.compand.empty()) {
        tokens.emplace_back("compand", fs.compand);
    }
    if (!fs.cap6.empty()) {
        tokens.emplace_back("cap6", fs.cap6);
    }
    if (!fs.cap4.empty()) {
        tokens.emplace_back("cap4", fs.cap4);
    }
    if (tokens.empty()) {
        return {};
    }

    std::ostringstream out;
    out << "NVFP4{";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << tokens[i].first << '=' << tokens[i].second;
    }
    out << '}';
    return out.str();
}

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

static std::string canonical_mixed_policy(std::string value) {
    value = trim(value);
    std::replace(value.begin(), value.end(), '-', '_');
    const std::string lower = lower_copy(value);
    if (lower.empty() || lower == "auto") {
        return value.empty() ? std::string() : "auto";
    }
    if (lower == "off" || lower == "none") {
        return "off";
    }
    if (lower == "nvfp4_quality_boost" ||
            lower == "nvfp4_primary" ||
            lower == "quality_boost" ||
            lower == "promote_mxfp6" ||
            lower == "nv4_promote_mx6") {
        return "nv4_promote_mx6";
    }
    if (lower == "mxfp6_primary" ||
            lower == "mxfp6_quality" ||
            lower == "quality_first" ||
            lower == "demote_nvfp4" ||
            lower == "mx6_demote_nv4") {
        return "mx6_demote_nv4";
    }
    if (lower == "mx6_slot" || lower == "mxfp6_slot") {
        return "mx6_slot";
    }
    if (lower == "bf16_mx6" || lower == "bf16_to_mxfp6") {
        return "bf16_mx6";
    }
    if (lower == "bf16_mx6_sse" || lower == "bf16_to_mxfp6_sse") {
        return "bf16_mx6_sse";
    }
    return value;
}

static void append_unique(std::vector<std::string> & values, const std::string & value) {
    if (!value.empty() && std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

static void clear_nvfp4_controls(Recipe & r) {
    r.nvfp4.preset.clear();
    r.nvfp4.cfg.clear();
    r.nvfp4.correction_denom.clear();
    r.nvfp4.input_scale_policy.clear();
    r.nvfp4.calibration_families.clear();
    r.nvfp4.scale_tie.clear();
    r.nvfp4.rsf.mode.clear();
    r.nvfp4.rsf.depth.clear();
    r.nvfp4.encoder.max_blocks.clear();
    r.nvfp4.encoder.threads.clear();
    r.nvfp4.four_six.choose46.clear();
    r.nvfp4.four_six.refit_iters.clear();
    r.nvfp4.four_six.compand.clear();
    r.nvfp4.four_six.cap6.clear();
    r.nvfp4.four_six.cap4.clear();
}

static void clear_mxfp6_controls(Recipe & r) {
    r.mxfp6.tensor_scale.clear();
    r.mxfp6.min_savings_bytes.clear();
    r.mxfp6.input_scale_denom.clear();
    r.mxfp6.input_scale_quantile.clear();
    r.mxfp6.tensor_scale_sample_blocks.clear();
    r.mxfp6.tensor_scale_steps.clear();
    r.mxfp6.selector_scale_top.clear();
    r.mxfp6.selector_scale_candidates.clear();
}

static void clear_mixed_controls(Recipe & r) {
    r.nv4mx6.policy.clear();
    r.nv4mx6.mx6_penalty.clear();
    r.nv4mx6.bf16_mx6_threshold.clear();
    r.nv4mx6.sample_blocks.clear();
    r.nv4mx6.sample_cap.clear();
    r.nv4mx6.imatrix_weight_blend.clear();
    r.nv4mx6.imatrix_weight_power.clear();
    r.nv4mx6.imatrix_weight_min.clear();
    r.nv4mx6.imatrix_weight_max.clear();
}

static void apply_native_policy_set(Recipe & r) {
    const std::string policy = lower_copy(trim(r.quantizer.policy_set));
    if (policy == "manual" || policy == "custom") {
        return;
    }

    const bool uses_nvfp4 = quant_type_uses_nvfp4(r.base.ftype) || quant_type_uses_nvfp4(r.target.precision_mode);
    const bool uses_mxfp6 = quant_type_uses_mxfp6(r.base.ftype) || quant_type_uses_mxfp6(r.target.precision_mode);
    if (!uses_nvfp4 && !uses_mxfp6) {
        return;
    }

    const bool core = policy == "native-core" || policy == "core";
    const bool full = policy.empty() || policy == "native-full" || policy == "full" || policy == "local-full";

    if (core && uses_nvfp4) {
        r.stock_ftype.technique_candidates = { "ptq", "kld-best" };
        r.nvfp4.calibration_families = { "max", "kld_best" };
        r.nvfp4.scale_tie = "none";
    } else if (full && uses_nvfp4) {
        r.stock_ftype.technique_candidates = {
            "ptq",
            "kld-best",
            "auto_search",
            "no_quantize_choice",
            "awq_lite",
            "awq_clip",
            "awq_full",
            "smoothquant",
            "mse_scale_sweep",
            "nvfp4_rsf",
            "kl_div_sensitivity",
        };
        r.nvfp4.calibration_families = {
            "max",
            "kld_best",
            "awq_lite",
            "awq_clip",
            "awq_full",
            "smoothquant",
            "mse_scale_sweep",
            "nvfp4_rsf",
            "kl_div_sensitivity",
        };
        if (r.nvfp4.scale_tie.empty() || r.nvfp4.scale_tie == "none") {
            r.nvfp4.scale_tie = "qkv_gate_up_expert";
        }
    }

    if (uses_nvfp4) {
        for (const char * technique : { "ptq", "kld-best" }) {
            append_unique(r.stock_ftype.technique_candidates, technique);
        }
        for (const char * family : { "max", "kld_best" }) {
            append_unique(r.nvfp4.calibration_families, family);
        }
        for (const char * type : { "BF16", "Q8_0", "Q6_K" }) {
            append_unique(r.stock_ftype.token_embedding_candidates, type);
            append_unique(r.stock_ftype.output_tensor_candidates, type);
        }
        r.stock_ftype.sweep_tensor_policy = true;
        r.stock_ftype.sweep_sensitive_tensors = true;
    }

    if (full && uses_nvfp4) {
        for (const char * technique : {
                "auto_search",
                "no_quantize_choice",
                "awq_lite",
                "awq_clip",
                "awq_full",
                "smoothquant",
                "mse_scale_sweep",
                "nvfp4_rsf",
                "kl_div_sensitivity",
        }) {
            append_unique(r.stock_ftype.technique_candidates, technique);
        }
        for (const char * family : {
                "awq_lite",
                "awq_clip",
                "awq_full",
                "smoothquant",
                "mse_scale_sweep",
                "nvfp4_rsf",
                "kl_div_sensitivity",
        }) {
            append_unique(r.nvfp4.calibration_families, family);
        }

        if (r.nvfp4.scale_tie.empty() || r.nvfp4.scale_tie == "none") {
            r.nvfp4.scale_tie = "qkv_gate_up_expert";
        }
    }

    if (uses_mxfp6) {
        for (const char * type : { "MXFP6_E2M3", "BF16", "Q8_0", "Q6_K" }) {
            append_unique(r.stock_ftype.token_embedding_candidates, type);
            append_unique(r.stock_ftype.output_tensor_candidates, type);
        }
    }
}

static void dump_string(std::ostringstream & out, const char * key, const std::string & value) {
    if (!value.empty()) {
        out << key << " = " << std::quoted(value) << "\n";
    }
}

static void dump_bool(std::ostringstream & out, const char * key, bool value) {
    if (value) {
        out << key << " = true\n";
    }
}

static void dump_bool_explicit(std::ostringstream & out, const char * key, bool value) {
    out << key << " = " << (value ? "true" : "false") << "\n";
}

static void dump_double(std::ostringstream & out, const char * key, double value) {
    if (value > 0.0) {
        out << key << " = " << value << "\n";
    }
}

static void dump_int(std::ostringstream & out, const char * key, int value) {
    if (value > 0) {
        out << key << " = " << value << "\n";
    }
}

static void dump_string_list(std::ostringstream & out, const char * key, const std::vector<std::string> & values) {
    if (values.empty()) {
        return;
    }
    out << key << " = [";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << std::quoted(values[i]);
    }
    out << "]\n";
}

} // namespace

std::string canonical_quant_type(std::string value) {
    value = trim(value);
    std::replace(value.begin(), value.end(), '-', '_');
    const std::string lower = lower_copy(value);
    if (lower == "nvfp4_mxfp6") {
        return "NVFP4_MXFP6";
    }
    if (lower == "nvfp4") {
        return "NVFP4";
    }
    if (lower == "mxfp6" || lower == "mxfp6_e2m3") {
        return "MXFP6";
    }
    if (lower == "mxfp8" || lower == "mxfp8_e4m3") {
        return "MXFP8";
    }
    if (lower == "q8" || lower == "q8_0") {
        return "Q8_0";
    }
    return value;
}

std::string sanitize_tensor_type_token(std::string value) {
    value = trim(value);
    while (value.size() >= 2 &&
            ((value.front() == '\'' && value.back() == '\'') ||
             (value.front() == '"' && value.back() == '"'))) {
        value = trim(value.substr(1, value.size() - 2));
    }

    bool only_quotes = !value.empty();
    for (const unsigned char c : value) {
        if (c != '\'' && c != '"' && !std::isspace(c)) {
            only_quotes = false;
            break;
        }
    }
    return only_quotes ? std::string() : value;
}

bool quant_type_uses_nvfp4(const std::string & value) {
    const std::string quant_type = canonical_quant_type(value);
    return quant_type == "NVFP4" || quant_type == "NVFP4_MXFP6";
}

bool quant_type_uses_mxfp6(const std::string & value) {
    const std::string quant_type = canonical_quant_type(value);
    return quant_type == "MXFP6" || quant_type == "NVFP4_MXFP6";
}

bool quant_type_uses_mxfp8(const std::string & value) {
    return canonical_quant_type(value) == "MXFP8";
}

std::vector<std::string> quant_type_choices() {
    std::vector<std::string> out;
    out.reserve(QUANT_TYPE_PROFILES.size());
    for (const QuantTypeProfile & choice : QUANT_TYPE_PROFILES) {
        out.push_back(choice.type);
    }
    return out;
}

LoadedRecipe load_recipe_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open recipe: " + path);
    }

    LoadedRecipe loaded;
    std::string section;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        line = trim(strip_comment(line));
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            if (section.empty()) {
                throw std::runtime_error("empty section at line " + std::to_string(line_no));
            }
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("expected key=value at line " + std::to_string(line_no));
        }
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key.empty()) {
            throw std::runtime_error("empty key at line " + std::to_string(line_no));
        }
        const std::string path_key = section.empty() ? key : section + "." + key;
        try {
            set_value(loaded, path_key, value);
        } catch (const std::exception & e) {
            throw std::runtime_error("line " + std::to_string(line_no) + ": " + e.what());
        }
    }

    return loaded;
}

void apply_override(LoadedRecipe & loaded, const std::string & assignment) {
    const size_t eq = assignment.find('=');
    if (eq == std::string::npos || eq == 0) {
        throw std::runtime_error("--set expects path=value, got: " + assignment);
    }
    set_value(loaded, trim(assignment.substr(0, eq)), trim(assignment.substr(eq + 1)));
}

std::vector<std::string> validate_recipe(const Recipe & recipe, bool require_io) {
    std::vector<std::string> errors;
    if (require_io && recipe.io.input.empty()) {
        errors.push_back("io.input is required");
    }
    if (require_io && recipe.io.output.empty()) {
        errors.push_back("io.output is required");
    }
    if (recipe.base.dry_run) {
        errors.push_back("base.dry_run is no longer supported; use advanced-gguf-quantizer plan for inspection and advanced-gguf-quantizer run for real saved artifacts");
    }
    if (recipe.base.ftype.empty()) {
        errors.push_back("base.ftype is required");
    }
    const auto check_tensor_type = [&errors](const char * name, const std::string & raw) {
        const std::string value = sanitize_tensor_type_token(raw);
        if (value.empty()) {
            return;
        }
        if (value.find('\'') != std::string::npos || value.find('"') != std::string::npos) {
            errors.push_back(std::string(name) + " contains quote characters; leave it blank for the profile default or use a tensor type like Q6_K");
            return;
        }
        if (std::any_of(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); })) {
            errors.push_back(std::string(name) + " must be a single tensor type token with no spaces");
        }
    };
    check_tensor_type("base.output_tensor_type", recipe.base.output_tensor_type);
    check_tensor_type("base.token_embedding_type", recipe.base.token_embedding_type);
    check_tensor_type("base.mtp_tensor_type", recipe.base.mtp_tensor_type);
    const bool advanced_precision_quant =
        quant_type_uses_nvfp4(recipe.base.ftype) ||
        quant_type_uses_mxfp6(recipe.base.ftype) ||
        quant_type_uses_mxfp8(recipe.base.ftype) ||
        quant_type_uses_nvfp4(recipe.target.precision_mode) ||
        quant_type_uses_mxfp6(recipe.target.precision_mode) ||
        quant_type_uses_mxfp8(recipe.target.precision_mode);
    const std::string quantizer_mode = lower_copy(trim(recipe.quantizer.mode));
    if (quantizer_mode != "fast" &&
            quantizer_mode != "normal" &&
            quantizer_mode != "deep") {
        errors.push_back("quantizer.mode must be one of: fast, normal, deep");
    }
    if (advanced_precision_quant && recipe.quantizer.require_kld && recipe.evaluation.kld_base.empty() && recipe.selector.kld.empty()) {
        errors.push_back("real Blackwell quantizer evidence requires evaluation.kld_base or selector.kld");
    }
    if (advanced_precision_quant && recipe.quantizer.require_corpus && recipe.evaluation.corpus.empty()) {
        errors.push_back("quality Blackwell quantizer evidence requires evaluation.corpus so PPL/KLD uses a named corpus");
    }
    if (advanced_precision_quant && recipe.quantizer.require_imatrix && recipe.calibration.imatrix.empty()) {
        errors.push_back("quality Blackwell quantizer evidence requires calibration.imatrix; generate or select an imatrix for the same corpus before claiming a best-quality run");
    }
    if (advanced_precision_quant && recipe.quantizer.require_imatrix && recipe.calibration.corpus.empty()) {
        errors.push_back("quality Blackwell quantizer evidence requires calibration.corpus so imatrix and local calibration use the same training corpus");
    }
    if (recipe.calibration.include_weights.size() > 0 && recipe.calibration.exclude_weights.size() > 0) {
        errors.push_back("calibration.include_weights and calibration.exclude_weights cannot both be set");
    }
    const std::string rsf_mode = lower_copy(trim(recipe.nvfp4.rsf.mode));
    if (!rsf_mode.empty() &&
            rsf_mode != "tensor" &&
            rsf_mode != "slice" &&
            rsf_mode != "expert" &&
            rsf_mode != "group") {
        errors.push_back("nvfp4.rsf.mode must be one of: tensor, slice, expert, group");
    }
    const std::string rsf_depth = lower_copy(trim(recipe.nvfp4.rsf.depth));
    if (!rsf_depth.empty() &&
            rsf_depth != "normal" &&
            rsf_depth != "deep" &&
            rsf_depth != "deeper" &&
            rsf_depth != "exhaustive") {
        errors.push_back("nvfp4.rsf.depth must be one of: normal, deep, deeper, exhaustive");
    }
    if (recipe.evaluation.kld_mode == "make_base" && recipe.evaluation.bf16_reference.empty() && recipe.io.input.empty()) {
        errors.push_back("evaluation.bf16_reference or io.input is required when evaluation.kld_mode=make_base");
    }
    if (recipe.evaluation.kld_mode == "make_base" && recipe.evaluation.corpus.empty()) {
        errors.push_back("evaluation.corpus is required when evaluation.kld_mode=make_base");
    }
    if ((recipe.evaluation.kld_mode == "existing" || recipe.evaluation.kld_mode == "make_base") &&
            recipe.evaluation.kld_base.empty()) {
        errors.push_back("evaluation.kld_base is required for existing or make_base KLD workflows");
    }
    return errors;
}

std::string dump_recipe_toml(const Recipe & r) {
    std::ostringstream out;
    out << "version = 1\n\n";

    out << "[target]\n";
    dump_string(out, "precision_mode", r.target.precision_mode);
    dump_double(out, "model_params_b", r.target.model_params_b);
    dump_int(out, "vram_gb", r.target.vram_gb);
    dump_int(out, "ram_gb", r.target.ram_gb);
    dump_double(out, "target_bpw", r.target.target_bpw);
    dump_double(out, "weight_budget_gib", r.target.weight_budget_gib);
    dump_double(out, "kv_cache_gib", r.target.kv_cache_gib);
    dump_double(out, "activation_headroom_gib", r.target.activation_headroom_gib);
    dump_bool(out, "fit_to_vram", r.target.fit_to_vram);
    dump_string(out, "sizing_note", r.target.sizing_note);

    out << "\n[quantizer]\n";
    dump_bool_explicit(out, "enabled", r.quantizer.enabled);
    dump_string(out, "mode", r.quantizer.mode);
    dump_string(out, "objective", r.quantizer.objective);
    dump_string(out, "evidence", r.quantizer.evidence);
    dump_string(out, "policy_set", r.quantizer.policy_set);
    dump_bool_explicit(out, "require_kld", r.quantizer.require_kld);
    dump_bool_explicit(out, "require_corpus", r.quantizer.require_corpus);
    dump_bool_explicit(out, "require_imatrix", r.quantizer.require_imatrix);
    dump_bool_explicit(out, "allow_diagnostic", r.quantizer.allow_diagnostic);

    out << "\n[stock_ftype]\n";
    dump_string(out, "source", r.stock_ftype.source);
    dump_string(out, "mostly_type", r.stock_ftype.mostly_type);
    dump_bool_explicit(out, "preserve_embeddings", r.stock_ftype.preserve_embeddings);
    dump_string(out, "output_policy", r.stock_ftype.output_policy);
    dump_bool_explicit(out, "sweep_tensor_policy", r.stock_ftype.sweep_tensor_policy);
    dump_string_list(out, "token_embedding_candidates", r.stock_ftype.token_embedding_candidates);
    dump_string_list(out, "output_tensor_candidates", r.stock_ftype.output_tensor_candidates);
    dump_double(out, "min_quant_savings_mib", r.stock_ftype.min_quant_savings_mib);
    dump_string_list(out, "technique_candidates", r.stock_ftype.technique_candidates);
    dump_string(out, "rationale", r.stock_ftype.rationale);

    out << "\n[io]\n";
    dump_string(out, "input", r.io.input);
    dump_string(out, "output", r.io.output);
    dump_string(out, "patch_base", r.io.patch_base);
    dump_bool(out, "keep_split", r.io.keep_split);

    out << "\n[base]\n";
    dump_string(out, "ftype", r.base.ftype);
    if (r.base.threads > 0) {
        out << "threads = " << r.base.threads << "\n";
    }
    dump_string(out, "output_tensor_type", r.base.output_tensor_type);
    dump_string(out, "token_embedding_type", r.base.token_embedding_type);
    dump_string(out, "mtp_tensor_type", r.base.mtp_tensor_type);
    dump_bool(out, "allow_requantize", r.base.allow_requantize);
    dump_bool(out, "leave_output_tensor", r.base.leave_output_tensor);
    dump_bool(out, "pure", r.base.pure);
    dump_bool(out, "copy_only", r.base.copy_only);

    out << "\n[model]\n";
    dump_string(out, "prune_layers", r.model.prune_layers);

    out << "\n[metadata]\n";
    dump_string_list(out, "overrides", r.metadata.overrides);

    out << "\n[calibration]\n";
    dump_string(out, "imatrix", r.calibration.imatrix);
    dump_string(out, "corpus", r.calibration.corpus);
    dump_string(out, "imatrix_bin", r.calibration.imatrix_bin);
    dump_string(out, "ctx_size", r.calibration.ctx_size);
    dump_string(out, "batch_size", r.calibration.batch_size);
    dump_string(out, "ubatch_size", r.calibration.ubatch_size);
    dump_string(out, "n_gpu_layers", r.calibration.n_gpu_layers);
    dump_string(out, "threads", r.calibration.threads);
    dump_string(out, "threads_batch", r.calibration.threads_batch);
    dump_string(out, "extra_args", r.calibration.extra_args);
    dump_string_list(out, "include_weights", r.calibration.include_weights);
    dump_string_list(out, "exclude_weights", r.calibration.exclude_weights);

    out << "\n[evaluation]\n";
    dump_string(out, "kld_mode", r.evaluation.kld_mode);
    dump_string(out, "bf16_reference", r.evaluation.bf16_reference);
    dump_string(out, "corpus", r.evaluation.corpus);
    dump_string(out, "kld_base", r.evaluation.kld_base);
    dump_string(out, "bundle", r.evaluation.bundle);
    dump_string(out, "perplexity_bin", r.evaluation.perplexity_bin);

    out << "\n[tensor_overrides]\n";
    dump_string_list(out, "files", r.tensor_overrides.files);
    dump_string_list(out, "entries", r.tensor_overrides.entries);

    const bool show_low_level = !r.quantizer.enabled || r.quantizer.allow_diagnostic;
    const bool uses_nvfp4 =
        quant_type_uses_nvfp4(r.base.ftype) ||
        quant_type_uses_nvfp4(r.target.precision_mode);
    const bool uses_mxfp6 =
        quant_type_uses_mxfp6(r.base.ftype) ||
        quant_type_uses_mxfp6(r.target.precision_mode);
    const bool has_nvfp4_base =
        !r.nvfp4.preset.empty() ||
        !r.nvfp4.cfg.empty() ||
        !r.nvfp4.correction_denom.empty() ||
        !r.nvfp4.input_scale_policy.empty() ||
        !r.nvfp4.calibration_families.empty() ||
        !r.nvfp4.scale_tie.empty();
    const bool has_nvfp4_rsf = !r.nvfp4.rsf.mode.empty() || !r.nvfp4.rsf.depth.empty();
    const bool has_nvfp4_encoder = !r.nvfp4.encoder.max_blocks.empty() || !r.nvfp4.encoder.threads.empty();
    const bool has_nvfp4_four_six =
        !r.nvfp4.four_six.choose46.empty() ||
        !r.nvfp4.four_six.refit_iters.empty() ||
        !r.nvfp4.four_six.compand.empty() ||
        !r.nvfp4.four_six.cap6.empty() ||
        !r.nvfp4.four_six.cap4.empty();
    const bool has_mxfp6_controls =
        !r.mxfp6.tensor_scale.empty() ||
        !r.mxfp6.min_savings_bytes.empty() ||
        !r.mxfp6.input_scale_denom.empty() ||
        !r.mxfp6.input_scale_quantile.empty() ||
        !r.mxfp6.tensor_scale_sample_blocks.empty() ||
        !r.mxfp6.tensor_scale_steps.empty() ||
        !r.mxfp6.selector_scale_top.empty() ||
        !r.mxfp6.selector_scale_candidates.empty();
    const bool has_mixed_controls =
        !r.nv4mx6.policy.empty() ||
        !r.nv4mx6.mx6_penalty.empty() ||
        !r.nv4mx6.bf16_mx6_threshold.empty() ||
        !r.nv4mx6.sample_blocks.empty() ||
        !r.nv4mx6.sample_cap.empty() ||
        !r.nv4mx6.imatrix_weight_blend.empty() ||
        !r.nv4mx6.imatrix_weight_power.empty() ||
        !r.nv4mx6.imatrix_weight_min.empty() ||
        !r.nv4mx6.imatrix_weight_max.empty();

    if (uses_nvfp4 || has_nvfp4_base || has_nvfp4_rsf || has_nvfp4_encoder || has_nvfp4_four_six) {
        out << "\n[nvfp4]\n";
        dump_string(out, "preset", r.nvfp4.preset);
        dump_string(out, "cfg", r.nvfp4.cfg);
        dump_string(out, "correction_denom", r.nvfp4.correction_denom);
        dump_string(out, "input_scale_policy", r.nvfp4.input_scale_policy);
        dump_string_list(out, "calibration_families", r.nvfp4.calibration_families);
        dump_string(out, "scale_tie", r.nvfp4.scale_tie);
    }

    if (uses_nvfp4 || has_nvfp4_rsf) {
        out << "\n[nvfp4.rsf]\n";
        dump_string(out, "mode", r.nvfp4.rsf.mode);
        dump_string(out, "depth", r.nvfp4.rsf.depth);
    }

    if ((uses_nvfp4 || has_nvfp4_encoder || has_nvfp4_four_six) && show_low_level) {
        out << "\n[nvfp4.encoder]\n";
        dump_string(out, "max_blocks", r.nvfp4.encoder.max_blocks);
        dump_string(out, "threads", r.nvfp4.encoder.threads);

        out << "\n[nvfp4.four_six]\n";
        dump_string(out, "choose46", r.nvfp4.four_six.choose46);
        dump_string(out, "refit_iters", r.nvfp4.four_six.refit_iters);
        dump_string(out, "compand", r.nvfp4.four_six.compand);
        dump_string(out, "cap6", r.nvfp4.four_six.cap6);
        dump_string(out, "cap4", r.nvfp4.four_six.cap4);
    }

    if (uses_mxfp6 || has_mxfp6_controls) {
        out << "\n[mxfp6]\n";
        dump_string(out, "tensor_scale", r.mxfp6.tensor_scale);
        dump_string(out, "min_savings_bytes", r.mxfp6.min_savings_bytes);
        dump_string(out, "input_scale_denom", r.mxfp6.input_scale_denom);
        dump_string(out, "input_scale_quantile", r.mxfp6.input_scale_quantile);
        dump_string(out, "tensor_scale_sample_blocks", r.mxfp6.tensor_scale_sample_blocks);
        dump_string(out, "tensor_scale_steps", r.mxfp6.tensor_scale_steps);
        dump_string(out, "selector_scale_top", r.mxfp6.selector_scale_top);
        dump_string(out, "selector_scale_candidates", r.mxfp6.selector_scale_candidates);
    }

    if ((uses_nvfp4 && uses_mxfp6) || has_mixed_controls) {
        out << "\n[mixed]\n";
        dump_string(out, "policy", r.nv4mx6.policy);
        dump_string(out, "mx6_penalty", r.nv4mx6.mx6_penalty);
        dump_string(out, "bf16_mx6_threshold", r.nv4mx6.bf16_mx6_threshold);
        dump_string(out, "sample_blocks", r.nv4mx6.sample_blocks);
        dump_string(out, "sample_cap", r.nv4mx6.sample_cap);
        dump_string(out, "imatrix_weight_blend", r.nv4mx6.imatrix_weight_blend);
        dump_string(out, "imatrix_weight_power", r.nv4mx6.imatrix_weight_power);
        dump_string(out, "imatrix_weight_min", r.nv4mx6.imatrix_weight_min);
        dump_string(out, "imatrix_weight_max", r.nv4mx6.imatrix_weight_max);
    }

    out << "\n[selector]\n";
    dump_string(out, "effort", r.selector.effort);
    dump_string(out, "kld", r.selector.kld);
    dump_string(out, "checkpoint_model", r.selector.checkpoint_model);
    dump_string(out, "cache_dir", r.selector.cache_dir);
    dump_string(out, "skip_file", r.selector.skip_file);
    dump_string(out, "ledger", r.selector.ledger);
    if (show_low_level) {
        dump_string(out, "search", r.selector.search);
        dump_string(out, "local_top_k", r.selector.local_top_k);
        dump_string(out, "group_units", r.selector.group_units);
        dump_string(out, "beam_width", r.selector.beam_width);
        dump_string(out, "exact_budget", r.selector.exact_budget);
        dump_string(out, "delta_mode", r.selector.delta_mode);
        dump_bool(out, "keep_checkpoint", r.selector.keep_checkpoint);
        dump_bool(out, "require_runtime_cache", r.selector.require_runtime_cache);
        dump_string(out, "stagea_sample_blocks", r.selector.stagea_sample_blocks);
        dump_string(out, "stagea_max_policies", r.selector.stagea_max_policies);
        dump_string(out, "refine_top", r.selector.refine_top);
        dump_string(out, "refine_budget", r.selector.refine_budget);
        dump_string(out, "survey_top", r.selector.survey_top);
        dump_string(out, "survey_sample_blocks", r.selector.survey_sample_blocks);
        dump_string(out, "max_tensors", r.selector.max_tensors);
        dump_bool(out, "trace", r.selector.trace);
        dump_string(out, "policy_threads", r.selector.policy_threads);
        dump_string(out, "threads", r.selector.threads);
        dump_string(out, "kld_threads", r.selector.kld_threads);
        dump_bool(out, "only", r.selector.only);
        dump_string(out, "eval_top", r.selector.eval_top);
        dump_string(out, "n_seq", r.selector.n_seq);
    }
    dump_string(out, "sensitivity_report", r.selector.sensitivity_report);
    dump_string(out, "sensitivity_top", r.selector.sensitivity_top);
    dump_string(out, "sensitivity_layer", r.selector.sensitivity_layer);
    dump_string(out, "sensitivity_tensor", r.selector.sensitivity_tensor);
    dump_string(out, "sensitivity_sample_blocks", r.selector.sensitivity_sample_blocks);
    dump_string(out, "rsf_report", r.selector.rsf_report);

    out << "\n[selector.ranking]\n";
    dump_string(out, "kld_penalty", r.selector.ranking.kld_penalty);
    dump_string(out, "p99_penalty", r.selector.ranking.p99_penalty);
    dump_string(out, "p999_penalty", r.selector.ranking.p999_penalty);
    dump_string(out, "max_kld_penalty", r.selector.ranking.max_kld_penalty);
    dump_string(out, "kld_threshold", r.selector.ranking.kld_threshold);
    dump_string(out, "p99_threshold", r.selector.ranking.p99_threshold);
    dump_string(out, "p999_threshold", r.selector.ranking.p999_threshold);
    dump_string(out, "max_kld_threshold", r.selector.ranking.max_kld_threshold);
    dump_bool(out, "kld_hard_gate", r.selector.ranking.kld_hard_gate);
    dump_bool(out, "p99_hard_gate", r.selector.ranking.p99_hard_gate);
    dump_bool(out, "p999_hard_gate", r.selector.ranking.p999_hard_gate);
    dump_bool(out, "max_kld_hard_gate", r.selector.ranking.max_kld_hard_gate);

    out << "\n[rescue]\n";
    dump_bool(out, "enabled", r.rescue.enabled);
    dump_string(out, "type", r.rescue.type);
    dump_string(out, "top", r.rescue.top);
    dump_string(out, "report_top", r.rescue.report_top);
    dump_string(out, "budget_mb", r.rescue.budget_mb);
    dump_string(out, "bf16_budget_mb", r.rescue.bf16_budget_mb);
    dump_string(out, "class_limit", r.rescue.class_limit);
    dump_string(out, "encoder_top", r.rescue.encoder_top);
    dump_string(out, "sample_blocks", r.rescue.sample_blocks);
    dump_string(out, "coarse_max_blocks", r.rescue.coarse_max_blocks);
    dump_string(out, "refine_max_blocks", r.rescue.refine_max_blocks);
    dump_string(out, "guard_max_blocks", r.rescue.guard_max_blocks);
    dump_string(out, "report", r.rescue.report);
    dump_string(out, "tensor_types", r.rescue.tensor_types);

    out << "\n[artifacts]\n";
    dump_string(out, "run_dir", r.artifacts.run_dir);

    return out.str();
}

std::string default_recipe_toml(const std::string & profile) {
    return dump_recipe_toml(default_recipe(profile));
}

void apply_quantizer_mode(Recipe & r) {
    if (!r.quantizer.enabled) {
        return;
    }

    const std::string mode = lower_copy(trim(r.quantizer.mode));
    const bool fast = mode == "fast";
    const bool normal = mode == "normal";
    if (fast) {
        r.quantizer.mode = "fast";
        r.selector.effort = "fast";
    } else if (normal) {
        r.quantizer.mode = "normal";
        r.selector.effort = "normal";
    } else {
        r.quantizer.mode = "deep";
        r.selector.effort = "deep";
    }

    if (r.quantizer.objective.empty()) {
        r.quantizer.objective = "kld-first";
    }
    if (r.quantizer.evidence.empty()) {
        r.quantizer.evidence = "real-ppl-kld";
    }
    r.quantizer.require_kld = !fast;
    r.quantizer.require_corpus = !fast;
    r.quantizer.require_imatrix = !fast;

    apply_native_policy_set(r);

    const bool uses_nvfp4 = quant_type_uses_nvfp4(r.base.ftype) || quant_type_uses_nvfp4(r.target.precision_mode);
    const bool uses_mxfp6 = quant_type_uses_mxfp6(r.base.ftype) || quant_type_uses_mxfp6(r.target.precision_mode);

    if (!uses_nvfp4 && !uses_mxfp6) {
        return;
    }

    r.selector.keep_checkpoint = true;
    r.selector.require_runtime_cache = !fast && r.quantizer.evidence == "real-ppl-kld";
    auto assign_if_empty = [](std::string & value, const std::string & fallback) {
        if (value.empty()) {
            value = fallback;
        }
    };
    assign_if_empty(r.selector.search, "tensor-policy");
    assign_if_empty(r.selector.local_top_k, "0");
    assign_if_empty(r.selector.group_units, "auto");
    assign_if_empty(r.selector.beam_width, "1");
    assign_if_empty(r.selector.exact_budget, "auto");
    assign_if_empty(r.selector.delta_mode, "estimate");
    if (fast) {
        assign_if_empty(r.selector.stagea_sample_blocks, "2048");
        assign_if_empty(r.selector.stagea_max_policies, "0");
        assign_if_empty(r.selector.refine_top, "12");
        assign_if_empty(r.selector.refine_budget, "96");
        assign_if_empty(r.selector.survey_top, "48");
        assign_if_empty(r.selector.survey_sample_blocks, "2048");
        assign_if_empty(r.selector.max_tensors, "0");
        assign_if_empty(r.selector.eval_top, "16");
        assign_if_empty(r.selector.ranking.kld_penalty, "4.0");
        assign_if_empty(r.selector.ranking.p99_penalty, "1.5");
        assign_if_empty(r.selector.ranking.p999_penalty, "0.75");
        assign_if_empty(r.selector.ranking.max_kld_penalty, "0.10");
        r.selector.ranking.kld_hard_gate = false;
        r.selector.ranking.p99_hard_gate = false;
        r.selector.ranking.p999_hard_gate = false;
        r.selector.ranking.max_kld_hard_gate = false;
        if (uses_nvfp4) {
        assign_if_empty(r.nvfp4.encoder.max_blocks, "8192");
            assign_if_empty(r.nvfp4.rsf.depth, "deeper");
            assign_if_empty(r.nvfp4.four_six.choose46, "adaptive");
            assign_if_empty(r.nvfp4.four_six.refit_iters, "8");
            assign_if_empty(r.nvfp4.four_six.compand, "1");
            assign_if_empty(r.nvfp4.four_six.cap6, "448");
            assign_if_empty(r.nvfp4.four_six.cap4, "256");
        }
        if (uses_mxfp6) {
            if (uses_nvfp4) {
                assign_if_empty(r.nv4mx6.policy, "nv4_promote_mx6");
                assign_if_empty(r.nv4mx6.mx6_penalty, "3.5");
            }
            assign_if_empty(r.mxfp6.selector_scale_top, "16");
        }
        r.rescue.enabled = false;
        assign_if_empty(r.rescue.top, "0");
        assign_if_empty(r.rescue.report_top, "0");
        return;
    }

    const bool normal_mode = r.quantizer.mode == "normal";
    assign_if_empty(r.selector.stagea_sample_blocks, normal_mode ? "8192" : "16384");
    assign_if_empty(r.selector.stagea_max_policies, "0");
    assign_if_empty(r.selector.refine_top, "24");
    assign_if_empty(r.selector.refine_budget, "192");
    assign_if_empty(r.selector.survey_top, "64");
    assign_if_empty(r.selector.survey_sample_blocks, normal_mode ? "8192" : "16384");
    assign_if_empty(r.selector.max_tensors, "0");
    assign_if_empty(r.selector.eval_top, "24");

    if (uses_nvfp4) {
        assign_if_empty(r.nvfp4.encoder.max_blocks, normal_mode ? "32768" : "65536");
        assign_if_empty(r.nvfp4.rsf.depth, normal_mode ? "deeper" : "exhaustive");
        assign_if_empty(r.nvfp4.four_six.choose46, "adaptive");
        assign_if_empty(r.nvfp4.four_six.refit_iters, "16");
        assign_if_empty(r.nvfp4.four_six.compand, "1");
        assign_if_empty(r.nvfp4.four_six.cap6, "448");
        assign_if_empty(r.nvfp4.four_six.cap4, "224");
        assign_if_empty(r.selector.ranking.kld_penalty, "4.0");
        assign_if_empty(r.selector.ranking.p99_penalty, "3.0");
        assign_if_empty(r.selector.ranking.p999_penalty, "1.5");
        assign_if_empty(r.selector.ranking.max_kld_penalty, "0.35");
        r.selector.ranking.kld_hard_gate = false;
        r.selector.ranking.p99_hard_gate = !normal_mode;
        r.selector.ranking.p999_hard_gate = !normal_mode;
        r.selector.ranking.max_kld_hard_gate = false;
    } else if (uses_mxfp6) {
        assign_if_empty(r.selector.ranking.kld_penalty, "8.0");
        assign_if_empty(r.selector.ranking.p99_penalty, "4.0");
        assign_if_empty(r.selector.ranking.p999_penalty, "1.5");
        assign_if_empty(r.selector.ranking.max_kld_penalty, "0.03");
        r.selector.ranking.p99_hard_gate = true;
        r.selector.ranking.p999_hard_gate = true;
    }

    if (uses_mxfp6) {
        if (uses_nvfp4) {
            assign_if_empty(r.nv4mx6.policy, "nv4_promote_mx6");
            assign_if_empty(r.nv4mx6.mx6_penalty, normal_mode ? "3.0" : "3.5");
        }
        assign_if_empty(r.mxfp6.selector_scale_top, normal_mode ? "32" : "48");
        assign_if_empty(r.mxfp6.selector_scale_candidates, "0.771105,0.840896,0.917004,0.957603,1,1.04427,1.09051,1.18921,1.29684");
    } else {
        r.nv4mx6.policy.clear();
        r.nv4mx6.mx6_penalty.clear();
        r.nv4mx6.bf16_mx6_threshold.clear();
        r.mxfp6.selector_scale_top.clear();
        r.mxfp6.selector_scale_candidates.clear();
    }

    if (r.rescue.enabled) {
        assign_if_empty(r.rescue.sample_blocks, normal_mode ? "4096" : "8192");
        assign_if_empty(r.rescue.encoder_top, uses_nvfp4 && !normal_mode ? "16" : "0");
        assign_if_empty(r.rescue.coarse_max_blocks, normal_mode ? "16384" : "32768");
        assign_if_empty(r.rescue.refine_max_blocks, "32768");
        assign_if_empty(r.rescue.guard_max_blocks, normal_mode ? "32768" : "65536");
    }
}

static void apply_deep_quality_defaults(Recipe & r) {
    r.quantizer.enabled = true;
    r.quantizer.mode = "deep";
    r.quantizer.objective = "kld-first";
    r.quantizer.evidence = "real-ppl-kld";
    r.quantizer.require_kld = true;
    r.quantizer.require_corpus = true;
    r.quantizer.require_imatrix = true;
    r.quantizer.allow_diagnostic = false;
    apply_quantizer_mode(r);
}

Recipe default_recipe(const std::string & profile) {
    Recipe r;
    if (profile == "mxfp8") {
        r.target.precision_mode = "MXFP8";
        r.target.target_bpw = 8.25;
        r.base.ftype = "MXFP8";
        r.base.output_tensor_type = "MXFP8";
        r.base.token_embedding_type = "MXFP8";
        r.base.mtp_tensor_type = "MXFP8";
        r.quantizer.enabled = false;
        r.quantizer.objective.clear();
        r.quantizer.evidence.clear();
        r.quantizer.policy_set.clear();
        r.quantizer.require_kld = false;
        r.quantizer.require_corpus = false;
        r.quantizer.require_imatrix = false;
        clear_nvfp4_controls(r);
        clear_mixed_controls(r);
        clear_mxfp6_controls(r);
        r.selector = {};
        r.selector.effort.clear();
        r.rescue.enabled = false;
        r.rescue.type.clear();
        r.stock_ftype.mostly_type = "MOSTLY_MXFP8";
        r.stock_ftype.token_embedding_candidates = { "MXFP8" };
        r.stock_ftype.output_tensor_candidates = { "MXFP8" };
        r.stock_ftype.sweep_tensor_policy = false;
        r.stock_ftype.sweep_sensitive_tensors = false;
        r.stock_ftype.technique_candidates = { "ptq" };
        r.stock_ftype.rationale = "OCP MXFP8 E4M3 recipe with one E8M0 scale per 32-value sub-block.";
    } else if (profile == "mxfp6") {
        r.target.precision_mode = "MXFP6";
        r.base.ftype = "MXFP6";
        clear_nvfp4_controls(r);
        r.base.output_tensor_type = "MXFP6_E2M3";
        r.base.token_embedding_type = "MXFP6_E2M3";
        r.base.mtp_tensor_type = "MXFP6_E2M3";
        apply_deep_quality_defaults(r);
        r.mxfp6.tensor_scale = "on";
        r.stock_ftype.mostly_type = "MOSTLY_MXFP6_E2M3";
        r.stock_ftype.token_embedding_candidates = { "MXFP6_E2M3" };
        r.stock_ftype.output_tensor_candidates = { "MXFP6_E2M3" };
        r.stock_ftype.rationale = "Explicit local MXFP6_E2M3 recipe; embeddings, output, and MTP matrix weights use MXFP6_E2M3.";
    } else if (profile == "nvfp4_mxfp6") {
        r.target.precision_mode = "NVFP4_MXFP6";
        r.base.ftype = "NVFP4_MXFP6";
        r.base.output_tensor_type = "MXFP6_E2M3";
        r.base.token_embedding_type = "MXFP6_E2M3";
        r.base.mtp_tensor_type = "NVFP4";
        r.nvfp4.preset = "baseline";
        r.nvfp4.correction_denom = "2688";
        r.nvfp4.input_scale_policy = "imatrix-rms";
        apply_deep_quality_defaults(r);
        r.nv4mx6.policy = "nv4_promote_mx6";
        r.nv4mx6.mx6_penalty = "3.5";
        r.selector.ranking.kld_penalty = "12.0";
        r.selector.ranking.p99_penalty = "7.0";
        r.selector.ranking.p999_penalty = "2.0";
        r.selector.ranking.max_kld_penalty = "0.04";
        r.rescue.enabled = true;
        r.rescue.type = "MXFP6_E2M3";
        r.rescue.top = "-1";
        r.rescue.budget_mb = "3000";
        r.rescue.class_limit = "0";
        r.rescue.encoder_top = "16";
        apply_quantizer_mode(r);
        r.stock_ftype.mostly_type = "MOSTLY_NVFP4";
        r.stock_ftype.token_embedding_candidates = { "MXFP6_E2M3" };
        r.stock_ftype.output_tensor_candidates = { "MXFP6_E2M3" };
        r.stock_ftype.rationale = "Local mixed NVFP4/MXFP6 recipe starts compact, keeps MTP matrix weights NVFP4 with the active NVFP4 RSF policy, and promotes measured high-risk tensors to MXFP6_E2M3.";
    } else if (profile == "mxfp6-primary") {
        r.target.precision_mode = "NVFP4_MXFP6";
        r.base.ftype = "MXFP6";
        r.base.output_tensor_type = "MXFP6_E2M3";
        r.base.token_embedding_type = "MXFP6_E2M3";
        r.base.mtp_tensor_type = "MXFP6_E2M3";
        r.nvfp4.preset = "baseline";
        r.nvfp4.correction_denom = "2688";
        r.nvfp4.input_scale_policy = "imatrix-rms";
        apply_deep_quality_defaults(r);
        r.nv4mx6.policy = "mx6_demote_nv4";
        r.nv4mx6.mx6_penalty = "1.75";
        r.rescue.enabled = true;
        r.rescue.type = "MXFP6_E2M3";
        r.rescue.top = "0";
        r.rescue.report_top = "12";
        r.rescue.budget_mb = "0";
        r.rescue.class_limit = "0";
        apply_quantizer_mode(r);
        r.stock_ftype.mostly_type = "MOSTLY_MXFP6_E2M3";
        r.stock_ftype.token_embedding_candidates = { "MXFP6_E2M3" };
        r.stock_ftype.output_tensor_candidates = { "MXFP6_E2M3" };
        r.stock_ftype.rationale = "Local mixed quality recipe starts MXFP6_E2M3-first and demotes measured safe tensors to NVFP4 for speed/size.";
    } else if (profile == "nvfp4") {
        r.target.precision_mode = "NVFP4";
        r.base.ftype = "NVFP4";
        r.base.output_tensor_type = "Q6_K";
        r.base.token_embedding_type = "NVFP4";
        r.base.mtp_tensor_type = "NVFP4";
        r.quantizer.enabled = true;
        r.quantizer.mode = "normal";
        r.quantizer.objective = "kld-first";
        r.quantizer.evidence = "real-ppl-kld";
        r.quantizer.require_kld = true;
        r.quantizer.require_corpus = true;
        r.quantizer.require_imatrix = true;
        r.quantizer.allow_diagnostic = false;
        apply_quantizer_mode(r);
        r.rescue.type.clear();
        clear_mixed_controls(r);
        clear_mxfp6_controls(r);
        r.stock_ftype.mostly_type = "MOSTLY_NVFP4";
        r.stock_ftype.token_embedding_candidates = { "NVFP4" };
        r.stock_ftype.output_tensor_candidates = { "Q6_K" };
        r.stock_ftype.rationale = "Default RSF selector search: compact KLD-guided real-artifact budget with NVFP4 MTP matrix weights.";
    } else if (profile == "q8_0") {
        r.target.precision_mode = "Q8_0";
        r.base.ftype = "Q8_0";
        r.quantizer.enabled = false;
        r.quantizer.require_kld = false;
        r.quantizer.require_corpus = false;
        r.quantizer.require_imatrix = false;
        r.base.output_tensor_type = "Q8_0";
        r.base.token_embedding_type = "Q8_0";
        clear_nvfp4_controls(r);
        clear_mixed_controls(r);
        r.selector = {};
        r.selector.effort.clear();
        r.rescue.enabled = false;
        r.rescue.type.clear();
        clear_mxfp6_controls(r);
        r.stock_ftype.mostly_type = "MOSTLY_Q8_0";
        r.stock_ftype.token_embedding_candidates = { "Q8_0" };
        r.stock_ftype.output_tensor_candidates = { "Q8_0" };
        r.stock_ftype.sweep_tensor_policy = false;
        r.stock_ftype.sweep_sensitive_tensors = false;
        r.stock_ftype.rationale = "Plain GGUF Q8_0 recipe for users who want a high-compatibility baseline.";
    } else {
        throw std::runtime_error("unknown profile: " + profile);
    }
    return r;
}

Recipe default_recipe_for_quant_type(const std::string & precision_mode) {
    const std::string quant_type = canonical_quant_type(precision_mode);
    for (const QuantTypeProfile & choice : QUANT_TYPE_PROFILES) {
        if (quant_type == choice.type) {
            return default_recipe(choice.profile);
        }
    }
    throw std::runtime_error("unsupported quant type: " + precision_mode);
}

Recipe default_recipe() {
    return default_recipe_for_quant_type(Recipe().target.precision_mode);
}

static std::string quantizer_work_mode_arg(const std::string & mode) {
    const std::string normalized = lower_copy(trim(mode));
    if (normalized == "fast") {
        return "fast";
    }
    if (normalized == "normal") {
        return "normal";
    }
    if (normalized == "deep") {
        return "deep";
    }
    return std::string();
}

std::vector<std::string> build_quantize_args(const Recipe & r, bool force_dry_run) {
    (void) force_dry_run;
    std::vector<std::string> args;
    args.push_back("llama-quantize");
    const std::string quant_type = canonical_quant_type(
        !r.target.precision_mode.empty() ? r.target.precision_mode : r.base.ftype);
    const std::string mixed_policy = canonical_mixed_policy(r.nv4mx6.policy);
    const bool mixed_mx6_primary = quant_type == "NVFP4_MXFP6" && mixed_policy == "mx6_demote_nv4";
    const std::string ftype = r.base.copy_only ? "COPY" :
        (quant_type == "NVFP4_MXFP6" ? (mixed_mx6_primary ? "MXFP6" : "NVFP4_MXFP6") :
            (quant_type == "NVFP4" || quant_type == "MXFP6" || quant_type == "MXFP8" ? quant_type : r.base.ftype));
    const bool uses_nvfp4 = quant_type == "NVFP4" || quant_type == "NVFP4_MXFP6" || ftype == "NVFP4";
    const bool uses_mxfp6 = quant_type == "MXFP6" || quant_type == "NVFP4_MXFP6" || ftype == "MXFP6";
    const bool use_mode_defaults = r.quantizer.enabled && !r.quantizer.allow_diagnostic;
    auto push_pair = [&args](const char * flag, const std::string & value) {
        if (!value.empty()) {
            args.push_back(flag);
            args.push_back(value);
        }
    };
    auto push_bool = [&args](const char * flag, bool value) {
        if (value) {
            args.push_back(flag);
        }
    };
    push_bool("--allow-requantize", r.base.allow_requantize);
    push_bool("--leave-output-tensor", r.base.leave_output_tensor);
    push_bool("--pure", r.base.pure);
    push_bool("--keep-split", r.io.keep_split);
    push_pair("--patch-base", r.io.patch_base);
    push_pair("--prune-layers", r.model.prune_layers);
    push_pair("--output-tensor-type", sanitize_tensor_type_token(r.base.output_tensor_type));
    push_pair("--token-embedding-type", sanitize_tensor_type_token(r.base.token_embedding_type));
    push_pair("--mtp-tensor-type", sanitize_tensor_type_token(r.base.mtp_tensor_type));
    for (const std::string & item : r.metadata.overrides) {
        push_pair("--override-kv", item);
    }
    push_pair("--imatrix", r.calibration.imatrix);
    for (const std::string & item : r.calibration.include_weights) {
        push_pair("--include-weights", item);
    }
    for (const std::string & item : r.calibration.exclude_weights) {
        push_pair("--exclude-weights", item);
    }
    for (const std::string & item : r.tensor_overrides.files) {
        push_pair("--tensor-type-file", item);
    }
    for (const std::string & item : r.tensor_overrides.entries) {
        push_pair("--tensor-type", item);
    }

    if (r.quantizer.enabled) {
        push_pair("--mode", quantizer_work_mode_arg(r.quantizer.mode));
    }

    if (uses_nvfp4) {
        const std::string four_six_cfg = nvfp4_four_six_cfg(r.nvfp4.four_six);
        if (four_six_cfg.empty()) {
            push_pair("--nvfp4-preset", r.nvfp4.preset);
            push_pair("--nvfp4-cfg", r.nvfp4.cfg);
        } else {
            push_pair("--nvfp4-cfg", four_six_cfg);
        }
        push_pair("--nvfp4-correction-denom", r.nvfp4.correction_denom);
        push_pair("--nvfp4-input-scale-policy", r.nvfp4.input_scale_policy);
        if (!use_mode_defaults) {
            push_pair("--nvfp4-encoder-max-blocks", r.nvfp4.encoder.max_blocks);
            push_pair("--nvfp4-encoder-threads", r.nvfp4.encoder.threads);
            push_pair("--selector-rsf-mode", r.nvfp4.rsf.mode);
            push_pair("--selector-rsf-depth", r.nvfp4.rsf.depth);
        }
    }

    const bool uses_mxfp6_controls =
        uses_mxfp6 ||
        (r.rescue.enabled && r.rescue.type == "MXFP6_E2M3") ||
        !r.mxfp6.selector_scale_top.empty() ||
        !r.mxfp6.selector_scale_candidates.empty();
    if (uses_mxfp6_controls) {
        push_pair("--mxfp6_e2m3-tensor-scale", r.mxfp6.tensor_scale);
        push_pair("--mxfp6_e2m3-min-savings", r.mxfp6.min_savings_bytes);
        push_pair("--mxfp6_e2m3-input-scale-denom", r.mxfp6.input_scale_denom);
        push_pair("--mxfp6_e2m3-input-scale-quantile", r.mxfp6.input_scale_quantile);
        push_pair("--mxfp6_e2m3-tensor-scale-sample-blocks", r.mxfp6.tensor_scale_sample_blocks);
        push_pair("--mxfp6_e2m3-tensor-scale-steps", r.mxfp6.tensor_scale_steps);
        push_pair("--mxfp6_e2m3-selector-scale-top", r.mxfp6.selector_scale_top);
        push_pair("--mxfp6_e2m3-selector-scale-candidates", r.mxfp6.selector_scale_candidates);
    }
    if (uses_nvfp4 && uses_mxfp6) {
        push_pair("--mixed-format-policy", mixed_policy);
        push_pair("--mixed-mx6-penalty", r.nv4mx6.mx6_penalty);
        push_pair("--mixed-bf16-mx6-threshold", r.nv4mx6.bf16_mx6_threshold);
        push_pair("--mixed-sample-blocks", r.nv4mx6.sample_blocks);
        push_pair("--mixed-sample-cap", r.nv4mx6.sample_cap);
        push_pair("--mixed-imatrix-weight-blend", r.nv4mx6.imatrix_weight_blend);
        push_pair("--mixed-imatrix-weight-power", r.nv4mx6.imatrix_weight_power);
        push_pair("--mixed-imatrix-weight-min", r.nv4mx6.imatrix_weight_min);
        push_pair("--mixed-imatrix-weight-max", r.nv4mx6.imatrix_weight_max);
    }

    push_pair("--selector-kld", r.selector.kld);
    push_pair("--selector-checkpoint-model", r.selector.checkpoint_model);
    push_pair("--selector-cache-dir", r.selector.cache_dir);
    push_pair("--selector-skip-file", r.selector.skip_file);
    push_pair("--selector-ledger", r.selector.ledger);
    push_bool("--selector-keep-checkpoint", r.selector.keep_checkpoint);
    push_bool("--selector-require-runtime-cache", r.selector.require_runtime_cache);
    if (!use_mode_defaults) {
        push_pair("--selector-stagea-sample-blocks", r.selector.stagea_sample_blocks);
        push_pair("--selector-stagea-max-policies", r.selector.stagea_max_policies);
        push_pair("--selector-refine-top", r.selector.refine_top);
        push_pair("--selector-refine-budget", r.selector.refine_budget);
        push_pair("--selector-survey-top", r.selector.survey_top);
        push_pair("--selector-survey-sample-blocks", r.selector.survey_sample_blocks);
        push_pair("--selector-max-tensors", r.selector.max_tensors);
    }
    push_bool("--selector-trace", r.selector.trace);
    if (r.quantizer.enabled && !use_mode_defaults) {
        push_pair("--selector-policy-threads", r.selector.policy_threads);
        push_pair("--selector-threads", r.selector.threads);
        push_pair("--selector-kld-threads", r.selector.kld_threads);
    }
    push_bool("--selector-only", r.selector.only);
    if (!use_mode_defaults) {
        push_pair("--selector-eval-top", r.selector.eval_top);
        push_pair("--selector-n-seq", r.selector.n_seq);
    }
    push_pair("--selector-sensitivity-report", r.selector.sensitivity_report);
    push_pair("--selector-sensitivity-top", r.selector.sensitivity_top);
    push_pair("--selector-sensitivity-layer", r.selector.sensitivity_layer);
    push_pair("--selector-sensitivity-tensor", r.selector.sensitivity_tensor);
    push_pair("--selector-sensitivity-sample-blocks", r.selector.sensitivity_sample_blocks);
    push_pair("--selector-rsf-report", r.selector.rsf_report);

    if (!use_mode_defaults) {
        push_pair("--selector-kld-penalty", r.selector.ranking.kld_penalty);
        push_pair("--selector-p99-penalty", r.selector.ranking.p99_penalty);
        push_pair("--selector-p999-penalty", r.selector.ranking.p999_penalty);
        push_pair("--selector-max-kld-penalty", r.selector.ranking.max_kld_penalty);
        push_pair("--selector-rank-kld-threshold", r.selector.ranking.kld_threshold);
        push_pair("--selector-rank-p99-threshold", r.selector.ranking.p99_threshold);
        push_pair("--selector-rank-p999-threshold", r.selector.ranking.p999_threshold);
        push_pair("--selector-rank-max-kld-threshold", r.selector.ranking.max_kld_threshold);
        push_bool("--selector-rank-kld-hard-gate", r.selector.ranking.kld_hard_gate);
        push_bool("--selector-rank-p99-hard-gate", r.selector.ranking.p99_hard_gate);
        push_bool("--selector-rank-p999-hard-gate", r.selector.ranking.p999_hard_gate);
        push_bool("--selector-rank-max-kld-hard-gate", r.selector.ranking.max_kld_hard_gate);
    }

    if (r.rescue.enabled) {
        args.push_back("--selector-auto-rescue");
        push_pair("--selector-candidate-types", r.rescue.type);
        push_pair("--selector-rescue-top", r.rescue.top);
        push_pair("--selector-rescue-report-top", r.rescue.report_top);
        push_pair("--selector-rescue-budget-mb", r.rescue.budget_mb);
        push_pair("--selector-rescue-bf16-budget-mb", r.rescue.bf16_budget_mb);
        push_pair("--selector-rescue-class-limit", r.rescue.class_limit);
        push_pair("--selector-rescue-encoder-top", r.rescue.encoder_top);
        push_pair("--selector-rescue-sample-blocks", r.rescue.sample_blocks);
        push_pair("--selector-rescue-coarse-max-blocks", r.rescue.coarse_max_blocks);
        push_pair("--selector-rescue-refine-max-blocks", r.rescue.refine_max_blocks);
        push_pair("--selector-rescue-guard-max-blocks", r.rescue.guard_max_blocks);
        push_pair("--selector-rescue-report", r.rescue.report);
        push_pair("--selector-rescue-tensor-types", r.rescue.tensor_types);
    }

    args.push_back(r.io.input);
    if (!r.io.output.empty()) {
        args.push_back(r.io.output);
    }
    args.push_back(ftype);
    if (r.base.threads > 0) {
        args.push_back(std::to_string(r.base.threads));
    }
    return args;
}

} // namespace bq
