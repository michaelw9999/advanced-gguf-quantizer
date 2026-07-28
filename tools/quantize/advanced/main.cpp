#include "best.h"
#include "build-info.h"
#include "candidate_metrics.h"
#include "inspect.h"
#include "project.h"
#include "recipe.h"
#include "quantize.h"
#include "run_plan.h"
#include "run_monitor.h"
#include "shell_workflow.h"
#include "terminal_ui.h"
#include "tui.h"
#include "utils.h"
#include "log.h"
#include "../../../src/llama-quant.h"
#include "quantize-selector-ledger.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

int llama_perplexity(int argc, char ** argv);

namespace {

using bq::KldBaseInfo;
using bq::PRODUCT_COMMAND;
using bq::ShellState;
using bq::canonical_quant_type;
using bq::default_config_path;
using bq::default_project_path;
using bq::display_path;
using bq::format_kld_info;
using bq::file_size_or_zero;
using bq::default_worker_threads;
using bq::json_escape;
using bq::mib_string;
using bq::now_iso_like;
using bq::parse_int_or_default;
using bq::read_text_file_trimmed;
using bq::read_kld_base_info;
using bq::RunStatusThread;
using bq::sanitize_tensor_type_token;
using bq::set_if_empty;
using bq::quant_type_uses_mxfp6;
using bq::quant_type_uses_mxfp8;
using bq::quant_type_uses_nvfp4;
using bq::shellish_args;
using bq::shell_clear;
using bq::shell_executable_path;
using bq::shell_home_menu_options;
using bq::shell_infer_run_phase;
using bq::shell_latest_progress_line;
using bq::shell_options_menu_options;
using bq::shell_pid_alive;
using bq::shell_preflight_errors;
using bq::shell_print_status;
using bq::shell_project_menu_options;
using bq::shell_recent_meaningful_log_lines;
using bq::shell_run_finished;
using bq::shell_settings_dir;
using bq::string_is_auto;
using bq::tui::consume_escape_sequence_after_esc;
using bq::tui::Key;
using bq::tui::menu_select;
using bq::tui::PromptCancelled;
using bq::tui::PromptReadResult;
using bq::tui::RawTerminal;
using bq::tui::read_input_char;
using bq::tui::read_key;
using bq::tui::read_prompt_line;
using bq::tui::stdin_is_tty;
using bq::tui::strip_control_input;
using bq::write_text_file;

static void usage() {
    std::cout <<
        "usage:\n"
        "  advanced-gguf-quantizer recipe init --profile <name> [--output recipe.toml]\n"
        "  advanced-gguf-quantizer recipe validate <recipe.toml> [--set path=value...]\n"
        "  advanced-gguf-quantizer candidates <recipe.toml> [--output-dir dir]\n"
        "  advanced-gguf-quantizer best <candidates.jsonl|candidates.csv> [--min field] [--max field]\n"
        "  advanced-gguf-quantizer inspect <model.gguf> [--json] [--tensors] [--keys]\n"
        "  advanced-gguf-quantizer kld-info <logits.kld>\n"
        "  advanced-gguf-quantizer kld-command <recipe.toml> [--set path=value...]\n"
        "  advanced-gguf-quantizer imatrix-command <recipe.toml> [--set path=value...]\n"
        "  advanced-gguf-quantizer evidence inspect <selector-ledger.jsonl> [--top N] [--json]\n"
        "  advanced-gguf-quantizer layer-policy\n"
        "  advanced-gguf-quantizer plan <recipe.toml> [--set path=value...]\n"
        "  advanced-gguf-quantizer what-if <sensitivity-report> [--tensor NAME|--layer N]\n"
        "  advanced-gguf-quantizer project init [--output project.bwqproj] [--name name] [--recipe recipe.toml]\n"
        "  advanced-gguf-quantizer project open <project.bwqproj>\n"
        "  advanced-gguf-quantizer project add-candidates <project.bwqproj> <manifest.jsonl>\n"
        "  advanced-gguf-quantizer project record-metrics <project.bwqproj> --variant ID --json '{...}'\n"
        "  advanced-gguf-quantizer project export-metrics <project.bwqproj> [--output metrics.jsonl]\n"
        "  advanced-gguf-quantizer size --params-b N [--precision-mode NVFP4|MXFP6|MXFP8|NVFP4_MXFP6] [--vram-gb N] [--ram-gb N]\n"
        "  advanced-gguf-quantizer run <recipe.toml> [--yes] [--project project] [--variant id] [--set path=value...]\n"
        "  advanced-gguf-quantizer shell\n"
        "  advanced-gguf-quantizer wizard [--output recipe.toml] [--run] [--yes]\n\n"
        "advanced creator:\n"
        "  llama-quantize is the retained model-writing engine used by run/plan.\n\n"
        "profiles:\n"
        "  nvfp4\n"
        "  mxfp6\n"
        "  mxfp8\n"
        "  nvfp4_mxfp6\n"
        "  mxfp6-primary\n"
        "  q8_0\n\n"
        "MXFP6_E2M3 is experimental and unsupported by NVIDIA/llama.cpp.\n"
        "Future official MXFP6 formats may differ, so GGUFs made here may not\n"
        "remain compatible. Feedback branch: https://github.com/michaelw9999/llama.cpp/tree/mxfp6-cuda\n";
}

static void write_inspect_event(std::ofstream & events, const char * event, const bq::InspectSummary & summary) {
    events << "{\"event\":\"" << event
           << "\",\"path\":\"" << json_escape(summary.path)
           << "\",\"file_bytes\":" << summary.file_bytes
           << ",\"tensor_bytes\":" << summary.tensor_bytes
           << ",\"tensors\":" << summary.tensors
           << ",\"nvfp4_tensors\":" << summary.nvfp4_tensors
           << ",\"mxfp6_e2m3_tensors\":" << summary.mxfp6_tensors
           << ",\"mxfp8_tensors\":" << summary.mxfp8_tensors
           << ",\"scale_tensors\":" << summary.scale_tensors
           << ",\"input_scale_tensors\":" << summary.input_scale_tensors
           << ",\"has_mtp\":" << (summary.has_mtp ? "true" : "false")
           << ",\"mtp_keys\":" << summary.mtp_keys
           << ",\"mtp_tensors\":" << summary.mtp_tensors
           << "}\n";
}

struct FileKey {
    std::string path;
    bool exists = false;
    uint64_t bytes = 0;
    std::string mode = "missing";
    std::string hash64;
};

static uint64_t fnv1a64_update(uint64_t hash, const char * data, size_t size) {
    constexpr uint64_t prime = 1099511628211ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= (uint8_t) data[i];
        hash *= prime;
    }
    return hash;
}

static std::string hex64(uint64_t value) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016" PRIx64, value);
    return buf;
}

static FileKey file_key(const std::string & path) {
    FileKey fp;
    fp.path = path;
    if (path.empty()) {
        return fp;
    }
    std::error_code ec;
    fp.exists = std::filesystem::exists(path, ec);
    if (!fp.exists) {
        return fp;
    }
    fp.bytes = (uint64_t) std::filesystem::file_size(path, ec);
    const uint64_t sample = 1024ULL * 1024ULL;
    const uint64_t full_limit = 64ULL * 1024ULL * 1024ULL;
    fp.mode = fp.bytes <= full_limit ? "full-fnv1a64" : "sampled-fnv1a64-1m-start-mid-end";

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fp.mode = "unreadable";
        return fp;
    }

    uint64_t hash = 1469598103934665603ULL;
    std::array<char, 1024 * 1024> buf{};
    if (fp.bytes <= full_limit) {
        while (in) {
            in.read(buf.data(), (std::streamsize) buf.size());
            const std::streamsize got = in.gcount();
            if (got > 0) {
                hash = fnv1a64_update(hash, buf.data(), (size_t) got);
            }
        }
    } else {
        std::vector<uint64_t> offsets = { 0, fp.bytes / 2, fp.bytes > sample ? fp.bytes - sample : 0 };
        std::sort(offsets.begin(), offsets.end());
        offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
        for (const uint64_t off : offsets) {
            in.clear();
            in.seekg((std::streamoff) off, std::ios::beg);
            const uint64_t want = std::min<uint64_t>(sample, fp.bytes - off);
            in.read(buf.data(), (std::streamsize) want);
            const std::streamsize got = in.gcount();
            if (got > 0) {
                hash = fnv1a64_update(hash, buf.data(), (size_t) got);
            }
        }
    }
    hash = fnv1a64_update(hash, reinterpret_cast<const char *>(&fp.bytes), sizeof(fp.bytes));
    fp.hash64 = hex64(hash);
    return fp;
}

static std::string text_hash64(const std::string & text) {
    uint64_t hash = fnv1a64_update(1469598103934665603ULL, text.data(), text.size());
    const uint64_t size = (uint64_t) text.size();
    hash = fnv1a64_update(hash, reinterpret_cast<const char *>(&size), sizeof(size));
    return hex64(hash);
}

static void append_json_file_key(std::ostream & out, const std::string & key, const FileKey & fp, const char * suffix) {
    out << "    \"" << key << "\": {"
        << "\"path\":\"" << json_escape(fp.path)
        << "\",\"exists\":" << (fp.exists ? "true" : "false")
        << ",\"bytes\":" << fp.bytes
        << ",\"mode\":\"" << json_escape(fp.mode)
        << "\",\"hash64\":\"" << json_escape(fp.hash64) << "\"}"
        << suffix << "\n";
}

struct RunKey {
    FileKey source;
    FileKey calibration_corpus;
    FileKey evaluation_corpus;
    FileKey imatrix;
    FileKey kld_base;
    FileKey recipe_lock;
    std::string recipe_hash64;
    std::string selector_flags_hash64;
    std::string command_hash64;
    std::string code_commit;
    std::string json;
};

static RunKey build_run_key(
        const bq::Recipe & recipe,
        const bq::QuantizeRunPlan & plan,
        const std::filesystem::path & run_dir) {
    RunKey fp;
    const std::string recipe_lock_text = bq::dump_recipe_toml(recipe);
    fp.source = file_key(plan.input);
    fp.calibration_corpus = file_key(recipe.calibration.corpus);
    fp.evaluation_corpus = file_key(recipe.evaluation.corpus);
    fp.imatrix = file_key(recipe.calibration.imatrix);
    fp.kld_base = file_key(!recipe.selector.kld.empty() ? recipe.selector.kld : recipe.evaluation.kld_base);
    fp.recipe_lock = file_key((run_dir / "recipe.lock.toml").string());
    fp.recipe_hash64 = text_hash64(recipe_lock_text);
    fp.selector_flags_hash64 = text_hash64(shellish_args(plan.argv));
    fp.command_hash64 = text_hash64(shellish_args(plan.argv));
    fp.code_commit = llama_commit() != nullptr ? llama_commit() : "unknown";

    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"advanced-gguf-quantizer-checkpoint-key-v1\",\n";
    out << "  \"code_commit\": \"" << json_escape(fp.code_commit) << "\",\n";
    out << "  \"recipe_hash64\": \"" << fp.recipe_hash64 << "\",\n";
    out << "  \"selector_flags_hash64\": \"" << fp.selector_flags_hash64 << "\",\n";
    out << "  \"command_hash64\": \"" << fp.command_hash64 << "\",\n";
    out << "  \"files\": {\n";
    append_json_file_key(out, "source_model", fp.source, ",");
    append_json_file_key(out, "calibration_corpus", fp.calibration_corpus, ",");
    append_json_file_key(out, "evaluation_corpus", fp.evaluation_corpus, ",");
    append_json_file_key(out, "imatrix", fp.imatrix, ",");
    append_json_file_key(out, "kld_base", fp.kld_base, ",");
    append_json_file_key(out, "recipe_lock", fp.recipe_lock, "");
    out << "  }\n";
    out << "}\n";
    fp.json = out.str();
    return fp;
}

static void write_or_check_run_key(const std::filesystem::path & run_dir, const RunKey & fp) {
    const std::filesystem::path path = run_dir / "checkpoint-key.json";
    if (std::filesystem::exists(path)) {
        std::ifstream in(path);
        std::stringstream old;
        old << in.rdbuf();
        const std::string old_json = old.str();
        if (old_json != fp.json) {
            auto has_line = [&](const std::string & s) {
                return old_json.find(s) != std::string::npos;
            };
            auto has_file = [&](const char * key, const FileKey & file) {
                std::ostringstream s;
                append_json_file_key(s, key, file, "");
                std::string needle = s.str();
                if (!needle.empty() && needle.back() == '\n') {
                    needle.pop_back();
                }
                return has_line(needle);
            };
            const bool same_material =
                has_line("\"recipe_hash64\": \"" + fp.recipe_hash64 + "\"") &&
                has_file("source_model", fp.source) &&
                has_file("calibration_corpus", fp.calibration_corpus) &&
                has_file("evaluation_corpus", fp.evaluation_corpus) &&
                has_file("imatrix", fp.imatrix) &&
                has_file("kld_base", fp.kld_base);
            if (!same_material) {
                write_text_file(run_dir / "checkpoint-key.mismatch.json", fp.json);
                throw std::runtime_error(
                    "run checkpoint key mismatch in " + run_dir.string() +
                    "; use a new artifacts.run_dir or remove stale checkpoint/cache files");
            }
            write_text_file(run_dir / "checkpoint-key.previous.json", old_json);
            fprintf(stderr,
                "%s: run checkpoint key changed in code/selector command only; reusing checkpoint because source, corpora, imatrix, KLD base, and recipe hash match\n",
                __func__);
        }
    }
    write_text_file(path, fp.json);
}

static std::string pipeline_stage_json(const bq::Recipe & recipe) {
    std::ostringstream out;
    const bool has_kld = !recipe.selector.kld.empty() || !recipe.evaluation.kld_base.empty();
    out << "["
        << "{\"name\":\"probe\",\"status\":\"enabled\",\"detail\":\"GGUF inspect and model-profile validation\"},"
        << "{\"name\":\"candidate_search\",\"status\":\"" << (recipe.quantizer.enabled ? "enabled" : "disabled")
        << "\",\"detail\":\"proxy policy survey and non-dominated shortlist\"},"
        << "{\"name\":\"measured_ppl_kld_eval\",\"status\":\"" << (has_kld ? "enabled" : "missing-input")
        << "\",\"detail\":\"saved-logit PPL/KLD measured ranking\"},"
        << "{\"name\":\"quality_repair\",\"status\":\"" << (recipe.rescue.enabled ? "enabled" : "disabled")
        << "\",\"detail\":\"same GGUF writer path plus runtime patch evaluator\"},"
        << "{\"name\":\"export\",\"status\":\"enabled\",\"detail\":\"write GGUF, inspect, smoke-ready artifacts\"}"
        << "]";
    return out.str();
}

static std::string profile_validation_summary(
        const bq::Recipe & recipe,
        const bq::InspectSummary * input,
        const bq::InspectSummary * output,
        int rc) {
    std::ostringstream out;
    const bool expect_mtp = input != nullptr && input->has_mtp;
    const bool mtp_ok = !expect_mtp || (output != nullptr && output->has_mtp);
    const bool expects_nvfp4 =
        quant_type_uses_nvfp4(recipe.target.precision_mode) ||
        quant_type_uses_nvfp4(recipe.base.ftype);
    const bool scale_ok = output == nullptr || !expects_nvfp4 || output->nvfp4_tensors == 0 ||
        (output->scale_tensors >= output->nvfp4_tensors && output->input_scale_tensors >= output->nvfp4_tensors);
    std::string mtp_policy = recipe.base.mtp_tensor_type;
    if (mtp_policy.empty()) {
        if (quant_type_uses_mxfp6(recipe.base.ftype) && !quant_type_uses_nvfp4(recipe.base.ftype)) {
            mtp_policy = "MXFP6_E2M3";
        } else if (quant_type_uses_mxfp8(recipe.base.ftype)) {
            mtp_policy = "MXFP8";
        } else if (expects_nvfp4) {
            mtp_policy = "NVFP4";
        } else {
            mtp_policy = "Q8_0";
        }
    }
    out << "model_profile={"
        << "tensor_matching:" << (input != nullptr ? "checked" : "missing")
        << ", mtp_preservation:" << (mtp_ok ? "pass" : "fail")
        << ", output_policy:" << (recipe.base.output_tensor_type.empty() ? "default" : recipe.base.output_tensor_type)
        << ", token_embedding_policy:" << (recipe.base.token_embedding_type.empty() ? "default" : recipe.base.token_embedding_type)
        << ", mtp_policy:" << mtp_policy
        << ", expected_scale_tensors:" << (scale_ok ? "pass" : "fail")
        << ", run_rc:" << rc
        << "}";
    return out.str();
}

static void write_assignment_jsonl(
        const bq::Recipe & recipe,
        const bq::QuantizeRunPlan & plan,
        const std::filesystem::path & run_dir,
        const bq::InspectSummary * output) {
    std::ofstream out(run_dir / "assignment.jsonl", std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to write " + (run_dir / "assignment.jsonl").string());
    }
    out << "{\"schema\":\"advanced-gguf-quantizer-assignment-v1\",\"event\":\"run_assignment_summary\""
        << ",\"input\":\"" << json_escape(plan.input) << "\""
        << ",\"output\":\"" << json_escape(plan.output) << "\""
        << ",\"precision_mode\":\"" << json_escape(recipe.target.precision_mode) << "\""
        << ",\"mixed_policy\":\"" << json_escape(recipe.nv4mx6.policy) << "\""
        << ",\"threads\":" << recipe.base.threads
        << ",\"output_available\":" << (output != nullptr ? "true" : "false");
    if (output != nullptr) {
        out << ",\"tensors\":" << output->tensors
            << ",\"nvfp4_tensors\":" << output->nvfp4_tensors
            << ",\"mxfp6_e2m3_tensors\":" << output->mxfp6_tensors
            << ",\"mxfp8_tensors\":" << output->mxfp8_tensors
            << ",\"scale_tensors\":" << output->scale_tensors
            << ",\"input_scale_tensors\":" << output->input_scale_tensors
            << ",\"file_bytes\":" << output->file_bytes;
    }
    out << "}\n";
    out << "{\"schema\":\"advanced-gguf-quantizer-assignment-v1\",\"event\":\"selector_evidence\""
        << ",\"selector_ledger\":\"" << json_escape(recipe.selector.ledger) << "\""
        << ",\"note\":\"exact PPL/KLD gates and final artifact evaluation are release evidence\"}\n";
    out << "{\"schema\":\"advanced-gguf-quantizer-assignment-v1\",\"event\":\"fused_decision_units\""
        << ",\"units\":[\"qkv\",\"gate_up\",\"expert_pairs\",\"mtp_heads\",\"lm_head_and_embeddings\"]"
        << ",\"policy\":\"record groups as coherent layer decisions before per-tensor overrides\"}\n";
    out << "{\"schema\":\"advanced-gguf-quantizer-assignment-v1\",\"event\":\"tail_quality_gates\""
        << ",\"p99_penalty\":\"" << json_escape(recipe.selector.ranking.p99_penalty) << "\""
        << ",\"p999_penalty\":\"" << json_escape(recipe.selector.ranking.p999_penalty) << "\""
        << ",\"p99_threshold\":\"" << json_escape(recipe.selector.ranking.p99_threshold) << "\""
        << ",\"p999_threshold\":\"" << json_escape(recipe.selector.ranking.p999_threshold) << "\""
        << ",\"p99_hard_gate\":" << (recipe.selector.ranking.p99_hard_gate ? "true" : "false")
        << ",\"p999_hard_gate\":" << (recipe.selector.ranking.p999_hard_gate ? "true" : "false")
        << "}\n";
}

struct FinalQualityEvidence {
    bool attempted = false;
    bool available = false;
    int rc = -1;
    std::string reason;
    std::string command;
    std::string log_path;
    std::string summary_block;
    std::vector<std::pair<std::string, std::string>> summary_values;
};

static std::string trim_copy_local(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

static std::string read_text_file_or_empty(const std::filesystem::path & path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

static std::string value_after_metric_label(const std::string & line, const std::string & label) {
    const size_t pos = line.find(label);
    if (pos == std::string::npos) {
        return {};
    }
    const size_t colon = line.find(':', pos + label.size());
    if (colon == std::string::npos) {
        return {};
    }
    return trim_copy_local(line.substr(colon + 1));
}

static void add_metric_value_if_found(
        std::vector<std::pair<std::string, std::string>> & values,
        const std::string & line,
        const std::string & label,
        const std::string & display) {
    const std::string value = value_after_metric_label(line, label);
    if (!value.empty()) {
        values.emplace_back(display, value);
    }
}

static void parse_final_quality_log(FinalQualityEvidence & evidence) {
    const std::string text = read_text_file_or_empty(evidence.log_path);
    if (text.empty()) {
        return;
    }

    std::istringstream in(text);
    std::string line;
    bool in_summary = false;
    std::ostringstream summary;
    while (std::getline(in, line)) {
        if (line.find("====== Perplexity statistics ======") != std::string::npos) {
            in_summary = true;
        }
        if (in_summary) {
            summary << line << '\n';
        }

        add_metric_value_if_found(evidence.summary_values, line, "Mean PPL(Q)", "Mean PPL(Q)");
        add_metric_value_if_found(evidence.summary_values, line, "Mean PPL(base)", "Mean PPL(base)");
        add_metric_value_if_found(evidence.summary_values, line, "Mean ln(PPL(Q)/PPL(base))", "Mean ln(PPL ratio)");
        add_metric_value_if_found(evidence.summary_values, line, "Mean PPL(Q)/PPL(base)", "Mean PPL ratio");
        add_metric_value_if_found(evidence.summary_values, line, "Mean PPL(Q)-PPL(base)", "Mean PPL delta");
        add_metric_value_if_found(evidence.summary_values, line, "Mean    KLD", "Mean KLD");
        add_metric_value_if_found(evidence.summary_values, line, "Maximum KLD", "Maximum KLD");
        add_metric_value_if_found(evidence.summary_values, line, "99.9%   KLD", "99.9% KLD");
        add_metric_value_if_found(evidence.summary_values, line, "99.0%   KLD", "99.0% KLD");
        add_metric_value_if_found(evidence.summary_values, line, "95.0%   KLD", "95.0% KLD");
        add_metric_value_if_found(evidence.summary_values, line, "90.0%   KLD", "90.0% KLD");
        add_metric_value_if_found(evidence.summary_values, line, "Median  KLD", "Median KLD");
        add_metric_value_if_found(evidence.summary_values, line, "Minimum KLD", "Minimum KLD");
        add_metric_value_if_found(evidence.summary_values, line, "RMS ", "RMS probability delta");
        add_metric_value_if_found(evidence.summary_values, line, "Same top p", "Same top probability");
        add_metric_value_if_found(evidence.summary_values, line, "Top flip weight", "Top flip weight");
        add_metric_value_if_found(evidence.summary_values, line, "Top prob RMSE", "Top probability RMSE");
        add_metric_value_if_found(evidence.summary_values, line, "Entropy RMSE", "Entropy RMSE");

        if (in_summary && line.find("Entropy RMSE") != std::string::npos) {
            in_summary = false;
        }
    }
    evidence.summary_block = summary.str();
    evidence.available = evidence.rc == 0 && !evidence.summary_block.empty();
}

#ifndef _WIN32
class ScopedOutputRedirect {
public:
    explicit ScopedOutputRedirect(const std::filesystem::path & path) {
        fflush(stdout);
        fflush(stderr);
        fd_ = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (fd_ < 0) {
            return;
        }
        stdout_fd_ = dup(STDOUT_FILENO);
        stderr_fd_ = dup(STDERR_FILENO);
        if (stdout_fd_ < 0 || stderr_fd_ < 0) {
            restore();
            return;
        }
        if (dup2(fd_, STDOUT_FILENO) < 0 || dup2(fd_, STDERR_FILENO) < 0) {
            restore();
            return;
        }
        active_ = true;
    }

    ScopedOutputRedirect(const ScopedOutputRedirect &) = delete;
    ScopedOutputRedirect & operator=(const ScopedOutputRedirect &) = delete;

    ~ScopedOutputRedirect() {
        restore();
    }

    bool active() const {
        return active_;
    }

    void restore() {
        if (active_) {
            fflush(stdout);
            fflush(stderr);
            if (stdout_fd_ >= 0) {
                dup2(stdout_fd_, STDOUT_FILENO);
            }
            if (stderr_fd_ >= 0) {
                dup2(stderr_fd_, STDERR_FILENO);
            }
        }
        if (stdout_fd_ >= 0) {
            close(stdout_fd_);
            stdout_fd_ = -1;
        }
        if (stderr_fd_ >= 0) {
            close(stderr_fd_);
            stderr_fd_ = -1;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
        active_ = false;
    }

private:
    int fd_ = -1;
    int stdout_fd_ = -1;
    int stderr_fd_ = -1;
    bool active_ = false;
};
#endif

static int call_llama_perplexity_logged(
        const std::vector<std::string> & args,
        const std::filesystem::path & log_path) {
    std::vector<std::string> owned_args = args;
    std::vector<char *> argv;
    argv.reserve(owned_args.size());
    for (std::string & arg : owned_args) {
        argv.push_back(arg.data());
    }

#ifndef _WIN32
    ScopedOutputRedirect redirect(log_path);
    if (!redirect.active()) {
        return 1;
    }
#else
    (void) log_path;
#endif
    const int rc = llama_perplexity((int) argv.size(), argv.data());
    common_log_flush(common_log_main());
    return rc;
}

static FinalQualityEvidence run_final_quality_eval(
        const bq::Recipe & recipe,
        const bq::QuantizeRunPlan & plan,
        const std::filesystem::path & run_dir) {
    FinalQualityEvidence evidence;
    const std::string kld = !recipe.evaluation.kld_base.empty() ? recipe.evaluation.kld_base : recipe.selector.kld;
    if (plan.output.empty()) {
        evidence.reason = "no output model";
        return evidence;
    }
    if (recipe.evaluation.corpus.empty()) {
        evidence.reason = "evaluation.corpus is not set";
        return evidence;
    }
    if (kld.empty()) {
        evidence.reason = "evaluation.kld_base/selector.kld is not set";
        return evidence;
    }

    const KldBaseInfo info = read_kld_base_info(kld);
    if (!info.valid) {
        evidence.reason = "KLD base is invalid: " + format_kld_info(info);
        return evidence;
    }
    if (!info.complete) {
        evidence.reason = "KLD base is incomplete: " + format_kld_info(info);
        return evidence;
    }

    const int threads = recipe.base.threads > 0 ? recipe.base.threads : default_worker_threads();
    const int ctx = info.n_ctx > 0 ? info.n_ctx : 512;
    const int batch = std::max(2048, ctx);
    const int ubatch = std::min(batch, std::max(512, ctx));
    evidence.log_path = (run_dir / "final-ppl-kld.log").string();
    std::vector<std::string> args = {
        recipe.evaluation.perplexity_bin.empty() ? "llama-perplexity" : recipe.evaluation.perplexity_bin,
        "-m", plan.output,
        "-f", recipe.evaluation.corpus,
        "-c", std::to_string(ctx),
        "-b", std::to_string(batch),
        "-ub", std::to_string(ubatch),
        "-np", std::to_string(std::max(1, batch / std::max(1, ctx))),
        "-t", std::to_string(threads),
        "-tb", std::to_string(threads),
        "-ngl", "9999",
        "--kl-divergence",
        "--kl-divergence-base", kld,
        "--log-colors", "off",
        "--no-log-prefix",
        "--no-log-timestamps",
    };
    evidence.attempted = true;
    evidence.command = shellish_args(args);
    evidence.rc = call_llama_perplexity_logged(args, evidence.log_path);
    parse_final_quality_log(evidence);
    if (!evidence.available && evidence.reason.empty()) {
        evidence.reason = evidence.rc == 0 ? "final evaluator did not produce a summary block" : "final evaluator failed";
    }
    return evidence;
}

static void write_run_manifest_and_report(
        const bq::Recipe & recipe,
        const bq::QuantizeRunPlan & plan,
        const std::filesystem::path & run_dir,
        const RunKey & key,
        const bq::InspectSummary * input,
        const bq::InspectSummary * output,
        const FinalQualityEvidence * final_quality,
        int rc) {
    write_assignment_jsonl(recipe, plan, run_dir, output);

    const std::string profile = profile_validation_summary(recipe, input, output, rc);
    std::ostringstream manifest;
    manifest << "{\n";
    manifest << "  \"schema\": \"advanced-gguf-quantizer-run-manifest-v1\",\n";
    manifest << "  \"return_code\": " << rc << ",\n";
    manifest << "  \"code_commit\": \"" << json_escape(key.code_commit) << "\",\n";
    manifest << "  \"pipeline\": " << pipeline_stage_json(recipe) << ",\n";
    manifest << "  \"profile_validation\": \"" << json_escape(profile) << "\",\n";
    manifest << "  \"artifacts\": {\n";
    manifest << "    \"recipe_lock\": \"" << json_escape((run_dir / "recipe.lock.toml").string()) << "\",\n";
    manifest << "    \"run_log\": \"" << json_escape((run_dir / "run.jsonl").string()) << "\",\n";
    manifest << "    \"assignment\": \"" << json_escape((run_dir / "assignment.jsonl").string()) << "\",\n";
    manifest << "    \"report\": \"" << json_escape((run_dir / "quantization-report.md").string()) << "\",\n";
    if (final_quality != nullptr && final_quality->attempted) {
        manifest << "    \"final_ppl_kld_log\": \"" << json_escape(final_quality->log_path) << "\",\n";
    }
    manifest << "    \"checkpoint_key\": \"" << json_escape((run_dir / "checkpoint-key.json").string()) << "\",\n";
    manifest << "    \"validation_smoke\": \"" << json_escape((run_dir / "validate-output-smoke.sh").string()) << "\"\n";
    manifest << "  },\n";
    manifest << "  \"input\": \"" << json_escape(plan.input) << "\",\n";
    manifest << "  \"output\": \"" << json_escape(plan.output) << "\"\n";
    manifest << "}\n";
    write_text_file(run_dir / "run-manifest.json", manifest.str());

    std::ostringstream report;
    report << "# Quantization Report\n\n";
    report << "- Command: `" << PRODUCT_COMMAND << " run`\n";
    report << "- Return code: `" << rc << "`\n";
    report << "- Commit: `" << key.code_commit << "`\n";
    report << "- Input: `" << plan.input << "`\n";
    report << "- Output: `" << plan.output << "`\n";
    report << "- Threads: `" << recipe.base.threads << "`\n\n";
    report << "## Staged Pipeline\n\n";
    report << "1. probe: GGUF inspect and model-profile validation.\n";
    report << "2. candidate_search: proxy policy survey and non-dominated shortlist.\n";
    report << "3. measured_ppl_kld_eval: saved-logit PPL/KLD ranking when `selector.kld` or `evaluation.kld_base` is present.\n";
    report << "4. quality_repair: repair/edit operations use the same `llama_model_quantize` GGUF writer path and resident runtime patch evaluator.\n";
    report << "5. export: write GGUF, inspect tensor mix, preserve MTP, and prepare smoke validation.\n\n";
    report << "## Model Profile Validation\n\n";
    report << "- " << profile << "\n";
    if (input != nullptr) {
        report << "- Source tensors: `" << input->tensors << "`, MTP: `" << (input->has_mtp ? "yes" : "no") << "`\n";
    }
    if (output != nullptr) {
        report << "- Output tensors: `" << output->tensors << "`, NVFP4: `" << output->nvfp4_tensors
               << "`, MXFP6_E2M3: `" << output->mxfp6_tensors << "`, MXFP8: `" << output->mxfp8_tensors
               << "`, scale tensors: `"
               << output->scale_tensors << "`, input scales: `" << output->input_scale_tensors << "`\n";
    }
    report << "\n## Final PPL/KLD\n\n";
    if (final_quality != nullptr && final_quality->attempted) {
        report << "- Command: `" << final_quality->command << "`\n";
        report << "- Return code: `" << final_quality->rc << "`\n";
        report << "- Raw output: `" << final_quality->log_path << "`\n";
        if (final_quality->available) {
            report << "\n| Metric | Value |\n";
            report << "| --- | --- |\n";
            for (const auto & item : final_quality->summary_values) {
                report << "| " << item.first << " | `" << item.second << "` |\n";
            }
            report << "\n```text\n" << final_quality->summary_block << "```\n\n";
        } else {
            report << "- Result: unavailable (" << final_quality->reason << ")\n\n";
        }
    } else {
        const std::string reason = final_quality != nullptr && !final_quality->reason.empty()
            ? final_quality->reason
            : "evaluation.corpus and evaluation.kld_base/selector.kld are required";
        report << "- Result: unavailable (" << reason << ")\n\n";
    }
    report << "\n## Tail Gates\n\n";
    report << "- p99 penalty: `" << recipe.selector.ranking.p99_penalty << "`, hard gate: `"
           << (recipe.selector.ranking.p99_hard_gate ? "on" : "off") << "`, threshold: `"
           << (recipe.selector.ranking.p99_threshold.empty() ? "baseline" : recipe.selector.ranking.p99_threshold) << "`\n";
    report << "- p999 penalty: `" << recipe.selector.ranking.p999_penalty << "`, hard gate: `"
           << (recipe.selector.ranking.p999_hard_gate ? "on" : "off") << "`, threshold: `"
           << (recipe.selector.ranking.p999_threshold.empty() ? "baseline" : recipe.selector.ranking.p999_threshold) << "`\n\n";
    report << "## Selector Evidence\n\n";
    report << "- Evidence ledger: `" << (recipe.selector.ledger.empty() ? "disabled" : recipe.selector.ledger) << "`\n";
    report << "- Exact PPL/KLD gates and final artifact evaluation are release evidence.\n\n";
    report << "## Keys\n\n";
    report << "- Source: `" << key.source.hash64 << "` (" << key.source.mode << ", " << key.source.bytes << " bytes)\n";
    report << "- Calibration corpus: `" << key.calibration_corpus.hash64 << "` (" << key.calibration_corpus.mode << ", " << key.calibration_corpus.bytes << " bytes)\n";
    report << "- Evaluation corpus: `" << key.evaluation_corpus.hash64 << "` (" << key.evaluation_corpus.mode << ", " << key.evaluation_corpus.bytes << " bytes)\n";
    report << "- Imatrix: `" << key.imatrix.hash64 << "` (" << key.imatrix.mode << ", " << key.imatrix.bytes << " bytes)\n";
    report << "- KLD base: `" << key.kld_base.hash64 << "` (" << key.kld_base.mode << ", " << key.kld_base.bytes << " bytes)\n";
    report << "- Recipe: `" << key.recipe_hash64 << "`\n";
    report << "- Selector flags: `" << key.selector_flags_hash64 << "`\n\n";
    report << "## Artifacts\n\n";
    report << "- `recipe.lock.toml`\n";
    report << "- `generated-quantize.args`\n";
    report << "- `run.jsonl`\n";
    report << "- `assignment.jsonl`\n";
    report << "- `run-manifest.json`\n";
    report << "- `checkpoint-key.json`\n";
    report << "- `validate-output-smoke.sh`\n";
    write_text_file(run_dir / "quantization-report.md", report.str());
}

static std::string prompt(const std::string & label, const std::string & fallback);

static size_t rendered_line_count(const std::string & text) {
    if (text.empty()) {
        return 0;
    }
    return (size_t) std::count(text.begin(), text.end(), '\n') + (text.back() == '\n' ? 0 : 1);
}
static void print_project_summary_clean(const std::string & path, std::ostream & out);

static void apply_selector_kld_evidence_defaults(bq::Recipe & recipe) {
    (void) recipe;
}

static void apply_search_effort_preset(bq::Recipe & recipe, const std::string & effort) {
    const bool fast = effort == "fast";
    const bool normal = effort == "normal";
    recipe.quantizer.enabled = true;
    recipe.quantizer.mode = fast ? "fast" : (normal ? "normal" : "deep");
    recipe.quantizer.objective = "kld-first";
    recipe.quantizer.evidence = "real-ppl-kld";
    recipe.quantizer.require_kld = !fast;
    recipe.quantizer.require_corpus = !fast;
    recipe.quantizer.require_imatrix = !fast;
    recipe.quantizer.allow_diagnostic = false;
    const std::string threads = std::to_string(default_worker_threads());
    set_if_empty(recipe.selector.kld, recipe.evaluation.kld_base);
    recipe.base.threads = recipe.base.threads > 0 ? recipe.base.threads : default_worker_threads();
    set_if_empty(recipe.nvfp4.encoder.threads, threads);
    set_if_empty(recipe.selector.kld_threads, threads);
    recipe.selector.keep_checkpoint = true;
    recipe.selector.require_runtime_cache = !fast;
    if (recipe.selector.cache_dir.empty() && !recipe.artifacts.run_dir.empty()) {
        recipe.selector.cache_dir = (std::filesystem::path(recipe.artifacts.run_dir) / "selector-cache").string();
    }
    recipe.stock_ftype.sweep_tensor_policy = true;
    recipe.stock_ftype.sweep_sensitive_tensors = true;
    if (fast || normal) {
        recipe.selector.ranking.kld_penalty = "4.0";
        recipe.selector.ranking.p99_penalty = "1.5";
        recipe.selector.ranking.p999_penalty = "0.75";
        recipe.selector.ranking.max_kld_penalty = "0.10";
        recipe.selector.ranking.kld_hard_gate = false;
        recipe.selector.ranking.p99_hard_gate = false;
        recipe.selector.ranking.p999_hard_gate = false;
        recipe.selector.ranking.max_kld_hard_gate = false;
    } else {
        recipe.selector.ranking.kld_penalty = "12.0";
        recipe.selector.ranking.p99_penalty = "7.0";
        recipe.selector.ranking.p999_penalty = "2.0";
        recipe.selector.ranking.max_kld_penalty = "0.04";
        recipe.selector.ranking.kld_hard_gate = false;
        recipe.selector.ranking.p99_hard_gate = true;
        recipe.selector.ranking.p999_hard_gate = true;
        recipe.selector.ranking.max_kld_hard_gate = false;
    }
    set_if_empty(recipe.mxfp6.min_savings_bytes, "2097152");
    bq::apply_quantizer_mode(recipe);
}

static void apply_execution_defaults(bq::Recipe & recipe) {
    const int threads = default_worker_threads();
    const std::string thread_text = std::to_string(threads);
    if (recipe.base.threads <= 0) {
        recipe.base.threads = threads;
    }
    set_if_empty(recipe.selector.kld, recipe.evaluation.kld_base);
    set_if_empty(recipe.nvfp4.encoder.threads, thread_text);
    set_if_empty(recipe.selector.kld_threads, thread_text);
    if (recipe.quantizer.enabled) {
        bq::apply_quantizer_mode(recipe);
    }
    apply_selector_kld_evidence_defaults(recipe);
}

static std::vector<std::string> recipe_file_preflight_errors(const bq::Recipe & recipe, bool require_io) {
    std::vector<std::string> errors;
    std::error_code ec;
    if (require_io && !recipe.io.input.empty() && !std::filesystem::exists(recipe.io.input, ec)) {
        errors.push_back("model input does not exist: " + recipe.io.input);
    }
    if (require_io && !recipe.io.output.empty()) {
        const std::filesystem::path out(recipe.io.output);
        const std::filesystem::path parent = out.has_parent_path() ? out.parent_path() : std::filesystem::path(".");
        if (!std::filesystem::exists(parent, ec)) {
            errors.push_back("model output directory does not exist: " + parent.string());
        }
    }
    if (!recipe.evaluation.bf16_reference.empty() && !std::filesystem::exists(recipe.evaluation.bf16_reference, ec)) {
        errors.push_back("BF16 reference model does not exist: " + recipe.evaluation.bf16_reference);
    }
    if (!recipe.evaluation.corpus.empty() && !std::filesystem::exists(recipe.evaluation.corpus, ec)) {
        errors.push_back("PPL/KLD corpus does not exist: " + recipe.evaluation.corpus);
    }
    if (!recipe.calibration.corpus.empty() && !std::filesystem::exists(recipe.calibration.corpus, ec)) {
        errors.push_back("calibration corpus does not exist: " + recipe.calibration.corpus);
    }
    if (!recipe.calibration.imatrix.empty() && !std::filesystem::exists(recipe.calibration.imatrix, ec)) {
        errors.push_back("imatrix file does not exist: " + recipe.calibration.imatrix);
    }

    const std::string kld = !recipe.selector.kld.empty() ? recipe.selector.kld : recipe.evaluation.kld_base;
    if (kld.empty()) {
        return errors;
    }
    if (!std::filesystem::exists(kld, ec)) {
        errors.push_back("KLD base is selected but does not exist yet: " + kld);
        return errors;
    }

    const KldBaseInfo info = read_kld_base_info(kld);
    if (!info.valid) {
        errors.push_back("KLD base is not a valid logits/KLD file: " + kld + " (" + format_kld_info(info) + ")");
        return errors;
    }
    return errors;
}

static std::filesystem::path run_dir_for(const bq::Recipe & recipe) {
    if (!recipe.artifacts.run_dir.empty()) {
        return recipe.artifacts.run_dir;
    }
    if (!recipe.io.output.empty()) {
        return recipe.io.output + ".run";
    }
    return std::string(PRODUCT_COMMAND) + "-" + now_iso_like();
}

static void write_run_files(
        const bq::Recipe & recipe,
        const std::vector<std::string> & args,
        const std::filesystem::path & run_dir) {
    std::filesystem::create_directories(run_dir);
    write_text_file(run_dir / "recipe.lock.toml", bq::dump_recipe_toml(recipe));
    write_text_file(run_dir / "generated-quantize.args", shellish_args(args) + "\n");

    std::ofstream events(run_dir / "run.jsonl", std::ios::app);
    if (!events) {
        throw std::runtime_error("failed to write " + (run_dir / "run.jsonl").string());
    }
    events << "{\"event\":\"planned\",\"argv\":\"" << json_escape(shellish_args(args)) << "\"}\n";
}

static std::string default_kld_base_path(const bq::Recipe & recipe) {
    if (!recipe.evaluation.kld_base.empty()) {
        return recipe.evaluation.kld_base;
    }
    const std::string base = !recipe.evaluation.bf16_reference.empty() ? recipe.evaluation.bf16_reference : recipe.io.input;
    if (base.empty()) {
        return {};
    }
    return std::filesystem::path(base).replace_extension(".kld").string();
}

static std::string default_imatrix_path(const bq::Recipe & recipe) {
    if (!recipe.calibration.imatrix.empty()) {
        return recipe.calibration.imatrix;
    }
    const std::string base = !recipe.evaluation.bf16_reference.empty() ? recipe.evaluation.bf16_reference : recipe.io.input;
    if (base.empty()) {
        return "blackwell-imatrix.gguf";
    }
    return std::filesystem::path(base).replace_extension(".imatrix.gguf").string();
}

static std::string kld_base_command_shell(const bq::Recipe & recipe) {
    const std::string model = !recipe.evaluation.bf16_reference.empty() ? recipe.evaluation.bf16_reference : recipe.io.input;
    if (model.empty() || recipe.evaluation.corpus.empty() || recipe.evaluation.kld_base.empty()) {
        return {};
    }

    std::vector<std::string> args;
    args.push_back(recipe.evaluation.perplexity_bin.empty() ? "llama-perplexity" : recipe.evaluation.perplexity_bin);
    args.push_back("-m");
    args.push_back(model);
    args.push_back("-f");
    args.push_back(recipe.evaluation.corpus);
    args.push_back("--save-all-logits");
    args.push_back(recipe.evaluation.kld_base);

    std::string command = shellish_args(args);
    return command;
}

static std::string imatrix_command_shell(const bq::Recipe & recipe) {
    const std::string model = !recipe.evaluation.bf16_reference.empty() ? recipe.evaluation.bf16_reference : recipe.io.input;
    if (model.empty() || recipe.calibration.corpus.empty() || recipe.calibration.imatrix.empty()) {
        return {};
    }

    const int fallback_threads = recipe.base.threads > 0 ? recipe.base.threads : default_worker_threads();
    const std::string threads = recipe.calibration.threads.empty() ? std::to_string(fallback_threads) : recipe.calibration.threads;
    const std::string threads_batch = recipe.calibration.threads_batch.empty() ? threads : recipe.calibration.threads_batch;
    const std::string ctx_size = recipe.calibration.ctx_size;
    const std::string batch_size = recipe.calibration.batch_size;
    const std::string ubatch_size = recipe.calibration.ubatch_size.empty() ? batch_size : recipe.calibration.ubatch_size;
    const std::string n_gpu_layers = recipe.calibration.n_gpu_layers;
    std::vector<std::string> args;
    args.push_back(recipe.calibration.imatrix_bin.empty() ? "llama-imatrix" : recipe.calibration.imatrix_bin);
    args.push_back("-m");
    args.push_back(model);
    args.push_back("-f");
    args.push_back(recipe.calibration.corpus);
    args.push_back("-o");
    args.push_back(recipe.calibration.imatrix);
    if (!threads.empty()) {
        args.push_back("-t");
        args.push_back(threads);
    }
    if (!threads_batch.empty()) {
        args.push_back("-tb");
        args.push_back(threads_batch);
    }
    if (!ctx_size.empty()) {
        args.push_back("-c");
        args.push_back(ctx_size);
    }
    if (!batch_size.empty()) {
        args.push_back("-b");
        args.push_back(batch_size);
    }
    if (!ubatch_size.empty()) {
        args.push_back("-ub");
        args.push_back(ubatch_size);
    }
    if (!n_gpu_layers.empty()) {
        args.push_back("-ngl");
        args.push_back(n_gpu_layers);
    }

    std::string command = shellish_args(args);
    if (!recipe.calibration.extra_args.empty()) {
        command += " ";
        command += recipe.calibration.extra_args;
    }
    return command;
}

static bq::ProjectQualityInputs project_quality_inputs_from_recipe(const bq::Recipe & recipe) {
    bq::ProjectQualityInputs quality;
    quality.bf16_reference = recipe.evaluation.bf16_reference;
    quality.kld_base = recipe.evaluation.kld_base.empty() ? recipe.selector.kld : recipe.evaluation.kld_base;
    quality.eval_corpus = recipe.evaluation.corpus;
    quality.calibration_corpus = recipe.calibration.corpus;
    quality.imatrix = recipe.calibration.imatrix;
    return quality;
}

static std::filesystem::path bundle_manifest_path(const std::string & bundle) {
    std::filesystem::path path(bundle);
    if (path.extension() == ".json") {
        return path;
    }
    return path / "manifest.json";
}

static void write_evaluation_artifacts(
        const bq::Recipe & recipe,
        const bq::QuantizeRunPlan & plan,
        const std::filesystem::path & run_dir) {
    if (recipe.evaluation.kld_mode == "make_base" || recipe.evaluation.kld_mode == "bundle") {
        const std::string command = kld_base_command_shell(recipe);
        if (!command.empty()) {
            write_text_file(run_dir / "make-kld-base.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n");
        }
    }

    {
        const std::string command = imatrix_command_shell(recipe);
        if (!command.empty()) {
            write_text_file(run_dir / "make-imatrix.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n");
        }
    }

    if (!plan.output.empty()) {
        std::vector<std::string> inspect_args = { "llama-quantize", "inspect", plan.output };
        std::vector<std::string> smoke_args = {
            "llama-completion", "-m", plan.output, "-p", "The model is", "-n", "32"
        };
        write_text_file(run_dir / "validate-output-smoke.sh",
                "#!/usr/bin/env bash\nset -euo pipefail\n" +
                shellish_args(inspect_args) + "\n" +
                shellish_args(smoke_args) + "\n");
    }

    if (recipe.evaluation.bundle.empty()) {
        return;
    }

    const std::filesystem::path manifest = bundle_manifest_path(recipe.evaluation.bundle);
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"advanced-gguf-quantizer-eval-bundle-v1\",\n";
    out << "  \"recipe_lock\": \"" << json_escape((run_dir / "recipe.lock.toml").string()) << "\",\n";
    out << "  \"generated_quantize_args\": \"" << json_escape((run_dir / "generated-quantize.args").string()) << "\",\n";
    out << "  \"run_log\": \"" << json_escape((run_dir / "run.jsonl").string()) << "\",\n";
    out << "  \"input\": \"" << json_escape(plan.input) << "\",\n";
    out << "  \"output\": \"" << json_escape(plan.output) << "\",\n";
    out << "  \"ftype\": \"" << json_escape(plan.ftype) << "\",\n";
    out << "  \"kld\": {\n";
    out << "    \"mode\": \"" << json_escape(recipe.evaluation.kld_mode) << "\",\n";
    out << "    \"bf16_reference\": \"" << json_escape(recipe.evaluation.bf16_reference) << "\",\n";
    out << "    \"corpus\": \"" << json_escape(recipe.evaluation.corpus) << "\",\n";
    out << "    \"kld_base\": \"" << json_escape(recipe.evaluation.kld_base) << "\",\n";
    out << "    \"make_base_command\": \"" << json_escape(kld_base_command_shell(recipe)) << "\"\n";
    out << "  },\n";
    out << "  \"calibration\": {\n";
    out << "    \"corpus\": \"" << json_escape(recipe.calibration.corpus) << "\",\n";
    out << "    \"imatrix\": \"" << json_escape(recipe.calibration.imatrix) << "\",\n";
    out << "    \"make_imatrix_command\": \"" << json_escape(imatrix_command_shell(recipe)) << "\"\n";
    out << "  }\n";
    out << "}\n";
    write_text_file(manifest, out.str());
}

static int run_recipe(
        const std::string & recipe_path,
        const std::vector<std::string> & sets,
        bool yes,
        bool force_dry_run,
        const std::string & project_path = {},
        const std::string & variant = {}) {
    bq::LoadedRecipe loaded = bq::load_recipe_file(recipe_path);
    for (const std::string & set : sets) {
        bq::apply_override(loaded, set);
    }
    if (force_dry_run) {
        loaded.recipe.base.dry_run = true;
    }
    if (force_dry_run || loaded.recipe.base.dry_run) {
        std::cerr << PRODUCT_COMMAND << " run writes real model artifacts only; use `" << PRODUCT_COMMAND
                  << " plan` to inspect commands without running.\n";
        return 2;
    }
    apply_execution_defaults(loaded.recipe);

    auto errors = bq::validate_recipe(loaded.recipe, true);
    const auto file_errors = recipe_file_preflight_errors(loaded.recipe, true);
    errors.insert(errors.end(), file_errors.begin(), file_errors.end());
    if (!errors.empty()) {
        for (const std::string & error : errors) {
            std::cerr << "recipe error: " << error << "\n";
        }
        return 2;
    }

    bq::QuantizeRunPlan plan = bq::make_quantize_run_plan(loaded.recipe, force_dry_run);
    const std::filesystem::path run_dir = run_dir_for(loaded.recipe);
    const std::string assignment_path = (run_dir / "assignment.jsonl").string();
    plan.argv.insert(plan.argv.begin() + std::min<size_t>(1, plan.argv.size()), {
        "--assignment-jsonl",
        assignment_path,
    });
    const bool writes_output = !plan.dry_run && !plan.output.empty() && !loaded.recipe.selector.only;
    write_run_files(loaded.recipe, plan.argv, run_dir);
    const RunKey run_key = build_run_key(loaded.recipe, plan, run_dir);
    write_or_check_run_key(run_dir, run_key);
    write_evaluation_artifacts(loaded.recipe, plan, run_dir);
    if (!project_path.empty()) {
        bq::project_append_run_event(
                project_path, "run_planned", variant.empty() ? std::filesystem::path(recipe_path).stem().string() : variant,
                recipe_path, plan.output, run_dir.string(), -1, file_size_or_zero(plan.output),
                project_quality_inputs_from_recipe(loaded.recipe));
    }

    bq::InspectSummary input_summary;
    bool have_input_summary = false;
    {
        std::ofstream events(run_dir / "run.jsonl", std::ios::app);
        try {
            input_summary = bq::inspect_gguf(plan.input);
            have_input_summary = true;
            if (events) {
                write_inspect_event(events, "input_inspected", input_summary);
            }
        } catch (const std::exception & e) {
            if (events) {
                events << "{\"event\":\"input_inspect_failed\",\"error\":\"" << json_escape(e.what()) << "\"}\n";
                events << "{\"event\":\"finished\",\"return_code\":3}\n";
            }
            write_run_manifest_and_report(loaded.recipe, plan, run_dir, run_key, nullptr, nullptr, nullptr, 3);
            std::cerr << "input inspect failed: " << e.what() << "\n";
            return 3;
        }
    }

    std::cout << "run dir: " << run_dir.string() << "\n";
    std::cout << "internal quantize call:\n  " << shellish_args(plan.argv) << "\n";
    std::cout << std::flush;
    if (!yes) {
        std::cout << "continue? [y/N] " << std::flush;
        std::string answer;
        std::getline(std::cin, answer);
        if (answer != "y" && answer != "Y" && answer != "yes" && answer != "YES") {
            std::cout << "aborted\n";
            if (!project_path.empty()) {
                bq::project_append_run_event(
                        project_path, "run_aborted", variant.empty() ? std::filesystem::path(recipe_path).stem().string() : variant,
                        recipe_path, plan.output, run_dir.string(), 0, file_size_or_zero(plan.output),
                        project_quality_inputs_from_recipe(loaded.recipe));
            }
            write_run_manifest_and_report(loaded.recipe, plan, run_dir, run_key,
                    have_input_summary ? &input_summary : nullptr, nullptr, nullptr, 0);
            return 0;
        }
    }

    std::vector<std::string> owned_args = plan.argv;
    std::vector<char *> argv;
    argv.reserve(owned_args.size());
    for (std::string & arg : owned_args) {
        argv.push_back(arg.data());
    }
    RunStatusThread status(run_dir, plan.output, writes_output);
    int rc = llama_quantize((int) argv.size(), argv.data());
    status.stop();

    bq::InspectSummary output_summary;
    bool have_output_summary = false;
    std::ofstream events(run_dir / "run.jsonl", std::ios::app);
    if (events) {
        if (rc == 0 && writes_output) {
            try {
                output_summary = bq::inspect_gguf(plan.output);
                have_output_summary = true;
                write_inspect_event(events, "output_inspected", output_summary);
                if (have_input_summary && input_summary.has_mtp) {
                    if (output_summary.has_mtp) {
                        events << "{\"event\":\"mtp_preserved\",\"input_mtp_tensors\":" << input_summary.mtp_tensors
                               << ",\"output_mtp_tensors\":" << output_summary.mtp_tensors
                               << ",\"input_mtp_keys\":" << input_summary.mtp_keys
                               << ",\"output_mtp_keys\":" << output_summary.mtp_keys
                               << "}\n";
                    } else {
                        events << "{\"event\":\"mtp_preservation_failed\",\"input_mtp_tensors\":" << input_summary.mtp_tensors
                               << ",\"input_mtp_keys\":" << input_summary.mtp_keys
                               << "}\n";
                        rc = 4;
                    }
                }
            } catch (const std::exception & e) {
                events << "{\"event\":\"output_inspect_failed\",\"error\":\"" << json_escape(e.what()) << "\"}\n";
            }
        }
    }
    FinalQualityEvidence final_quality;
    if (rc == 0 && writes_output && have_output_summary) {
        std::cout << "final PPL/KLD evaluation:\n";
        final_quality = run_final_quality_eval(loaded.recipe, plan, run_dir);
        if (final_quality.attempted) {
            std::cout << "  " << final_quality.log_path << " (rc=" << final_quality.rc << ")\n";
        } else {
            std::cout << "  skipped: " << final_quality.reason << "\n";
        }
        if (events) {
            events << "{\"event\":\"final_ppl_kld\""
                   << ",\"attempted\":" << (final_quality.attempted ? "true" : "false")
                   << ",\"available\":" << (final_quality.available ? "true" : "false")
                   << ",\"return_code\":" << final_quality.rc
                   << ",\"log\":\"" << json_escape(final_quality.log_path) << "\""
                   << ",\"reason\":\"" << json_escape(final_quality.reason) << "\""
                   << "}\n";
        }
    } else {
        final_quality.reason = "quantization did not complete with an inspected output model";
    }
    if (events) {
        events << "{\"event\":\"finished\",\"return_code\":" << rc << "}\n";
    }

    if (!project_path.empty()) {
        bq::project_append_run_event(
                project_path, "run_finished", variant.empty() ? std::filesystem::path(recipe_path).stem().string() : variant,
                recipe_path, plan.output, run_dir.string(), rc, file_size_or_zero(plan.output),
        project_quality_inputs_from_recipe(loaded.recipe));
    }
    write_run_manifest_and_report(loaded.recipe, plan, run_dir, run_key,
            have_input_summary ? &input_summary : nullptr,
            have_output_summary ? &output_summary : nullptr,
            &final_quality,
            rc);
    return rc;
}

static int recipe_init(int argc, char ** argv) {
    std::string profile;
    std::string output;
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile" && i + 1 < argc) {
            profile = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else {
            throw std::runtime_error("unknown recipe init argument: " + arg);
        }
    }

    const std::string text = profile.empty() ?
        bq::dump_recipe_toml(bq::default_recipe()) :
        bq::default_recipe_toml(profile);
    if (output.empty()) {
        std::cout << text;
    } else {
        write_text_file(output, text);
        std::cout << "wrote " << output << "\n";
    }
    return 0;
}

static int recipe_validate(int argc, char ** argv) {
    if (argc < 1) {
        throw std::runtime_error("recipe validate requires a recipe path");
    }
    std::vector<std::string> sets;
    std::string path = argv[0];
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--set" && i + 1 < argc) {
            sets.push_back(argv[++i]);
        } else {
            throw std::runtime_error("unknown recipe validate argument: " + arg);
        }
    }
    bq::LoadedRecipe loaded = bq::load_recipe_file(path);
    for (const std::string & set : sets) {
        bq::apply_override(loaded, set);
    }
    apply_execution_defaults(loaded.recipe);
    auto errors = bq::validate_recipe(loaded.recipe, true);
    const auto file_errors = recipe_file_preflight_errors(loaded.recipe, true);
    errors.insert(errors.end(), file_errors.begin(), file_errors.end());
    if (!errors.empty()) {
        for (const std::string & error : errors) {
            std::cerr << "recipe error: " << error << "\n";
        }
        return 2;
    }
    std::cout << "recipe ok: " << path << "\n";
    return 0;
}

static int kld_command_main(int argc, char ** argv) {
    if (argc < 1) {
        throw std::runtime_error("kld-command requires a recipe path");
    }
    std::vector<std::string> sets;
    const std::string path = argv[0];
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--set" && i + 1 < argc) {
            sets.push_back(argv[++i]);
        } else {
            throw std::runtime_error("unknown kld-command argument: " + arg);
        }
    }

    bq::LoadedRecipe loaded = bq::load_recipe_file(path);
    for (const std::string & set : sets) {
        bq::apply_override(loaded, set);
    }

    const std::string command = kld_base_command_shell(loaded.recipe);
    if (command.empty()) {
        throw std::runtime_error("recipe needs evaluation.bf16_reference, evaluation.corpus, and evaluation.kld_base");
    }
    std::cout << command << "\n";
    return 0;
}

static int imatrix_command_main(int argc, char ** argv) {
    if (argc < 1) {
        throw std::runtime_error("imatrix-command requires a recipe path");
    }
    std::vector<std::string> sets;
    const std::string path = argv[0];
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--set" && i + 1 < argc) {
            sets.push_back(argv[++i]);
        } else {
            throw std::runtime_error("unknown imatrix-command argument: " + arg);
        }
    }

    bq::LoadedRecipe loaded = bq::load_recipe_file(path);
    for (const std::string & set : sets) {
        bq::apply_override(loaded, set);
    }

    const std::string command = imatrix_command_shell(loaded.recipe);
    if (command.empty()) {
        throw std::runtime_error("recipe needs io.input or evaluation.bf16_reference, calibration.corpus, and calibration.imatrix");
    }
    std::cout << command << "\n";
    return 0;
}

static int kld_info_main(int argc, char ** argv) {
    if (argc < 1) {
        throw std::runtime_error("kld-info requires a KLD/logits file path");
    }
    const KldBaseInfo info = read_kld_base_info(argv[0]);
    if (!info.valid) {
        std::cerr << "invalid KLD base: " << argv[0] << " (" << format_kld_info(info) << ")\n";
        return 2;
    }
    std::cout << "path: " << argv[0] << "\n";
    std::cout << "ctx: " << info.n_ctx << "\n";
    std::cout << "vocab: " << info.n_vocab << "\n";
    std::cout << "chunks: " << info.n_chunks << "\n";
    std::cout << "available_chunks: " << info.available_chunks << "\n";
    std::cout << "size_mib: " << mib_string(info.file_bytes) << "\n";
    std::cout << "expected_mib: " << mib_string(info.expected_bytes) << "\n";
    std::cout << "complete: " << (info.complete ? "yes" : "no") << "\n";
    if (!info.complete) {
        return 3;
    }
    return 0;
}

static int plan_recipe(int argc, char ** argv) {
    if (argc < 1) {
        throw std::runtime_error("plan requires a recipe path");
    }
    std::vector<std::string> sets;
    const std::string path = argv[0];
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--set" && i + 1 < argc) {
            sets.push_back(argv[++i]);
        } else {
            throw std::runtime_error("unknown plan argument: " + arg);
        }
    }

    bq::LoadedRecipe loaded = bq::load_recipe_file(path);
    for (const std::string & set : sets) {
        bq::apply_override(loaded, set);
    }
    apply_execution_defaults(loaded.recipe);
    auto errors = bq::validate_recipe(loaded.recipe, true);
    const auto file_errors = recipe_file_preflight_errors(loaded.recipe, true);
    errors.insert(errors.end(), file_errors.begin(), file_errors.end());
    if (!errors.empty()) {
        for (const std::string & error : errors) {
            std::cerr << "recipe error: " << error << "\n";
        }
        return 2;
    }

    const bq::QuantizeRunPlan plan = bq::make_quantize_run_plan(loaded.recipe, false);
    std::cout << "input: " << plan.input << "\n";
    std::cout << "output: " << plan.output << "\n";
    std::cout << "ftype: " << plan.ftype << "\n";
    if (!loaded.recipe.target.sizing_note.empty()) {
        std::cout << "sizing: " << loaded.recipe.target.sizing_note << "\n";
    }
    if (loaded.recipe.target.target_bpw > 0.0) {
        std::cout << "target_bpw: " << loaded.recipe.target.target_bpw << "\n";
    }
    if (!loaded.recipe.nvfp4.calibration_families.empty()) {
        std::cout << "native_families: ";
        for (size_t i = 0; i < loaded.recipe.nvfp4.calibration_families.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << loaded.recipe.nvfp4.calibration_families[i];
        }
        std::cout << "\n";
    }
    if (!loaded.recipe.stock_ftype.technique_candidates.empty()) {
        std::cout << "native_techniques: ";
        for (size_t i = 0; i < loaded.recipe.stock_ftype.technique_candidates.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << loaded.recipe.stock_ftype.technique_candidates[i];
        }
        std::cout << "\n";
    }
    if (!loaded.recipe.nvfp4.scale_tie.empty()) {
        std::cout << "native_scale_tie: " << loaded.recipe.nvfp4.scale_tie << "\n";
    }
    const std::string selected_kld = !loaded.recipe.selector.kld.empty()
        ? loaded.recipe.selector.kld
        : loaded.recipe.evaluation.kld_base;
    if (!selected_kld.empty()) {
        const KldBaseInfo info = read_kld_base_info(selected_kld);
        std::cout << "KLD base: " << selected_kld << " (" << format_kld_info(info) << ")\n";
        if (info.valid) {
            std::cout << "selector_evidence: full KLD base (" << info.available_chunks << " chunks)\n";
        }
    }
    std::cout << "selector_evidence: ledger="
              << (loaded.recipe.selector.ledger.empty() ? "disabled" : loaded.recipe.selector.ledger) << "\n";
    std::cout << "internal quantize call:\n  " << shellish_args(plan.argv) << "\n";
    const std::string kld_command = kld_base_command_shell(loaded.recipe);
    if (!kld_command.empty()) {
        std::cout << "KLD base command:\n  " << kld_command << "\n";
    }
    const std::string imatrix_command = imatrix_command_shell(loaded.recipe);
    if (!imatrix_command.empty()) {
        std::cout << "imatrix command:\n  " << imatrix_command << "\n";
    }
    return 0;
}

static std::string output_for_candidate(const bq::Recipe & recipe, const std::string & id) {
    std::filesystem::path base = !recipe.io.output.empty() ? recipe.io.output : recipe.io.input;
    if (base.empty()) {
        return id + ".gguf";
    }
    const std::string stem = base.stem().string();
    const std::filesystem::path parent = base.has_parent_path() ? base.parent_path() : std::filesystem::path();
    return (parent / (stem + "." + id + ".gguf")).string();
}

static std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

static void append_unique_type(std::vector<std::string> & out, const std::string & raw) {
    const std::string value = trim_copy(raw);
    if (value.empty()) {
        return;
    }
    if (std::find(out.begin(), out.end(), value) == out.end()) {
        out.push_back(value);
    }
}

static std::vector<std::string> dedupe_types(const std::vector<std::string> & values) {
    std::vector<std::string> out;
    for (const std::string & value : values) {
        append_unique_type(out, value);
    }
    return out;
}

static std::string lower_token(std::string value) {
    value = trim_copy(value);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return c == '-' ? '_' : (char) std::tolower(c);
    });
    return value;
}

static bool recipe_has_token(const std::vector<std::string> & values, const std::string & token) {
    const std::string needle = lower_token(token);
    return std::any_of(values.begin(), values.end(), [&](const std::string & value) {
        return lower_token(value) == needle;
    });
}

static void set_type_tokens(std::vector<std::string> & values, const std::vector<std::string> & tokens, bool enabled) {
    for (const std::string & token : tokens) {
        if (enabled) {
            append_unique_type(values, token);
            continue;
        }
        const std::string needle = lower_token(token);
        values.erase(std::remove_if(values.begin(), values.end(), [&](const std::string & value) {
            return lower_token(value) == needle;
        }), values.end());
    }
}

static std::vector<std::string> split_type_csv(const std::string & csv) {
    std::vector<std::string> out;
    std::string current;
    for (const char c : csv) {
        if (c == ',' || c == ';') {
            append_unique_type(out, current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    append_unique_type(out, current);
    return out;
}

static std::string join_type_csv(const std::vector<std::string> & values) {
    std::ostringstream out;
    bool first = true;
    for (const std::string & value : values) {
        const std::string trimmed = trim_copy(value);
        if (trimmed.empty()) {
            continue;
        }
        if (!first) {
            out << ", ";
        }
        out << trimmed;
        first = false;
    }
    return out.str();
}

static std::string candidate_id_token(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    bool last_dash = false;
    for (const unsigned char c : value) {
        if (std::isalnum(c)) {
            out.push_back((char) std::tolower(c));
            last_dash = false;
        } else if (c == '_') {
            out.push_back('_');
            last_dash = false;
        } else if (!last_dash) {
            out.push_back('-');
            last_dash = true;
        }
    }
    while (!out.empty() && out.front() == '-') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out.empty() ? "custom" : out;
}

static std::vector<std::string> nvfp4_selector_preset_candidates() {
    std::vector<std::string> out;
    for (const llama_nvfp4_named_preset & preset : llama_nvfp4_preset_catalog()) {
        if (std::string(preset.name) != "baseline") {
            out.emplace_back(preset.name);
        }
    }
    return out;
}

static void write_candidate_recipe(
        const bq::Recipe & base,
        const std::filesystem::path & output_dir,
        std::ofstream & manifest,
        const std::string & id,
        const std::string & description,
        const std::function<void(bq::Recipe &)> & edit) {
    bq::Recipe recipe = base;
    edit(recipe);
    recipe.io.output = output_for_candidate(base, id);
    recipe.artifacts.run_dir = recipe.io.output + ".run";
    const std::filesystem::path path = output_dir / (id + ".toml");
    write_text_file(path, bq::dump_recipe_toml(recipe));
    manifest << "{\"id\":\"" << json_escape(id)
             << "\",\"recipe\":\"" << json_escape(path.string())
             << "\",\"output\":\"" << json_escape(recipe.io.output)
             << "\",\"description\":\"" << json_escape(description)
             << "\"}\n";
}

static int candidates_main(int argc, char ** argv) {
    if (argc < 1) {
        throw std::runtime_error("candidates requires a recipe path");
    }
    const std::string path = argv[0];
    std::filesystem::path output_dir = "advanced-gguf-candidates";
    std::vector<std::string> sets;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--set" && i + 1 < argc) {
            sets.push_back(argv[++i]);
        } else {
            throw std::runtime_error("unknown candidates argument: " + arg);
        }
    }

    bq::LoadedRecipe loaded = bq::load_recipe_file(path);
    for (const std::string & set : sets) {
        bq::apply_override(loaded, set);
    }
    std::filesystem::create_directories(output_dir);
    std::ofstream manifest(output_dir / "manifest.jsonl");
    if (!manifest) {
        throw std::runtime_error("failed to write " + (output_dir / "manifest.jsonl").string());
    }

    bq::Recipe base = loaded.recipe;
    apply_execution_defaults(base);
    const auto has_native_candidate = [&](const char * token) {
        return recipe_has_token(base.stock_ftype.technique_candidates, token);
    };
    const auto has_native_family = [&](const char * token) {
        return recipe_has_token(base.nvfp4.calibration_families, token) || has_native_candidate(token);
    };
    const bool family_awq_lite = has_native_family("awq_lite");
    const bool family_awq_full = has_native_family("awq_full");
    const bool family_awq_clip = has_native_family("awq_clip");
    const bool family_awq = family_awq_lite || family_awq_full || family_awq_clip;
    const bool family_smoothquant = has_native_family("smoothquant");
    const bool family_mse = has_native_family("mse_scale_sweep");
    const bool family_rsf = has_native_family("nvfp4_rsf");
    const bool family_kl = has_native_family("kl_div_sensitivity");
    const bool family_gradient_sidecar = has_native_family("gradient_or_hessian_sidecar");
    const bool native_auto_search = has_native_candidate("auto_search");
    const bool native_no_quantize_choice = has_native_candidate("no_quantize_choice");

    if (native_no_quantize_choice) {
        append_unique_type(base.stock_ftype.token_embedding_candidates, "BF16");
        append_unique_type(base.stock_ftype.output_tensor_candidates, "BF16");
        base.stock_ftype.sweep_tensor_policy = true;
        base.stock_ftype.sweep_sensitive_tensors = true;
    }

    if (native_auto_search) {
        write_candidate_recipe(base, output_dir, manifest,
                "auto-search-w4a4",
                "Native Blackwell W4A4 auto-search seed measured by real PPL/KLD selector scoring",
                [family_kl, family_gradient_sidecar](bq::Recipe & r) {
                    r.target.precision_mode = "NVFP4";
                    r.base.ftype = "NVFP4";
                    if (r.nvfp4.preset.empty()) {
                        r.nvfp4.preset = "baseline";
                    }
                    if (family_kl) {
                        r.selector.effort = "kl-div-sensitivity";
                        r.selector.require_runtime_cache = true;
                    } else if (family_gradient_sidecar) {
                        r.selector.effort = "gradient-or-hessian-sidecar";
                    } else if (r.selector.effort.empty()) {
                        r.selector.effort = "kld-best";
                    }
                });

        for (const std::string & preset : nvfp4_selector_preset_candidates()) {
            write_candidate_recipe(base, output_dir, manifest,
                    "preset-" + candidate_id_token(preset),
                    "Native Blackwell preset " + preset + " measured by the same real PPL/KLD selector path",
                    [preset](bq::Recipe & r) {
                        r.target.precision_mode = "NVFP4";
                        r.base.ftype = "NVFP4";
                        r.nvfp4.preset = preset;
                        r.nvfp4.cfg.clear();
                        r.nvfp4.four_six = {};
                        if (r.selector.effort.empty()) {
                            r.selector.effort = "kld-best";
                        }
                    });
        }
    }

    if (family_awq) {
        write_candidate_recipe(base, output_dir, manifest,
                "awq-asym-tail",
                "AWQ-style local proxy: asym-tail NVFP4 candidate, then measured by real PPL/KLD",
                [](bq::Recipe & r) {
                    r.target.precision_mode = "NVFP4";
                    r.base.ftype = "NVFP4";
                    r.nvfp4.preset = "asym_tail";
                    r.nvfp4.four_six = {};
                    r.selector.effort = "awq-local";
                    r.selector.ranking.p99_penalty = r.selector.ranking.p99_penalty.empty() ? "1.5" : r.selector.ranking.p99_penalty;
                    r.selector.ranking.p999_penalty = r.selector.ranking.p999_penalty.empty() ? "2.0" : r.selector.ranking.p999_penalty;
                });
    }

    if (family_awq_clip) {
        write_candidate_recipe(base, output_dir, manifest,
                "awq-clip-tail",
                "Activation-aware clipped-tail NVFP4 candidate, then measured by real PPL/KLD",
                [](bq::Recipe & r) {
                    r.target.precision_mode = "NVFP4";
                    r.base.ftype = "NVFP4";
                    r.nvfp4.preset = "asym_tail";
                    r.nvfp4.four_six.choose46 = "adaptive";
                    r.nvfp4.four_six.refit_iters = "8";
                    r.nvfp4.four_six.compand = "1";
                    r.nvfp4.four_six.cap6 = "384";
                    r.nvfp4.four_six.cap4 = "224";
                    r.selector.effort = "awq-clip-local";
                });
    }

    if (family_awq_full) {
        write_candidate_recipe(base, output_dir, manifest,
                "awq-full-tail",
                "Heavier AWQ-style NVFP4 candidate with deeper refit/search, then real PPL/KLD ranking",
                [](bq::Recipe & r) {
                    r.target.precision_mode = "NVFP4";
                    r.base.ftype = "NVFP4";
                    r.nvfp4.preset = "asym_tail";
                    r.nvfp4.four_six.choose46 = "adaptive";
                    r.nvfp4.four_six.refit_iters = "16";
                    r.nvfp4.four_six.compand = "1";
                    r.nvfp4.four_six.cap6 = "448";
                    r.nvfp4.four_six.cap4 = "256";
                    r.nvfp4.encoder.max_blocks = r.nvfp4.encoder.max_blocks.empty() ? "32768" : r.nvfp4.encoder.max_blocks;
                    r.selector.effort = "awq-full-local";
                });
    }

    if (family_smoothquant) {
        write_candidate_recipe(base, output_dir, manifest,
                "smoothquant-input-scale",
                "Activation/input-scale balanced NVFP4 candidate, then real PPL/KLD ranking",
                [](bq::Recipe & r) {
                    r.target.precision_mode = "NVFP4";
                    r.base.ftype = "NVFP4";
                    r.nvfp4.input_scale_policy = r.nvfp4.input_scale_policy.empty() ? "imatrix-rms" : r.nvfp4.input_scale_policy;
                    r.nvfp4.correction_denom = r.nvfp4.correction_denom.empty() ? "2688" : r.nvfp4.correction_denom;
                    r.selector.effort = "smoothquant-local";
                });
    }

    if (family_mse) {
        write_candidate_recipe(base, output_dir, manifest,
                "mse-scale-sweep",
                "CUDA NVFP4 scale/cap sweep candidate, then real PPL/KLD ranking",
                [](bq::Recipe & r) {
                    r.target.precision_mode = "NVFP4";
                    r.base.ftype = "NVFP4";
                    r.nvfp4.preset.clear();
                    r.nvfp4.cfg.clear();
                    r.nvfp4.four_six.choose46 = "adaptive";
                    r.nvfp4.four_six.refit_iters = "16";
                    r.nvfp4.four_six.compand = "1";
                    r.nvfp4.four_six.cap6 = "448";
                    r.nvfp4.four_six.cap4 = "224";
                    r.nvfp4.encoder.max_blocks = r.nvfp4.encoder.max_blocks.empty() ? "32768" : r.nvfp4.encoder.max_blocks;
                    r.selector.effort = "mse-scale-sweep";
                });
    }

    if (family_rsf) {
        write_candidate_recipe(base, output_dir, manifest,
                "nvfp4-rsf",
                "NVFP4 refined scale fit (RSF) variants for the normal selector policies",
                [](bq::Recipe & r) {
                    r.target.precision_mode = "NVFP4";
                    r.base.ftype = "NVFP4";
                    append_unique_type(r.stock_ftype.technique_candidates, "nvfp4_rsf");
                    append_unique_type(r.nvfp4.calibration_families, "nvfp4_rsf");
                    r.selector.effort = "nvfp4-rsf";
                    r.selector.stagea_sample_blocks = r.selector.stagea_sample_blocks.empty() ? "4096" : r.selector.stagea_sample_blocks;
                    r.selector.eval_top = r.selector.eval_top.empty() ? "6" : r.selector.eval_top;
                    r.selector.require_runtime_cache = true;
                });
    }

    if (family_kl) {
        write_candidate_recipe(base, output_dir, manifest,
                "kl-div-sensitivity",
                "Saved-logit KL-sensitivity search using resident tensor patching",
                [](bq::Recipe & r) {
                    r.target.precision_mode = "NVFP4";
                    r.base.ftype = "NVFP4";
                    r.selector.effort = "kl-div-sensitivity";
                    r.selector.require_runtime_cache = true;
                    if (r.selector.eval_top.empty()) {
                        r.selector.eval_top = "4";
                    }
                });
    }

    const bool allow_mxfp6_candidates =
        quant_type_uses_mxfp6(base.target.precision_mode) ||
        quant_type_uses_mxfp6(base.base.ftype) ||
        (base.rescue.enabled && base.rescue.type == "MXFP6_E2M3");
    if (allow_mxfp6_candidates) {
        write_candidate_recipe(base, output_dir, manifest, "mixed-quality", "Mixed NVFP4/MXFP6 quality-first candidate with lower MXFP6 penalty and larger edit budget", [](bq::Recipe & r) {
            r.target.precision_mode = "NVFP4_MXFP6";
            r.base.ftype = "NVFP4_MXFP6";
            r.nvfp4.preset.clear();
            r.nvfp4.cfg.clear();
            r.nvfp4.four_six.choose46 = "adaptive";
            r.nvfp4.four_six.refit_iters = "16";
            r.nvfp4.four_six.compand = "1";
            r.nvfp4.four_six.cap6 = "448";
            r.nvfp4.four_six.cap4 = "224";
            r.rescue.enabled = true;
            const int base_budget = r.rescue.budget_mb.empty() ? 3000 : std::stoi(r.rescue.budget_mb);
            r.rescue.budget_mb = std::to_string(std::min(6144, std::max(2048, base_budget + 1024)));
        });
        write_candidate_recipe(base, output_dir, manifest, "mxfp6-scale", "Pure MXFP6 with KLD/PPL tensor-scale refinement enabled", [](bq::Recipe & r) {
            r.target.precision_mode = "MXFP6";
            r.base.ftype = "MXFP6";
            r.base.output_tensor_type.clear();
            r.base.token_embedding_type.clear();
            r.nvfp4.preset.clear();
            r.nvfp4.cfg.clear();
            r.nvfp4.correction_denom.clear();
            r.nvfp4.input_scale_policy.clear();
            r.nvfp4.four_six = {};
            r.nv4mx6.policy.clear();
            r.nv4mx6.mx6_penalty.clear();
            r.nv4mx6.bf16_mx6_threshold.clear();
            r.rescue.enabled = false;
            r.mxfp6.selector_scale_top = "96";
            r.mxfp6.selector_scale_candidates = "0.771105,0.840896,0.917004,0.957603,1,1.04427,1.09051,1.18921,1.29684";
        });
    }

    if (base.stock_ftype.sweep_tensor_policy || base.stock_ftype.sweep_sensitive_tensors) {
        std::vector<std::string> token_types = dedupe_types(base.stock_ftype.token_embedding_candidates);
        std::vector<std::string> output_types = dedupe_types(base.stock_ftype.output_tensor_candidates);
        const bool pure_nvfp4 =
            base.target.precision_mode == "NVFP4" ||
            base.base.ftype == "NVFP4";
        if (token_types.empty()) {
            token_types = pure_nvfp4 ? std::vector<std::string>{ "NVFP4" }
                                     : std::vector<std::string>{ "Q8_0", "Q6_K", "BF16" };
        }
        if (output_types.empty()) {
            output_types = pure_nvfp4 ? std::vector<std::string>{ "Q6_K" }
                                      : std::vector<std::string>{ "Q8_0", "Q6_K", "BF16" };
        }

        for (const std::string & type : token_types) {
            write_candidate_recipe(
                    base, output_dir, manifest,
                    "tokemb-" + candidate_id_token(type),
                    "Token embeddings as " + type + "; evaluate PPL/KLD before making this tensor policy default",
                    [type](bq::Recipe & r) {
                        r.base.token_embedding_type = type;
                    });
        }

        for (const std::string & type : output_types) {
            write_candidate_recipe(
                    base, output_dir, manifest,
                    "output-" + candidate_id_token(type),
                    "Output tensor as " + type + "; evaluate PPL/KLD before making this tensor policy default",
                    [type](bq::Recipe & r) {
                        r.base.output_tensor_type = type;
                        r.base.leave_output_tensor = false;
                    });
        }

        std::vector<std::string> paired_types;
        for (const std::string & type : token_types) {
            if (std::find(output_types.begin(), output_types.end(), type) != output_types.end()) {
                append_unique_type(paired_types, type);
            }
        }
        for (const std::string & type : paired_types) {
            write_candidate_recipe(
                    base, output_dir, manifest,
                    "tensor-policy-" + candidate_id_token(type),
                    "Token embeddings and output tensor both as " + type + "; measured tensor-policy candidate",
                    [type](bq::Recipe & r) {
                        r.base.token_embedding_type = type;
                        r.base.output_tensor_type = type;
                        r.base.leave_output_tensor = false;
                    });
        }
    }

    std::cout << "wrote candidate recipes to " << output_dir.string() << "\n";
    std::cout << "manifest: " << (output_dir / "manifest.jsonl").string() << "\n";
    return 0;
}

static int project_main(int argc, char ** argv) {
    if (argc < 1) {
        throw std::runtime_error("project requires a subcommand");
    }
    const std::string sub = argv[0];

    if (sub == "init") {
        std::string output = std::string(PRODUCT_COMMAND) + ".bwqproj.jsonl";
        bq::ProjectInit init;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) {
                output = argv[++i];
            } else if (arg == "--name" && i + 1 < argc) {
                init.name = argv[++i];
            } else if (arg == "--recipe" && i + 1 < argc) {
                init.recipe = argv[++i];
            } else if (arg == "--input" && i + 1 < argc) {
                init.input = argv[++i];
            } else if (arg == "--bf16-reference" && i + 1 < argc) {
                init.bf16_reference = argv[++i];
            } else if (arg == "--kld-base" && i + 1 < argc) {
                init.kld_base = argv[++i];
            } else if (arg == "--corpus" && i + 1 < argc) {
                init.corpus = argv[++i];
            } else if (arg == "--calibration-corpus" && i + 1 < argc) {
                init.calibration_corpus = argv[++i];
            } else if (arg == "--imatrix" && i + 1 < argc) {
                init.imatrix = argv[++i];
            } else {
                throw std::runtime_error("unknown project init argument: " + arg);
            }
        }
        if (init.name.empty()) {
            init.name = std::filesystem::path(output).stem().string();
        }
        bq::project_init_file(output, init);
        std::cout << "wrote project " << output << "\n";
        return 0;
    }

    if (sub == "open" || sub == "list") {
        if (argc < 2) {
            throw std::runtime_error("project open requires a project path");
        }
        print_project_summary_clean(argv[1], std::cout);
        return 0;
    }

    if (sub == "add-candidates") {
        if (argc < 3) {
            throw std::runtime_error("project add-candidates requires <project> <manifest.jsonl>");
        }
        bq::project_append_candidate_manifest(argv[1], argv[2]);
        std::cout << "added candidates from " << argv[2] << " to " << argv[1] << "\n";
        return 0;
    }

    if (sub == "record-metrics") {
        if (argc < 2) {
            throw std::runtime_error("project record-metrics requires a project path");
        }
        const std::string project = argv[1];
        std::string variant;
        std::string json = "{}";
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--variant" && i + 1 < argc) {
                variant = argv[++i];
            } else if (arg == "--json" && i + 1 < argc) {
                json = argv[++i];
            } else {
                throw std::runtime_error("unknown project record-metrics argument: " + arg);
            }
        }
        if (variant.empty()) {
            throw std::runtime_error("project record-metrics requires --variant ID");
        }
        bq::project_append_metric_event(project, variant, json);
        std::cout << "recorded metrics for " << variant << " in " << project << "\n";
        return 0;
    }

    if (sub == "export-metrics") {
        if (argc < 2) {
            throw std::runtime_error("project export-metrics requires a project path");
        }
        const std::string project = argv[1];
        std::string output;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) {
                output = argv[++i];
            } else {
                throw std::runtime_error("unknown project export-metrics argument: " + arg);
            }
        }

        if (output.empty()) {
            const uint64_t count = bq::project_export_metrics(project, std::cout);
            std::cerr << "exported " << count << " metric records\n";
        } else {
            std::ofstream out(output);
            if (!out) {
                throw std::runtime_error("failed to write " + output);
            }
            const uint64_t count = bq::project_export_metrics(project, out);
            std::cout << "exported " << count << " metric records to " << output << "\n";
        }
        return 0;
    }

    throw std::runtime_error("unknown project subcommand: " + sub);
}

static int layer_policy_main() {
    std::cout <<
        "Stock MOSTLY_x layer policy comes from src/llama-quant.cpp:llama_tensor_get_type_impl.\n"
        "\n"
        "Key rules to keep visible in Blackwell recipes:\n"
        "- Output tensors use --output-tensor-type when set; otherwise low-bit stock recipes often bump output to Q6_K/Q8_0 or the local ftype when safe.\n"
        "- Tied token embeddings follow output policy; untied token embeddings use --token-embedding-type when set and are preserved for local NVFP4 by default.\n"
        "- attn_v is often higher bitrate for GQA, 70B, and 8-expert MoE models because size impact is small relative to quality risk.\n"
        "- attn_k gets a Q8_0 bump for 8-expert MoE stock policy.\n"
        "- ffn_down often receives early/late layer promotions under the stock use_more_bits pattern; Q4_K_M/Q5_K_M can promote selected layers to Q6_K.\n"
        "- attention output and fused QKV get conservative bumps for low-bit K-quants and MoE cases.\n"
        "- Local NVFP4 Nemotron handling keeps dense SSM/control paths BF16 or Q8_0 and spends NVFP4 on large expert tensors.\n"
        "\n"
        "Blackwell project policy treats these as priors, not facts. Embeddings, output, and other small/high-impact tensors should be swept\n"
        "as configurable candidates (for example BF16, Q8_0, Q6_K, and MXFP6_E2M3 when that type is in play) and selected by measured\n"
        "mean/tail PPL+KLD deltas. A single bad-token max KLD can be acceptable when holdout mean/tail quality improves, so hard gates\n"
        "should be explicit rather than assumed.\n"
        "\n"
        "Fused decision units used by the release policy:\n"
        "- q/k/v attention siblings should move as one unit unless a measured exact tensor override explains the split.\n"
        "- gate/up FFN siblings should stay coherent; ffn_down is its own quality/speed pressure point.\n"
        "- MoE expert gate/up and paired expert tensors should be grouped per expert and per layer before per-tensor repair.\n"
        "- MTP/NextN heads are preserved by default and should have an explicit policy when quantized differently.\n"
        "- token embeddings and lm_head/output are one public policy decision even when the tensors are untied.\n"
        "\n"
        "Project runs write assignment.jsonl, quantization-report.md, run-manifest.json, and checkpoint-key.json so these policy choices\n"
        "are justified by measured deltas and can be resumed safely.\n";
    return 0;
}

static std::string record_string_field(const bq::CandidateRecord & record, const std::string & field) {
    const auto it = record.fields.find(field);
    return it == record.fields.end() ? std::string() : it->second;
}

static double record_numeric_field(const bq::CandidateRecord & record, const std::string & field, double fallback = 0.0) {
    const auto it = record.numeric.find(field);
    return it == record.numeric.end() ? fallback : it->second;
}

static bool record_is_event(const bq::CandidateRecord & record, const std::string & event) {
    return record_string_field(record, "event") == event;
}

static bool record_bool_field(const bq::CandidateRecord & record, const std::string & field, bool fallback = false) {
    const std::string value = lower_token(record_string_field(record, field));
    if (value == "true" || value == "1" || value == "yes") {
        return true;
    }
    if (value == "false" || value == "0" || value == "no" || value == "null") {
        return false;
    }
    return fallback;
}

struct EvidenceTensorMetric {
    std::string tensor;
    std::string policy;
    std::string phase;
    std::string cls;
    int layer = -1;
    double proxy_score = 0.0;
    double rmse = 0.0;
    double abs_mean = 0.0;
    double max_abs = 0.0;
    int64_t count = 0;
};

static int evidence_main(int argc, char ** argv) {
    if (argc < 2) {
        throw std::runtime_error("evidence requires: inspect <selector-ledger.jsonl>");
    }
    const std::string sub = argv[0];
    if (sub != "inspect") {
        throw std::runtime_error("unknown evidence subcommand: " + sub);
    }

    const std::string path = argv[1];
    int top = 20;
    bool json = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--top" && i + 1 < argc) {
            top = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--json") {
            json = true;
        } else {
            throw std::runtime_error("unknown evidence argument: " + arg);
        }
    }

    const bq::CandidateTable table = bq::load_candidate_table(path, bq::CandidateInputFormat::Jsonl);

    std::map<std::string, size_t> row_counts;
    std::vector<EvidenceTensorMetric> tensors;
    const size_t rows = table.records.size();
    size_t ignored = 0;
    size_t exact_eval = 0;
    double best_exact_score = std::numeric_limits<double>::infinity();
    std::string best_exact_policy;
    for (const bq::CandidateRecord & record : table.records) {
        if (record_string_field(record, "schema") != selector_ledger::schema) {
            ++ignored;
            continue;
        }
        std::string row_type = record_string_field(record, "row_type");
        if (row_type.empty()) {
            row_type = "unknown";
        }
        ++row_counts[row_type];
        if (row_type == "tensor.local_metric" && record_bool_field(record, "proxy_ok", false)) {
            EvidenceTensorMetric m;
            m.tensor = record_string_field(record, "tensor");
            m.policy = record_string_field(record, "policy");
            m.phase = record_string_field(record, "phase");
            m.cls = record_string_field(record, "class");
            const double layer = record_numeric_field(record, "layer", -1.0);
            const double count = record_numeric_field(record, "count", 0.0);
            m.layer = std::isfinite(layer) ? (int) layer : -1;
            m.proxy_score = record_numeric_field(record, "proxy_score", 0.0);
            m.rmse = record_numeric_field(record, "rmse", 0.0);
            m.abs_mean = record_numeric_field(record, "abs_mean", 0.0);
            m.max_abs = record_numeric_field(record, "max_abs", 0.0);
            m.count = std::isfinite(count) && count > 0.0 ? (int64_t) count : 0;
            if (!m.tensor.empty()) {
                tensors.push_back(std::move(m));
            }
        } else if (row_type == "exact_eval") {
            ++exact_eval;
            const double score = record_numeric_field(record, "score", std::numeric_limits<double>::infinity());
            if (std::isfinite(score) && score < best_exact_score) {
                best_exact_score = score;
                best_exact_policy = record_string_field(record, "policy");
            }
        }
    }

    std::sort(tensors.begin(), tensors.end(), [](const auto & a, const auto & b) {
        if (a.proxy_score != b.proxy_score) return a.proxy_score > b.proxy_score;
        if (a.max_abs != b.max_abs) return a.max_abs > b.max_abs;
        return a.tensor < b.tensor;
    });
    if (top > 0 && tensors.size() > (size_t) top) {
        tensors.resize((size_t) top);
    }

    if (json) {
        std::cout << "{\n";
        std::cout << "  \"schema\": \"advanced-gguf-selector-evidence-inspect-v1\",\n";
        std::cout << "  \"path\": \"" << json_escape(path) << "\",\n";
        std::cout << "  \"rows\": " << rows << ",\n";
        std::cout << "  \"ignored\": " << ignored << ",\n";
        std::cout << "  \"row_counts\": {";
        bool first = true;
        for (const auto & kv : row_counts) {
            std::cout << (first ? "" : ", ") << "\"" << json_escape(kv.first) << "\": " << kv.second;
            first = false;
        }
        std::cout << "},\n";
        std::cout << "  \"exact_eval_rows\": " << exact_eval << ",\n";
        std::cout << "  \"best_exact_policy\": \"" << json_escape(best_exact_policy) << "\",\n";
        std::cout << "  \"best_exact_score\": ";
        if (std::isfinite(best_exact_score)) {
            std::cout << best_exact_score;
        } else {
            std::cout << "null";
        }
        std::cout << ",\n";
        std::cout << "  \"top_tensor_local_metrics\": [\n";
        for (size_t i = 0; i < tensors.size(); ++i) {
            const EvidenceTensorMetric & m = tensors[i];
            std::cout
                << "    {\"tensor\": \"" << json_escape(m.tensor)
                << "\", \"policy\": \"" << json_escape(m.policy)
                << "\", \"phase\": \"" << json_escape(m.phase)
                << "\", \"class\": \"" << json_escape(m.cls)
                << "\", \"layer\": " << m.layer
                << ", \"proxy_score\": " << m.proxy_score
                << ", \"rmse\": " << m.rmse
                << ", \"abs_mean\": " << m.abs_mean
                << ", \"max_abs\": " << m.max_abs
                << ", \"count\": " << m.count
                << "}" << (i + 1 == tensors.size() ? "" : ",") << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
        return 0;
    }

    std::cout << "selector evidence ledger: " << path << "\n";
    std::cout << "rows: " << rows << " ignored: " << ignored << "\n";
    for (const auto & kv : row_counts) {
        std::cout << "  " << kv.first << ": " << kv.second << "\n";
    }
    if (exact_eval > 0) {
        std::cout << "exact_eval: " << exact_eval;
        if (!best_exact_policy.empty()) {
            std::cout << " best=" << best_exact_policy << " score=" << best_exact_score;
        }
        std::cout << "\n";
    }
    if (!tensors.empty()) {
        std::cout << "top local tensor metrics (higher proxy score is worse):\n";
        for (size_t i = 0; i < tensors.size(); ++i) {
            const EvidenceTensorMetric & m = tensors[i];
            std::cout
                << "  " << (i + 1)
                << ". " << m.tensor
                << " policy=" << m.policy
                << " phase=" << m.phase
                << " cls=" << m.cls
                << " layer=" << m.layer
                << " proxy=" << m.proxy_score
                << " rmse=" << m.rmse
                << " abs=" << m.abs_mean
                << " max=" << m.max_abs
                << "\n";
        }
    }
    std::cout << "note: ledger rankings are search evidence; release claims still need exact PPL/KLD on the final artifact.\n";
    return 0;
}

static int what_if_main(int argc, char ** argv) {
    if (argc < 1) {
        throw std::runtime_error("what-if requires a selector sensitivity report");
    }
    const std::string path = argv[0];
    std::string tensor_filter;
    int layer_filter = std::numeric_limits<int>::min();
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tensor" && i + 1 < argc) {
            tensor_filter = argv[++i];
        } else if (arg == "--layer" && i + 1 < argc) {
            layer_filter = std::stoi(argv[++i]);
        } else if (arg == "--json") {
            json = true;
        } else {
            throw std::runtime_error("unknown what-if argument: " + arg);
        }
    }

    const bq::CandidateTable table = bq::load_candidate_table(path, bq::CandidateInputFormat::Jsonl);
    if (table.records.empty()) {
        throw std::runtime_error("sensitivity file is empty: " + path);
    }

    const bq::CandidateRecord * baseline = nullptr;
    std::vector<const bq::CandidateRecord *> deltas;
    std::vector<const bq::CandidateRecord *> exact_groups;
    for (const bq::CandidateRecord & record : table.records) {
        if (record_is_event(record, "baseline") && baseline == nullptr) {
            baseline = &record;
            continue;
        }
        if (record_is_event(record, "layer_delta")) {
            if (layer_filter != std::numeric_limits<int>::min() && tensor_filter.empty()) {
                const double layer = record_numeric_field(record, "layer", std::numeric_limits<double>::quiet_NaN());
                if (std::isfinite(layer) && (int) layer == layer_filter) {
                    exact_groups.push_back(&record);
                }
            }
            continue;
        }
        if (!record_is_event(record, "tensor_delta")) {
            continue;
        }
        const std::string tensor = record_string_field(record, "tensor");
        if (!tensor_filter.empty() && tensor != tensor_filter && tensor.find(tensor_filter) == std::string::npos) {
            continue;
        }
        if (layer_filter != std::numeric_limits<int>::min()) {
            const double layer = record_numeric_field(record, "layer", std::numeric_limits<double>::quiet_NaN());
            if (!std::isfinite(layer) || (int) layer != layer_filter) {
                continue;
            }
        }
        deltas.push_back(&record);
    }
    if (!exact_groups.empty()) {
        deltas = exact_groups;
    }

    if (baseline == nullptr) {
        throw std::runtime_error("sensitivity file has no baseline event: " + path);
    }
    if (deltas.empty()) {
        throw std::runtime_error("no sensitivity rows matched the what-if filter");
    }

    const bool using_exact_group = !exact_groups.empty();
    const bool linear_estimate = !using_exact_group && deltas.size() > 1;
    size_t matched_tensors = deltas.size();
    if (using_exact_group) {
        matched_tensors = 0;
        for (const bq::CandidateRecord * row : deltas) {
            matched_tensors += (size_t) std::max(0.0, record_numeric_field(*row, "tensor_count", 1.0));
        }
    }
    const std::string estimate_mode = using_exact_group
        ? "exact layer measurement"
        : (deltas.size() == 1 ? "exact single-tensor measurement" : "linear sum of independently measured tensor deltas");
    const std::string measurement_id = using_exact_group
        ? "exact_layer"
        : (deltas.size() == 1 ? "exact_tensor" : "linear_estimate");
    bool candidate_valid = true;
    for (const bq::CandidateRecord * row : deltas) {
        if (!record_bool_field(*row, "valid", true)) {
            candidate_valid = false;
            break;
        }
    }

    struct metric_spec {
        const char * name;
        const char * delta;
        bool higher_is_better = false;
    };
    const std::vector<metric_spec> metrics = {
        { "ppl", "delta_ppl", false },
        { "ln_ratio", "delta_ln_ratio", false },
        { "mean_kld", "delta_mean_kld", false },
        { "kld_p95", "delta_kld_p95", false },
        { "kld_p99", "delta_kld_p99", false },
        { "kld_p999", "delta_kld_p999", false },
        { "kld_tail_mean", "delta_kld_tail_mean", false },
        { "max_kld", "delta_max_kld", false },
        { "rms_dp", "delta_rms_dp", false },
        { "same_top", "delta_same_top", true },
        { "top_flip_weight", "delta_top_flip_weight", false },
        { "top_prob_rmse", "delta_top_prob_rmse", false },
        { "entropy_rmse", "delta_entropy_rmse", false },
    };

    struct metric_value {
        std::string name;
        double current = 0.0;
        double delta = 0.0;
        double predicted = 0.0;
        bool valid = true;
        bool higher_is_better = false;
    };
    std::vector<metric_value> values;
    values.reserve(metrics.size());
    for (const auto & metric : metrics) {
        metric_value value;
        value.name = metric.name;
        value.current = record_numeric_field(*baseline, metric.name, 0.0);
        value.higher_is_better = metric.higher_is_better;
        value.valid = candidate_valid;
        if (value.valid) {
            for (const bq::CandidateRecord * row : deltas) {
                const double delta = record_numeric_field(*row, metric.delta, std::numeric_limits<double>::quiet_NaN());
                if (!std::isfinite(delta)) {
                    value.valid = false;
                    break;
                }
                value.delta += delta;
            }
        }
        value.predicted = value.valid ? value.current + value.delta : std::numeric_limits<double>::quiet_NaN();
        values.push_back(value);
    }

    if (json) {
        std::cout << "{\n";
        std::cout << "  \"sensitivity_file\": \"" << json_escape(path) << "\",\n";
        std::cout << "  \"matched_tensors\": " << matched_tensors << ",\n";
        std::cout << "  \"valid\": " << (candidate_valid ? "true" : "false") << ",\n";
        std::cout << "  \"measurement\": \"" << measurement_id << "\",\n";
        std::cout << "  \"exact_measured\": " << (!linear_estimate ? "true" : "false") << ",\n";
        std::cout << "  \"linear_estimate\": " << (linear_estimate ? "true" : "false") << ",\n";
        std::cout << "  \"estimate_mode\": \"" << json_escape(estimate_mode) << "\",\n";
        std::cout << "  \"metrics\": {";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            const auto & value = values[i];
            std::cout << "\n    \"" << value.name << "\": {\"current\": " << value.current
                      << ", \"delta\": ";
            if (value.valid) {
                std::cout << value.delta;
            } else {
                std::cout << "null";
            }
            std::cout << ", \"predicted\": ";
            if (value.valid) {
                std::cout << value.predicted;
            } else {
                std::cout << "null";
            }
            std::cout << "}";
        }
        std::cout << "\n  }\n}\n";
        return 0;
    }

    const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
    bq::tui::print(std::cout, bq::tui::render_section_header("What-If Sensitivity", "", caps));
    std::vector<std::string> summary;
    summary.push_back("Report       " + path);
    if (!tensor_filter.empty()) {
        summary.push_back("Tensor       " + tensor_filter);
    }
    if (layer_filter != std::numeric_limits<int>::min()) {
        summary.push_back("Layer        " + std::to_string(layer_filter));
    }
    summary.push_back("Tensors      " + std::to_string(matched_tensors));
    if (using_exact_group) {
        summary.push_back("Measurement  exact layer delta, measured with all matching tensors patched together");
    } else if (deltas.size() == 1) {
        summary.push_back("Measurement  exact tensor delta, measured by patching this tensor");
    } else {
        summary.push_back("Measurement  linear estimate from independently measured tensor deltas");
        summary.push_back("Caution      run with selector.sensitivity_layer for an exact joint layer delta");
    }
    summary.push_back(std::string("Candidate    ") + (candidate_valid ? "valid on this subset" : "invalid, produced non-finite metrics"));
    bq::tui::BoxOptions summary_box;
    summary_box.title = "Scope";
    summary_box.wrap = true;
    bq::tui::print(std::cout, bq::tui::render_box(summary, summary_box, caps));
    if (!candidate_valid) {
        std::cout << bq::tui::paint(
            "Invalid candidates are shown without predicted values; they should not be selected or ranked as wins.",
            bq::tui::error(), caps) << "\n\n";
    }
    auto format_cell = [candidate_valid](double value, bool valid) {
        if (!valid || !std::isfinite(value)) {
            return candidate_valid ? std::string("n/a") : std::string("invalid");
        }
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6) << value;
        return ss.str();
    };
    bq::tui::print(std::cout, bq::tui::render_section_header("Metric Deltas", "", caps));
    std::cout << std::left << std::setw(18) << "metric"
              << std::right << std::setw(16) << "current"
              << std::setw(16) << "delta"
              << std::setw(16) << "predicted"
              << "  meaning\n";
    std::cout << std::string(78, '-') << "\n";
    for (const auto & value : values) {
        std::cout << std::left << std::setw(18) << value.name
                  << std::right << std::setw(16) << format_cell(value.current, true)
                  << std::setw(16) << format_cell(value.delta, value.valid)
                  << std::setw(16) << format_cell(value.predicted, value.valid)
                  << "  " << (value.higher_is_better ? "higher better" : "lower better")
                  << "\n";
    }
    return 0;
}

static bool prompt_value(const std::string & label, const std::string & fallback, std::string & out_value) {
    const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
    bq::tui::InputPrompt input;
    input.label = label;
    input.value = fallback;
    input.placeholder = fallback.empty() ?
        "Type a value, or press Enter to leave blank. ESC to go back." :
        "Type a new name, or press Enter to proceed as is. ESC to go back.";
    input.hint = "Use backspace to edit.";
    std::cout << bq::tui::render_input_prompt(input, caps) << std::flush;
    const PromptReadResult read = read_prompt_line();
    if (read.cancelled || read.eof) {
        return false;
    }
    out_value = read.value.empty() ? fallback : read.value;
    return true;
}

static std::string prompt(const std::string & label, const std::string & fallback = "") {
    std::string value;
    if (!prompt_value(label, fallback, value)) {
        throw PromptCancelled();
    }
    return value;
}

static double prompt_double(const std::string & label, double fallback) {
    std::ostringstream ss;
    ss << std::setprecision(10) << fallback;
    const std::string fallback_str = ss.str();
    const std::string value = prompt(label, fallback_str);
    if (value.empty()) {
        return fallback;
    }
    return std::stod(value);
}

static int prompt_int(const std::string & label, int fallback) {
    const std::string fallback_str = std::to_string(fallback);
    const std::string value = prompt(label, fallback_str);
    if (value.empty()) {
        return fallback;
    }
    return std::stoi(value);
}

static bool prompt_bool(const std::string & label, bool fallback) {
    while (true) {
        const std::string value = prompt(label + " (y/n)", fallback ? "y" : "n");
        if (value.empty()) {
            return fallback;
        }
        const char c = (char) std::tolower((unsigned char) value.front());
        if (c == 'y' || c == '1' || c == 't') {
            return true;
        }
        if (c == 'n' || c == '0' || c == 'f') {
            return false;
        }
        std::cout << " Enter Y or N. ESC to cancel.\n";
    }
}

static double estimate_gib(double params_b, double bpw) {
    if (params_b <= 0.0 || bpw <= 0.0) {
        return 0.0;
    }
    return params_b * bpw / 8.0 / 1.073741824;
}

static double assumed_bpw_for_type(std::string type) {
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    if (type == "f32") {
        return 32.0;
    }
    if (type == "f16" || type == "bf16") {
        return 16.0;
    }
    if (type == "q8_0" || type == "q8_1") {
        return 8.5;
    }
    if (type == "q6_k") {
        return 6.5625;
    }
    if (type == "q5_k" || type == "q5_k_m" || type == "q5_0" || type == "q5_1") {
        return 5.5;
    }
    if (type == "q4_k" || type == "q4_k_m" || type == "q4_0" || type == "q4_1" || type == "nvfp4") {
        return 4.5;
    }
    if (type == "mxfp6_e2m3") {
        return 6.62;
    }
    if (type == "mxfp8" || type == "mxfp8_e4m3") {
        return 8.25;
    }
    return 0.0;
}

static double estimate_params_b_from_summary(const bq::InspectSummary & summary) {
    double params = 0.0;
    for (const auto & entry : summary.type_stats) {
        const double bpw = assumed_bpw_for_type(entry.first);
        if (bpw > 0.0) {
            params += ((double) entry.second.bytes * 8.0) / bpw;
        }
    }
    return params / 1000000000.0;
}

static void apply_vram_target(
        bq::Recipe & recipe,
        const std::string & precision_mode,
        double params_b,
        int vram_gb,
        int ram_gb,
        double kv_cache_gib,
        double activation_headroom_gib) {
    const std::string quant_type = canonical_quant_type(precision_mode);
    recipe = bq::default_recipe_for_quant_type(quant_type);

    const double nv4_gib = estimate_gib(params_b, 4.5);
    const double mx6_gib = estimate_gib(params_b, 6.62);
    const double mx8_gib = estimate_gib(params_b, 8.25);
    const double usable_gib = vram_gb > 0 ? vram_gb * 0.88 : 0.0;
    const double weight_budget_gib = std::max(0.0, usable_gib - kv_cache_gib - activation_headroom_gib);
    const double target_bpw = params_b > 0.0 && weight_budget_gib > 0.0 ?
        weight_budget_gib * 8.0 * 1.073741824 / params_b : 0.0;

    recipe.target.precision_mode = quant_type;
    recipe.target.model_params_b = params_b;
    recipe.target.vram_gb = vram_gb;
    recipe.target.ram_gb = ram_gb;
    recipe.target.target_bpw = target_bpw;
    recipe.target.weight_budget_gib = weight_budget_gib;
    recipe.target.kv_cache_gib = kv_cache_gib;
    recipe.target.activation_headroom_gib = activation_headroom_gib;
    recipe.target.fit_to_vram = vram_gb > 0;

    std::ostringstream note;
    note.setf(std::ios::fixed);
    note.precision(2);
    note << "rough weights-only estimate: NVFP4=" << nv4_gib << " GiB, MXFP6=" << mx6_gib
         << " GiB, MXFP8=" << mx8_gib << " GiB";
    if (vram_gb > 0) {
        note << ", target usable=" << usable_gib << " GiB"
             << ", weight budget=" << weight_budget_gib << " GiB"
             << ", required average <= " << target_bpw << " bpw";
    }
    if (ram_gb > 0) {
        note << ", host RAM target=" << ram_gb << " GiB";
    }
    const std::string sizing_note = note.str();
    recipe.target.sizing_note = sizing_note;

    if (quant_type == "NVFP4_MXFP6" && params_b > 0.0 && weight_budget_gib > 0.0) {
        const double nv4_bpw = 4.5;
        const double mx6_bpw = 6.62;
        if (target_bpw >= mx6_bpw) {
            recipe = bq::default_recipe("mxfp6-primary");
            recipe.target.precision_mode = "NVFP4_MXFP6";
            recipe.target.model_params_b = params_b;
            recipe.target.vram_gb = vram_gb;
            recipe.target.ram_gb = ram_gb;
            recipe.target.target_bpw = target_bpw;
            recipe.target.weight_budget_gib = weight_budget_gib;
            recipe.target.kv_cache_gib = kv_cache_gib;
            recipe.target.activation_headroom_gib = activation_headroom_gib;
            recipe.target.fit_to_vram = vram_gb > 0;
            recipe.target.sizing_note = sizing_note;
        } else if (target_bpw > nv4_bpw) {
            const double mx6_fraction = std::clamp((target_bpw - nv4_bpw) / (mx6_bpw - nv4_bpw), 0.0, 1.0);
            const double extra_mb = std::max(0.0, (weight_budget_gib - nv4_gib) * 1024.0);
            recipe.nv4mx6.policy = "nv4_promote_mx6";
            recipe.nv4mx6.mx6_penalty = mx6_fraction > 0.65 ? "2.5" : "3.5";
            recipe.rescue.enabled = true;
            recipe.rescue.type = "MXFP6_E2M3";
            recipe.rescue.budget_mb = std::to_string((int) std::round(extra_mb));
        } else {
            recipe.nv4mx6.policy = "nv4_promote_mx6";
            recipe.nv4mx6.mx6_penalty = "5.0";
            recipe.rescue.budget_mb = "0";
        }
    } else if (quant_type == "NVFP4" && recipe.target.fit_to_vram) {
        recipe.rescue.enabled = false;
        recipe.rescue.top.clear();
        recipe.rescue.report_top.clear();
        recipe.rescue.budget_mb.clear();
        recipe.rescue.bf16_budget_mb.clear();
        recipe.rescue.class_limit.clear();
        recipe.rescue.encoder_top.clear();
    }

    std::ostringstream alloc;
    alloc.setf(std::ios::fixed);
    alloc.precision(2);
    alloc << "; allocator types: NVFP4=" << assumed_bpw_for_type("NVFP4")
          << " bpw, MXFP6_E2M3=" << assumed_bpw_for_type("MXFP6_E2M3")
          << " bpw, MXFP8=" << assumed_bpw_for_type("MXFP8")
          << " bpw, Q4_0=" << assumed_bpw_for_type("Q4_0")
          << " bpw, Q6_K=" << assumed_bpw_for_type("Q6_K")
          << " bpw, Q8_0=" << assumed_bpw_for_type("Q8_0")
          << " bpw, BF16/F16=16.00 bpw";
    if (quant_type == "NVFP4_MXFP6") {
        alloc << "; mixed policy=" << (recipe.nv4mx6.policy.empty() ? "auto" : recipe.nv4mx6.policy)
              << ", mx6_penalty=" << (recipe.nv4mx6.mx6_penalty.empty() ? "auto" : recipe.nv4mx6.mx6_penalty)
              << ", repair_type=" << (recipe.rescue.type.empty() ? "none" : recipe.rescue.type)
              << ", repair_budget_mb=" << (recipe.rescue.budget_mb.empty() ? "0" : recipe.rescue.budget_mb);
    }
    recipe.target.sizing_note += alloc.str();
}

static bool configure_precision_vram(bq::Recipe & recipe, bool preserve_context) {
    const bq::Recipe::Io io = recipe.io;
    const bq::Recipe::Base base = recipe.base;
    const bq::Recipe::Model model = recipe.model;
    const bq::Recipe::Metadata metadata = recipe.metadata;
    const bq::Recipe::Calibration calibration = recipe.calibration;
    const bq::Recipe::Evaluation evaluation = recipe.evaluation;
    const bq::Recipe::TensorOverrides tensor_overrides = recipe.tensor_overrides;
    const bq::Recipe::Nvfp4 nvfp4 = recipe.nvfp4;
    const bq::Recipe::Mxfp6 mxfp6 = recipe.mxfp6;
    const bq::Recipe::Artifacts artifacts = recipe.artifacts;

    std::vector<std::string> quant_labels = bq::quant_type_choices();
    quant_labels.push_back("Exit");
    const int mode = menu_select("Quant Formula", quant_labels);
    if (mode < 0 || mode >= (int) quant_labels.size() - 1) {
        return false;
    }
    const std::string precision_mode = quant_labels[mode];
    const int vram_idx = menu_select("VRAM target", {
        "8 GB",
        "12 GB",
        "16 GB",
        "24 GB",
        "32 GB",
        "Unlimited",
        "Back",
    });
    if (vram_idx == 6) {
        return false;
    }
    const std::vector<int> vram_values = { 8, 12, 16, 24, 32, 0 };
    const int ram_idx = menu_select("Host RAM target", {
        "Same as VRAM target - keep host budget simple",
        "16 GB",
        "32 GB",
        "64 GB",
        "No fixed target",
        "Back",
    });
    if (ram_idx == 5) {
        return false;
    }
    const std::vector<int> ram_values = {
        vram_values[vram_idx],
        16,
        32,
        64,
        0,
    };
    const double params_b = prompt_double("model parameters in billions", recipe.target.model_params_b);
    const double kv_cache_gib = prompt_double("KV/cache reserve GiB", recipe.target.kv_cache_gib);
    const double activation_headroom_gib = prompt_double("activation/headroom reserve GiB", recipe.target.activation_headroom_gib);
    apply_vram_target(recipe, precision_mode, params_b, vram_values[vram_idx], ram_values[ram_idx], kv_cache_gib, activation_headroom_gib);

    if (preserve_context) {
        recipe.io = io;
        recipe.base.allow_requantize = base.allow_requantize;
        recipe.base.leave_output_tensor = base.leave_output_tensor;
        recipe.base.pure = base.pure;
        recipe.base.copy_only = base.copy_only;
        recipe.model = model;
        recipe.metadata = metadata;
        recipe.calibration = calibration;
        recipe.evaluation = evaluation;
        recipe.tensor_overrides = tensor_overrides;
        if (!nvfp4.preset.empty()) {
            recipe.nvfp4 = nvfp4;
        }
        if (!mxfp6.tensor_scale.empty()) {
            recipe.mxfp6 = mxfp6;
        }
        recipe.artifacts = artifacts;
        recipe.base.threads = base.threads;
        recipe.base.dry_run = false;
    }
    return true;
}

static void configure_evaluation_source(bq::Recipe & recipe) {
    const int choice = menu_select("PPL + KLD evaluation", {
        "Use existing BF16 KLD base to calculate best ranking candidate",
        "Build BF16 reference logits, then use them for KLD",
        "Create a reusable bundle with recipe, corpus, KLD base, and artifacts",
        "Back",
    });
    if (choice == 3) {
        return;
    }

    if (choice == 0) {
        recipe.evaluation.kld_mode = "existing";
        recipe.evaluation.kld_base = prompt("existing KLD base", recipe.evaluation.kld_base.empty() ? recipe.selector.kld : recipe.evaluation.kld_base);
        recipe.evaluation.corpus = prompt("PPL/KLD corpus file", recipe.evaluation.corpus);
        const KldBaseInfo info = read_kld_base_info(recipe.evaluation.kld_base);
        if (info.valid) {
            std::cout << "KLD base: " << format_kld_info(info) << "\n";
        } else if (!recipe.evaluation.kld_base.empty()) {
            std::cout << "KLD base warning: " << format_kld_info(info) << "\n";
        }
        recipe.selector.kld = recipe.evaluation.kld_base;
        recipe.calibration.corpus = prompt(
            "calibration file for imatrix",
            recipe.calibration.corpus.empty() ? recipe.evaluation.corpus : recipe.calibration.corpus);
        return;
    }

    if (choice == 1 || choice == 2) {
        recipe.evaluation.kld_mode = choice == 1 ? "make_base" : "bundle";
        recipe.evaluation.bf16_reference = prompt(
            "BF16 reference GGUF",
            recipe.evaluation.bf16_reference.empty() ? recipe.io.input : recipe.evaluation.bf16_reference);
        recipe.evaluation.corpus = prompt("PPL/KLD corpus file", recipe.evaluation.corpus);
        recipe.evaluation.kld_base = prompt("KLD base output", default_kld_base_path(recipe));
        recipe.evaluation.perplexity_bin = prompt("perplexity executable", recipe.evaluation.perplexity_bin);
        recipe.selector.kld = recipe.evaluation.kld_base;
        if (choice == 2) {
            recipe.evaluation.bundle = prompt(
                "eval bundle path",
                recipe.evaluation.bundle.empty() && !recipe.artifacts.run_dir.empty() ?
                    (std::filesystem::path(recipe.artifacts.run_dir) / "eval-bundle").string() :
                    recipe.evaluation.bundle);
        }

        const std::string command = kld_base_command_shell(recipe);
        if (!command.empty()) {
            std::cout << "\nKLD base command:\n  " << command << "\n";
        }
    }
    recipe.calibration.corpus = prompt(
        "calibration file for imatrix",
        recipe.calibration.corpus.empty() ? recipe.evaluation.corpus : recipe.calibration.corpus);
}

static void configure_standard_quantize_options(bq::Recipe & recipe) {
    const int choice = menu_select("Standard quantize options", {
        "Skip - keep defaults from the recipe/profile",
        "Configure output tensors, pruning, metadata, and tensor overrides",
        "Back",
    });
    if (choice != 1) {
        return;
    }

    recipe.base.output_tensor_type = sanitize_tensor_type_token(prompt("output.weight tensor type", recipe.base.output_tensor_type));
    recipe.base.token_embedding_type = sanitize_tensor_type_token(prompt("token embedding tensor type", recipe.base.token_embedding_type));
    recipe.base.mtp_tensor_type = sanitize_tensor_type_token(prompt("MTP/NextN tensor type (blank follows profile/ftype default)", recipe.base.mtp_tensor_type));
    recipe.base.leave_output_tensor = prompt_bool("leave output.weight unquantized", recipe.base.leave_output_tensor);
    recipe.stock_ftype.sweep_tensor_policy = prompt_bool("measure embeddings/output as candidates", recipe.stock_ftype.sweep_tensor_policy || recipe.stock_ftype.sweep_sensitive_tensors);
    recipe.stock_ftype.sweep_sensitive_tensors = recipe.stock_ftype.sweep_tensor_policy;
    recipe.stock_ftype.token_embedding_candidates = split_type_csv(prompt(
        "token embedding candidate types",
        join_type_csv(recipe.stock_ftype.token_embedding_candidates)));
    recipe.stock_ftype.output_tensor_candidates = split_type_csv(prompt(
        "output tensor candidate types",
        join_type_csv(recipe.stock_ftype.output_tensor_candidates)));
    recipe.stock_ftype.min_quant_savings_mib = prompt_double("minimum tensor savings MiB before quantizing", recipe.stock_ftype.min_quant_savings_mib);
    recipe.stock_ftype.technique_candidates = split_type_csv(prompt(
        "technique candidates",
        join_type_csv(recipe.stock_ftype.technique_candidates)));
    recipe.base.allow_requantize = prompt_bool("allow requantizing already-quantized tensors", recipe.base.allow_requantize);
    recipe.base.pure = prompt_bool("pure mode, no built-in type mixtures", recipe.base.pure);
    recipe.base.copy_only = prompt_bool("copy only, do not quantize", recipe.base.copy_only);
    recipe.io.keep_split = prompt_bool("keep input split/shard layout", recipe.io.keep_split);
    recipe.model.prune_layers = prompt("prune layers CSV", recipe.model.prune_layers);

    const std::string kv_override = prompt("add metadata override KEY=TYPE:VALUE", "");
    if (!kv_override.empty()) {
        recipe.metadata.overrides.push_back(kv_override);
    }
    const std::string tensor_file = prompt("add tensor type override file", "");
    if (!tensor_file.empty()) {
        recipe.tensor_overrides.files.push_back(tensor_file);
    }
    const std::string tensor_entry = prompt("add tensor override tensor=type", "");
    if (!tensor_entry.empty()) {
        recipe.tensor_overrides.entries.push_back(tensor_entry);
    }

}

static void configure_four_six_mixed(bq::Recipe & recipe) {
    const int choice = menu_select("NVFP4 4/6 encoder", {
        "Balanced adaptive 4/6 - current default, lets CUDA choose M4 or M6 per block",
        "NVFP4_MXFP6 quality-first - spend more MXFP6 budget on risky tensors",
        "Keep current recipe settings",
        "Back",
    });
    if (choice == 2 || choice == 3) {
        return;
    }

    recipe.nvfp4.preset.clear();
    recipe.nvfp4.cfg.clear();
    recipe.nvfp4.four_six.compand = "1";

    if (choice == 0) {
        recipe.nvfp4.four_six.choose46 = "adaptive";
        recipe.nvfp4.four_six.refit_iters = "8";
        recipe.nvfp4.four_six.cap6 = "448";
        recipe.nvfp4.four_six.cap4 = "256";
    } else if (choice == 1) {
        recipe.target.precision_mode = "NVFP4_MXFP6";
        recipe.base.ftype = "NVFP4_MXFP6";
        recipe.nvfp4.four_six.choose46 = "adaptive";
        recipe.nvfp4.four_six.refit_iters = "16";
        recipe.nvfp4.four_six.cap6 = "448";
        recipe.nvfp4.four_six.cap4 = "224";
        if (recipe.rescue.enabled && !recipe.rescue.budget_mb.empty()) {
            const int budget = std::max(0, std::stoi(recipe.rescue.budget_mb));
            recipe.rescue.budget_mb = std::to_string(std::min(6144, std::max(2048, budget + 1024)));
        }
    }
}

static int size_main(int argc, char ** argv) {
    std::string precision_mode = bq::Recipe().target.precision_mode;
    double params_b = 0.0;
    int vram_gb = 0;
    int ram_gb = 0;
    double kv_cache_gib = 0.0;
    double activation_headroom_gib = 0.0;
    std::string inspect_path;

    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--precision-mode" && i + 1 < argc) {
            precision_mode = argv[++i];
        } else if (arg == "--params-b" && i + 1 < argc) {
            params_b = std::stod(argv[++i]);
        } else if (arg == "--vram-gb" && i + 1 < argc) {
            vram_gb = std::stoi(argv[++i]);
        } else if (arg == "--ram-gb" && i + 1 < argc) {
            ram_gb = std::stoi(argv[++i]);
        } else if (arg == "--kv-cache-gib" && i + 1 < argc) {
            kv_cache_gib = std::stod(argv[++i]);
        } else if (arg == "--activation-headroom-gib" && i + 1 < argc) {
            activation_headroom_gib = std::stod(argv[++i]);
        } else if (!arg.empty() && arg[0] != '-' && inspect_path.empty()) {
            inspect_path = arg;
        } else {
            throw std::runtime_error("unknown size argument: " + arg);
        }
    }

    precision_mode = canonical_quant_type(precision_mode);
    if (params_b <= 0.0) {
        throw std::runtime_error("size requires --params-b N");
    }

    bq::Recipe recipe;
    apply_vram_target(recipe, precision_mode, params_b, vram_gb, ram_gb, kv_cache_gib, activation_headroom_gib);

    std::cout << recipe.target.sizing_note << "\n";
    std::cout << "estimated NVFP4 weights: " << estimate_gib(params_b, 4.5) << " GiB\n";
    std::cout << "estimated MXFP6 weights: " << estimate_gib(params_b, 6.62) << " GiB\n";
    std::cout << "estimated MXFP8 weights: " << estimate_gib(params_b, 8.25) << " GiB\n";
    if (recipe.target.target_bpw > 0.0) {
        std::cout << "target average BPW: " << recipe.target.target_bpw << "\n";
    }
    if (precision_mode == "NVFP4_MXFP6") {
        std::cout << "suggested MXFP6 edit budget: " << recipe.rescue.budget_mb << " MiB\n";
    }
    if (!inspect_path.empty()) {
        const bq::InspectSummary summary = bq::inspect_gguf(inspect_path);
        std::cout << "inspected tensor bytes: " << mib_string(summary.tensor_bytes) << " MiB"
                  << ", NVFP4 tensors=" << summary.nvfp4_tensors
                  << ", MXFP6 tensors=" << summary.mxfp6_tensors
                  << ", MXFP8 tensors=" << summary.mxfp8_tensors
                  << ", MTP=" << (summary.has_mtp ? "yes" : "no") << "\n";
    }
    return 0;
}

static void shell_pause(const bq::tui::TerminalCapabilities & caps) {
    if (!caps.is_tty) {
        return;
    }
    std::cout << "\n" << bq::tui::paint("Press Enter to continue.", bq::tui::muted(), caps) << std::flush;
    (void) read_prompt_line();
}

class ShellAlternateScreen {
public:
    ShellAlternateScreen() {
        const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
        enabled = caps.ansi;
        if (enabled) {
            std::cout << "\033[?1049h\033[H\033[2J\033[3J" << std::flush;
        }
    }

    ~ShellAlternateScreen() {
        if (enabled) {
            std::cout << "\033[?1049l" << std::flush;
        }
    }

    ShellAlternateScreen(const ShellAlternateScreen &) = delete;
    ShellAlternateScreen & operator=(const ShellAlternateScreen &) = delete;

private:
    bool enabled = false;
};

static std::string shell_prompt_command(const bq::tui::TerminalCapabilities & caps, const std::string & prefix = "") {
    bq::tui::print(std::cout, bq::tui::render_branded_header(bq::shell_product(), caps));
    bq::tui::print(std::cout, bq::tui::render_section_header("Command", "", caps));
    bq::tui::InputPrompt input;
    input.label = "Command";
    input.value = prefix;
    input.placeholder = "Type /help, /status, /run, or press Esc to cancel.";
    input.hint = "Commands can also be typed directly at any menu by pressing slash.";
    std::cout << bq::tui::render_input_prompt(input, caps) << std::flush;
    const PromptReadResult rest = read_prompt_line();
    if (rest.cancelled || rest.eof) {
        return {};
    }
    return prefix + rest.value;
}

static int shell_menu_select(
        const ShellState & state,
        const std::string & title,
        const std::vector<bq::tui::MenuOption> & options,
        std::string & slash_command) {
    slash_command.clear();

#if !defined(_WIN32)
    if (stdin_is_tty()) {
        RawTerminal raw;
        int selected = 0;
        while (true) {
            const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
            shell_clear(caps);
            const bool compact_status = caps.is_tty && caps.rows > 0 && caps.rows <= 28;
            std::string status_panel;
            if (!compact_status) {
                status_panel += bq::tui::render_branded_header(bq::shell_product(), caps);
            }
            status_panel += bq::tui::render_status_panel(
                bq::shell_product(),
                compact_status ? shell_status_items_compact(state) : shell_status_items(state),
                caps);
            bq::tui::print(std::cout, status_panel);
            const size_t used_rows = rendered_line_count(status_panel) + 2;
            const size_t reserved_rows = 2;
            const size_t menu_rows = caps.is_tty && caps.rows > 0 && (size_t) caps.rows > used_rows + reserved_rows ?
                (size_t) caps.rows - used_rows - reserved_rows :
                5;
            bq::tui::print(std::cout, bq::tui::render_menu(title, options, (size_t) selected, caps, menu_rows));
            const std::string esc_hint = title == "Main Menu" ? "Esc to quit." : "Esc to go back.";
            std::cout << bq::tui::paint("Use Up/Down or j/k, Enter to select, / for commands, " + esc_hint, bq::tui::muted(), caps)
                      << "\n"
                      << std::flush;
            switch (read_key()) {
                case Key::UP:
                    selected = (selected + (int) options.size() - 1) % (int) options.size();
                    break;
                case Key::DOWN:
                    selected = (selected + 1) % (int) options.size();
                    break;
                case Key::ENTER:
                    shell_clear(caps);
                    return selected;
                case Key::SLASH:
                    slash_command = "/";
                    shell_clear(caps);
                    return -2;
                case Key::ESC:
                    shell_clear(caps);
                    return (int) options.size() - 1;
                default:
                    break;
            }
        }
    }
#endif

    const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
    const std::string status_panel =
        bq::tui::render_branded_header(bq::shell_product(), caps) +
        bq::tui::render_status_panel(bq::shell_product(), shell_status_items(state), caps);
    bq::tui::print(std::cout, status_panel);
    bq::tui::print(std::cout, bq::tui::render_menu(title, options, 0, caps));
    std::cout << "choice or /command: " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) {
        return (int) options.size() - 1;
    }
    line = strip_control_input(line);
    if (line.empty() || line == "q" || line == "quit" || line == "back" || line == "cancel") {
        return (int) options.size() - 1;
    }
    if (!line.empty() && line[0] == '/') {
        slash_command = line;
        return -2;
    }
    const int choice = std::stoi(line);
    if (choice < 1 || choice > (int) options.size()) {
        throw std::runtime_error("menu choice out of range");
    }
    return choice - 1;
}

static void shell_begin_page(const ShellState & state, const std::string & title) {
    const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
    shell_clear(caps);
    shell_print_status(state, caps);
    bq::tui::print(std::cout, bq::tui::render_section_header(title, "", caps));
}

static std::string shell_prompt_on_page(
        const ShellState & state,
        const std::string & title,
        const std::string & label,
        const std::string & fallback = "") {
    shell_begin_page(state, title);
    return prompt(label, fallback);
}

static double shell_prompt_double_on_page(
        const ShellState & state,
        const std::string & title,
        const std::string & label,
        double fallback) {
    std::ostringstream ss;
    ss << std::setprecision(10) << fallback;
    const std::string value = shell_prompt_on_page(state, title, label, ss.str());
    if (value.empty()) {
        return fallback;
    }
    return std::stod(value);
}

static int shell_prompt_int_on_page(
        const ShellState & state,
        const std::string & title,
        const std::string & label,
        int fallback) {
    const std::string value = shell_prompt_on_page(state, title, label, std::to_string(fallback));
    if (value.empty()) {
        return fallback;
    }
    return std::stoi(value);
}

static bool shell_prompt_bool_on_page(
        const ShellState & state,
        const std::string & title,
        const std::string & label,
        bool fallback) {
    while (true) {
        const std::string value = shell_prompt_on_page(state, title, label + " (y/n)", fallback ? "y" : "n");
        if (value.empty()) {
            return fallback;
        }
        const char c = (char) std::tolower((unsigned char) value.front());
        if (c == 'y' || c == '1' || c == 't') {
            return true;
        }
        if (c == 'n' || c == '0' || c == 'f') {
            return false;
        }
        std::cout << " Enter Y or N. ESC to cancel.\n";
        shell_pause(bq::tui::detect_terminal(stdout));
    }
}

static bool shell_handle_command(ShellState & state, std::string line, const bq::tui::TerminalCapabilities & caps);

static int shell_submenu_select(ShellState & state, const std::string & title, const std::vector<std::string> & labels) {
    std::vector<bq::tui::MenuOption> options;
    options.reserve(labels.size());
    for (const std::string & label : labels) {
        options.push_back({ label, "", "" });
    }

    while (true) {
        std::string slash_command;
        const int action = shell_menu_select(state, title, options, slash_command);
        const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
        if (action != -2) {
            return action;
        }
        try {
            shell_handle_command(state, slash_command, caps);
        } catch (const PromptCancelled &) {
            state.status = "Cancelled";
            continue;
        }
        if (state.quit) {
            return -1;
        }
        shell_pause(caps);
    }
}

static std::vector<std::string> split_words(const std::string & line) {
    std::istringstream in(line);
    std::vector<std::string> words;
    std::string word;
    while (in >> word) {
        words.push_back(word);
    }
    return words;
}

static bool has_any_extension(const std::filesystem::path & path, const std::vector<std::string> & extensions) {
    if (extensions.empty()) {
        return true;
    }
    const std::string ext = path.extension().string();
    for (const std::string & candidate : extensions) {
        if (ext == candidate) {
            return true;
        }
    }
    return false;
}

static bool contains_case_insensitive(std::string haystack, std::string needle) {
    if (needle.empty()) {
        return true;
    }
    std::transform(haystack.begin(), haystack.end(), haystack.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    std::transform(needle.begin(), needle.end(), needle.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return haystack.find(needle) != std::string::npos;
}

static std::filesystem::path picker_absolute(std::filesystem::path path) {
    std::error_code ec;
    std::filesystem::path absolute = std::filesystem::absolute(path, ec);
    if (!ec && !absolute.empty()) {
        return absolute.lexically_normal();
    }
    return path.lexically_normal();
}

static std::filesystem::path picker_start_directory(const std::string & fallback, bool fallback_is_directory) {
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::current_path(ec);
    if (!fallback.empty()) {
        std::filesystem::path path(fallback);
        dir = fallback_is_directory ? path : path.parent_path();
    }
    if (dir.empty()) {
        dir = ".";
    }
    dir = picker_absolute(dir);
    if (!std::filesystem::is_directory(dir, ec) && dir.has_parent_path()) {
        dir = picker_absolute(dir.parent_path());
    }
    if (!std::filesystem::is_directory(dir, ec)) {
        std::error_code cwd_ec;
        dir = std::filesystem::current_path(cwd_ec);
        if (cwd_ec || dir.empty()) {
            dir = ".";
        }
        dir = picker_absolute(dir);
    }
    return dir;
}

static std::filesystem::path picker_parent_directory(const std::filesystem::path & dir) {
    std::filesystem::path parent = dir.parent_path();
    if (parent.empty() || parent == dir) {
        return {};
    }
    return picker_absolute(parent);
}

static std::string picker_lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return text;
}

static bool picker_hidden_name(const std::filesystem::path & path) {
    const std::string name = path.filename().string();
    return !name.empty() && name[0] == '.';
}

static bool picker_query_allows_hidden(const std::string & query) {
    return !query.empty() && query[0] == '.';
}

static bool picker_prune_directory(const std::filesystem::path & path, const std::string & query) {
    const std::string name = path.filename().string();
    if (!picker_query_allows_hidden(query) && picker_hidden_name(path)) {
        return true;
    }
    static const std::array<const char *, 15> pruned = {
        ".git",
        ".cache",
        ".cargo",
        ".npm",
        ".rustup",
        ".venv",
        "__pycache__",
        "build",
        "cmake-build-debug",
        "cmake-build-release",
        "node_modules",
        "target",
        "venv",
        "env",
        "dist",
    };
    return std::find(pruned.begin(), pruned.end(), name) != pruned.end();
}

static std::vector<std::string> picker_query_tokens(const std::string & query) {
    std::istringstream in(picker_lower(query));
    std::vector<std::string> tokens;
    std::string token;
    while (in >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

static bool picker_matches_query(const std::filesystem::path & path, const std::string & query) {
    if (query.empty()) {
        return true;
    }
    const std::string path_text = picker_lower(path.string());
    for (const std::string & token : picker_query_tokens(query)) {
        if (path_text.find(token) == std::string::npos) {
            return false;
        }
    }
    return true;
}

static int picker_match_score(
        const std::filesystem::path & path,
        const std::string & query,
        bool is_directory,
        bool directories_only,
        int root_rank) {
    const std::string q = picker_lower(query);
    const std::string name = picker_lower(path.filename().string());
    const std::string full = picker_lower(path.string());
    int score = root_rank * 100000;
    if (!directories_only && !is_directory) {
        score -= 20000;
    }
    if (name == q) {
        score -= 40000;
    } else if (name.find(q) == 0) {
        score -= 30000;
    } else if (name.find(q) != std::string::npos) {
        score -= 20000;
    } else if (full.find(q) != std::string::npos) {
        score -= 10000;
    }
    score += (int) std::min<size_t>(999, std::distance(path.begin(), path.end())) * 2000;
    score += (int) std::min<size_t>(999, path.string().size());
    return score;
}

static bool picker_same_path(const std::filesystem::path & lhs, const std::filesystem::path & rhs) {
    return picker_absolute(lhs).string() == picker_absolute(rhs).string();
}

static void picker_add_root(std::vector<std::filesystem::path> & roots, const std::filesystem::path & candidate) {
    if (candidate.empty() || picker_absolute(candidate) == picker_absolute(candidate).root_path()) {
        return;
    }
    std::error_code ec;
    const std::filesystem::path root = picker_absolute(candidate);
    if (!std::filesystem::is_directory(root, ec)) {
        return;
    }
    for (const std::filesystem::path & existing : roots) {
        if (picker_same_path(existing, root)) {
            return;
        }
    }
    roots.push_back(root);
}

static void picker_add_root_with_parents(
        std::vector<std::filesystem::path> & roots,
        std::filesystem::path root,
        int parent_count) {
    root = picker_absolute(root);
    for (int i = 0; i <= parent_count && !root.empty(); ++i) {
        picker_add_root(roots, root);
        const std::filesystem::path parent = picker_parent_directory(root);
        if (parent.empty()) {
            break;
        }
        root = parent;
    }
}

static std::vector<std::filesystem::path> picker_search_roots(const std::filesystem::path & dir) {
    std::vector<std::filesystem::path> roots;
    picker_add_root_with_parents(roots, dir, 1);
    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) {
        picker_add_root_with_parents(roots, cwd, 2);
    }
    picker_add_root_with_parents(roots, shell_settings_dir(), 2);
    return roots;
}

struct PickerScoredOption {
    bq::tui::MenuOption option;
    int score = 0;
    std::string path;
};

struct PickerSearchResults {
    std::vector<bq::tui::MenuOption> entries;
    size_t total_matches = 0;
    bool limited = false;
};

struct PickerOptions {
    std::vector<bq::tui::MenuOption> options;
    size_t result_count = 0;
    bool result_count_limited = false;
};

static std::string picker_result_label(const std::filesystem::path & path, bool is_directory, bool show_full_path) {
    std::string label = show_full_path ? display_path(path.string()) : path.filename().string();
    if (is_directory && (label.empty() || label.back() != '/')) {
        label += "/";
    }
    return label;
}

static std::vector<bq::tui::MenuOption> picker_current_entries(
        const std::filesystem::path & dir,
        bool directories_only,
        const std::vector<std::string> & extensions,
        const std::string & query) {
    std::vector<std::filesystem::path> dirs;
    std::vector<std::filesystem::path> files;
    const bool include_hidden = picker_query_allows_hidden(query);

    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }

        std::error_code entry_ec;
        const std::filesystem::path path = entry.path();
        if (!include_hidden && picker_hidden_name(path)) {
            continue;
        }
        if (entry.is_directory(entry_ec)) {
            if (contains_case_insensitive(path.filename().string(), query)) {
                dirs.push_back(path);
            }
            continue;
        }

        entry_ec.clear();
        if (!directories_only && entry.is_regular_file(entry_ec) && has_any_extension(path, extensions)) {
            if (contains_case_insensitive(path.filename().string(), query)) {
                files.push_back(path);
            }
        }
    }

    auto path_less = [](const std::filesystem::path & lhs, const std::filesystem::path & rhs) {
        return picker_lower(lhs.filename().string()) < picker_lower(rhs.filename().string());
    };
    std::sort(dirs.begin(), dirs.end(), path_less);
    std::sort(files.begin(), files.end(), path_less);

    std::vector<bq::tui::MenuOption> entries;
    entries.reserve(dirs.size() + files.size());
    auto append_files = [&]() {
        for (const std::filesystem::path & path : files) {
            entries.push_back({ picker_result_label(path, false, false), "", "file:" + path.string() });
        }
    };
    auto append_dirs = [&]() {
        for (const std::filesystem::path & path : dirs) {
            entries.push_back({ picker_result_label(path, true, false), "", "dir:" + path.string() });
        }
    };
    if (directories_only) {
        append_dirs();
    } else {
        append_files();
        append_dirs();
    }
    return entries;
}

static void picker_maybe_add_search_result(
        std::vector<PickerScoredOption> & scored,
        std::vector<std::string> & seen,
        const std::filesystem::directory_entry & entry,
        bool directories_only,
        const std::vector<std::string> & extensions,
        const std::string & query,
        int root_rank) {
    std::error_code entry_ec;
    const std::filesystem::path path = entry.path();
    const bool is_dir = entry.is_directory(entry_ec);
    if (entry_ec) {
        return;
    }

    std::filesystem::path selected_path = path;
    std::filesystem::path score_path = path;
    bool selected_is_dir = is_dir;
    std::string description;

    if (directories_only) {
        if (is_dir) {
            if (!picker_matches_query(path, query)) {
                return;
            }
        } else if (entry.is_regular_file(entry_ec) && picker_matches_query(path, query)) {
            selected_path = path.parent_path();
            score_path = path;
            selected_is_dir = true;
            description = "contains " + path.filename().string();
        } else {
            return;
        }
    } else {
        const bool selectable = is_dir || (entry.is_regular_file(entry_ec) && has_any_extension(path, extensions));
        if (!selectable || !picker_matches_query(path, query)) {
            return;
        }
    }

    if (selected_path.empty()) {
        return;
    }

    const std::string key = picker_absolute(selected_path).string();
    if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
        return;
    }
    seen.push_back(key);
    const std::string command = selected_is_dir ? "dir:" + selected_path.string() : "file:" + selected_path.string();
    scored.push_back({
        { picker_result_label(selected_path, selected_is_dir, true), description, command },
        picker_match_score(score_path, query, selected_is_dir, directories_only, root_rank),
        key,
    });
}

static PickerSearchResults picker_search_entries(
        const std::filesystem::path & dir,
        bool directories_only,
        const std::vector<std::string> & extensions,
        const std::string & query) {
    std::vector<PickerScoredOption> scored;
    std::vector<std::string> seen;
    const std::vector<std::filesystem::path> roots = picker_search_roots(dir);
    constexpr int max_depth = 6;
    constexpr size_t max_seen = 250000;
    constexpr size_t max_results = 2000;
    size_t visited = 0;

    for (size_t root_index = 0; root_index < roots.size(); ++root_index) {
        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(roots[root_index], ec)) {
            if (ec) {
                break;
            }
            picker_maybe_add_search_result(scored, seen, entry, directories_only, extensions, query, (int) root_index);
        }
        ec.clear();

        std::filesystem::recursive_directory_iterator it(
            roots[root_index],
            std::filesystem::directory_options::skip_permission_denied,
            ec);
        std::filesystem::recursive_directory_iterator end;
        while (!ec && it != end && visited < max_seen) {
            const std::filesystem::directory_entry entry = *it;
            ++visited;
            std::error_code entry_ec;
            const std::filesystem::path path = entry.path();
            const bool is_dir = entry.is_directory(entry_ec);
            if (entry_ec) {
                it.increment(ec);
                continue;
            }
            if (is_dir && (it.depth() >= max_depth || picker_prune_directory(path, query))) {
                it.disable_recursion_pending();
            }
            picker_maybe_add_search_result(scored, seen, entry, directories_only, extensions, query, (int) root_index);
            it.increment(ec);
        }
        if (visited >= max_seen) {
            break;
        }
    }

    std::sort(scored.begin(), scored.end(), [](const PickerScoredOption & lhs, const PickerScoredOption & rhs) {
        if (lhs.score != rhs.score) {
            return lhs.score < rhs.score;
        }
        return lhs.path < rhs.path;
    });

    PickerSearchResults result;
    result.total_matches = scored.size();
    result.limited = visited >= max_seen || scored.size() > max_results;
    result.entries.reserve(std::min(max_results, scored.size()));
    for (size_t i = 0; i < scored.size() && i < max_results; ++i) {
        result.entries.push_back(scored[i].option);
    }
    return result;
}

static bool picker_query_looks_like_path(const std::string & query) {
    return query.find('/') != std::string::npos ||
           query.find('\\') != std::string::npos ||
           (!query.empty() && (query[0] == '.' || query[0] == '~'));
}

static std::filesystem::path picker_path_from_query(const std::filesystem::path & dir, const std::string & query) {
    if (query.empty()) {
        return {};
    }
    std::filesystem::path path(query);
    if (!query.empty() && query[0] == '~') {
        const std::filesystem::path settings_home = shell_settings_dir().parent_path().parent_path();
        const std::string rest = query.size() > 1 && (query[1] == '/' || query[1] == '\\') ? query.substr(2) : query.substr(1);
        path = settings_home / rest;
    } else if (!path.is_absolute()) {
        path = dir / path;
    }
    return picker_absolute(path);
}

static void picker_add_typed_path_option(
        std::vector<bq::tui::MenuOption> & options,
        const std::filesystem::path & dir,
        bool directories_only,
        const std::vector<std::string> & extensions,
        const std::string & query) {
    if (!picker_query_looks_like_path(query)) {
        return;
    }
    const std::filesystem::path path = picker_path_from_query(dir, query);
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        options.push_back({
            directories_only ? "Use typed directory" : "Open typed directory",
            display_path(path.string()),
            directories_only ? "use:" + path.string() : "dir:" + path.string(),
        });
        return;
    }
    ec.clear();
    if (!directories_only && std::filesystem::is_regular_file(path, ec) && has_any_extension(path, extensions)) {
        options.push_back({ "Use typed file", display_path(path.string()), "file:" + path.string() });
    }
}

static void picker_add_fallback_option(
        std::vector<bq::tui::MenuOption> & options,
        const std::filesystem::path & dir,
        bool directories_only,
        const std::vector<std::string> & extensions,
        const std::string & fallback) {
    if (fallback.empty()) {
        return;
    }
    const std::filesystem::path path = picker_absolute(fallback);
    std::error_code ec;
    if (directories_only && std::filesystem::is_directory(path, ec)) {
        if (picker_same_path(path, dir)) {
            return;
        }
        options.push_back({ "Use Selected Directory", display_path(path.string()), "use:" + path.string() });
        return;
    }
    ec.clear();
    if (!directories_only && std::filesystem::is_regular_file(path, ec) && has_any_extension(path, extensions)) {
        options.push_back({ "Use Selected File", display_path(path.string()), "file:" + path.string() });
    }
}

static PickerOptions picker_options(
        const std::filesystem::path & dir,
        bool directories_only,
        const std::vector<std::string> & extensions,
        const std::string & query,
        const std::string & fallback) {
    PickerOptions out;
    const std::filesystem::path parent = picker_parent_directory(dir);
    if (query.empty()) {
        if (directories_only) {
            out.options.push_back({ "Use This Directory", display_path(dir.string()), "use:" + dir.string() });
        }
        picker_add_fallback_option(out.options, dir, directories_only, extensions, fallback);
        out.options.push_back({ "Type Path", "enter a full file or directory path", "type" });
        if (!parent.empty()) {
            out.options.push_back({ "..", display_path(parent.string()), "dir:" + parent.string() });
        }
        std::vector<bq::tui::MenuOption> entries = picker_current_entries(dir, directories_only, extensions, query);
        out.result_count = entries.size();
        out.options.insert(out.options.end(), entries.begin(), entries.end());
        if (entries.empty()) {
            out.options.push_back({ directories_only ? "No folders here" : "No matching files here", "type to search nearby folders", "noop", true });
        }
    } else {
        picker_add_typed_path_option(out.options, dir, directories_only, extensions, query);
        out.options.push_back({ "Type Path", "enter a full file or directory path", "type" });
        if (query.size() < 2 && !picker_query_looks_like_path(query)) {
            out.options.push_back({ "Keep typing", "two or more characters searches nearby folders", "noop", true });
        } else {
            PickerSearchResults entries = picker_search_entries(dir, directories_only, extensions, query);
            out.result_count = entries.total_matches;
            out.result_count_limited = entries.limited;
            out.options.insert(out.options.end(), entries.entries.begin(), entries.entries.end());
            if (entries.entries.empty()) {
                out.options.push_back({ "No matches", "try fewer words or paste a full path", "noop", true });
            }
        }
        out.options.push_back({ "Clear search", query, "clear" });
    }
    out.options.push_back({ "Cancel", "", "cancel" });
    return out;
}

static bool picker_resolve_typed_path(
        const std::filesystem::path & dir,
        bool directories_only,
        const std::vector<std::string> & extensions,
        const std::string & typed,
        std::filesystem::path & out_dir,
        std::string & selected_path) {
    const std::filesystem::path path = picker_path_from_query(dir, typed);
    std::error_code ec;
    if (directories_only) {
        if (std::filesystem::is_directory(path, ec)) {
            selected_path = path.string();
            return true;
        }
        return false;
    }
    if (std::filesystem::is_regular_file(path, ec) && has_any_extension(path, extensions)) {
        selected_path = path.string();
        return true;
    }
    ec.clear();
    if (std::filesystem::is_directory(path, ec)) {
        out_dir = path;
    }
    return false;
}

static void picker_move_selection(const std::vector<bq::tui::MenuOption> & options, size_t & selected, int delta) {
    if (options.empty()) {
        selected = 0;
        return;
    }
    const int count = (int) options.size();
    int next = (int) selected;
    for (int i = 0; i < count; ++i) {
        next = std::max(0, std::min(count - 1, next + delta));
        if (!options[(size_t) next].disabled) {
            selected = (size_t) next;
            return;
        }
        if (next == 0 || next == count - 1) {
            break;
        }
    }
}

static void picker_jump_selection(const std::vector<bq::tui::MenuOption> & options, size_t & selected, int delta) {
    if (options.empty()) {
        selected = 0;
        return;
    }
    const int count = (int) options.size();
    selected = (size_t) std::max(0, std::min(count - 1, (int) selected + delta));
    if (options[selected].disabled) {
        picker_move_selection(options, selected, delta >= 0 ? 1 : -1);
    }
}

static size_t picker_window_begin(size_t selected, size_t option_count, size_t visible_rows) {
    if (option_count <= visible_rows) {
        return 0;
    }
    const size_t half = visible_rows / 2;
    size_t begin = selected > half ? selected - half : 0;
    if (begin + visible_rows > option_count) {
        begin = option_count - visible_rows;
    }
    return begin;
}

static std::string picker_format_row(
        size_t index,
        const bq::tui::MenuOption & option,
        bool is_selected,
        const bq::tui::TerminalCapabilities & caps,
        size_t width) {
    std::ostringstream out;
    out << (is_selected ? "> " : "  ")
        << std::setw(3) << (index + 1) << ". ";

    std::string label = option.label;
    if (option.disabled) {
        label = bq::tui::paint(label, bq::tui::muted(), caps);
    } else if (is_selected) {
        label = bq::tui::paint(" " + label + " ", bq::tui::selected(), caps);
    } else {
        label = bq::tui::paint(label, bq::tui::normal(), caps);
    }
    out << label;
    if (!option.description.empty() && width >= 92) {
        out << "  " << bq::tui::paint(option.description, bq::tui::muted(), caps);
    }
    return bq::tui::truncate_to_width(out.str(), width);
}

static void picker_render(
        const ShellState & state,
        const std::string & title,
        const std::filesystem::path & dir,
        const std::string & query,
        const PickerOptions & picker,
        size_t selected) {
    (void) state;
    const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
    shell_clear(caps);

    const std::vector<bq::tui::MenuOption> & options = picker.options;
    const size_t width = std::min<size_t>(std::max(80, caps.columns), 120);
    const size_t header_rows = query.empty() ? 5 : 6;
    const size_t visible_rows = caps.is_tty && caps.rows > 0 && (size_t) caps.rows > header_rows + 3 ?
        std::max<size_t>(10, (size_t) caps.rows - header_rows - 3) :
        24;
    const size_t begin = picker_window_begin(selected, options.size(), visible_rows);
    const size_t end = std::min(options.size(), begin + visible_rows);

    std::cout << bq::tui::paint(title, bq::tui::accent(), caps) << "\n";
    std::cout << bq::tui::paint(std::string(std::min<size_t>(width, 100), '-'), bq::tui::muted(), caps) << "\n";
    std::cout << "Current  " << display_path(dir.string()) << "\n";
    std::cout << "Search   " << (query.empty() ? "-" : query) << "\n";
    std::cout << (query.empty() ? "Items    " : "Matches  ")
              << picker.result_count
              << (picker.result_count_limited ? "+" : "")
              << "  Showing " << (options.empty() ? 0 : begin + 1)
              << "-" << end
              << " of " << options.size()
              << "\n";
    std::cout << bq::tui::paint(std::string(std::min<size_t>(width, 100), '-'), bq::tui::muted(), caps) << "\n";
    if (begin > 0) {
        std::cout << bq::tui::paint("... " + std::to_string(begin) + " above", bq::tui::muted(), caps) << "\n";
    }
    for (size_t i = begin; i < end; ++i) {
        std::cout << picker_format_row(i, options[i], i == selected, caps, width) << "\n";
    }
    if (end < options.size()) {
        std::cout << bq::tui::paint("... " + std::to_string(options.size() - end) + " below", bq::tui::muted(), caps) << "\n";
    }
    std::cout << bq::tui::paint(std::string(std::min<size_t>(width, 100), '-'), bq::tui::muted(), caps) << "\n";
    std::cout << bq::tui::paint(
        "Type to search. Arrows move. Ctrl-F/B jumps. Enter selects. Esc clears search or cancels. Use .. to move up.",
        bq::tui::muted(),
        caps) << "\n" << std::flush;
}

static bool picker_apply_command(const std::string & command, std::filesystem::path & dir, std::string & query, std::string & selected_path) {
    if (command == "cancel") {
        selected_path.clear();
        return true;
    }
    if (command == "noop") {
        return false;
    }
    if (command == "clear") {
        query.clear();
        return false;
    }
    if (command.find("use:") == 0) {
        selected_path = command.substr(4);
        return true;
    }
    if (command.find("file:") == 0) {
        selected_path = command.substr(5);
        return true;
    }
    if (command.find("dir:") == 0) {
        dir = picker_absolute(command.substr(4));
        query.clear();
        return false;
    }
    return false;
}

static std::string choose_path_from_disk(
        ShellState & state,
        const std::string & title,
        bool directories_only,
        const std::vector<std::string> & extensions,
        const std::string & fallback) {
    std::filesystem::path dir = picker_start_directory(fallback, directories_only);
    std::string query;
    size_t selected = 0;

#if !defined(_WIN32)
    if (stdin_is_tty()) {
        while (true) {
            PickerOptions picker = picker_options(dir, directories_only, extensions, query, fallback);
            std::vector<bq::tui::MenuOption> & options = picker.options;
            selected = std::min(selected, options.empty() ? (size_t) 0 : options.size() - 1);
            if (!options.empty() && options[selected].disabled) {
                picker_move_selection(options, selected, 1);
            }
            picker_render(state, title, dir, query, picker, selected);

            char c = 0;
            Key escaped = Key::ESC;
            bool consumed_escape = false;
            {
                RawTerminal raw;
                if (!read_input_char(c, -1)) {
                    return {};
                }
                if (c == 27) {
                    consumed_escape = consume_escape_sequence_after_esc(escaped);
                }
            }
            const unsigned char uc = (unsigned char) c;
            if (c == 27) {
                if (consumed_escape) {
                    if (escaped == Key::UP) {
                        picker_move_selection(options, selected, -1);
                    } else if (escaped == Key::DOWN) {
                        picker_move_selection(options, selected, 1);
                    }
                    continue;
                }
                if (!query.empty()) {
                    query.clear();
                    selected = 0;
                    continue;
                }
                return {};
            }
            if (c == '\n' || c == '\r') {
                std::string selected_path;
                const std::string command = selected < options.size() ? options[selected].command : "cancel";
                if (command == "type") {
                    shell_begin_page(state, title);
                    const std::string typed = prompt("Path", fallback);
                    if (picker_resolve_typed_path(dir, directories_only, extensions, typed, dir, selected_path)) {
                        return selected_path;
                    }
                    query.clear();
                    selected = 0;
                    state.status = "Path is not selectable";
                    continue;
                }
                if (picker_apply_command(command, dir, query, selected_path)) {
                    return selected_path;
                }
                selected = 0;
                continue;
            }
            if (c == 21) {
                query.clear();
                selected = 0;
                continue;
            }
            if (c == 127 || c == 8) {
                if (!query.empty()) {
                    query.pop_back();
                    selected = 0;
                }
                continue;
            }
            if (c == 16) {
                picker_move_selection(options, selected, -1);
                continue;
            }
            if (c == 14) {
                picker_move_selection(options, selected, 1);
                continue;
            }
            if (c == 2) {
                picker_jump_selection(options, selected, -15);
                continue;
            }
            if (c == 6) {
                picker_jump_selection(options, selected, 15);
                continue;
            }
            if (std::isprint(uc) || c == '\t') {
                query.push_back(c);
                selected = 0;
            }
        }
    }
#endif

    while (true) {
        PickerOptions picker = picker_options(dir, directories_only, extensions, query, fallback);
        std::vector<bq::tui::MenuOption> & options = picker.options;
        picker_render(state, title, dir, query, picker, std::min(selected, options.empty() ? (size_t) 0 : options.size() - 1));
        std::cout << "search, path, choice, or blank to cancel: " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) {
            return {};
        }
        line = strip_control_input(line);
        if (line.empty()) {
            return {};
        }
        if (std::all_of(line.begin(), line.end(), [](unsigned char c) { return std::isdigit(c); })) {
            const int choice = std::stoi(line);
            if (choice < 1 || choice > (int) options.size()) {
                continue;
            }
            std::string selected_path;
            const std::string command = options[(size_t) choice - 1].command;
            if (command == "type") {
                const std::string typed = prompt("Path", fallback);
                if (picker_resolve_typed_path(dir, directories_only, extensions, typed, dir, selected_path)) {
                    return selected_path;
                }
                query.clear();
                selected = 0;
                state.status = "Path is not selectable";
                continue;
            }
            if (picker_apply_command(command, dir, query, selected_path)) {
                return selected_path;
            }
            continue;
        }
        query = line;
        selected = 0;
    }
}

static std::string choose_file_from_disk(
        ShellState & state,
        const std::string & title,
        const std::vector<std::string> & extensions,
        const std::string & fallback = {}) {
    return choose_path_from_disk(state, title, false, extensions, fallback);
}

static std::string choose_directory_from_disk(
        ShellState & state,
        const std::string & title,
        const std::string & fallback = {}) {
    return choose_path_from_disk(state, title, true, {}, fallback);
}

static void shell_sync_recipe_paths(ShellState & state) {
    state.recipe.io.input = state.input_model;
    state.recipe.io.output = state.output_model;
}

static std::string output_quant_type_token(std::string quant_type) {
    quant_type = canonical_quant_type(quant_type);
    std::replace(quant_type.begin(), quant_type.end(), '_', '-');
    return quant_type.empty() ? std::string("NVFP4") : quant_type;
}

static std::string shell_active_quant_type(const ShellState & state) {
    if (!state.recipe.target.precision_mode.empty()) {
        return state.recipe.target.precision_mode;
    }
    if (!state.precision_mode.empty()) {
        return state.precision_mode;
    }
    if (!state.recipe.base.ftype.empty()) {
        return state.recipe.base.ftype;
    }
    return bq::Recipe().target.precision_mode;
}

static std::string default_output_model_path(const std::string & input_model, const std::string & quant_type) {
    if (input_model.empty()) {
        return {};
    }
    std::filesystem::path path(input_model);
    std::string stem = path.stem().string();
    const std::string token = output_quant_type_token(quant_type);
    std::string lower_stem = stem;
    std::transform(lower_stem.begin(), lower_stem.end(), lower_stem.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    const size_t bf16 = lower_stem.rfind("bf16");
    if (bf16 != std::string::npos) {
        stem.replace(bf16, 4, token);
    } else {
        stem += "-" + token;
    }
    path.replace_filename(stem + ".gguf");
    return path.string();
}

static bool same_model_path_string(const std::string & lhs, const std::string & rhs) {
    if (lhs.empty() || rhs.empty()) {
        return lhs == rhs;
    }
    return std::filesystem::path(lhs).lexically_normal() == std::filesystem::path(rhs).lexically_normal();
}

static bool output_model_is_auto_default(
        const std::string & input_model,
        const std::string & output_model,
        const std::string & current_quant_type) {
    if (output_model.empty()) {
        return true;
    }
    if (same_model_path_string(output_model, default_output_model_path(input_model, current_quant_type))) {
        return true;
    }
    for (const std::string & quant_type : bq::quant_type_choices()) {
        if (same_model_path_string(output_model, default_output_model_path(input_model, quant_type))) {
            return true;
        }
    }
    return false;
}

static void shell_remember_recipe(ShellState & state, const std::string & path, const bq::Recipe & recipe) {
    state.recipe = recipe;
    const std::string kld = !state.recipe.selector.kld.empty() ? state.recipe.selector.kld : state.recipe.evaluation.kld_base;
    const KldBaseInfo info = read_kld_base_info(kld);
    if (info.valid) {
        if (state.recipe.evaluation.kld_base.empty()) {
            state.recipe.evaluation.kld_base = kld;
        }
        if (state.recipe.selector.kld.empty()) {
            state.recipe.selector.kld = kld;
        }
    }
    state.have_recipe = true;
    state.last_recipe = path;
    state.precision_mode = recipe.target.precision_mode.empty() ? recipe.base.ftype : recipe.target.precision_mode;
    if (!recipe.io.input.empty()) {
        state.input_model = recipe.io.input;
    }
    if (!recipe.io.output.empty()) {
        state.output_model = recipe.io.output;
    }
    shell_sync_recipe_paths(state);
    state.run_dir = run_dir_for(recipe).string();
    state.variant = std::filesystem::path(path).stem().string();
    if (!state.last_recipe.empty()) {
        write_text_file(state.last_recipe, bq::dump_recipe_toml(state.recipe));
    }
}

static void shell_load_recipe(ShellState & state, const std::string & path) {
    bq::LoadedRecipe loaded = bq::load_recipe_file(path);
    shell_remember_recipe(state, path, loaded.recipe);
    state.status = "Configuration loaded";
}

static void shell_print_project_overview(const ShellState & state);


static void shell_record_metrics(ShellState & state) {
    const std::string title = "Project > Evaluation and Best Candidates > Record Metrics";
    state.project_path = shell_prompt_on_page(state, title, "project path", state.project_path.empty() ? default_project_path() : state.project_path);
    const std::string variant = shell_prompt_on_page(state, title, "variant id", state.variant);
    const std::string ppl = shell_prompt_on_page(state, title, "PPL", "");
    const std::string mean_kld = shell_prompt_on_page(state, title, "mean KLD", "");
    const std::string p99_kld = shell_prompt_on_page(state, title, "p99 KLD", "");
    const std::string p999_kld = shell_prompt_on_page(state, title, "p999 KLD", "");
    const std::string max_kld = shell_prompt_on_page(state, title, "max KLD", "");
    const std::string bpw = shell_prompt_on_page(state, title, "BPW", "");
    auto metric_value = [](const std::string & value) {
        return value.empty() ? std::string("null") : value;
    };
    const std::string metrics =
        "{\"ppl\":" + metric_value(ppl) +
        ",\"mean_kld\":" + metric_value(mean_kld) +
        ",\"p99_kld\":" + metric_value(p99_kld) +
        ",\"p999_kld\":" + metric_value(p999_kld) +
        ",\"max_kld\":" + metric_value(max_kld) +
        ",\"bpw\":" + metric_value(bpw) + "}";
    if (!variant.empty()) {
        bq::project_append_metric_event(state.project_path, variant, metrics);
        state.variant = variant;
        state.status = "Metrics recorded";
        std::cout << "recorded metrics for " << variant << "\n";
    }
}

static std::string default_metrics_export_path(const std::string & project_path) {
    if (project_path.empty()) {
        return std::string(PRODUCT_COMMAND) + ".metrics.jsonl";
    }
    std::filesystem::path path(project_path);
    return (path.parent_path() / (path.filename().string() + ".metrics.jsonl")).string();
}

static void shell_project_best(ShellState & state, std::string metrics_path = {}) {
    if (metrics_path.empty()) {
        const std::string title = "Project > Evaluation and Best Candidates";
        state.project_path = shell_prompt_on_page(state, title, "project path", state.project_path.empty() ? default_project_path() : state.project_path);
        metrics_path = shell_prompt_on_page(state, title, "metrics export path", default_metrics_export_path(state.project_path));
        std::ofstream out(metrics_path);
        if (!out) {
            throw std::runtime_error("failed to write " + metrics_path);
        }
        const uint64_t count = bq::project_export_metrics(state.project_path, out);
        std::cout << "exported " << count << " metric records to " << metrics_path << "\n";
        if (count == 0) {
            state.status = "No metrics to rank";
            return;
        }
    }

    std::vector<std::string> owned = { metrics_path, "--real-ppl-kld" };
    std::vector<char *> args;
    for (std::string & item : owned) {
        args.push_back(item.data());
    }
    try {
        const int rc = bq::best_main((int) args.size(), args.data());
        state.status = rc == 0 ? "Best-candidate report complete" : "Best-candidate report failed";
    } catch (const std::exception & e) {
        state.status = "Metrics needed for best report";
        std::cout << "Best-candidate report needs recorded PPL/KLD metrics first.\n";
        std::cout << e.what() << "\n";
        std::cout << "Use Record PPL/KLD metrics after candidate runs have real evaluation numbers.\n";
    }
}

static void shell_inspect_model(ShellState & state, std::string model = {}) {
    if (model.empty()) {
        shell_begin_page(state, "Inspect GGUF");
        model = prompt("GGUF path", !state.output_model.empty() ? state.output_model : state.input_model);
    }
    if (model.empty()) {
        return;
    }
    std::vector<std::string> owned = { model, "--tensors" };
    std::vector<char *> args;
    for (std::string & item : owned) {
        args.push_back(item.data());
    }
    bq::inspect_main((int) args.size(), args.data());
    state.status = "GGUF inspected";
}

static bool shell_handle_command(ShellState & state, std::string line, const bq::tui::TerminalCapabilities & caps);

static std::string default_config_path_for_project(const std::string & project_path) {
    if (project_path.empty()) {
        return default_config_path();
    }
    std::filesystem::path path(project_path);
    const std::filesystem::path parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    return (parent / (path.stem().string() + ".config.toml")).string();
}

static std::string safe_project_slug(std::string name) {
    std::string out;
    out.reserve(name.size());
    bool last_dash = false;
    for (unsigned char c : name) {
        if (std::isalnum(c)) {
            out.push_back((char) std::tolower(c));
            last_dash = false;
        } else if (!last_dash) {
            out.push_back('-');
            last_dash = true;
        }
    }
    while (!out.empty() && out.front() == '-') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out.empty() ? PRODUCT_COMMAND : out;
}

static std::filesystem::path resolve_project_path(const std::string & project_path, const std::string & maybe_relative) {
    std::filesystem::path path(maybe_relative);
    if (path.empty() || path.is_absolute()) {
        return path;
    }
    const std::filesystem::path project_dir = std::filesystem::path(project_path).parent_path();
    if (project_dir.empty()) {
        return path;
    }
    return project_dir / path;
}

static std::string extract_json_field_shell(const std::string & line, const std::string & key) {
    const std::string needle = "\"" + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return {};
    }
    pos += needle.size();
    while (pos < line.size() && std::isspace((unsigned char) line[pos])) {
        ++pos;
    }
    if (pos >= line.size()) {
        return {};
    }
    if (line[pos] != '"') {
        const size_t start = pos;
        while (pos < line.size() &&
               (std::isdigit((unsigned char) line[pos]) || line[pos] == '-' || line[pos] == '+' ||
                line[pos] == '.' || line[pos] == 'e' || line[pos] == 'E')) {
            ++pos;
        }
        return pos > start ? line.substr(start, pos - start) : std::string();
    }
    std::string out;
    for (++pos; pos < line.size(); ++pos) {
        const char c = line[pos];
        if (c == '"') {
            return out;
        }
        if (c == '\\' && pos + 1 < line.size()) {
            const char escaped = line[++pos];
            switch (escaped) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                default:   out.push_back(escaped); break;
            }
        } else {
            out.push_back(c);
        }
    }
    return {};
}

struct ProjectSummaryClean {
    std::string path;
    std::string name;
    std::string recipe;
    std::string input;
    std::string output;
    std::string variant;
    std::string run_dir;
    std::string bf16_reference;
    std::string kld_base;
    std::string corpus;
    std::string calibration_corpus;
    std::string imatrix;
    std::string last_run_event;
    std::string last_run_rc;
    std::string last_metric_variant;
    uint64_t records = 0;
    uint64_t candidates = 0;
    uint64_t candidate_manifests = 0;
    uint64_t runs = 0;
    uint64_t metrics = 0;
};

static ProjectSummaryClean read_project_summary_clean(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open project: " + path);
    }

    ProjectSummaryClean summary;
    summary.path = path;

    std::string line;
    while (std::getline(in, line)) {
        ++summary.records;
        const std::string event = extract_json_field_shell(line, "event");
        if (event == "project_init") {
            summary.name = extract_json_field_shell(line, "name");
            const std::string recipe = extract_json_field_shell(line, "recipe");
            if (!recipe.empty()) {
                summary.recipe = resolve_project_path(path, recipe).string();
            }
            const std::string input = extract_json_field_shell(line, "input");
            if (!input.empty()) {
                summary.input = input;
            }
            summary.bf16_reference = extract_json_field_shell(line, "bf16_reference");
            summary.kld_base = extract_json_field_shell(line, "kld_base");
            summary.corpus = extract_json_field_shell(line, "eval_corpus");
            if (summary.corpus.empty()) {
                summary.corpus = extract_json_field_shell(line, "corpus");
            }
            summary.calibration_corpus = extract_json_field_shell(line, "calibration_corpus");
            summary.imatrix = extract_json_field_shell(line, "imatrix");
        } else if (event == "candidate") {
            ++summary.candidates;
        } else if (event == "candidate_manifest_added") {
            ++summary.candidate_manifests;
        } else if (event.find("run_") == 0) {
            ++summary.runs;
            summary.last_run_event = event;
            summary.last_run_rc = extract_json_field_shell(line, "return_code");
            const std::string recipe = extract_json_field_shell(line, "recipe");
            if (!recipe.empty()) {
                summary.recipe = resolve_project_path(path, recipe).string();
            }
            const std::string output = extract_json_field_shell(line, "output");
            if (!output.empty()) {
                summary.output = output;
            }
            const std::string variant = extract_json_field_shell(line, "variant");
            if (!variant.empty()) {
                summary.variant = variant;
            }
            const std::string run_dir = extract_json_field_shell(line, "run_dir");
            if (!run_dir.empty()) {
                summary.run_dir = run_dir;
            }
        } else if (event == "metrics") {
            ++summary.metrics;
            summary.last_metric_variant = extract_json_field_shell(line, "variant");
        }
    }

    return summary;
}

static std::string display_project_value(const std::string & value) {
    return value.empty() ? "-" : value;
}

static std::string display_project_path_value(const std::string & value) {
    const std::string rendered = display_path(value);
    return rendered.empty() ? "-" : rendered;
}

static void print_project_summary_clean(const std::string & path, std::ostream & out) {
    const ProjectSummaryClean summary = read_project_summary_clean(path);
    out << "Project\n";
    out << "  Path              " << display_project_path_value(summary.path) << "\n";
    out << "  Name              " << display_project_value(summary.name) << "\n";
    out << "  Config            " << display_project_path_value(summary.recipe) << "\n";
    out << "  Model Input       " << display_project_path_value(summary.input) << "\n";
    out << "  Model Output      " << display_project_path_value(summary.output) << "\n";
    out << "  BF16 Base         " << display_project_path_value(summary.bf16_reference) << "\n";
    out << "  KLD Base          " << display_project_path_value(summary.kld_base) << "\n";
    out << "  PPL/KLD Corpus    " << display_project_path_value(summary.corpus) << "\n";
    out << "  Calib Corpus      " << display_project_path_value(summary.calibration_corpus) << "\n";
    out << "  Imatrix           " << display_project_path_value(summary.imatrix) << "\n";
    out << "  Last Variant      " << display_project_value(summary.variant) << "\n";
    out << "  Run Directory     " << display_project_path_value(summary.run_dir) << "\n";
    out << "\nHistory\n";
    out << "  Saved Records     " << summary.records << "\n";
    out << "  Candidates        " << summary.candidates;
    if (summary.candidate_manifests > 0) {
        out << " across " << summary.candidate_manifests << " loaded manifest";
        if (summary.candidate_manifests != 1) {
            out << "s";
        }
    }
    out << "\n";
    out << "  Run Updates       " << summary.runs << "\n";
    out << "  Metric Records    " << summary.metrics << "\n";
    if (!summary.last_run_event.empty()) {
        out << "  Last Run          " << summary.last_run_event;
        if (!summary.last_run_rc.empty()) {
            out << " rc=" << summary.last_run_rc;
        }
        out << "\n";
    }
    if (!summary.last_metric_variant.empty()) {
        out << "  Last Metrics      " << summary.last_metric_variant << "\n";
    }
}

static bool shell_restore_project_state(ShellState & state, const std::string & project_path) {
    std::ifstream in(project_path);
    if (!in) {
        throw std::runtime_error("failed to open project: " + project_path);
    }

    std::string recipe_path;
    std::string input;
    std::string output;
    std::string variant;
    std::string run_dir;
    std::string bf16_reference;
    std::string kld_base;
    std::string corpus;
    std::string calibration_corpus;
    std::string imatrix;

    std::string line;
    while (std::getline(in, line)) {
        const std::string event = extract_json_field_shell(line, "event");
        if (event == "project_init") {
            const std::string init_recipe = extract_json_field_shell(line, "recipe");
            if (!init_recipe.empty()) {
                recipe_path = resolve_project_path(project_path, init_recipe).string();
            }
            const std::string init_input = extract_json_field_shell(line, "input");
            if (!init_input.empty()) {
                input = init_input;
            }
            bf16_reference = extract_json_field_shell(line, "bf16_reference");
            kld_base = extract_json_field_shell(line, "kld_base");
            corpus = extract_json_field_shell(line, "eval_corpus");
            if (corpus.empty()) {
                corpus = extract_json_field_shell(line, "corpus");
            }
            calibration_corpus = extract_json_field_shell(line, "calibration_corpus");
            imatrix = extract_json_field_shell(line, "imatrix");
        } else if (event.find("run_") == 0) {
            const std::string run_recipe = extract_json_field_shell(line, "recipe");
            if (!run_recipe.empty()) {
                recipe_path = resolve_project_path(project_path, run_recipe).string();
            }
            const std::string run_output = extract_json_field_shell(line, "output");
            if (!run_output.empty()) {
                output = run_output;
            }
            const std::string run_variant = extract_json_field_shell(line, "variant");
            if (!run_variant.empty()) {
                variant = run_variant;
            }
            const std::string saved_run_dir = extract_json_field_shell(line, "run_dir");
            if (!saved_run_dir.empty()) {
                run_dir = saved_run_dir;
            }
        }
    }

    state.project_path = project_path;
    if (!recipe_path.empty() && std::filesystem::exists(recipe_path)) {
        shell_load_recipe(state, recipe_path);
    } else if (!recipe_path.empty()) {
        state.last_recipe = recipe_path;
    }
    if (!input.empty()) {
        state.input_model = input;
    }
    if (!output.empty()) {
        state.output_model = output;
    }
    if (!variant.empty()) {
        state.variant = variant;
    }
    if (!run_dir.empty()) {
        state.run_dir = run_dir;
    }
    if (!bf16_reference.empty()) {
        state.recipe.evaluation.bf16_reference = bf16_reference;
    }
    if (!kld_base.empty()) {
        state.recipe.evaluation.kld_base = kld_base;
        state.recipe.selector.kld = kld_base;
    }
    if (!corpus.empty()) {
        state.recipe.evaluation.corpus = corpus;
    }
    if (!calibration_corpus.empty()) {
        state.recipe.calibration.corpus = calibration_corpus;
    }
    if (!imatrix.empty()) {
        state.recipe.calibration.imatrix = imatrix;
    }
    if (!state.have_recipe) {
        state.recipe = bq::default_recipe();
        state.have_recipe = true;
    }
    shell_sync_recipe_paths(state);
    state.status = "Project loaded";
    return !recipe_path.empty() && std::filesystem::exists(recipe_path);
}

static void shell_print_project_overview(const ShellState & state) {
    const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
    const std::string kld = !state.recipe.selector.kld.empty() ? state.recipe.selector.kld : state.recipe.evaluation.kld_base;

    std::vector<std::string> project_lines;
    project_lines.push_back("Project       " + display_project_path_value(state.project_path));
    project_lines.push_back("Config        " + display_project_path_value(state.last_recipe));
    project_lines.push_back("Model Input   " + display_project_path_value(state.input_model));
    project_lines.push_back("Model Output  " + display_project_path_value(state.output_model));
    project_lines.push_back("Quant Type    " + display_project_value(state.precision_mode.empty() ? state.recipe.target.precision_mode : state.precision_mode));
    project_lines.push_back("KLD Base      " + display_project_path_value(kld));
    project_lines.push_back("PPL/KLD Data  " + display_project_path_value(state.recipe.evaluation.corpus));
    project_lines.push_back("Calib Data    " + display_project_path_value(state.recipe.calibration.corpus));
    project_lines.push_back("Imatrix       " + display_project_path_value(state.recipe.calibration.imatrix));

    if (!state.project_path.empty() && std::filesystem::exists(state.project_path)) {
        const ProjectSummaryClean summary = read_project_summary_clean(state.project_path);
        project_lines.push_back("Records       " + std::to_string(summary.records));
        project_lines.push_back("Candidates    " + std::to_string(summary.candidates));
        project_lines.push_back("Run Updates   " + std::to_string(summary.runs));
        project_lines.push_back("Metrics       " + std::to_string(summary.metrics));
        if (!summary.last_run_event.empty()) {
            std::string last_run = summary.last_run_event;
            if (!summary.last_run_rc.empty()) {
                last_run += " rc=" + summary.last_run_rc;
            }
            project_lines.push_back("Last Run      " + last_run);
        }
        if (!summary.last_metric_variant.empty()) {
            project_lines.push_back("Last Metrics  " + summary.last_metric_variant);
        }
    }

    bq::tui::BoxOptions box;
    box.title = "Project Status";
    box.wrap = false;
    bq::tui::print(std::cout, bq::tui::render_box(project_lines, box, caps));
}

static void shell_select_quant_type(ShellState & state) {
    std::vector<std::string> labels = bq::quant_type_choices();
    labels.push_back("Back");
    const int choice = shell_submenu_select(state, "Project > Options > Quant Type", labels);
    if (choice < 0 || choice >= (int) labels.size() - 1) {
        return;
    }

    const std::string previous_quant_type = shell_active_quant_type(state);
    const bool output_was_auto = output_model_is_auto_default(state.input_model, state.output_model, previous_quant_type);
    const bq::Recipe::Io io = state.recipe.io;
    const bq::Recipe::Evaluation evaluation = state.recipe.evaluation;
    const bq::Recipe::Calibration calibration = state.recipe.calibration;
    const bq::Recipe::TensorOverrides tensor_overrides = state.recipe.tensor_overrides;
    const bq::Recipe::Artifacts artifacts = state.recipe.artifacts;
    state.recipe = bq::default_recipe_for_quant_type(labels[choice]);
    state.recipe.io = io;
    state.recipe.evaluation = evaluation;
    state.recipe.calibration = calibration;
    state.recipe.tensor_overrides = tensor_overrides;
    state.recipe.artifacts = artifacts;
    if (!state.input_model.empty() && output_was_auto) {
        state.output_model = default_output_model_path(state.input_model, shell_active_quant_type(state));
        state.recipe.io.output = state.output_model;
    }
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = quant_type_uses_mxfp6(labels[choice])
        ? "MXFP6 experimental; feedback requested"
        : "Quant type selected";
}

static void shell_choose_model_files(ShellState & state) {
    const int choice = shell_submenu_select(state, "Project > Model Files", {
        "Select model input from disk",
        "Type model input path",
        "Set model output path",
        "Back",
    });
    if (choice < 0 || choice == 3) {
        state.status = "Model files unchanged";
        return;
    }
    shell_begin_page(state, "Project > Model Files");
    if (choice == 0) {
        const std::string old_input = state.input_model;
        const bool output_was_auto = output_model_is_auto_default(old_input, state.output_model, shell_active_quant_type(state));
        const std::string selected = choose_file_from_disk(state, "Model input", { ".gguf" }, state.input_model);
        if (!selected.empty()) {
            state.input_model = selected;
            if (output_was_auto) {
                state.output_model = default_output_model_path(state.input_model, shell_active_quant_type(state));
            }
        }
    } else if (choice == 1) {
        const std::string old_input = state.input_model;
        const bool output_was_auto = output_model_is_auto_default(old_input, state.output_model, shell_active_quant_type(state));
        state.input_model = prompt("Model Input", state.input_model);
        if (!state.input_model.empty() && output_was_auto) {
            state.output_model = default_output_model_path(state.input_model, shell_active_quant_type(state));
        }
    } else if (choice == 2) {
        state.output_model = prompt("Model Output filename/path", state.output_model);
    }
    shell_sync_recipe_paths(state);
    if (state.have_recipe) {
        shell_remember_recipe(state, state.last_recipe, state.recipe);
    }
    state.status = "Model files updated";
}

static void shell_default_bf16_reference_to_input(ShellState & state) {
    if (state.recipe.evaluation.bf16_reference.empty() && !state.input_model.empty()) {
        state.recipe.evaluation.bf16_reference = state.input_model;
    }
}

static void shell_configure_kld_files(ShellState & state) {
    shell_sync_recipe_paths(state);
    const int choice = shell_submenu_select(state, "Project > Options > Quality Inputs", {
        "Use existing KLD base",
        "Make KLD base from BF16",
        "Select calibration corpus and imatrix",
        "Make imatrix from BF16",
        "Use or create eval bundle",
        "Back",
    });
    if (choice < 0 || choice == 5) {
        state.status = "Quality inputs unchanged";
        return;
    }
    shell_begin_page(state, "Project > Options > Quality Inputs");
    if (choice == 0) {
        state.recipe.evaluation.kld_mode = "existing";
        shell_default_bf16_reference_to_input(state);
        const std::string corpus = choose_file_from_disk(state, "PPL/KLD corpus", { ".txt", ".raw", ".jsonl", ".json" }, state.recipe.evaluation.corpus);
        if (!corpus.empty()) {
            state.recipe.evaluation.corpus = corpus;
        }
        const std::string kld = choose_file_from_disk(state, "KLD base file", { ".kld", ".json", ".bin" }, state.recipe.evaluation.kld_base.empty() ? state.recipe.selector.kld : state.recipe.evaluation.kld_base);
        if (!kld.empty()) {
            state.recipe.evaluation.kld_base = kld;
        }
        const KldBaseInfo info = read_kld_base_info(state.recipe.evaluation.kld_base);
        if (info.valid) {
            std::cout << "KLD base: " << format_kld_info(info) << "\n";
        } else if (!state.recipe.evaluation.kld_base.empty()) {
            std::cout << "KLD base warning: " << format_kld_info(info) << "\n";
        }
        state.recipe.selector.kld = state.recipe.evaluation.kld_base;
    } else if (choice == 1 || choice == 4) {
        state.recipe.evaluation.kld_mode = choice == 1 ? "make_base" : "bundle";
        shell_default_bf16_reference_to_input(state);
        if (!state.recipe.evaluation.bf16_reference.empty()) {
            std::cout << "BF16 reference: " << display_path(state.recipe.evaluation.bf16_reference) << "\n";
        }
        const std::string corpus = choose_file_from_disk(state, "PPL/KLD corpus", { ".txt", ".raw", ".jsonl", ".json" }, state.recipe.evaluation.corpus);
        if (!corpus.empty()) {
            state.recipe.evaluation.corpus = corpus;
        }
        const std::string title = choice == 1 ?
            "Project > Options > Quality Inputs > Make KLD Base" :
            "Project > Options > Quality Inputs > Eval Bundle";
        state.recipe.evaluation.kld_base = shell_prompt_on_page(state, title, "KLD base output", default_kld_base_path(state.recipe));
        std::error_code ec;
        const auto free_bytes = std::filesystem::space(std::filesystem::path(state.recipe.evaluation.kld_base).parent_path().empty() ?
                std::filesystem::path(".") : std::filesystem::path(state.recipe.evaluation.kld_base).parent_path(), ec).available;
        if (!ec) {
            std::cout << "available disk near KLD output: " << mib_string(free_bytes) << " MiB\n";
        }
        state.recipe.evaluation.perplexity_bin = shell_prompt_on_page(state, title, "perplexity executable", state.recipe.evaluation.perplexity_bin);
        state.recipe.selector.kld = state.recipe.evaluation.kld_base;
        if (choice == 4) {
            state.recipe.evaluation.bundle = shell_prompt_on_page(state, title, "eval bundle path", state.recipe.evaluation.bundle);
        }
        const std::string command = kld_base_command_shell(state.recipe);
        if (!command.empty()) {
            std::cout << "\nKLD base command:\n  " << command << "\n";
        }
    } else if (choice == 2) {
        const std::string default_corpus = state.recipe.calibration.corpus.empty() ? state.recipe.evaluation.corpus : state.recipe.calibration.corpus;
        const std::string corpus = choose_file_from_disk(state, "Calibration corpus", { ".txt", ".raw", ".jsonl", ".json" }, default_corpus);
        if (!corpus.empty()) {
            state.recipe.calibration.corpus = corpus;
        }
        const std::string imatrix = choose_file_from_disk(state, "Imatrix file", { ".gguf", ".dat", ".imat", ".imatrix", ".bin" }, state.recipe.calibration.imatrix);
        if (!imatrix.empty()) {
            state.recipe.calibration.imatrix = imatrix;
        }
    } else if (choice == 3) {
        shell_default_bf16_reference_to_input(state);
        if (!state.recipe.evaluation.bf16_reference.empty()) {
            std::cout << "BF16 reference: " << display_path(state.recipe.evaluation.bf16_reference) << "\n";
        }
        const std::string default_corpus = state.recipe.calibration.corpus.empty() ? state.recipe.evaluation.corpus : state.recipe.calibration.corpus;
        const std::string corpus = choose_file_from_disk(state, "Calibration corpus", { ".txt", ".raw", ".jsonl", ".json" }, default_corpus);
        if (!corpus.empty()) {
            state.recipe.calibration.corpus = corpus;
        }
        const std::string title = "Project > Options > Quality Inputs > Make Imatrix";
        state.recipe.calibration.imatrix = shell_prompt_on_page(state, title, "imatrix output", default_imatrix_path(state.recipe));
        state.recipe.calibration.imatrix_bin = shell_prompt_on_page(state, title, "llama-imatrix executable", state.recipe.calibration.imatrix_bin);
        const std::string default_threads = std::to_string(state.recipe.base.threads > 0 ? state.recipe.base.threads : default_worker_threads());
        state.recipe.calibration.threads = shell_prompt_on_page(state, title, "CPU threads", state.recipe.calibration.threads.empty() ? default_threads : state.recipe.calibration.threads);
        state.recipe.calibration.threads_batch = shell_prompt_on_page(state, title, "CPU batch/collector threads", state.recipe.calibration.threads_batch.empty() ? state.recipe.calibration.threads : state.recipe.calibration.threads_batch);
        state.recipe.calibration.ctx_size = shell_prompt_on_page(state, title, "imatrix context size", state.recipe.calibration.ctx_size);
        state.recipe.calibration.batch_size = shell_prompt_on_page(state, title, "imatrix batch size", state.recipe.calibration.batch_size);
        state.recipe.calibration.ubatch_size = shell_prompt_on_page(state, title, "imatrix ubatch size", state.recipe.calibration.ubatch_size);
        state.recipe.calibration.n_gpu_layers = shell_prompt_on_page(state, title, "imatrix GPU layers (auto/all/N)", state.recipe.calibration.n_gpu_layers);
        state.recipe.calibration.extra_args = shell_prompt_on_page(state, title, "extra llama-imatrix args", state.recipe.calibration.extra_args);
        const std::string command = imatrix_command_shell(state.recipe);
        if (!command.empty()) {
            std::cout << "\nimatrix command:\n  " << command << "\n";
        }
    }
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "Quality inputs updated";
}

static void shell_configure_target_budget(ShellState & state) {
    if (state.input_model.empty()) {
        state.status = "Select a model before sizing";
        shell_choose_model_files(state);
        if (state.input_model.empty()) {
            return;
        }
    }

    bq::InspectSummary summary = bq::inspect_gguf(state.input_model);
    double params_b = estimate_params_b_from_summary(summary);
    if (params_b <= 0.0) {
        params_b = state.recipe.target.model_params_b;
    }

    shell_begin_page(state, "Project > Options > Target BPW / VRAM");
    std::cout << "Loaded model: " << display_path(state.input_model) << "\n";
    std::cout << "Tensor payload: " << mib_string(summary.tensor_bytes) << " MiB";
    if (params_b > 0.0) {
        std::cout << ", estimated parameters: " << params_b << "B";
    }
    std::cout << "\n";

    const std::string mode = canonical_quant_type(
        state.recipe.target.precision_mode.empty() ? bq::Recipe().target.precision_mode : state.recipe.target.precision_mode);
    const int vram_idx = shell_submenu_select(state, "Project > Options > Target BPW / VRAM > Fit", {
        "8 GB",
        "12 GB",
        "16 GB",
        "24 GB",
        "32 GB",
        "Type custom VRAM",
        "No fixed target",
        "Back",
    });
    if (vram_idx < 0 || vram_idx == 7) {
        state.status = "Target budget unchanged";
        return;
    }
    const std::vector<int> vram_values = { 8, 12, 16, 24, 32, 0, 0 };
    int vram_gb = vram_values[vram_idx];
    if (vram_idx == 5) {
        vram_gb = shell_prompt_int_on_page(
            state,
            "Project > Options > Target BPW / VRAM",
            "Custom VRAM target in GB",
            state.recipe.target.vram_gb > 0 ? state.recipe.target.vram_gb : 24);
    }

    const std::string title = "Project > Options > Target BPW / VRAM";
    const double target_bpw = shell_prompt_double_on_page(state, title, "Target final average BPW (0 = derive from VRAM)", state.recipe.target.target_bpw);
    const double kv_cache_gib = shell_prompt_double_on_page(state, title, "KV/cache reserve GiB", state.recipe.target.kv_cache_gib);
    const double activation_headroom_gib = shell_prompt_double_on_page(state, title, "activation/headroom reserve GiB", state.recipe.target.activation_headroom_gib);

    const bq::Recipe::Io io = state.recipe.io;
    const bq::Recipe::Base base = state.recipe.base;
    const bq::Recipe::Evaluation evaluation = state.recipe.evaluation;
    const bq::Recipe::Calibration calibration = state.recipe.calibration;
    const bq::Recipe::TensorOverrides tensor_overrides = state.recipe.tensor_overrides;
    const bq::Recipe::Artifacts artifacts = state.recipe.artifacts;
    apply_vram_target(state.recipe, mode, params_b, vram_gb, vram_gb, kv_cache_gib, activation_headroom_gib);
    state.recipe.io = io;
    state.recipe.base.threads = base.threads;
    state.recipe.base.allow_requantize = base.allow_requantize;
    state.recipe.base.leave_output_tensor = base.leave_output_tensor;
    state.recipe.base.pure = base.pure;
    state.recipe.base.copy_only = base.copy_only;
    state.recipe.evaluation = evaluation;
    state.recipe.calibration = calibration;
    state.recipe.tensor_overrides = tensor_overrides;
    state.recipe.artifacts = artifacts;
    if (target_bpw > 0.0) {
        state.recipe.target.target_bpw = target_bpw;
        state.recipe.target.weight_budget_gib = estimate_gib(params_b, target_bpw);
        if (mode == "NVFP4_MXFP6" && params_b > 0.0) {
            const double nv4_gib = estimate_gib(params_b, 4.5);
            const double mx6_gib = estimate_gib(params_b, 6.62);
            if (state.recipe.target.weight_budget_gib > nv4_gib) {
                const double extra_mb = std::max(0.0, (state.recipe.target.weight_budget_gib - nv4_gib) * 1024.0);
                state.recipe.rescue.budget_mb = std::to_string((int) std::round(extra_mb));
                state.recipe.nv4mx6.policy = state.recipe.target.weight_budget_gib >= mx6_gib ? "mx6_demote_nv4" : "nv4_promote_mx6";
                state.recipe.nv4mx6.mx6_penalty = state.recipe.target.weight_budget_gib >= mx6_gib ? "1.75" : "3.5";
            }
        } else if (mode == "NVFP4") {
            state.recipe.rescue.enabled = false;
            state.recipe.rescue.top.clear();
            state.recipe.rescue.report_top.clear();
            state.recipe.rescue.budget_mb.clear();
            state.recipe.rescue.bf16_budget_mb.clear();
            state.recipe.rescue.class_limit.clear();
            state.recipe.rescue.encoder_top.clear();
        }
        std::ostringstream note;
        note.setf(std::ios::fixed);
        note.precision(2);
        note << state.recipe.target.sizing_note
             << ", target final average=" << target_bpw << " bpw"
             << ", target weight size=" << state.recipe.target.weight_budget_gib << " GiB";
        state.recipe.target.sizing_note = note.str();
    }
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "Target budget updated";
}

static void shell_configure_nvfp4_46_policy(ShellState & state) {
    const std::string title = "Project > Options > NVFP4 4/6 Policy";
    state.recipe.nvfp4.preset = shell_prompt_on_page(state, title, "NVFP4 preset", state.recipe.nvfp4.preset);
    state.recipe.nvfp4.cfg = shell_prompt_on_page(state, title, "raw NVFP4 cfg override", state.recipe.nvfp4.cfg);
    const int mode = shell_submenu_select(state, "Project > Options > NVFP4 4/6 > Lane Selection", {
        "Adaptive",
        "Back",
    });
    if (mode < 0 || mode == 1) {
        state.status = "NVFP4 4/6 policy unchanged";
        return;
    }
    state.recipe.nvfp4.four_six.choose46 = "adaptive";
    const std::string params_title = "Project > Options > NVFP4 4/6 Policy > Parameters";
    state.recipe.nvfp4.four_six.refit_iters = shell_prompt_on_page(state, params_title, "4/6 refit iterations", state.recipe.nvfp4.four_six.refit_iters);
    state.recipe.nvfp4.four_six.compand = shell_prompt_on_page(state, params_title, "4/6 companding enabled (0/1)", state.recipe.nvfp4.four_six.compand);
    state.recipe.nvfp4.four_six.cap6 = shell_prompt_on_page(state, params_title, "6-bit lane cap", state.recipe.nvfp4.four_six.cap6);
    state.recipe.nvfp4.four_six.cap4 = shell_prompt_on_page(state, params_title, "4-bit lane cap", state.recipe.nvfp4.four_six.cap4);
    state.recipe.nvfp4.correction_denom = shell_prompt_on_page(state, params_title, "NVFP4 scale correction denominator", state.recipe.nvfp4.correction_denom);
    state.recipe.nvfp4.input_scale_policy = shell_prompt_on_page(state, params_title, "input scale policy", state.recipe.nvfp4.input_scale_policy);
    state.recipe.nvfp4.encoder.max_blocks = shell_prompt_on_page(state, params_title, "max sample blocks", state.recipe.nvfp4.encoder.max_blocks);
    state.recipe.nvfp4.encoder.threads = shell_prompt_on_page(state, params_title, "CPU worker threads", state.recipe.nvfp4.encoder.threads);
    state.recipe.nvfp4.calibration_families = split_type_csv(shell_prompt_on_page(
        state,
        params_title,
        "calibration/search families",
        join_type_csv(state.recipe.nvfp4.calibration_families)));
    state.recipe.nvfp4.scale_tie = shell_prompt_on_page(state, params_title, "scale/group tie policy", state.recipe.nvfp4.scale_tie);
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "NVFP4 4/6 policy updated";
}

static void shell_configure_native_techniques(ShellState & state) {
    while (true) {
        const auto has_candidate = [&](const char * token) {
            return recipe_has_token(state.recipe.stock_ftype.technique_candidates, token);
        };
        const auto has_family = [&](const char * token) {
            return recipe_has_token(state.recipe.nvfp4.calibration_families, token) || has_candidate(token);
        };
        const auto label = [](const std::string & text, bool enabled) {
            return text + (enabled ? " [on]" : " [off]");
        };
        const bool auto_search = has_candidate("auto_search");
        const bool no_quantize = has_candidate("no_quantize_choice");
        const bool awq = has_family("awq_lite");
        const bool smoothquant = has_family("smoothquant");
        const bool mse = has_family("mse_scale_sweep");
        const bool rsf = has_family("nvfp4_rsf");
        const bool kl = has_family("kl_div_sensitivity");
        const bool gradient = has_family("gradient_or_hessian_sidecar");
        const bool grouped = state.recipe.nvfp4.scale_tie == "qkv_gate_up_expert";
        const bool tensor_sweep =
            state.recipe.stock_ftype.sweep_tensor_policy ||
            state.recipe.stock_ftype.sweep_sensitive_tensors;

        const int choice = shell_submenu_select(state, "Project > Options > Native Technique Families", {
            "Review current native families",
            "Use core PPL/KLD search",
            "Use full local GGUF search set",
            label("Auto-search candidate seed", auto_search),
            label("Tensor-policy sweep", tensor_sweep),
            label("Add BF16/no-quantize tensor choices", no_quantize),
            label("AWQ candidates", awq),
            label("SmoothQuant input-scale candidates", smoothquant),
            label("Scale/cap sweep candidates", mse),
            label("RSF variants", rsf),
            label("KL-divergence sensitivity scorer", kl),
            label("Gradient/Hessian sidecar scorer", gradient),
            label("Tie Q/K/V, gate/up, and experts as groups", grouped),
            "Manual native lists",
            "Back",
        });
        if (choice < 0 || choice == 14) {
            shell_remember_recipe(state, state.last_recipe, state.recipe);
            state.status = "Native technique families updated";
            return;
        }

        if (choice == 0) {
            shell_begin_page(state, "Project > Options > Native Technique Families");
            const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
            std::vector<std::string> lines;
            lines.push_back("Policy set         " + (state.recipe.quantizer.policy_set.empty() ? std::string("native-full") : state.recipe.quantizer.policy_set));
            lines.push_back("Technique list     " + join_type_csv(state.recipe.stock_ftype.technique_candidates));
            lines.push_back("Calibration list   " + join_type_csv(state.recipe.nvfp4.calibration_families));
            lines.push_back("Tensor sweep       " + std::string(tensor_sweep ? "on" : "off"));
            lines.push_back("Grouped decisions  " + std::string(grouped ? "on" : "off"));
            bq::tui::BoxOptions box;
            box.title = "Current Native Search";
            bq::tui::print(std::cout, bq::tui::render_box(lines, box, caps));
            state.status = "Native technique families unchanged";
            shell_pause(caps);
        } else if (choice == 1) {
            state.recipe.quantizer.policy_set = "native-core";
            state.recipe.stock_ftype.technique_candidates = { "ptq", "kld-best" };
            state.recipe.nvfp4.calibration_families = { "max", "kld_best" };
            state.recipe.nvfp4.scale_tie = "none";
            state.recipe.stock_ftype.sweep_tensor_policy = true;
            state.recipe.stock_ftype.sweep_sensitive_tensors = true;
        } else if (choice == 2) {
            state.recipe.quantizer.policy_set = "native-full";
            state.recipe.stock_ftype.technique_candidates = {
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
            state.recipe.nvfp4.calibration_families = {
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
            state.recipe.nvfp4.scale_tie = "qkv_gate_up_expert";
            state.recipe.stock_ftype.sweep_tensor_policy = true;
            state.recipe.stock_ftype.sweep_sensitive_tensors = true;
        } else if (choice == 3) {
            state.recipe.quantizer.policy_set = "manual";
            set_type_tokens(state.recipe.stock_ftype.technique_candidates, { "auto_search" }, !auto_search);
        } else if (choice == 4) {
            state.recipe.quantizer.policy_set = "manual";
            state.recipe.stock_ftype.sweep_tensor_policy = !tensor_sweep;
            state.recipe.stock_ftype.sweep_sensitive_tensors = !tensor_sweep;
        } else if (choice == 5) {
            state.recipe.quantizer.policy_set = "manual";
            set_type_tokens(state.recipe.stock_ftype.technique_candidates, { "no_quantize_choice" }, !no_quantize);
            if (!no_quantize) {
                append_unique_type(state.recipe.stock_ftype.token_embedding_candidates, "BF16");
                append_unique_type(state.recipe.stock_ftype.output_tensor_candidates, "BF16");
                state.recipe.stock_ftype.sweep_tensor_policy = true;
                state.recipe.stock_ftype.sweep_sensitive_tensors = true;
            }
        } else if (choice == 6) {
            state.recipe.quantizer.policy_set = "manual";
            set_type_tokens(state.recipe.stock_ftype.technique_candidates, { "awq_lite", "awq_clip", "awq_full" }, !awq);
            set_type_tokens(state.recipe.nvfp4.calibration_families, { "awq_lite", "awq_clip", "awq_full" }, !awq);
        } else if (choice == 7) {
            state.recipe.quantizer.policy_set = "manual";
            set_type_tokens(state.recipe.stock_ftype.technique_candidates, { "smoothquant" }, !smoothquant);
            set_type_tokens(state.recipe.nvfp4.calibration_families, { "smoothquant" }, !smoothquant);
        } else if (choice == 8) {
            state.recipe.quantizer.policy_set = "manual";
            set_type_tokens(state.recipe.stock_ftype.technique_candidates, { "mse_scale_sweep" }, !mse);
            set_type_tokens(state.recipe.nvfp4.calibration_families, { "mse_scale_sweep" }, !mse);
        } else if (choice == 9) {
            state.recipe.quantizer.policy_set = "manual";
            set_type_tokens(state.recipe.stock_ftype.technique_candidates, { "nvfp4_rsf" }, !rsf);
            set_type_tokens(state.recipe.nvfp4.calibration_families, { "nvfp4_rsf" }, !rsf);
            if (!rsf) {
                state.recipe.selector.require_runtime_cache = true;
                if (state.recipe.selector.eval_top.empty()) {
                    state.recipe.selector.eval_top = "6";
                }
            }
        } else if (choice == 10) {
            state.recipe.quantizer.policy_set = "manual";
            set_type_tokens(state.recipe.stock_ftype.technique_candidates, { "kl_div_sensitivity" }, !kl);
            set_type_tokens(state.recipe.nvfp4.calibration_families, { "kl_div_sensitivity" }, !kl);
            if (!kl) {
                state.recipe.selector.require_runtime_cache = true;
            }
        } else if (choice == 11) {
            state.recipe.quantizer.policy_set = "manual";
            set_type_tokens(state.recipe.stock_ftype.technique_candidates, { "gradient_or_hessian_sidecar" }, !gradient);
            set_type_tokens(state.recipe.nvfp4.calibration_families, { "gradient_or_hessian_sidecar" }, !gradient);
        } else if (choice == 12) {
            state.recipe.quantizer.policy_set = "manual";
            state.recipe.nvfp4.scale_tie = grouped ? "none" : "qkv_gate_up_expert";
        } else if (choice == 13) {
            state.recipe.quantizer.policy_set = "manual";
            const std::string title = "Project > Options > Native Technique Families > Manual";
            state.recipe.stock_ftype.technique_candidates = split_type_csv(shell_prompt_on_page(
                state,
                title,
                "technique candidates",
                join_type_csv(state.recipe.stock_ftype.technique_candidates)));
            state.recipe.nvfp4.calibration_families = split_type_csv(shell_prompt_on_page(
                state,
                title,
                "calibration/search families",
                join_type_csv(state.recipe.nvfp4.calibration_families)));
            state.recipe.nvfp4.scale_tie = shell_prompt_on_page(state, title, "scale/group tie policy", state.recipe.nvfp4.scale_tie);
        }
    }
}

static void shell_configure_candidate_search(ShellState & state) {
    const std::string current_effort = state.recipe.selector.effort.empty() ? std::string("unset") : state.recipe.selector.effort;
    const std::string current_mode = state.recipe.quantizer.mode.empty() ? std::string("unset") : state.recipe.quantizer.mode;
    const int effort = shell_submenu_select(state, "Candidate Search - current " + current_mode + " / " + current_effort, {
        "Select native technique families",
        "Keep current search settings",
        "Fast - compact tensor-policy search",
        "Normal - real KLD-first tensor-policy search",
        "Deep - broad KLD-first tensor-policy search",
        "Advanced low-level knobs",
        "Back",
    });
    if (effort < 0 || effort == 6) {
        return;
    }
    if (effort == 0) {
        shell_configure_native_techniques(state);
        return;
    }
    if (effort == 1) {
        state.status = "Candidate search unchanged";
        return;
    }
    if (effort == 2) {
        apply_search_effort_preset(state.recipe, "fast");
    } else if (effort == 3) {
        apply_search_effort_preset(state.recipe, "normal");
    } else if (effort == 4) {
        apply_search_effort_preset(state.recipe, "deep");
    }
    if (effort > 1 && effort < 5) {
        state.status = state.recipe.selector.effort + " search preset selected";
        shell_remember_recipe(state, state.last_recipe, state.recipe);
        return;
    }

    const std::string title = "Project > Options > Candidate Search";
    state.recipe.quantizer.enabled = false;
    state.recipe.selector.kld = shell_prompt_on_page(state, title, "KLD base file", state.recipe.selector.kld.empty() ? state.recipe.evaluation.kld_base : state.recipe.selector.kld);
    state.recipe.selector.ledger = shell_prompt_on_page(state, title, "selector evidence ledger", state.recipe.selector.ledger);
    state.recipe.selector.checkpoint_model = shell_prompt_on_page(state, title, "candidate search checkpoint GGUF", state.recipe.selector.checkpoint_model);
    state.recipe.selector.cache_dir = shell_prompt_on_page(state, title, "checkpoint cache directory", state.recipe.selector.cache_dir);
    state.recipe.selector.skip_file = shell_prompt_on_page(state, title, "skip remaining tuning request file", state.recipe.selector.skip_file);
    state.recipe.selector.effort = shell_prompt_on_page(state, title, "effort label", state.recipe.selector.effort);
    state.recipe.selector.stagea_sample_blocks = shell_prompt_on_page(state, title, "stage A sample blocks", state.recipe.selector.stagea_sample_blocks);
    state.recipe.selector.stagea_max_policies = shell_prompt_on_page(state, title, "stage A max policies", state.recipe.selector.stagea_max_policies);
    state.recipe.selector.refine_top = shell_prompt_on_page(state, title, "refine top policies", state.recipe.selector.refine_top);
    state.recipe.selector.refine_budget = shell_prompt_on_page(state, title, "refine budget", state.recipe.selector.refine_budget);
    state.recipe.selector.survey_top = shell_prompt_on_page(state, title, "survey top tensors", state.recipe.selector.survey_top);
    state.recipe.selector.survey_sample_blocks = shell_prompt_on_page(state, title, "survey sample blocks", state.recipe.selector.survey_sample_blocks);
    state.recipe.selector.max_tensors = shell_prompt_on_page(state, title, "max tensors to search", state.recipe.selector.max_tensors);
    state.recipe.selector.eval_top = shell_prompt_on_page(state, title, "full PPL/KLD candidates", state.recipe.selector.eval_top);
    state.recipe.selector.n_seq = shell_prompt_on_page(state, title, "eval sequences", state.recipe.selector.n_seq);
    state.recipe.selector.policy_threads = shell_prompt_on_page(state, title, "policy search threads", state.recipe.selector.policy_threads);
    state.recipe.selector.threads = shell_prompt_on_page(state, title, "selector threads", state.recipe.selector.threads);
    state.recipe.selector.kld_threads = shell_prompt_on_page(state, title, "PPL/KLD host reduction threads", state.recipe.selector.kld_threads.empty() ? state.recipe.selector.threads : state.recipe.selector.kld_threads);
    state.recipe.selector.keep_checkpoint = shell_prompt_bool_on_page(state, title, "keep generated checkpoint", state.recipe.selector.keep_checkpoint);
    state.recipe.selector.require_runtime_cache = shell_prompt_bool_on_page(state, title, "require resident tensor cache", state.recipe.selector.require_runtime_cache);
    state.recipe.selector.trace = shell_prompt_bool_on_page(state, title, "write selector trace", state.recipe.selector.trace);
    state.recipe.selector.only = shell_prompt_bool_on_page(state, title, "selector only, no output write", state.recipe.selector.only);
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "Candidate search updated";
}

static void shell_configure_quality_gates(ShellState & state) {
    const std::string title = "Project > Options > PPL/KLD Scoring";
    state.recipe.selector.ranking.kld_penalty = shell_prompt_on_page(state, title, "mean KLD penalty", state.recipe.selector.ranking.kld_penalty);
    state.recipe.selector.ranking.p99_penalty = shell_prompt_on_page(state, title, "p99 KLD penalty", state.recipe.selector.ranking.p99_penalty);
    state.recipe.selector.ranking.p999_penalty = shell_prompt_on_page(state, title, "p999 KLD penalty", state.recipe.selector.ranking.p999_penalty);
    state.recipe.selector.ranking.max_kld_penalty = shell_prompt_on_page(state, title, "max KLD penalty", state.recipe.selector.ranking.max_kld_penalty);
    state.recipe.selector.ranking.kld_threshold = shell_prompt_on_page(state, title, "mean KLD max delta", state.recipe.selector.ranking.kld_threshold);
    state.recipe.selector.ranking.p99_threshold = shell_prompt_on_page(state, title, "p99 KLD max delta", state.recipe.selector.ranking.p99_threshold);
    state.recipe.selector.ranking.p999_threshold = shell_prompt_on_page(state, title, "p999 KLD max delta", state.recipe.selector.ranking.p999_threshold);
    state.recipe.selector.ranking.max_kld_threshold = shell_prompt_on_page(state, title, "max KLD max delta", state.recipe.selector.ranking.max_kld_threshold);
    state.recipe.selector.ranking.kld_hard_gate = shell_prompt_bool_on_page(state, title, "hard gate mean KLD", state.recipe.selector.ranking.kld_hard_gate);
    state.recipe.selector.ranking.p99_hard_gate = shell_prompt_bool_on_page(state, title, "hard gate p99 KLD", state.recipe.selector.ranking.p99_hard_gate);
    state.recipe.selector.ranking.p999_hard_gate = shell_prompt_bool_on_page(state, title, "hard gate p999 KLD", state.recipe.selector.ranking.p999_hard_gate);
    state.recipe.selector.ranking.max_kld_hard_gate = shell_prompt_bool_on_page(state, title, "hard gate max KLD", state.recipe.selector.ranking.max_kld_hard_gate);
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "PPL/KLD scoring updated";
}

static void shell_configure_edit_existing_gguf(ShellState & state) {
    const std::string title = "Project > Options > Edit Existing GGUF";
    state.recipe.rescue.enabled = shell_prompt_bool_on_page(state, title, "enable Edit Existing GGUF pass", state.recipe.rescue.enabled);
    state.recipe.rescue.type = shell_prompt_on_page(state, title, "edit quant type", state.recipe.rescue.type);
    state.recipe.rescue.top = shell_prompt_on_page(state, title, "tensors to edit", state.recipe.rescue.top);
    state.recipe.rescue.report_top = shell_prompt_on_page(state, title, "report top tensors", state.recipe.rescue.report_top);
    state.recipe.rescue.budget_mb = shell_prompt_on_page(state, title, "edit budget MiB", state.recipe.rescue.budget_mb);
    state.recipe.rescue.bf16_budget_mb = shell_prompt_on_page(state, title, "BF16 edit budget MiB", state.recipe.rescue.bf16_budget_mb);
    state.recipe.rescue.class_limit = shell_prompt_on_page(state, title, "per-class edit limit", state.recipe.rescue.class_limit);
    state.recipe.rescue.encoder_top = shell_prompt_on_page(state, title, "encoder retest top", state.recipe.rescue.encoder_top);
    state.recipe.rescue.sample_blocks = shell_prompt_on_page(state, title, "edit sample blocks", state.recipe.rescue.sample_blocks);
    state.recipe.rescue.coarse_max_blocks = shell_prompt_on_page(state, title, "edit coarse max blocks", state.recipe.rescue.coarse_max_blocks);
    state.recipe.rescue.refine_max_blocks = shell_prompt_on_page(state, title, "edit refine max blocks", state.recipe.rescue.refine_max_blocks);
    state.recipe.rescue.guard_max_blocks = shell_prompt_on_page(state, title, "edit guard max blocks", state.recipe.rescue.guard_max_blocks);
    state.recipe.rescue.report = shell_prompt_on_page(state, title, "edit report file", state.recipe.rescue.report);
    state.recipe.rescue.tensor_types = shell_prompt_on_page(state, title, "edit tensor type overrides", state.recipe.rescue.tensor_types);
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "Edit Existing GGUF updated";
}

static void shell_configure_mxfp6_scale_refine(ShellState & state) {
    const std::string title = "Project > Options > MXFP6 Scale Refinement";
    state.recipe.mxfp6.tensor_scale = shell_prompt_on_page(state, title, "MXFP6 tensor scale mode", state.recipe.mxfp6.tensor_scale);
    state.recipe.mxfp6.min_savings_bytes = shell_prompt_on_page(state, title, "MXFP6 min savings bytes", state.recipe.mxfp6.min_savings_bytes);
    state.recipe.mxfp6.selector_scale_top = shell_prompt_on_page(state, title, "scale-refine tensor count", state.recipe.mxfp6.selector_scale_top);
    state.recipe.mxfp6.selector_scale_candidates = shell_prompt_on_page(state, title, "scale candidates", state.recipe.mxfp6.selector_scale_candidates);
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "MXFP6 scale refinement updated";
}

static void shell_configure_standard_quantize_options(ShellState & state) {
    const int choice = shell_submenu_select(state, "Project > Options > Standard Quantize Options", {
        "Keep defaults",
        "Select standard llama-quantize options",
        "Back",
    });
    if (choice != 1) {
        return;
    }

    const std::string title = "Project > Options > Standard Quantize Options";
    state.recipe.base.output_tensor_type = sanitize_tensor_type_token(shell_prompt_on_page(state, title, "output.weight tensor type", state.recipe.base.output_tensor_type));
    state.recipe.base.token_embedding_type = sanitize_tensor_type_token(shell_prompt_on_page(state, title, "token embedding tensor type", state.recipe.base.token_embedding_type));
    state.recipe.base.mtp_tensor_type = sanitize_tensor_type_token(shell_prompt_on_page(state, title, "MTP/NextN tensor type (blank follows profile/ftype default)", state.recipe.base.mtp_tensor_type));
    state.recipe.base.leave_output_tensor = shell_prompt_bool_on_page(state, title, "leave output.weight unquantized", state.recipe.base.leave_output_tensor);
    state.recipe.stock_ftype.sweep_tensor_policy = shell_prompt_bool_on_page(
        state,
        title,
        "measure embeddings/output as candidates",
        state.recipe.stock_ftype.sweep_tensor_policy || state.recipe.stock_ftype.sweep_sensitive_tensors);
    state.recipe.stock_ftype.sweep_sensitive_tensors = state.recipe.stock_ftype.sweep_tensor_policy;
    state.recipe.stock_ftype.token_embedding_candidates = split_type_csv(shell_prompt_on_page(
        state,
        title,
        "token embedding candidate types",
        join_type_csv(state.recipe.stock_ftype.token_embedding_candidates)));
    state.recipe.stock_ftype.output_tensor_candidates = split_type_csv(shell_prompt_on_page(
        state,
        title,
        "output tensor candidate types",
        join_type_csv(state.recipe.stock_ftype.output_tensor_candidates)));
    state.recipe.stock_ftype.min_quant_savings_mib = shell_prompt_double_on_page(state, title, "minimum tensor savings MiB before quantizing", state.recipe.stock_ftype.min_quant_savings_mib);
    state.recipe.stock_ftype.technique_candidates = split_type_csv(shell_prompt_on_page(
        state,
        title,
        "technique candidates",
        join_type_csv(state.recipe.stock_ftype.technique_candidates)));
    state.recipe.base.allow_requantize = shell_prompt_bool_on_page(state, title, "allow requantizing already-quantized tensors", state.recipe.base.allow_requantize);
    state.recipe.base.pure = shell_prompt_bool_on_page(state, title, "pure mode, no built-in type mixtures", state.recipe.base.pure);
    state.recipe.base.copy_only = shell_prompt_bool_on_page(state, title, "copy only, do not quantize", state.recipe.base.copy_only);
    state.recipe.io.keep_split = shell_prompt_bool_on_page(state, title, "keep input split/shard layout", state.recipe.io.keep_split);
    state.recipe.model.prune_layers = shell_prompt_on_page(state, title, "prune layers CSV", state.recipe.model.prune_layers);

    const std::string kv_override = shell_prompt_on_page(state, title, "add metadata override KEY=TYPE:VALUE", "");
    if (!kv_override.empty()) {
        state.recipe.metadata.overrides.push_back(kv_override);
    }
    const std::string tensor_file = shell_prompt_on_page(state, title, "add tensor type override file", "");
    if (!tensor_file.empty()) {
        state.recipe.tensor_overrides.files.push_back(tensor_file);
    }
    const std::string tensor_entry = shell_prompt_on_page(state, title, "add tensor override tensor=type", "");
    if (!tensor_entry.empty()) {
        state.recipe.tensor_overrides.entries.push_back(tensor_entry);
    }

    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "Standard options updated";
}

static void shell_tensor_rules(ShellState & state) {
    while (true) {
        const int choice = shell_submenu_select(state, "Project > Options > Tensor Rules", {
            "Browse model tensors",
            "Add tensor=type override",
            "Add tensor override file",
            "Set Edit Existing GGUF tensor type list",
            "Back",
        });
        if (choice < 0) {
            return;
        }
        if (choice == 0) {
            shell_inspect_model(state, state.input_model);
        } else if (choice == 1) {
            shell_begin_page(state, "Project > Options > Tensor Rules > Add Override");
            const std::string entry = prompt("tensor=type");
            if (!entry.empty()) {
                state.recipe.tensor_overrides.entries.push_back(entry);
            }
        } else if (choice == 2) {
            shell_begin_page(state, "Project > Options > Tensor Rules > Add Override File");
            const std::string file = prompt("tensor override file");
            if (!file.empty()) {
                state.recipe.tensor_overrides.files.push_back(file);
            }
        } else if (choice == 3) {
            shell_begin_page(state, "Project > Options > Tensor Rules > Edit Existing GGUF");
            state.recipe.rescue.tensor_types = prompt("edit tensor types", state.recipe.rescue.tensor_types);
        } else {
            break;
        }
    }
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "Tensor rules updated";
}

static void shell_save_active_config(ShellState & state) {
    shell_begin_page(state, "Project > Options > Save Configuration");
    shell_sync_recipe_paths(state);
    if (state.last_recipe.empty()) {
        state.last_recipe = default_config_path_for_project(state.project_path);
    }
    state.last_recipe = prompt("save configuration as", state.last_recipe);
    state.recipe.artifacts.run_dir = prompt("run directory", state.recipe.artifacts.run_dir.empty() && !state.output_model.empty() ? state.output_model + ".run" : state.recipe.artifacts.run_dir);
    write_text_file(state.last_recipe, bq::dump_recipe_toml(state.recipe));
    shell_remember_recipe(state, state.last_recipe, state.recipe);
    state.status = "Configuration saved";
    std::cout << "wrote " << state.last_recipe << "\n";
}

static void shell_options_menu(ShellState & state) {
    if (!state.have_recipe) {
        state.recipe = bq::default_recipe();
        state.have_recipe = true;
    }
    while (true) {
        std::string slash_command;
        const std::vector<bq::tui::MenuOption> options = shell_options_menu_options(state);
        const int action = shell_menu_select(state, "Project > Options", options, slash_command);
        const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
        if (action == -2) {
            try {
                shell_handle_command(state, slash_command, caps);
            } catch (const PromptCancelled &) {
                state.status = "Cancelled";
                continue;
            }
            if (state.quit) {
                return;
            }
            shell_pause(caps);
            continue;
        }
        try {
            const std::string command = action >= 0 && action < (int) options.size() ? options[action].command : "back";
            if (command == "quant") {
                shell_select_quant_type(state);
            } else if (command == "models") {
                shell_choose_model_files(state);
            } else if (command == "quality") {
                shell_configure_kld_files(state);
            } else if (command == "budget") {
                shell_configure_target_budget(state);
            } else if (command == "nvfp4") {
                shell_configure_nvfp4_46_policy(state);
            } else if (command == "candidate-search") {
                shell_configure_candidate_search(state);
            } else if (command == "gates") {
                shell_configure_quality_gates(state);
            } else if (command == "edit") {
                shell_configure_edit_existing_gguf(state);
            } else if (command == "mxfp6") {
                shell_configure_mxfp6_scale_refine(state);
            } else if (command == "tensor-rules") {
                shell_tensor_rules(state);
            } else if (command == "standard") {
                shell_configure_standard_quantize_options(state);
            } else if (command == "save") {
                shell_save_active_config(state);
            } else {
                break;
            }
        } catch (const PromptCancelled &) {
            state.status = "Cancelled";
            continue;
        }
        if (state.quit) {
            return;
        }
        if (state.status.find("unchanged") != std::string::npos || state.status.find("cancelled") != std::string::npos || state.status == "Cancelled") {
            continue;
        }
        shell_pause(caps);
    }
}

static int shell_launch_background_run(
        const bq::Recipe & recipe,
        const std::string & recipe_path,
        const std::string & project_path,
        const std::string & variant,
        const std::filesystem::path & run_dir,
        const std::filesystem::path & log_path,
        const std::filesystem::path & pid_file) {
    std::filesystem::create_directories(run_dir);
    bq::Recipe launch_recipe = recipe;
    if (launch_recipe.evaluation.kld_base.empty() && !launch_recipe.selector.kld.empty()) {
        launch_recipe.evaluation.kld_base = launch_recipe.selector.kld;
    }
    const auto q = [](const std::string & value) {
        return shellish_args({ value });
    };

    std::vector<std::string> args = {
        shell_executable_path(),
        "run",
        recipe_path,
        "--yes",
    };
    if (!project_path.empty()) {
        args.push_back("--project");
        args.push_back(project_path);
    }
    if (!variant.empty()) {
        args.push_back("--variant");
        args.push_back(variant);
    }
    const std::string run_command = shellish_args(args);

    const std::string kld = !launch_recipe.selector.kld.empty() ? launch_recipe.selector.kld : launch_recipe.evaluation.kld_base;
    const std::string kld_command =
        (launch_recipe.evaluation.kld_mode == "make_base" || launch_recipe.evaluation.kld_mode == "bundle") ?
            kld_base_command_shell(launch_recipe) : std::string();
    const std::string imatrix_command = imatrix_command_shell(launch_recipe);

    const std::filesystem::path script_path = run_dir / "pipeline.command.sh";
    std::ostringstream script;
    script << "#!/usr/bin/env bash\n";
    script << "set -euo pipefail\n";
    script << "RUN_JSONL=" << q((run_dir / "run.jsonl").string()) << "\n";
    script << "trap 'rc=$?; if [ \"$rc\" -ne 0 ]; then printf \"{\\\"event\\\":\\\"finished\\\",\\\"return_code\\\":%s}\\n\" \"$rc\" >> \"$RUN_JSONL\"; fi' EXIT\n";
    script << "mkdir -p " << q(run_dir.string()) << "\n";
    script << "printf '%s\\n' " << q("[pipeline] starting") << "\n";
    script << "printf '{\"event\":\"pipeline_started\"}\\n' >> \"$RUN_JSONL\"\n";
    if (!kld_command.empty() && !kld.empty()) {
        const std::filesystem::path parent = std::filesystem::path(kld).parent_path();
        if (!parent.empty()) {
            script << "mkdir -p " << q(parent.string()) << "\n";
        }
        script << "if [ ! -s " << q(kld) << " ]; then\n";
        script << "  printf '%s\\n' " << q("[pipeline] building KLD base: " + display_path(kld)) << "\n";
        script << "  " << kld_command << "\n";
        script << "else\n";
        script << "  printf '%s\\n' " << q("[pipeline] reusing KLD base: " + display_path(kld)) << "\n";
        script << "fi\n";
    }
    if (!imatrix_command.empty() && !launch_recipe.calibration.imatrix.empty()) {
        const std::filesystem::path parent = std::filesystem::path(launch_recipe.calibration.imatrix).parent_path();
        if (!parent.empty()) {
            script << "mkdir -p " << q(parent.string()) << "\n";
        }
        script << "if [ ! -s " << q(launch_recipe.calibration.imatrix) << " ]; then\n";
        script << "  printf '%s\\n' " << q("[pipeline] building imatrix: " + display_path(launch_recipe.calibration.imatrix)) << "\n";
        script << "  " << imatrix_command << "\n";
        script << "else\n";
        script << "  printf '%s\\n' " << q("[pipeline] reusing imatrix: " + display_path(launch_recipe.calibration.imatrix)) << "\n";
        script << "fi\n";
    }
    script << "printf '%s\\n' " << q("[pipeline] starting quantization") << "\n";
    script << run_command << "\n";
    write_text_file(script_path, script.str());

    const std::string command =
        "setsid nohup " + shellish_args({ "bash", script_path.string() }) +
        " > " + shellish_args({ log_path.string() }) +
        " 2>&1 < /dev/null & echo $! > " + shellish_args({ pid_file.string() });
    write_text_file(run_dir / "launch.command", command + "\n");
    return std::system(command.c_str());
}

static std::filesystem::path shell_skip_tuning_request_file(const std::filesystem::path & run_dir) {
    return run_dir / "skip-remaining-tuning.request";
}

static void shell_quantizing_page(
        ShellState & state,
        const std::filesystem::path & run_dir,
        const std::filesystem::path & log_path,
        const std::filesystem::path & pid_file) {
    const bool tty = stdin_is_tty();
    while (true) {
        const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
        const size_t log_count = caps.is_tty && caps.rows > 0 ?
            std::max<size_t>(4, std::min<size_t>(14, (size_t) caps.rows > 26 ? (size_t) caps.rows - 22 : 4)) :
            12;
        const std::vector<std::string> log_lines = shell_recent_meaningful_log_lines(log_path, log_count);
        std::string rc;
        const bool finished = shell_run_finished(run_dir, &rc);
        const bool alive = shell_pid_alive(pid_file);
        const std::string run_phase = shell_infer_run_phase(log_lines);
        const std::string run_progress = shell_latest_progress_line(log_lines);
        const std::filesystem::path skip_file = shell_skip_tuning_request_file(run_dir);
        const bool skip_requested = std::filesystem::exists(skip_file);
        ShellState view = state;
        view.status = finished ? (rc == "0" ? "Run finished" : "Run failed") : (alive ? "Quantizing" : "Starting");
        view.run_dir = run_dir.string();
        view.run_phase = run_phase;
        view.run_progress = run_progress;
        shell_clear(caps);
        shell_print_status(view, caps);
        bq::tui::print(std::cout, bq::tui::render_section_header(
            "Active Run",
            finished ? "Enter, B, Q, or Esc returns to the project." : "B, Q, or Esc detaches from this screen. The quantizer keeps running in the background.",
            caps));

        std::vector<std::string> run_lines;
        run_lines.push_back("Status       " + view.status);
        run_lines.push_back("Phase        " + run_phase);
        if (!run_progress.empty()) {
            run_lines.push_back("Progress     " + run_progress);
        }
        run_lines.push_back("PID          " + read_text_file_trimmed(pid_file));
        run_lines.push_back("Run Dir      " + display_path(run_dir.string()));
        run_lines.push_back("Log          " + display_path(log_path.string()));
        run_lines.push_back("Output       " + display_path(state.output_model));
        run_lines.push_back("Output Size  " + mib_string(file_size_or_zero(state.output_model)) + " MiB");
        run_lines.push_back("Tuning Skip  " + std::string(skip_requested ? "requested" : "available") + " (" + display_path(skip_file.string()) + ")");
        run_lines.push_back("State        " + std::string(finished ? ("finished rc=" + (rc.empty() ? "?" : rc)) : (alive ? "running" : "launching")));
        if (!finished) {
            run_lines.push_back("Detach       B, Q, or Esc returns to the UI; the background process is not killed");
        }
        bq::tui::BoxOptions run_box;
        run_box.title = "Quantize / Repair Status";
        run_box.border_style = finished ? (rc == "0" ? bq::tui::success() : bq::tui::error()) : bq::tui::warning();
        run_box.title_style = finished ? (rc == "0" ? bq::tui::success() : bq::tui::error()) : bq::tui::accent();
        bq::tui::print(std::cout, bq::tui::render_box(run_lines, run_box, caps));

        if (!log_lines.empty()) {
            bq::tui::BoxOptions log_box;
            log_box.title = "Recent Log";
            log_box.border_style = bq::tui::muted();
            bq::tui::print(std::cout, bq::tui::render_box(log_lines, log_box, caps));
        }

        std::cout << bq::tui::paint(
            finished ? "Enter/B/Q/Esc: back to project." : "S: skip remaining tuning. B/Q/Esc: detach and leave the run alive.",
            bq::tui::muted(), caps) << "\n" << std::flush;

        if (!tty) {
            return;
        }

#if !defined(_WIN32)
        RawTerminal raw;
        char c = 0;
        if (!read_input_char(c, 1000)) {
            continue;
        }
        if (c == 27) {
            Key escaped = Key::ESC;
            if (consume_escape_sequence_after_esc(escaped)) {
                continue;
            }
            return;
        }
        if (c == '\r' || c == '\n' || c == 'b' || c == 'B' || c == 'q' || c == 'Q') {
            return;
        }
        if (c == 's' || c == 'S') {
            write_text_file(skip_file, "skip remaining tuning requested from TUI\n");
            state.status = "Skip remaining tuning requested";
            continue;
        }
#else
        return;
#endif
    }
}

static void shell_start_quantization(ShellState & state) {
    while (true) {
        const std::vector<std::string> preflight = shell_preflight_errors(state);
        std::string slash_command;
        std::vector<bq::tui::MenuOption> options;
        if (preflight.empty()) {
            options.push_back({ "Start pipeline now", "builds missing KLD/imatrix files, then quantizes", "start" });
            state.status = "Ready to start";
        } else {
            options.push_back({ "Show blockers", std::to_string(preflight.size()) + " item(s) need attention", "blockers" });
            state.status = "Review blockers before starting";
        }
        options.push_back({ "Edit model files", display_path(state.input_model) + " -> " + display_path(state.output_model), "models" });
        options.push_back({ "Edit quant type and options", shell_active_quant_type(state), "options" });
        options.push_back({ "Edit quality inputs", display_path(!state.recipe.selector.kld.empty() ? state.recipe.selector.kld : state.recipe.evaluation.kld_base), "quality" });
        options.push_back({ "Edit candidate search", state.recipe.quantizer.mode + " / " + state.recipe.selector.effort, "candidate-search" });
        options.push_back({ "Save config", display_path(state.last_recipe), "save" });
        options.push_back({ "Back", "", "back" });

        const int action = shell_menu_select(state, "Project > Review and Start", options, slash_command);
        const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
        if (action == -2) {
            try {
                shell_handle_command(state, slash_command, caps);
            } catch (const PromptCancelled &) {
                state.status = "Cancelled";
                continue;
            }
            if (state.quit) {
                return;
            }
            shell_pause(caps);
            continue;
        }
        const std::string command = action >= 0 && action < (int) options.size() ? options[action].command : "back";
        if (command == "blockers") {
            shell_begin_page(state, "Project > Review and Start > Blockers");
            std::cout << "Resolve these before quantization:\n";
            for (const std::string & item : preflight) {
                std::cout << "  - " << item << "\n";
            }
            shell_pause(caps);
            continue;
        }
        if (command == "models") {
            shell_choose_model_files(state);
            continue;
        }
        if (command == "options") {
            shell_options_menu(state);
            continue;
        }
        if (command == "quality") {
            shell_configure_kld_files(state);
            continue;
        }
        if (command == "candidate-search") {
            shell_configure_candidate_search(state);
            continue;
        }
        if (command == "save") {
            shell_save_active_config(state);
            shell_pause(caps);
            continue;
        }
        if (command != "start") {
            return;
        }
        break;
    }

    shell_begin_page(state, "Project > Start Quantization");
    if (state.last_recipe.empty()) {
        state.last_recipe = default_config_path_for_project(state.project_path);
    }
    shell_sync_recipe_paths(state);
    state.variant = prompt("run name", state.variant.empty() ? std::filesystem::path(state.last_recipe).stem().string() : state.variant);

    bq::Recipe preview = state.recipe;
    preview.io.input = state.input_model;
    preview.io.output = state.output_model;
    preview.artifacts.run_dir = run_dir_for(preview).string();
    preview.selector.skip_file = shell_skip_tuning_request_file(std::filesystem::path(preview.artifacts.run_dir)).string();
    const bq::QuantizeRunPlan plan = bq::make_quantize_run_plan(preview, false);
    std::cout << "\nSaved config:\n  " << state.last_recipe << "\n";
    std::cout << "Run directory:\n  " << preview.artifacts.run_dir << "\n";
    std::cout << "Internal quantize call:\n  " << shellish_args(plan.argv) << "\n";
    const std::string preview_kld = !preview.selector.kld.empty() ? preview.selector.kld : preview.evaluation.kld_base;
    const std::string kld_command = kld_base_command_shell(preview);
    const std::string imatrix_command = imatrix_command_shell(preview);
    if (!kld_command.empty() || !imatrix_command.empty()) {
        std::cout << "Pipeline prep:\n";
        if (!kld_command.empty()) {
            std::cout << "  KLD base: " << display_path(preview_kld) << "\n";
        }
        if (!imatrix_command.empty()) {
            std::cout << "  imatrix:  " << display_path(preview.calibration.imatrix) << "\n";
        }
    }

    const bool yes = prompt_bool("start pipeline now", true);
    if (yes) {
        state.recipe.artifacts.run_dir = preview.artifacts.run_dir;
        state.recipe.selector.skip_file = preview.selector.skip_file;
        if (!state.recipe.selector.skip_file.empty()) {
            std::error_code ec;
            std::filesystem::remove(state.recipe.selector.skip_file, ec);
        }
        write_text_file(state.last_recipe, bq::dump_recipe_toml(state.recipe));
        const std::filesystem::path run_dir(preview.artifacts.run_dir);
        const std::filesystem::path log_path = run_dir / "quantize.log";
        const std::filesystem::path pid_file = run_dir / "quantize.pid";
        const int launch_rc = shell_launch_background_run(state.recipe, state.last_recipe, state.project_path, state.variant, run_dir, log_path, pid_file);
        if (launch_rc != 0) {
            state.status = "Run launch failed";
            std::cout << "launch failed with rc=" << launch_rc << "\n";
            return;
        }
        state.run_dir = run_dir.string();
        state.run_log = log_path.string();
        state.run_pid_file = pid_file.string();
        state.status = "Quantizing";
        shell_quantizing_page(state, run_dir, log_path, pid_file);
        std::string rc;
        if (shell_run_finished(run_dir, &rc)) {
            state.status = rc == "0" ? "Run finished" : "Run failed";
        } else if (shell_pid_alive(pid_file)) {
            state.status = "Run detached; still quantizing";
        } else {
            state.status = "Run monitor closed";
        }
    }
}

static void shell_evaluation_menu(ShellState & state) {
    while (true) {
        const int action = shell_submenu_select(state, "Project > Evaluation and Best Candidates", {
            "Record PPL/KLD metrics",
            "Run best-candidate report",
            "Generate candidate configs",
            "Back",
        });
        if (action < 0) {
            return;
        }
        if (action == 0) {
            shell_record_metrics(state);
        } else if (action == 1) {
            shell_project_best(state);
        } else if (action == 2) {
            shell_begin_page(state, "Project > Evaluation and Best Candidates > Generate Candidates");
            const std::string out_dir = prompt("candidate output directory", "advanced-gguf-candidates");
            std::vector<std::string> owned = { state.last_recipe, "--output-dir", out_dir };
            std::vector<char *> args;
            for (std::string & item : owned) {
                args.push_back(item.data());
            }
            candidates_main((int) args.size(), args.data());
            state.status = "Candidates generated";
        } else {
            break;
        }
        if (state.quit) {
            return;
        }
    }
}

static void shell_project_menu(ShellState & state) {
    while (true) {
        std::string slash_command;
        const std::vector<bq::tui::MenuOption> options = shell_project_menu_options(state);
        const int action = shell_menu_select(state, "Project", options, slash_command);
        const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
        if (action == -2) {
            try {
                shell_handle_command(state, slash_command, caps);
            } catch (const PromptCancelled &) {
                state.status = "Cancelled";
                continue;
            }
            if (state.quit) {
                return;
            }
            shell_pause(caps);
            continue;
        }
        try {
            const std::string command = action >= 0 && action < (int) options.size() ? options[action].command : "back";
            bool pause_after = false;
            if (command == "models") {
                shell_choose_model_files(state);
            } else if (command == "quant") {
                shell_select_quant_type(state);
            } else if (command == "options") {
                shell_options_menu(state);
            } else if (command == "quality") {
                shell_configure_kld_files(state);
            } else if (command == "start") {
                shell_start_quantization(state);
                pause_after = true;
            } else if (command == "status") {
                shell_print_status(state, caps);
                shell_print_project_overview(state);
                pause_after = true;
            } else if (command == "eval") {
                shell_evaluation_menu(state);
            } else if (command == "inspect") {
                shell_inspect_model(state);
                pause_after = true;
            } else {
                return;
            }
            if (state.quit) {
                return;
            }
            if (pause_after && state.status != "Cancelled" && state.status.find("cancelled") == std::string::npos) {
                shell_pause(caps);
            }
            continue;
        } catch (const PromptCancelled &) {
            state.status = "Cancelled";
        } catch (const std::exception & e) {
            state.status = std::string("Error: ") + e.what();
            std::cerr << PRODUCT_COMMAND << " shell: " << e.what() << "\n";
        }
        if (state.quit) {
            return;
        }
    }
}

static bool shell_create_new_project(ShellState & state) {
    shell_begin_page(state, "Create New Project");
    std::string project_name;
    if (!prompt_value("Project name", PRODUCT_COMMAND, project_name)) {
        state.status = "Project creation cancelled";
        return false;
    }
    const std::string slug = safe_project_slug(project_name);
    const std::filesystem::path default_dir = state.project_path.empty() ?
        shell_settings_dir() :
        std::filesystem::path(state.project_path).parent_path();
    const std::string directory = choose_directory_from_disk(state, "Create New Project > Save Location", default_dir.string());
    if (directory.empty()) {
        state.status = "Project creation cancelled";
        return false;
    }

    const std::filesystem::path dir(directory);
    state.project_path = (dir / (slug + ".bwqproj")).string();
    state.last_recipe = (dir / (slug + ".config.toml")).string();

    while (std::filesystem::exists(state.project_path)) {
        const int choice = shell_submenu_select(state, "Create New Project > Existing Project", {
            "Load existing project here",
            "Choose another project name",
            "Back",
        });
        if (choice == 0) {
            shell_restore_project_state(state, state.project_path);
            return true;
        }
        if (choice == 1) {
            shell_begin_page(state, "Create New Project");
            if (!prompt_value("Project name", project_name, project_name)) {
                state.status = "Project creation cancelled";
                return false;
            }
            state.project_path = (dir / (safe_project_slug(project_name) + ".bwqproj")).string();
            state.last_recipe = default_config_path_for_project(state.project_path);
            continue;
        }
        state.status = "Project creation cancelled";
        return false;
    }

    state.recipe = bq::default_recipe();
    state.have_recipe = true;
    shell_select_quant_type(state);
    shell_sync_recipe_paths(state);
    write_text_file(state.last_recipe, bq::dump_recipe_toml(state.recipe));
    shell_remember_recipe(state, state.last_recipe, state.recipe);

    bq::ProjectInit init;
    init.name = project_name;
    init.recipe = state.last_recipe;
    init.input = state.input_model;
    init.bf16_reference = state.recipe.evaluation.bf16_reference.empty() ? state.input_model : state.recipe.evaluation.bf16_reference;
    init.kld_base = state.recipe.evaluation.kld_base;
    init.corpus = state.recipe.evaluation.corpus;
    init.calibration_corpus = state.recipe.calibration.corpus;
    init.imatrix = state.recipe.calibration.imatrix;
    bq::project_init_file(state.project_path, init);
    state.status = "Project created";

    const int next = shell_submenu_select(state, "Create New Project > Next", {
        "Select model input/output now",
        "Open project",
        "Back",
    });
    if (next == 0) {
        shell_choose_model_files(state);
    }
    return true;
}

static void shell_load_existing_project(ShellState & state) {
    shell_begin_page(state, "Load Existing Project");
    const std::string selected = choose_file_from_disk(state, "Load project", { ".bwqproj", ".jsonl" }, state.project_path.empty() ? default_project_path() : state.project_path);
    if (selected.empty()) {
        return;
    }
    const bool loaded_config = shell_restore_project_state(state, selected);
    if (!loaded_config) {
        state.status = "Project loaded; choose or save a config";
    }
    shell_project_menu(state);
}

static bool shell_handle_command(ShellState & state, std::string line, const bq::tui::TerminalCapabilities & caps) {
    if (line == "/") {
        line = shell_prompt_command(caps, "/");
    }
    const std::vector<std::string> words = split_words(line);
    if (words.empty()) {
        return true;
    }
    const std::string command = words[0];

    if (command == "/help" || command == "/commands") {
        bq::tui::print(std::cout, bq::tui::render_slash_help(bq::tui::default_slash_commands(), caps));
        return true;
    }
    if (command == "/status") {
        shell_print_status(state, caps);
        return true;
    }
    if (command == "/quit" || command == "/exit") {
        state.quit = true;
        return true;
    }
    if (command == "/clear") {
        shell_clear(caps);
        return true;
    }
    if (command == "/project") {
        if (words.size() > 1) {
            state.project_path = words[1];
            if (!std::filesystem::exists(state.project_path)) {
                state.status = "Project not found";
                std::cout << "project not found: " << state.project_path << "\n";
                std::cout << "Use Create new project to create it deliberately.\n";
                return true;
            }
            shell_restore_project_state(state, state.project_path);
            shell_print_project_overview(state);
            state.status = "Project opened";
        } else {
            shell_load_existing_project(state);
        }
        return true;
    }
    if (command == "/config" || command == "/recipe") {
        if (words.size() <= 1) {
            shell_begin_page(state, "Command > Load Configuration");
        }
        const std::string path = words.size() > 1 ? words[1] : prompt("configuration file", state.last_recipe);
        shell_load_recipe(state, path);
        std::cout << "loaded " << path << "\n";
        return true;
    }
    if (command == "/model") {
        const std::string old_input = state.input_model;
        const bool output_was_auto = output_model_is_auto_default(old_input, state.output_model, shell_active_quant_type(state));
        if (words.size() <= 1) {
            shell_begin_page(state, "Command > Set Model");
        }
        state.input_model = words.size() > 1 ? words[1] : prompt("input GGUF", state.input_model);
        if (!state.input_model.empty() && output_was_auto) {
            state.output_model = default_output_model_path(state.input_model, shell_active_quant_type(state));
        }
        shell_sync_recipe_paths(state);
        if (state.have_recipe) {
            shell_remember_recipe(state, state.last_recipe, state.recipe);
        }
        state.status = "Model selected";
        return true;
    }
    if (command == "/output") {
        if (words.size() <= 1) {
            shell_begin_page(state, "Command > Set Output");
        }
        state.output_model = words.size() > 1 ? words[1] : prompt("output GGUF", state.output_model);
        shell_sync_recipe_paths(state);
        if (state.have_recipe) {
            shell_remember_recipe(state, state.last_recipe, state.recipe);
        }
        state.status = "Output selected";
        return true;
    }
    if (command == "/inspect") {
        shell_inspect_model(state, words.size() > 1 ? words[1] : std::string());
        return true;
    }
    if (command == "/run") {
        shell_start_quantization(state);
        return true;
    }
    if (command == "/metrics") {
        shell_record_metrics(state);
        return true;
    }
    if (command == "/best") {
        shell_project_best(state, words.size() > 1 ? words[1] : std::string());
        return true;
    }
    if (command == "/what-if") {
        std::vector<std::string> owned;
        if (words.size() > 1) {
            owned.assign(words.begin() + 1, words.end());
        } else {
            shell_begin_page(state, "Command > What-If Sensitivity");
            owned.push_back(prompt("sensitivity report path"));
        }
        std::vector<char *> args;
        for (std::string & item : owned) {
            args.push_back(item.data());
        }
        what_if_main((int) args.size(), args.data());
        return true;
    }
    if (command == "/candidates") {
        if (words.size() <= 1) {
            shell_begin_page(state, "Command > Generate Candidates");
        }
        const std::string out_dir = words.size() > 1 ? words[1] : prompt("candidate output dir", "advanced-gguf-candidates");
        std::vector<std::string> owned = { state.last_recipe, "--output-dir", out_dir };
        std::vector<char *> args;
        for (std::string & item : owned) {
            args.push_back(item.data());
        }
        candidates_main((int) args.size(), args.data());
        state.status = "Candidates generated";
        return true;
    }
    if (command == "/policy") {
        layer_policy_main();
        return true;
    }

    std::cout << "unknown command: " << command << "\n";
    return true;
}

static int shell_main() {
    ShellState state;
    ShellAlternateScreen screen;

    while (true) {
        const bq::tui::TerminalCapabilities caps = bq::tui::detect_terminal(stdout);
        std::string slash_command;
        const int action = shell_menu_select(state, "Main Menu", shell_home_menu_options(), slash_command);
        if (action == -2) {
            try {
                shell_handle_command(state, slash_command, caps);
            } catch (const PromptCancelled &) {
                state.status = "Cancelled";
                continue;
            } catch (const std::exception & e) {
                state.status = std::string("Error: ") + e.what();
                std::cerr << PRODUCT_COMMAND << " shell: " << e.what() << "\n";
            }
            if (state.quit) {
                return 0;
            }
            shell_pause(caps);
            continue;
        }

        bool need_pause = false;
        try {
            if (action == 0) {
                if (shell_create_new_project(state)) {
                    shell_project_menu(state);
                }
            } else if (action == 1) {
                shell_load_existing_project(state);
            } else if (action == 2) {
                shell_inspect_model(state);
                need_pause = true;
            } else {
                return 0;
            }
        } catch (const PromptCancelled &) {
            state.status = "Cancelled";
        } catch (const std::exception & e) {
            state.status = std::string("Error: ") + e.what();
            std::cerr << PRODUCT_COMMAND << " shell: " << e.what() << "\n";
            need_pause = true;
        }

        if (state.quit) {
            return 0;
        }

        if (need_pause) {
            shell_pause(caps);
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
#if !defined(_WIN32)
            if (stdin_is_tty()) {
                return shell_main();
            }
#endif
            usage();
            return 1;
        }

        const std::string command = argv[1];
        if (command == "--help" || command == "-h" || command == "help") {
            usage();
            return 0;
        }

        if (command == "recipe") {
            if (argc < 3) {
                throw std::runtime_error("recipe requires a subcommand");
            }
            const std::string sub = argv[2];
            if (sub == "init") {
                return recipe_init(argc - 3, argv + 3);
            }
            if (sub == "validate") {
                return recipe_validate(argc - 3, argv + 3);
            }
            throw std::runtime_error("unknown recipe subcommand: " + sub);
        }

        if (command == "run") {
            if (argc < 3) {
                throw std::runtime_error("run requires a recipe path");
            }
            std::string path = argv[2];
            std::vector<std::string> sets;
            bool yes = false;
            std::string project_path;
            std::string variant;
            for (int i = 3; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--yes" || arg == "-y") {
                    yes = true;
                } else if (arg == "--dry-run") {
                    throw std::runtime_error(std::string("run no longer supports --dry-run; use ") + PRODUCT_COMMAND + " plan to inspect commands");
                } else if (arg == "--project" && i + 1 < argc) {
                    project_path = argv[++i];
                } else if (arg == "--variant" && i + 1 < argc) {
                    variant = argv[++i];
                } else if (arg == "--set" && i + 1 < argc) {
                    sets.push_back(argv[++i]);
                } else {
                    throw std::runtime_error("unknown run argument: " + arg);
                }
            }
            return run_recipe(path, sets, yes, false, project_path, variant);
        }

        if (command == "plan") {
            return plan_recipe(argc - 2, argv + 2);
        }

        if (command == "project") {
            return project_main(argc - 2, argv + 2);
        }

        if (command == "layer-policy") {
            return layer_policy_main();
        }

        if (command == "what-if") {
            return what_if_main(argc - 2, argv + 2);
        }

        if (command == "evidence") {
            return evidence_main(argc - 2, argv + 2);
        }

        if (command == "size") {
            return size_main(argc - 2, argv + 2);
        }

        if (command == "candidates") {
            return candidates_main(argc - 2, argv + 2);
        }

        if (command == "best") {
            return bq::best_main(argc - 2, argv + 2);
        }

        if (command == "inspect") {
            return bq::inspect_main(argc - 2, argv + 2);
        }

        if (command == "kld-info") {
            return kld_info_main(argc - 2, argv + 2);
        }

        if (command == "kld-command") {
            return kld_command_main(argc - 2, argv + 2);
        }

        if (command == "imatrix-command") {
            return imatrix_command_main(argc - 2, argv + 2);
        }

        if (command == "shell") {
            return shell_main();
        }

        if (command == "wizard") {
            std::string output = std::string(PRODUCT_COMMAND) + ".recipe.toml";
            bool run_after = false;
            bool yes = false;
            std::string project_path;
            std::string variant;
            for (int i = 2; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--output" && i + 1 < argc) {
                    output = argv[++i];
                } else if (arg == "--run") {
                    run_after = true;
                } else if (arg == "--yes" || arg == "-y") {
                    yes = true;
                } else if (arg == "--project" && i + 1 < argc) {
                    project_path = argv[++i];
                } else if (arg == "--variant" && i + 1 < argc) {
                    variant = argv[++i];
                } else {
                    throw std::runtime_error("unknown wizard argument: " + arg);
                }
            }

            const int action = menu_select("advanced-gguf-quantizer", {
                "Create new recipe - choose precision, VRAM/RAM target, KLD setup",
                "Load existing Blackwell recipe - edit paths, sizing, and evaluation",
                "Exit",
            });
            if (action == 2) {
                return 0;
            }

            bq::Recipe recipe;
            if (action == 1) {
                const std::string path = prompt("recipe path");
                bq::LoadedRecipe loaded = bq::load_recipe_file(path);
                recipe = loaded.recipe;
                const int edit = menu_select("Loaded recipe", {
                    "Keep precision and sizing settings",
                    "Select precision, VRAM/RAM target, and BPW budget",
                    "Exit",
                });
                if (edit == 2) {
                    return 0;
                }
                if (edit == 1 && !configure_precision_vram(recipe, true)) {
                    return 0;
                }
            } else {
                if (!configure_precision_vram(recipe, false)) {
                    return 0;
                }
            }

            recipe.io.input = prompt("input GGUF", recipe.io.input);
            recipe.io.output = prompt("output GGUF", recipe.io.output);
            recipe.base.threads = prompt_int("threads", recipe.base.threads > 0 ? recipe.base.threads : default_worker_threads());
            recipe.calibration.corpus = prompt(
                "calibration file for imatrix",
                recipe.calibration.corpus.empty() ? recipe.evaluation.corpus : recipe.calibration.corpus);
            recipe.calibration.imatrix = prompt("imatrix file", recipe.calibration.imatrix);
            recipe.io.patch_base = prompt("patch base", recipe.io.patch_base);
            configure_four_six_mixed(recipe);
            configure_standard_quantize_options(recipe);
            recipe.artifacts.run_dir = prompt("run dir", recipe.artifacts.run_dir.empty() && !recipe.io.output.empty() ? recipe.io.output + ".run" : recipe.artifacts.run_dir);
            configure_evaluation_source(recipe);

            if (!recipe.target.sizing_note.empty()) {
                std::cout << "\n" << recipe.target.sizing_note << "\n";
            }

            write_text_file(output, bq::dump_recipe_toml(recipe));
            std::cout << "wrote " << output << "\n";
            if (!run_after && stdin_is_tty()) {
                const int next = menu_select("Next", {
                    "Run now - write GGUF, report, manifest, assignment, and smoke script",
                    "Only save recipe",
                });
                run_after = next == 0;
            }
            if (run_after) {
                return run_recipe(output, {}, yes, false, project_path, variant);
            }
            return 0;
        }

        throw std::runtime_error("unknown command: " + command);
    } catch (const PromptCancelled &) {
        std::cerr << PRODUCT_COMMAND << ": cancelled\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << PRODUCT_COMMAND << ": " << e.what() << "\n\n";
        usage();
        return 1;
    }
}
