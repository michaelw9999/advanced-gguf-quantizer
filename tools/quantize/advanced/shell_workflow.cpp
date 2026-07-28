#include "shell_workflow.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace bq {
namespace {

static int parse_int_or_default(const std::string & value, int fallback) {
    if (value.empty()) {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str()) {
        return fallback;
    }
    return (int) parsed;
}

static bool recipe_uses_mxfp6(const Recipe & recipe, const std::string & precision_mode = {}) {
    return quant_type_uses_mxfp6(precision_mode) ||
        quant_type_uses_mxfp6(recipe.target.precision_mode) ||
        quant_type_uses_mxfp6(recipe.base.ftype);
}

static const char * mxfp6_tui_notice() {
    return "unsupported by NVIDIA/llama.cpp; future official format may differ";
}

static const char * mxfp6_feedback_branch() {
    return "github.com/michaelw9999/llama.cpp/tree/mxfp6-cuda";
}

} // namespace

std::string mib_string(uint64_t bytes) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(1);
    out << (double) bytes / (1024.0 * 1024.0);
    return out.str();
}

KldBaseInfo read_kld_base_info(const std::string & path) {
    KldBaseInfo info;
    if (path.empty()) {
        info.error = "-";
        return info;
    }

    std::error_code ec;
    info.file_bytes = std::filesystem::file_size(path, ec);
    if (ec) {
        info.error = "not readable";
        return info;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        info.error = "open failed";
        return info;
    }

    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, "_logits_", sizeof(magic)) != 0) {
        info.error = "not a llama logits/KLD base";
        return info;
    }

    in.read(reinterpret_cast<char *>(&info.n_ctx), sizeof(info.n_ctx));
    in.read(reinterpret_cast<char *>(&info.n_vocab), sizeof(info.n_vocab));
    in.read(reinterpret_cast<char *>(&info.n_chunks), sizeof(info.n_chunks));
    if (!in || info.n_ctx <= 0 || info.n_vocab <= 0 || info.n_chunks <= 0) {
        info.error = "invalid header";
        return info;
    }

    const uint64_t header_bytes = 8 + 3 * (uint64_t) sizeof(int32_t);
    const uint64_t token_bytes = (uint64_t) info.n_chunks * (uint64_t) info.n_ctx * (uint64_t) sizeof(int32_t);
    const uint64_t n_score = (uint64_t) info.n_ctx - 1U - (uint64_t) info.n_ctx / 2U;
    const uint64_t nv = 2U * (((uint64_t) info.n_vocab + 1U) / 2U) + 4U;
    const uint64_t chunk_logp_bytes = n_score * nv * (uint64_t) sizeof(uint16_t);
    const long double expected =
        (long double) header_bytes +
        (long double) token_bytes +
        (long double) info.n_chunks * (long double) chunk_logp_bytes;

    info.expected_bytes = expected > (long double) std::numeric_limits<uint64_t>::max()
        ? std::numeric_limits<uint64_t>::max()
        : (uint64_t) expected;
    info.valid = true;
    info.complete = info.file_bytes >= info.expected_bytes;
    if (info.file_bytes >= header_bytes + token_bytes && chunk_logp_bytes > 0) {
        const uint64_t logp_bytes = info.file_bytes - header_bytes - token_bytes;
        info.available_chunks = (int32_t) std::min<uint64_t>((uint64_t) info.n_chunks, logp_bytes / chunk_logp_bytes);
    } else {
        info.available_chunks = 0;
    }
    if (!info.complete) {
        info.error = "truncated";
    }
    return info;
}

std::string format_kld_info(const KldBaseInfo & info) {
    if (!info.valid) {
        return info.error.empty() ? "unknown" : info.error;
    }
    std::ostringstream out;
    out << "ctx=" << info.n_ctx
        << ", chunks=" << info.available_chunks << "/" << info.n_chunks
        << ", vocab=" << info.n_vocab
        << ", " << mib_string(info.file_bytes) << " MiB";
    if (!info.complete) {
        out << ", incomplete expected " << mib_string(info.expected_bytes) << " MiB";
    }
    return out.str();
}

std::filesystem::path shell_settings_dir() {
    const char * xdg_config = std::getenv("XDG_CONFIG_HOME");
    if (xdg_config != nullptr && xdg_config[0] != '\0') {
        return std::filesystem::path(xdg_config) / PRODUCT_CONFIG_DIR;
    }
    const char * home = std::getenv("HOME");
    if (home != nullptr && home[0] != '\0') {
        return std::filesystem::path(home) / ".config" / PRODUCT_CONFIG_DIR;
    }
    return std::filesystem::path(".advanced-gguf-quantizer");
}

std::string default_project_path() {
    return (shell_settings_dir() / "default.bwqproj").string();
}

std::string default_config_path() {
    return (shell_settings_dir() / "default.config.toml").string();
}

tui::ProductInfo shell_product() {
    tui::ProductInfo product;
    product.name = PRODUCT_NAME;
    product.version = PRODUCT_VERSION;
    product.subtitle = "GGUF NVFP4 / MXFP8 / experimental MXFP6 quantization shell";
    return product;
}

std::string display_path(const std::string & value) {
    if (value.empty()) {
        return {};
    }
    std::filesystem::path path(value);
    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) {
        std::filesystem::path rel = std::filesystem::relative(path, cwd, ec);
        const std::string rel_str = rel.string();
        if (!ec && !rel.empty() && rel_str.find("..") != 0 && rel_str.size() < value.size()) {
            path = rel;
        }
    }
    std::string out = path.string();
    const char * home = std::getenv("HOME");
    if (home != nullptr) {
        const std::string home_str(home);
        if (out.find(home_str + "/") == 0) {
            out = "~" + out.substr(home_str.size());
        }
    }
    if (out.size() > 78) {
        out = "..." + out.substr(out.size() - 75);
    }
    return out;
}

tui::Style shell_status_style(const std::string & status) {
    std::string lower = status;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    if (lower.find("failed") != std::string::npos || lower.find("error") != std::string::npos) {
        return tui::error();
    }
    if (lower.find("running") != std::string::npos || lower.find("quantizing") != std::string::npos ||
            lower.find("detached") != std::string::npos || lower.find("finished") != std::string::npos ||
            lower.find("loaded") != std::string::npos || lower.find("recorded") != std::string::npos) {
        return tui::success();
    }
    return tui::warning();
}

std::vector<tui::StatusItem> shell_status_items(const ShellState & state) {
    std::vector<tui::StatusItem> items;
    const std::string quant = state.precision_mode.empty() ? state.recipe.target.precision_mode : state.precision_mode;
    const std::string kld = !state.recipe.selector.kld.empty() ? state.recipe.selector.kld : state.recipe.evaluation.kld_base;
    items.push_back({ "Status", state.status, "", shell_status_style(state.status) });
    if (!state.run_phase.empty()) {
        items.push_back({ "Run Phase", state.run_phase, "", tui::accent() });
    }
    if (!state.run_progress.empty()) {
        items.push_back({ "Run ETA", state.run_progress, "", tui::warning() });
    }
    items.push_back({ "Project", display_path(state.project_path), "", tui::normal() });
    items.push_back({ "Config File", display_path(state.last_recipe), "", tui::accent() });
    items.push_back({ "Input File", display_path(state.input_model), "", tui::normal() });
    items.push_back({ "Output File", display_path(state.output_model), "", tui::normal() });
    items.push_back({ "Quant Type", quant, "", tui::warning() });
    if (recipe_uses_mxfp6(state.recipe, quant)) {
        items.push_back({ "MXFP6", "EXPERIMENTAL", mxfp6_tui_notice(), tui::warning() });
        items.push_back({ "MXFP6 Feedback", "requested", mxfp6_feedback_branch(), tui::warning() });
    }
    items.push_back({ "KLD Base", display_path(kld), "", tui::normal() });
    if (!kld.empty()) {
        const KldBaseInfo info = read_kld_base_info(kld);
        items.push_back({ "KLD Info", format_kld_info(info), "", info.valid ? (info.complete ? tui::normal() : tui::warning()) : tui::error() });
    }
    return items;
}

std::vector<tui::StatusItem> shell_status_items_compact(const ShellState & state) {
    const std::string quant = state.precision_mode.empty() ? state.recipe.target.precision_mode : state.precision_mode;
    std::vector<tui::StatusItem> items;
    items.push_back({ "Status", state.status, "", shell_status_style(state.status) });
    if (!state.run_phase.empty()) {
        items.push_back({ "Phase", state.run_phase, "", tui::accent() });
    }
    items.push_back({ "Project", display_path(state.project_path), "", tui::normal() });
    items.push_back({ "Input", display_path(state.input_model), "", tui::normal() });
    items.push_back({ "Output", display_path(state.output_model), "", tui::normal() });
    items.push_back({ "Type", quant, "", tui::warning() });
    if (recipe_uses_mxfp6(state.recipe, quant)) {
        items.push_back({ "MXFP6", "experimental", "unsupported; format may change; feedback requested", tui::warning() });
        items.push_back({ "Branch", "mxfp6-cuda", mxfp6_feedback_branch(), tui::warning() });
    }
    return items;
}

void shell_print_status(const ShellState & state, const tui::TerminalCapabilities & caps) {
    tui::print(std::cout, tui::render_branded_header(shell_product(), caps));
    tui::print(std::cout, tui::render_status_panel(shell_product(), shell_status_items(state), caps));
}

void shell_clear(const tui::TerminalCapabilities & caps) {
    if (caps.ansi) {
        std::cout << "\033[H\033[2J\033[3J";
    }
}

std::vector<tui::MenuOption> shell_home_menu_options() {
    return {
        { "Create new project", "", "" },
        { "Load existing project", "", "" },
        { "Inspect GGUF", "", "" },
        { "Quit", "", "" },
    };
}

std::vector<std::string> shell_preflight_errors(const ShellState & state) {
    std::vector<std::string> errors;
    if (!state.have_recipe) {
        errors.push_back("select or create a configuration");
        return errors;
    }

    Recipe recipe = state.recipe;
    recipe.io.input = state.input_model;
    recipe.io.output = state.output_model;
    const auto recipe_errors = validate_recipe(recipe, true);
    errors.insert(errors.end(), recipe_errors.begin(), recipe_errors.end());

    std::error_code ec;
    if (!state.input_model.empty() && !std::filesystem::exists(state.input_model, ec)) {
        errors.push_back("model input file not found: " + state.input_model);
    }
    if (!state.output_model.empty()) {
        const std::filesystem::path out(state.output_model);
        const std::filesystem::path parent = out.has_parent_path() ? out.parent_path() : std::filesystem::path(".");
        if (!std::filesystem::exists(parent, ec)) {
            errors.push_back("model output directory does not exist: " + parent.string());
        }
    }
    if (!recipe.evaluation.bf16_reference.empty() && !std::filesystem::exists(recipe.evaluation.bf16_reference, ec)) {
        errors.push_back("Selected BF16 reference model file not found " + recipe.evaluation.bf16_reference);
    }
    if (!recipe.evaluation.corpus.empty() && !std::filesystem::exists(recipe.evaluation.corpus, ec)) {
        errors.push_back("PPL/KLD input file not found: " + recipe.evaluation.corpus);
    }
    if (!recipe.calibration.corpus.empty() && !std::filesystem::exists(recipe.calibration.corpus, ec)) {
        errors.push_back("calibration corpus not found: " + recipe.calibration.corpus);
    }
    const bool imatrix_can_be_generated =
        !recipe.calibration.imatrix.empty() &&
        !recipe.calibration.corpus.empty() &&
        (!recipe.evaluation.bf16_reference.empty() || !recipe.io.input.empty());
    if (!recipe.calibration.imatrix.empty() && !std::filesystem::exists(recipe.calibration.imatrix, ec) && !imatrix_can_be_generated) {
        errors.push_back("imatrix file not found: " + recipe.calibration.imatrix);
    }
    const std::string kld = !recipe.selector.kld.empty() ? recipe.selector.kld : recipe.evaluation.kld_base;
    const bool kld_can_be_generated =
        !kld.empty() &&
        (recipe.evaluation.kld_mode == "make_base" || recipe.evaluation.kld_mode == "bundle") &&
        !recipe.evaluation.corpus.empty() &&
        (!recipe.evaluation.bf16_reference.empty() || !recipe.io.input.empty());
    if (!kld.empty() && !std::filesystem::exists(kld, ec) && !kld_can_be_generated) {
        errors.push_back("KLD base is selected but file was not found: " + kld);
    } else if (!kld.empty() && std::filesystem::exists(kld, ec)) {
        const KldBaseInfo info = read_kld_base_info(kld);
        if (!info.valid) {
            errors.push_back("KLD input file is not a valid logits/KLD file: " + kld + " (" + format_kld_info(info) + ")");
        } else if (!info.complete) {
            errors.push_back("KLD base is incomplete; selector evaluation requires the full saved-logit base: " + format_kld_info(info));
        }
    }
    if (!recipe.selector.eval_top.empty() && kld.empty()) {
        errors.push_back("PPL/KLD candidate evaluation needs a KLD base file");
    }
    if (recipe.evaluation.kld_mode == "bundle" && recipe.evaluation.bundle.empty()) {
        errors.push_back("eval bundle path is required when KLD mode is bundle");
    }
    return errors;
}

bool shell_ready_to_quantize(const ShellState & state) {
    return shell_preflight_errors(state).empty();
}

std::vector<tui::MenuOption> shell_project_menu_options(const ShellState & state) {
    const auto step = [](const std::string & label, bool done) {
        return std::string(done ? "[x] " : "[ ] ") + label;
    };
    const std::string kld = !state.recipe.selector.kld.empty() ? state.recipe.selector.kld : state.recipe.evaluation.kld_base;
    const bool model_ready = !state.input_model.empty() && !state.output_model.empty();
    const bool quant_ready = !state.recipe.target.precision_mode.empty();
    const bool kld_planned =
        !kld.empty() &&
        (std::filesystem::exists(kld) ||
         ((state.recipe.evaluation.kld_mode == "make_base" || state.recipe.evaluation.kld_mode == "bundle") &&
          !state.recipe.evaluation.corpus.empty() &&
          (!state.recipe.evaluation.bf16_reference.empty() || !state.input_model.empty())));
    const bool imatrix_planned =
        !state.recipe.calibration.imatrix.empty() &&
        (std::filesystem::exists(state.recipe.calibration.imatrix) ||
         (!state.recipe.calibration.corpus.empty() &&
          (!state.recipe.evaluation.bf16_reference.empty() || !state.input_model.empty())));
    const bool kld_ready = !state.recipe.quantizer.require_kld || kld_planned;
    const bool imatrix_ready = !state.recipe.quantizer.require_imatrix || imatrix_planned;
    const bool quality_ready = kld_ready && imatrix_ready;
    const std::string quality_note = !state.recipe.quantizer.require_kld && !state.recipe.quantizer.require_imatrix ?
        "not required for current effort" :
        (kld.empty() ? "KLD not selected" : display_path(kld));

    std::vector<tui::MenuOption> options = {
        { step("Pick BF16/input GGUF and output file", model_ready), display_path(state.input_model), "models" },
        { step("Choose quant type", quant_ready),
            state.recipe.target.precision_mode + (recipe_uses_mxfp6(state.recipe) ? " / MXFP6 experimental" : ""),
            "quant" },
        { step("Choose options", true), state.recipe.quantizer.mode + " / " + state.recipe.selector.effort, "options" },
        { step("Choose quality inputs", quality_ready), quality_note, "quality" },
        { step("Review and start pipeline", shell_ready_to_quantize(state)), "", "start" },
    };
    options.push_back({ "Show status", "", "status" });
    options.push_back({ "Evaluation and best candidates", "", "eval" });
    options.push_back({ "Inspect GGUF", "", "inspect" });
    options.push_back({ "Back", "", "back" });
    return options;
}

std::vector<tui::MenuOption> shell_options_menu_options(const ShellState & state) {
    const auto step = [](const std::string & label, bool done) {
        return std::string(done ? "[x] " : "[ ] ") + label;
    };
    const std::string kld = !state.recipe.selector.kld.empty() ? state.recipe.selector.kld : state.recipe.evaluation.kld_base;
    const bool has_quality_inputs =
        !kld.empty() &&
        !state.recipe.evaluation.corpus.empty() &&
        !state.recipe.calibration.corpus.empty() &&
        !state.recipe.calibration.imatrix.empty();
    const bool has_target_budget =
        state.recipe.target.fit_to_vram ||
        state.recipe.target.target_bpw > 0.0 ||
        state.recipe.target.weight_budget_gib > 0.0;
    const bool uses_nvfp4 =
        quant_type_uses_nvfp4(state.recipe.target.precision_mode) ||
        quant_type_uses_nvfp4(state.recipe.base.ftype);
    const bool uses_mxfp6 =
        quant_type_uses_mxfp6(state.recipe.target.precision_mode) ||
        quant_type_uses_mxfp6(state.recipe.base.ftype);
    const bool native_ready =
        !state.recipe.selector.effort.empty() ||
        !state.recipe.stock_ftype.technique_candidates.empty() ||
        !state.recipe.nvfp4.calibration_families.empty();
    const bool gates_ready =
        !state.recipe.selector.ranking.kld_penalty.empty() ||
        !state.recipe.selector.ranking.p99_penalty.empty() ||
        !state.recipe.selector.ranking.p999_penalty.empty();
    const bool nvfp4_ready =
        !uses_nvfp4 ||
        !state.recipe.nvfp4.preset.empty() ||
        !state.recipe.nvfp4.four_six.choose46.empty();
    const bool mxfp6_ready =
        !uses_mxfp6 ||
        !state.recipe.mxfp6.tensor_scale.empty() ||
        !state.recipe.mxfp6.selector_scale_top.empty();

    std::vector<tui::MenuOption> options = {
        { step("Select primary quant type", !state.recipe.target.precision_mode.empty()), "", "quant" },
        { step("Model input and output", !state.input_model.empty() && !state.output_model.empty()), "", "models" },
        { step("Quality inputs", has_quality_inputs), "", "quality" },
        { step("Set target BPW / VRAM", has_target_budget), "", "budget" },
    };
    if (uses_nvfp4) {
        options.push_back({ step("NVFP4 4/6 policy", nvfp4_ready), "", "nvfp4" });
    }
    options.push_back({ step("Native candidate search", native_ready), "", "candidate-search" });
    options.push_back({ step("PPL/KLD scoring gates", gates_ready), "", "gates" });
    options.push_back({ step("Edit Existing GGUF", !state.recipe.rescue.enabled || !state.recipe.rescue.type.empty()), "", "edit" });
    if (uses_mxfp6) {
        options.push_back({ step("MXFP6 scale refinement", mxfp6_ready), "", "mxfp6" });
    }
    options.push_back({ step("Tensor rules", !state.recipe.tensor_overrides.entries.empty() || !state.recipe.tensor_overrides.files.empty()), "", "tensor-rules" });
    options.push_back({ step("Standard quantize options",
                !state.recipe.base.output_tensor_type.empty() ||
                !state.recipe.base.token_embedding_type.empty() ||
                !state.recipe.base.mtp_tensor_type.empty()), "", "standard" });
    options.push_back({ "Save config", "", "save" });
    options.push_back({ "Back", "", "back" });
    return options;
}

} // namespace bq
