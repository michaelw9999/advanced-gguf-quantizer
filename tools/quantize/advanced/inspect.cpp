#include "inspect.h"

#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bq {
namespace {

struct GgufDeleter {
    void operator()(gguf_context * ctx) const {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

struct InspectOptions {
    bool json = false;
    bool list_tensors = false;
    bool list_keys = false;
    bool require_mtp = false;
    std::string tensor_filter;
    std::string path;
};

static void inspect_usage() {
    std::cout <<
        "usage:\n"
        "  advanced-gguf-quantizer inspect <model.gguf> [options]\n\n"
        "options:\n"
        "  --json                 write machine-readable JSON\n"
        "  --tensors              list tensor names, types, and sizes\n"
        "  --keys                 list metadata keys and compact values\n"
        "  --tensor-filter <text> only list tensors whose name contains text\n"
        "  --require-mtp          exit nonzero if no MTP metadata/tensors are found\n";
}

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

static bool contains_ci(const std::string & haystack, const std::string & needle) {
    if (needle.empty()) {
        return true;
    }
    return lower_copy(haystack).find(lower_copy(needle)) != std::string::npos;
}

static bool is_mtp_name(const std::string & name) {
    const std::string lower = lower_copy(name);
    return lower.find("mtp") != std::string::npos ||
           lower.find("nextn") != std::string::npos ||
           lower.find("multi_token") != std::string::npos ||
           lower.find("draft") != std::string::npos;
}

static std::string json_escape(const std::string & value) {
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

static std::string human_bytes(uint64_t bytes) {
    static const char * units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = (double) bytes;
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    if (unit == 0) {
        out << bytes << " " << units[unit];
    } else {
        out << std::fixed << std::setprecision(2) << value << " " << units[unit];
    }
    return out.str();
}

static std::string compact_value(const gguf_context * ctx, int64_t id) {
    const gguf_type type = gguf_get_kv_type(ctx, id);
    std::ostringstream out;
    switch (type) {
        case GGUF_TYPE_UINT8:   out << (uint32_t) gguf_get_val_u8(ctx, id); break;
        case GGUF_TYPE_INT8:    out << (int32_t) gguf_get_val_i8(ctx, id); break;
        case GGUF_TYPE_UINT16:  out << gguf_get_val_u16(ctx, id); break;
        case GGUF_TYPE_INT16:   out << gguf_get_val_i16(ctx, id); break;
        case GGUF_TYPE_UINT32:  out << gguf_get_val_u32(ctx, id); break;
        case GGUF_TYPE_INT32:   out << gguf_get_val_i32(ctx, id); break;
        case GGUF_TYPE_FLOAT32: out << gguf_get_val_f32(ctx, id); break;
        case GGUF_TYPE_BOOL:    out << (gguf_get_val_bool(ctx, id) ? "true" : "false"); break;
        case GGUF_TYPE_STRING: {
            std::string value = gguf_get_val_str(ctx, id);
            if (value.size() > 120) {
                value = value.substr(0, 117) + "...";
            }
            out << '"' << value << '"';
            break;
        }
        case GGUF_TYPE_UINT64:  out << gguf_get_val_u64(ctx, id); break;
        case GGUF_TYPE_INT64:   out << gguf_get_val_i64(ctx, id); break;
        case GGUF_TYPE_FLOAT64: out << gguf_get_val_f64(ctx, id); break;
        case GGUF_TYPE_ARRAY:
            out << gguf_type_name(type) << "[" << gguf_get_arr_n(ctx, id)
                << "] of " << gguf_type_name(gguf_get_arr_type(ctx, id));
            break;
        default:
            out << gguf_type_name(type);
            break;
    }
    return out.str();
}

static InspectOptions parse_options(int argc, char ** argv) {
    InspectOptions options;
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h" || arg == "help") {
            inspect_usage();
            std::exit(0);
        } else if (arg == "--json") {
            options.json = true;
        } else if (arg == "--tensors") {
            options.list_tensors = true;
        } else if (arg == "--keys") {
            options.list_keys = true;
        } else if (arg == "--require-mtp") {
            options.require_mtp = true;
        } else if (arg == "--tensor-filter" && i + 1 < argc) {
            options.tensor_filter = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown inspect argument: " + arg);
        } else if (options.path.empty()) {
            options.path = arg;
        } else {
            throw std::runtime_error("unexpected inspect argument: " + arg);
        }
    }
    if (options.path.empty()) {
        throw std::runtime_error("inspect requires a GGUF path");
    }
    return options;
}

} // namespace

InspectSummary inspect_gguf(const std::string & path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    std::unique_ptr<gguf_context, GgufDeleter> ctx(gguf_init_from_file(path.c_str(), params));
    if (!ctx) {
        throw std::runtime_error("failed to open GGUF: " + path);
    }

    std::vector<std::string> mtp_keys;
    std::vector<std::string> mtp_tensors;
    InspectSummary summary;
    summary.path = path;
    summary.version = gguf_get_version(ctx.get());
    summary.file_bytes = std::filesystem::exists(path) ? (uint64_t) std::filesystem::file_size(path) : 0;
    summary.metadata_keys = gguf_get_n_kv(ctx.get());
    summary.tensors = gguf_get_n_tensors(ctx.get());

    for (int64_t i = 0; i < gguf_get_n_kv(ctx.get()); ++i) {
        const std::string key = gguf_get_key(ctx.get(), i);
        if (is_mtp_name(key)) {
            mtp_keys.push_back(key);
        }
    }

    for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
        const std::string name = gguf_get_tensor_name(ctx.get(), i);
        const ggml_type type = gguf_get_tensor_type(ctx.get(), i);
        const uint64_t size = (uint64_t) gguf_get_tensor_size(ctx.get(), i);
        const std::string type_name = ggml_type_name(type);
        summary.type_stats[type_name].count += 1;
        summary.type_stats[type_name].bytes += size;
        summary.tensor_bytes += size;

        if (type == GGML_TYPE_NVFP4) {
            ++summary.nvfp4_tensors;
        } else if (type == GGML_TYPE_MXFP6_E2M3) {
            ++summary.mxfp6_tensors;
        } else if (type == GGML_TYPE_MXFP8) {
            ++summary.mxfp8_tensors;
        }
        if (name.size() >= 6 && name.compare(name.size() - 6, 6, ".scale") == 0) {
            ++summary.scale_tensors;
        }
        if (name.size() >= 12 && name.compare(name.size() - 12, 12, ".input_scale") == 0) {
            ++summary.input_scale_tensors;
        }
        if (is_mtp_name(name)) {
            mtp_tensors.push_back(name);
        }
    }

    summary.has_mtp = !mtp_keys.empty() || !mtp_tensors.empty();
    summary.mtp_keys = mtp_keys.size();
    summary.mtp_tensors = mtp_tensors.size();
    return summary;
}

int inspect_main(int argc, char ** argv) {
    const InspectOptions options = parse_options(argc, argv);

    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    std::unique_ptr<gguf_context, GgufDeleter> ctx(gguf_init_from_file(options.path.c_str(), params));
    if (!ctx) {
        throw std::runtime_error("failed to open GGUF: " + options.path);
    }

    const InspectSummary summary = inspect_gguf(options.path);
    std::vector<int64_t> listed_tensors;
    std::vector<std::string> mtp_keys;
    std::vector<std::string> mtp_tensors;

    for (int64_t i = 0; i < gguf_get_n_kv(ctx.get()); ++i) {
        const std::string key = gguf_get_key(ctx.get(), i);
        if (is_mtp_name(key)) {
            mtp_keys.push_back(key);
        }
    }

    for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
        const std::string name = gguf_get_tensor_name(ctx.get(), i);
        if (is_mtp_name(name)) {
            mtp_tensors.push_back(name);
        }
        if (options.list_tensors && contains_ci(name, options.tensor_filter)) {
            listed_tensors.push_back(i);
        }
    }

    if (options.require_mtp && !summary.has_mtp) {
        std::cerr << "inspect: no MTP metadata or tensors found in " << options.path << "\n";
        return 3;
    }

    if (options.json) {
        std::cout << "{\n";
        std::cout << "  \"path\": \"" << json_escape(summary.path) << "\",\n";
        std::cout << "  \"version\": " << summary.version << ",\n";
        std::cout << "  \"file_bytes\": " << summary.file_bytes << ",\n";
        std::cout << "  \"metadata_keys\": " << summary.metadata_keys << ",\n";
        std::cout << "  \"tensors\": " << summary.tensors << ",\n";
        std::cout << "  \"tensor_bytes\": " << summary.tensor_bytes << ",\n";
        std::cout << "  \"nvfp4_tensors\": " << summary.nvfp4_tensors << ",\n";
        std::cout << "  \"mxfp6_e2m3_tensors\": " << summary.mxfp6_tensors << ",\n";
        std::cout << "  \"mxfp8_tensors\": " << summary.mxfp8_tensors << ",\n";
        std::cout << "  \"scale_tensors\": " << summary.scale_tensors << ",\n";
        std::cout << "  \"input_scale_tensors\": " << summary.input_scale_tensors << ",\n";
        std::cout << "  \"has_mtp\": " << (summary.has_mtp ? "true" : "false") << ",\n";
        std::cout << "  \"mtp_keys\": " << summary.mtp_keys << ",\n";
        std::cout << "  \"mtp_tensors\": " << summary.mtp_tensors << ",\n";
        std::cout << "  \"types\": {";
        bool first = true;
        for (const auto & entry : summary.type_stats) {
            if (!first) {
                std::cout << ",";
            }
            first = false;
            std::cout << "\n    \"" << json_escape(entry.first) << "\": {\"count\": " << entry.second.count
                      << ", \"bytes\": " << entry.second.bytes << "}";
        }
        std::cout << "\n  }\n";
        std::cout << "}\n";
        return 0;
    }

    std::cout << "path: " << options.path << "\n";
    std::cout << "version: " << summary.version << "\n";
    std::cout << "file: " << human_bytes(summary.file_bytes) << "\n";
    std::cout << "metadata keys: " << summary.metadata_keys << "\n";
    std::cout << "tensors: " << summary.tensors << " (" << human_bytes(summary.tensor_bytes) << " tensor payload)\n";
    std::cout << "blackwell: NVFP4=" << summary.nvfp4_tensors
              << " MXFP6_E2M3=" << summary.mxfp6_tensors
              << " MXFP8=" << summary.mxfp8_tensors
              << " .scale=" << summary.scale_tensors
              << " .input_scale=" << summary.input_scale_tensors << "\n";
    std::cout << "mtp: " << (summary.has_mtp ? "present" : "not found")
              << " (keys=" << summary.mtp_keys << ", tensors=" << summary.mtp_tensors << ")\n";

    std::cout << "\ntensor types:\n";
    for (const auto & entry : summary.type_stats) {
        std::cout << "  " << std::setw(16) << std::left << entry.first
                  << " count=" << std::setw(5) << std::right << entry.second.count
                  << " bytes=" << human_bytes(entry.second.bytes) << "\n";
    }

    if (!mtp_keys.empty()) {
        std::cout << "\nmtp metadata keys:\n";
        for (const std::string & key : mtp_keys) {
            std::cout << "  " << key << "\n";
        }
    }
    if (!mtp_tensors.empty()) {
        std::cout << "\nmtp tensors:\n";
        for (const std::string & name : mtp_tensors) {
            std::cout << "  " << name << "\n";
        }
    }

    if (options.list_keys) {
        std::cout << "\nmetadata:\n";
        for (int64_t i = 0; i < gguf_get_n_kv(ctx.get()); ++i) {
            std::cout << "  " << gguf_get_key(ctx.get(), i)
                      << " = " << compact_value(ctx.get(), i) << "\n";
        }
    }
    if (options.list_tensors) {
        std::cout << "\ntensors:\n";
        for (const int64_t id : listed_tensors) {
            const std::string name = gguf_get_tensor_name(ctx.get(), id);
            const ggml_type type = gguf_get_tensor_type(ctx.get(), id);
            const uint64_t size = (uint64_t) gguf_get_tensor_size(ctx.get(), id);
            std::cout << "  " << std::setw(18) << std::left << ggml_type_name(type)
                      << " " << std::setw(12) << std::right << human_bytes(size)
                      << "  " << name << "\n";
        }
    }

    return 0;
}

} // namespace bq
