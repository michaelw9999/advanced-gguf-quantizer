#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace bq {

struct TypeStats {
    int64_t count = 0;
    uint64_t bytes = 0;
};

struct InspectSummary {
    std::string path;
    uint32_t version = 0;
    uint64_t file_bytes = 0;
    int64_t metadata_keys = 0;
    int64_t tensors = 0;
    uint64_t tensor_bytes = 0;
    int64_t nvfp4_tensors = 0;
    int64_t mxfp6_tensors = 0;
    int64_t mxfp8_tensors = 0;
    int64_t scale_tensors = 0;
    int64_t input_scale_tensors = 0;
    bool has_mtp = false;
    size_t mtp_keys = 0;
    size_t mtp_tensors = 0;
    std::map<std::string, TypeStats> type_stats;
};

InspectSummary inspect_gguf(const std::string & path);
int inspect_main(int argc, char ** argv);

} // namespace bq
