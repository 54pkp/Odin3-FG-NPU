// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "htp_probe_timing_common.hpp"

#include <utility>

namespace odin3_htp_pair {
namespace fs = htp_probe::fs;
using htp_probe::Clock;
using htp_probe::elapsed_us;

struct TensorAbi {
    Qnn_DataType_t type{};
    std::array<uint32_t, 4> shape{};
    std::size_t bytes{};
    std::optional<Qnn_ScaleOffset_t> encoding;

    bool matches(const TensorAbi& other) const {
        return type == other.type && shape == other.shape && bytes == other.bytes &&
               encoding.has_value() == other.encoding.has_value() &&
               (!encoding || (encoding->scale == other.encoding->scale && encoding->offset == other.encoding->offset));
    }
};

struct Execution {
    Clock::time_point start, done;

    double duration_us() const { return elapsed_us(start, done); }
};

struct FrameSize {
    uint32_t width = htp_probe::width;
    uint32_t height = htp_probe::height;
};

inline std::array<uint32_t, 4> field_shape(FrameSize frame, uint32_t block) {
    const auto blocks = [block](uint32_t size) { return size / block + (size % block != 0); };
    return {1, blocks(frame.height), blocks(frame.width), 2};
}

inline unsigned configured_analysis_stride() {
    const auto* value = std::getenv("ODIN3_HTP_RAW_FLOW_STRIDE");
    if (!value) return 4;
    const std::string_view stride(value);
    if (stride == "8") return 8;
    if (stride == "4") return 4;
    if (stride == "2") return 2;
    throw std::runtime_error("Unsupported static raw HTP analysis stride");
}

// The caller owns producer/consumer synchronization and serializes calls for each
// pipeline. Borrowed DMA-BUF descriptors stay valid for the pipeline's lifetime.
namespace detail {
using htp_probe::check;
using htp_probe::Htp;
using htp_probe::read_file;

struct GraphDescription {
    const char* name;
    const Qnn_Tensor_t* inputs;
    const Qnn_Tensor_t* outputs;
    uint32_t input_count, output_count;
};

template<class Info>
GraphDescription describe_graph(const Info& info) {
    return {info.graphName, info.graphInputs, info.graphOutputs, info.numGraphInputs, info.numGraphOutputs};
}

inline GraphDescription describe_graph(const QnnSystemContext_GraphInfo_t& graph) {
    switch (graph.version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: return describe_graph(graph.graphInfoV1);
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: return describe_graph(graph.graphInfoV2);
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: return describe_graph(graph.graphInfoV3);
    default: throw std::runtime_error("Unsupported QNN context graph-info version");
    }
}

template<std::size_t SlotCount>
struct Context {
    const Htp& htp;
    const fs::path path;
    std::vector<uint8_t> binary;
    QnnSystemContext_Handle_t metadata{};
    Qnn_ContextHandle_t handle{};
    std::vector<GraphDescription> graphs;
    std::array<Qnn_MemHandle_t, SlotCount> registrations{};

    Context(const Htp& htp, fs::path path) : htp(htp), path(std::move(path)) {}
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ~Context() {
        if (handle) htp.q.contextFree(handle, nullptr);
        if (metadata) htp.s.systemContextFree(metadata);
    }

    template<class Info>
    void read_graphs(const Info& info) {
        if (info.backendId != QNN_BACKEND_ID_HTP || !info.graphs || !info.numGraphs)
            throw std::runtime_error("Expected HTP graph metadata in " + path.string());
        for (uint32_t i = 0; i < info.numGraphs; ++i)
            graphs.push_back(describe_graph(info.graphs[i]));
    }

    void read_metadata() {
        binary = read_file(path);
        check(htp.s.systemContextCreate(&metadata), "Create raw replay metadata handle");
        const QnnSystemContext_BinaryInfo_t* info = nullptr;
        Qnn_ContextBinarySize_t info_bytes = 0;
        check(htp.s.systemContextGetBinaryInfo(metadata, binary.data(), binary.size(), &info, &info_bytes),
              "Read raw replay context metadata");
        if (!info)
            throw std::runtime_error("Missing QNN context metadata in " + path.string());
        switch (info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: read_graphs(info->contextBinaryInfoV1); break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: read_graphs(info->contextBinaryInfoV2); break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: read_graphs(info->contextBinaryInfoV3); break;
        default: throw std::runtime_error("Unsupported QNN context metadata version");
        }
    }

    const GraphDescription& find_graph(std::string_view name) const {
        const auto matches_name = [name](const GraphDescription& graph) { return graph.name && graph.name == name; };
        const auto graph = std::find_if(graphs.begin(), graphs.end(), matches_name);
        if (graph == graphs.end()) {
            if (graphs.size() == 1) return graphs[0];
            throw std::runtime_error("Expected one graph named " + std::string(name) + " in " + path.string());
        }
        if (std::find_if(graph + 1, graphs.end(), matches_name) != graphs.end())
            throw std::runtime_error("Duplicate graph named " + std::string(name) + " in " + path.string());
        return *graph;
    }

    void load() {
        check(htp.q.contextCreateFromBinary(htp.backend, htp.device, nullptr, binary.data(), binary.size(),
                                            &handle, nullptr), ("Load context " + path.string()).c_str());
    }
};

} // namespace detail
} // namespace odin3_htp_pair
