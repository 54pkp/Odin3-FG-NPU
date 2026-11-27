// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "odin3_htp_pair_types.hpp"
#include <utility>
#include "odin3_shared_dma.hpp"

namespace odin3_direct_pair {
namespace fs = htp_probe::fs;
using htp_probe::Clock;
using htp_probe::DmaArena;
using htp_probe::elapsed_us;

using odin3_htp_pair::Execution;
using odin3_htp_pair::FrameSize;
using odin3_htp_pair::TensorAbi;
using odin3_htp_pair::configured_analysis_stride;

namespace detail {
using htp_probe::check;
using htp_probe::Htp;
using htp_probe::tensor_name;

enum Slot { frame_a, frame_b, motion, first_frame, second_frame, spare_source, luma_0, luma_1,
            spare_source_2, spare_source_3, spare_source_4, spare_source_5, spare_source_6, spare_source_7,
            spare_source_8, spare_source_9, spare_source_10, slot_count };
inline constexpr std::array<std::string_view, slot_count> slot_names{
    "frame_a", "frame_b", "motion", "frame", "frame", "spare_source", "history_a", "history_b",
    "spare_source_2", "spare_source_3", "spare_source_4", "spare_source_5", "spare_source_6", "spare_source_7",
    "spare_source_8", "spare_source_9", "spare_source_10"
};
using odin3_htp_pair::field_shape;

template<class Descriptor>
TensorAbi inspect(const Descriptor& desc, Slot slot, bool output, FrameSize frame) {
    const std::array<uint32_t, 4> rgba_shape{1, frame.height, frame.width, 4};
    const bool luma = slot == luma_0 || slot == luma_1;
    const std::string name(slot_names[luma ? (output ? luma_1 : luma_0) : slot]);
    if (!desc.name || desc.name != name || desc.rank != rgba_shape.size() || !desc.dimensions ||
        desc.dataFormat != QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER ||
        desc.type != (output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_APP_WRITE))
        throw std::runtime_error("Unexpected static NHWC descriptor for " + name);

    std::array<uint32_t, 4> shape{};
    std::copy_n(desc.dimensions, shape.size(), shape.begin());
    const uint32_t pad_h32 = ((frame.height + 31u) / 32u) * 32u;
    const uint32_t pad_w32 = ((frame.width + 31u) / 32u) * 32u;
    const bool pure_latent = slot == motion && desc.dataType == QNN_DATATYPE_UFIXED_POINT_8 && shape[0] == 1 &&
        ((shape[1] == pad_h32 / 4u && shape[2] == pad_w32 / 4u) ||
         (shape[1] == pad_h32 / 8u && shape[2] == pad_w32 / 8u)) &&
        shape[3] >= 1 && shape[3] <= 512;
    const bool supported_shape = slot == motion
        ? (pure_latent || shape == field_shape(frame, 16) || shape == field_shape(frame, 8) ||
           shape == std::array<uint32_t, 4>{1, frame.height / 4, frame.width / 4, 28})
        : luma ? shape == std::array<uint32_t, 4>{1, frame.height / 4, frame.width / 4, 1} : shape == rgba_shape;
    if (!supported_shape)
        throw std::runtime_error("Unexpected NHWC shape for " + name);
    if (slot == motion && desc.dataType != QNN_DATATYPE_FLOAT_16 && !pure_latent)
        throw std::runtime_error("Expected signed FP16 displacement or UFIXED8 latent for motion");
    if (slot != motion && desc.dataType != QNN_DATATYPE_UINT_8 && desc.dataType != QNN_DATATYPE_UFIXED_POINT_8)
        throw std::runtime_error("Expected native UINT8 or UFIXED8 RGBA for " + name);

    const std::size_t elem_bytes = (slot == motion && desc.dataType == QNN_DATATYPE_FLOAT_16) ? 2 : 1;
    const auto bytes = static_cast<std::size_t>(shape[1]) * shape[2] * shape[3] * elem_bytes;
    TensorAbi abi{desc.dataType, shape, bytes, {}};
    const auto encoding = desc.quantizeParams.quantizationEncoding;
    if (luma && (desc.dataType != QNN_DATATYPE_UFIXED_POINT_8 ||
            encoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
            desc.quantizeParams.scaleOffsetEncoding.scale != 1.0f / 255.0f ||
            desc.quantizeParams.scaleOffsetEncoding.offset != 0))
        throw std::runtime_error("Expected shared 1/255 U8 luma encoding for " + name);
    // QAIRT preserves the graph's terminal QuantizeLinear as native bytes
    // without serializing a scale for the application-facing output tensor.
    if (output && slot != motion && desc.dataType == QNN_DATATYPE_UFIXED_POINT_8 &&
        encoding == QNN_QUANTIZATION_ENCODING_UNDEFINED &&
        desc.quantizeParams.encodingDefinition == QNN_DEFINITION_UNDEFINED)
        return abi;
    if (desc.dataType == QNN_DATATYPE_UFIXED_POINT_8) {
        if (encoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
            throw std::runtime_error("Expected per-tensor scale-offset encoding for " + name);
        abi.encoding = desc.quantizeParams.scaleOffsetEncoding;
        if (!std::isfinite(abi.encoding->scale) || abi.encoding->scale <= 0)
            throw std::runtime_error("Invalid quantization scale for " + name);
        if (!pure_latent) {
            const bool rgba_scale = abi.encoding->scale == (1.0f / 255.0f) ||
                                    (output && abi.encoding->scale == 1.0f);
            if (!rgba_scale || abi.encoding->offset != 0)
                throw std::runtime_error("Unexpected RGBA8 encoding for " + name);
        }
    } else if (encoding != QNN_QUANTIZATION_ENCODING_UNDEFINED) {
        throw std::runtime_error("Unexpected quantization encoding for " + name);
    }
    return abi;
}

inline TensorAbi inspect(const Qnn_Tensor_t& tensor, Slot slot, bool output, FrameSize frame) {
    if (tensor.version == QNN_TENSOR_VERSION_1)
        return inspect(tensor.v1, slot, output, frame);
    if (tensor.version != QNN_TENSOR_VERSION_2)
        throw std::runtime_error("Unsupported QNN tensor descriptor version");
    const auto* dynamic = tensor.v2.isDynamicDimensions;
    if (tensor.v2.rank == 4 && dynamic && std::any_of(dynamic, dynamic + 4, [](uint8_t d) { return d != 0; }))
        throw std::runtime_error("The raw HTP pipeline only accepts static tensors");
    return inspect(tensor.v2, slot, output, frame);
}

using Context = odin3_htp_pair::detail::Context<slot_count>;

using Catalog = std::array<std::optional<TensorAbi>, slot_count>;
using Buffers = std::array<std::unique_ptr<DmaArena>, slot_count>;

struct Invocation {
    Context* context{};
    const char* name{};
    Qnn_GraphHandle_t graph{};
    std::array<Qnn_Tensor_t, 3> inputs{};
    std::array<Slot, 3> input_slots{};
    uint32_t input_count{};
    std::array<Qnn_Tensor_t, 2> outputs{};
    uint32_t output_count{};
    Slot output_slot{};

    void configure(Context& owner, std::string_view graph_name, Slot result, Catalog& catalog,
            FrameSize frame, bool write_luma = false, bool read_luma = false, bool joined = false) {
        const auto& desc = owner.find_graph(graph_name);
        context = &owner;
        name = desc.name;
        output_slot = result;
        input_count = result == motion || joined ? 2 : 3;
        output_count = write_luma ? 2 : 1;
        if (!desc.inputs || !desc.outputs || desc.input_count != input_count || desc.output_count != output_count)
            throw std::runtime_error("Unexpected I/O tensor count in " + std::string(name));
        const std::array expected_slots{read_luma ? luma_0 : frame_a, frame_b, motion};
        const auto end = expected_slots.begin() + input_count;
        std::array<bool, 3> seen{};
        for (uint32_t i = 0; i < input_count; ++i) {
            inputs[i] = desc.inputs[i];
            const char* input_name = tensor_name(inputs[i]);
            const auto expected = input_name ? std::find_if(expected_slots.begin(), end, [input_name](Slot slot) {
                return slot_names[slot] == input_name;
            }) : end;
            if (expected == end || seen[expected - expected_slots.begin()])
                throw std::runtime_error("Unexpected or duplicate input in " + std::string(name));
            const auto slot = *expected;
            seen[expected - expected_slots.begin()] = true;
            input_slots[i] = slot;
            inspect_slot(inputs[i], slot, false, catalog, frame);
        }
        outputs[0] = desc.outputs[0];
        inspect_slot(outputs[0], output_slot, true, catalog, frame);
        if (write_luma) {
            outputs[1] = desc.outputs[1];
            inspect_slot(outputs[1], luma_0, true, catalog, frame);
        }
    }

    void inspect_slot(const Qnn_Tensor_t& tensor, Slot slot, bool is_output, Catalog& catalog, FrameSize frame) const {
        const auto abi = inspect(tensor, slot, is_output, frame);
        if (catalog[slot] && !catalog[slot]->matches(abi))
            throw std::runtime_error("Shared tensor ABI mismatch for " + std::string(slot_names[slot]) +
                                     " in " + name);
        catalog[slot] = abi;
    }

    void retrieve() {
        check(context->htp.q.graphRetrieve(context->handle, name, &graph),
              ("Retrieve graph " + std::string(name)).c_str());
    }

    template<class Descriptor>
    void bind(Descriptor& desc, Slot slot, const Buffers& buffers) {
        auto& registration = context->registrations[slot];
        if (!registration) {
            auto& arena = *buffers[slot];
            arena.register_tensor(desc, context->handle, 0, arena.size());
            registration = desc.memHandle;
        } else {
            desc.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            desc.memHandle = registration;
        }
    }

    void bind(Qnn_Tensor_t& tensor, Slot slot, const Buffers& buffers) {
        if (tensor.version == QNN_TENSOR_VERSION_1)
            bind(tensor.v1, slot, buffers);
        else
            bind(tensor.v2, slot, buffers);
    }

    void register_buffers(const Buffers& buffers) {
        for (uint32_t i = 0; i < input_count; ++i)
            bind(inputs[i], input_slots[i], buffers);
        bind(outputs[0], output_slot, buffers);
        if (output_count == 2) {
            bind(outputs[1], luma_0, buffers);
            auto descriptor = outputs[1];
            bind(descriptor, luma_1, buffers);
        }
    }

    void register_spare(const Buffers& buffers, Slot spare) {
        for (uint32_t i = 0; i < input_count; ++i) {
            if (input_slots[i] != frame_a && input_slots[i] != frame_b) continue;
            auto descriptor = inputs[i];
            bind(descriptor, spare, buffers);
            return;
        }
        throw std::logic_error("Missing endpoint descriptor for the spare source");
    }
    void retarget_endpoints(const Buffers& buffers, Slot a, Slot b) {
        for (uint32_t i = 0; i < input_count; ++i) {
            const auto slot = input_slots[i];
            if (slot == frame_a || slot == frame_b)
                bind(inputs[i], slot == frame_a ? a : b, buffers);
        }
    }

    void retarget_history(const Buffers& buffers, unsigned previous, unsigned next) {
        for (uint32_t i = 0; i < input_count; ++i)
            if (input_slots[i] == luma_0) bind(inputs[i], previous ? luma_1 : luma_0, buffers);
        bind(outputs[1], next ? luma_1 : luma_0, buffers);
    }

    Execution execute() { return execute_with_outputs(outputs.data()); }

    Execution execute_with_outputs(Qnn_Tensor_t* descriptors, Qnn_ProfileHandle_t profile = nullptr) {
        const auto start = Clock::now();
        const auto status = context->htp.q.graphExecute(graph, inputs.data(), input_count,
            descriptors, output_count, profile, nullptr);
        const auto done = Clock::now();
        check(status, name);
        return {start, done};
    }
};

} // namespace detail

class Pipeline {
    htp_probe::Htp htp;
    std::vector<std::unique_ptr<detail::Context>> contexts;
    detail::Catalog catalog;
    detail::Invocation motion_graph;
    detail::Invocation cold_graph, warm_graph;
    std::array<detail::Invocation, 2> phases;
    detail::Invocation half_in_triple;
    detail::Buffers buffers;
    struct PooledOutput {
        std::unique_ptr<DmaArena> arena;
        Qnn_Tensor_t tensor{};
    };
    std::array<PooledOutput, 4> extra_output_slots;
    std::array<PooledOutput, 2> half_in_triple_extra_slots;
    struct History {
        uint64_t ordinal;
        unsigned source_b, luma_buffer;
    };
    std::optional<History> history;
    const unsigned phase_total;
    const bool joined_motion;
    Clock::time_point setup_start;

    detail::Context& get_context(const fs::path& path) {
        const auto absolute = fs::canonical(path);
        for (const auto& context : contexts)
            if (context->path == absolute)
                return *context;
        auto context = std::make_unique<detail::Context>(htp, absolute);
        context->read_metadata();
        return *contexts.emplace_back(std::move(context));
    }

    detail::Slot input_slot(unsigned index) const {
        constexpr std::array sources{detail::frame_a, detail::frame_b, detail::spare_source, detail::spare_source_2,
                                     detail::spare_source_3, detail::spare_source_4, detail::spare_source_5, detail::spare_source_6,
                                     detail::spare_source_7, detail::spare_source_8, detail::spare_source_9, detail::spare_source_10};
        if (index >= sources.size() || !buffers[sources[index]])
            throw std::out_of_range("Raw HTP input index");
        return sources[index];
    }

    detail::Slot output_slot(unsigned index) const {
        if (index >= phase_total)
            throw std::out_of_range("Raw HTP phase index");
        return index == 0 ? detail::first_frame : detail::second_frame;
    }

public:
    Pipeline(const fs::path& motion_context, const fs::path& first_context,
             const fs::path& second_context, bool top_power, bool map_for_cpu = false,
             unsigned analysis_stride = 4, std::array<int, 2> output_fds = {-1, -1},
             std::array<int, 12> input_fds = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, const char* phase_op_package = nullptr,
             std::pair<fs::path, fs::path> luma_contexts = {}, FrameSize frame = {}, bool joined = false,
             const odin3::shared_frame::DmaSourcePool* described_sources = nullptr,
             const fs::path& half_in_triple_context = {})
        : phase_total(second_context.empty() ? 1 : 2), joined_motion(joined) {
        if (analysis_stride != 2 && analysis_stride != 4 && analysis_stride != 8)
            throw std::runtime_error("Unsupported static raw HTP analysis stride");
        if (frame.width < 64 || frame.width % 16 != 0 || frame.height < 64 || frame.height % 4 != 0)
            throw std::runtime_error("Raw HTP frame extent must be a positive multiple of 16x8");
        if (phase_op_package && *phase_op_package &&
            (std::string_view(phase_op_package) != "libOdinPhaseStrip.so" ||
             (phase_total != 1 && phase_total != 2) || analysis_stride != 4))
            throw std::runtime_error("The QHPI phase package requires 25% 2x or 3x phases");
        if (joined && (analysis_stride != 4 || !motion_context.empty() ||
                       !luma_contexts.first.empty() || !luma_contexts.second.empty()))
            throw std::runtime_error("The joined pure HTP graph requires empty motion/luma contexts");
        if (described_sources)
            odin3::shared_frame::validate_dma_source_pool(*described_sources, input_fds, frame.width, frame.height);
        htp.open();
        if (phase_op_package && *phase_op_package) {
            if (!htp.q.backendRegisterOpPackage)
                throw std::runtime_error("The QNN backend cannot register the QHPI phase package");
            htp_probe::check(htp.q.backendRegisterOpPackage(
                htp.backend, phase_op_package, "OdinPhaseStripInterfaceProvider", "HTP"),
                "Register the QHPI phase package");
        }
        const char* motion_op_package = std::getenv("ODIN3_HTP_RAW_MOTION_OP_PACKAGE");
        if (motion_op_package && std::string_view(motion_op_package) == "libOdinOdin3FgMotion.so")
            htp_probe::check(htp.q.backendRegisterOpPackage(htp.backend, "libOdinOdin3FgMotion.so",
                "OdinOdin3FgMotionInterfaceProvider", "HTP"), "Register the QHPI motion package");
        if (!htp.q.memRegister || !htp.q.memDeRegister)
            throw std::runtime_error("HTP memory registration APIs are unavailable");
        if (top_power)
            htp.enable_top_power();

        const char* motionName = analysis_stride == 8 ? "fg4_motion_s8" :
                                 analysis_stride == 2 ? "fg4_motion_s2" : "fg4_motion";
        const bool extendedHalo = analysis_stride == 8;
        const char* firstName = phase_total == 2
            ? (extendedHalo ? "fg4_phase_1_3_s8" : "fg4_phase_1_3")
            : (extendedHalo ? "fg4_phase_1_2_s8" : "fg4_phase_1_2");
        if (has_separate_motion())
            motion_graph.configure(get_context(motion_context), motionName, detail::motion, catalog, frame);
        if (!luma_contexts.first.empty() || !luma_contexts.second.empty()) {
            if (input_fds[2] < 0 || analysis_stride != 4 || !phase_op_package ||
                    luma_contexts.first.empty() || luma_contexts.second.empty())
                throw std::runtime_error("Luma history requires both contexts and 25% QHPI shared sources");
            cold_graph.configure(get_context(luma_contexts.first), "fg4_motion_cold", detail::motion, catalog, frame, true);
            warm_graph.configure(get_context(luma_contexts.second), "fg4_motion_warm", detail::motion, catalog, frame, true, true);
            catalog[detail::luma_1] = catalog[detail::luma_0];
        }
        phases[0].configure(get_context(first_context),
                            joined ? (phase_total == 2 ? "fg4_phase_1_3" : "fg4_joined_1_2") : firstName,
                            detail::first_frame, catalog, frame, false, false, joined);
        if (phase_total == 2) {
            const char* secondName = extendedHalo ? "fg4_phase_2_3_s8" : "fg4_phase_2_3";
            phases[1].configure(get_context(second_context), secondName, detail::second_frame, catalog, frame, false, false, joined);
            if (!half_in_triple_context.empty() && fs::is_regular_file(half_in_triple_context)) {
                half_in_triple.configure(get_context(half_in_triple_context),
                    extendedHalo ? "fg4_phase_1_2_s8" : "fg4_phase_1_2",
                    detail::first_frame, catalog, frame, false, false, false);
            }
        }
        for (const auto& context : contexts)
            context->load();
        if (has_separate_motion())
            motion_graph.retrieve();
        for (unsigned index = 0; index < phase_total; ++index)
            phases[index].retrieve();
        if (half_in_triple.context)
            half_in_triple.retrieve();
        setup_start = Clock::now();
        constexpr std::array spare_slots{detail::spare_source, detail::spare_source_2, detail::spare_source_3,
                                         detail::spare_source_4, detail::spare_source_5, detail::spare_source_6,
                                         detail::spare_source_7, detail::spare_source_8, detail::spare_source_9, detail::spare_source_10};
        if (input_fds[2] >= 0) {
            if (input_fds[0] < 0 || input_fds[1] < 0 || !catalog[detail::frame_a]->matches(*catalog[detail::frame_b]))
                throw std::runtime_error("Shared sources require compatible linear DMA imports");
            for (size_t k = 0; k < spare_slots.size(); ++k) {
                if (input_fds[2 + k] >= 0) catalog[spare_slots[k]] = catalog[detail::frame_a];
            }
        }

        // Separate allocations allow independent cache sync and consumer ownership
        // for each output. The motion buffer remains device-owned during execution.
        for (std::size_t slot = 0; slot < buffers.size(); ++slot) {
            int borrowed = -1;
            if (slot == detail::frame_a || slot == detail::frame_b) borrowed = input_fds[slot == detail::frame_b];
            for (size_t k = 0; k < spare_slots.size(); ++k)
                if (slot == spare_slots[k]) borrowed = input_fds[2 + k];
            if (slot == detail::first_frame || slot == detail::second_frame) borrowed = output_fds[slot == detail::second_frame];
            if (catalog[slot])
                buffers[slot] = std::make_unique<DmaArena>(htp, catalog[slot]->bytes, contexts.size(),
                                                           map_for_cpu && slot != detail::motion, borrowed);
        }
        if (has_separate_motion())
            motion_graph.register_buffers(buffers);
        for (unsigned index = 0; index < phase_total; ++index)
            phases[index].register_buffers(buffers);
        if (half_in_triple.context)
            half_in_triple.register_buffers(buffers);
        for (auto spare : spare_slots) {
            if (!buffers[spare]) continue;
            if (has_separate_motion()) motion_graph.register_spare(buffers, spare);
            for (unsigned index = 0; index < phase_total; ++index) phases[index].register_spare(buffers, spare);
            if (half_in_triple.context) half_in_triple.register_spare(buffers, spare);
        }
        if (cold_graph.context) {
            for (auto* graph : {&cold_graph, &warm_graph}) {
                graph->retrieve();
                graph->register_buffers(buffers);
                for (auto spare : spare_slots)
                    if (buffers[spare]) graph->register_spare(buffers, spare);
            }
        }
        if (described_sources) {
            for (size_t slot = 0; slot < described_sources->sources.size(); ++slot) {
                const auto& source = described_sources->sources[slot];
                std::fprintf(stderr, "Odin3 shared DMA: epoch=%ju slot=%zu device=%ju inode=%ju bytes=%ju width=%u height=%u stride=%u accepted=1\n",
                    uintmax_t(described_sources->epoch), slot, uintmax_t(source.identity.device),
                    uintmax_t(source.identity.inode), uintmax_t(source.layout.allocation_bytes),
                    source.layout.width, source.layout.height, source.layout.row_stride);
            }
        }
    }

    // Slot zero is the regular half-phase output. Register each additional
    // pixmap's DMA-BUF once, in the same context as the half-phase graph.
    void register_half_phase_output_slots(std::array<int, 2> fds) {
        if (phase_total != 1 || fds[0] < 0 || fds[1] < 0 || extra_output_slots[0].arena)
            throw std::invalid_argument("Invalid half-phase DMA-BUF pool");
        auto& graph = phases[0];
        const auto bytes = buffers[detail::first_frame]->size();
        for (size_t i = 0; i < fds.size(); ++i) {
            auto& slot = extra_output_slots[i];
            slot.arena = std::make_unique<DmaArena>(htp, bytes, 1, false, fds[i]);
            slot.tensor = graph.outputs[0];
            slot.arena->register_tensor(slot.tensor, graph.context->handle, 0, bytes);
        }
    }

    void register_triple_phase_output_slots(std::array<int, 4> fds) {
        if (phase_total != 2 || std::any_of(fds.begin(), fds.end(), [](int fd) { return fd < 0; }) || extra_output_slots[0].arena)
            throw std::invalid_argument("Invalid triple-phase DMA-BUF pool");
        for (size_t group = 0; group < 2; ++group) {
            for (size_t phase = 0; phase < 2; ++phase) {
                auto& graph = phases[phase];
                const auto bytes = buffers[phase == 0 ? detail::first_frame : detail::second_frame]->size();
                auto& slot = extra_output_slots[group * 2 + phase];
                slot.arena = std::make_unique<DmaArena>(htp, bytes, 1, false, fds[group * 2 + phase]);
                slot.tensor = graph.outputs[0];
                slot.arena->register_tensor(slot.tensor, graph.context->handle, 0, bytes);
            }
        }
        if (half_in_triple.context) {
            const auto bytes = buffers[detail::first_frame]->size();
            for (size_t group = 0; group < 2; ++group) {
                auto& slot = half_in_triple_extra_slots[group];
                slot.arena = std::make_unique<DmaArena>(htp, bytes, 1, false, fds[group * 2]);
                slot.tensor = half_in_triple.outputs[0];
                slot.arena->register_tensor(slot.tensor, half_in_triple.context->handle, 0, bytes);
            }
        }
    }

    Execution render_phase_to_group(unsigned phase, unsigned group, Qnn_ProfileHandle_t profile = nullptr) {
        if (phase_total == 1) {
            if (phase != 0) throw std::out_of_range("Half-phase index out of range");
            return render_half_phase_to_slot(group, profile);
        }
        if (phase >= 2 || group >= 3) throw std::out_of_range("Triple-phase pool index out of range");
        if (group == 0) return phases[phase].execute_with_outputs(phases[phase].outputs.data(), profile);
        auto& destination = extra_output_slots.at((group - 1) * 2 + phase);
        if (!destination.arena) throw std::logic_error("Missing triple-phase pool registration");
        return phases[phase].execute_with_outputs(&destination.tensor, profile);
    }

    DmaArena& phase_buffer(unsigned phase, unsigned group) const {
        if (phase_total == 1) return half_phase_buffer(group);
        if (phase >= 2 || group >= 3) throw std::out_of_range("Triple-phase buffer index out of range");
        if (group == 0) return output(phase);
        const auto& destination = extra_output_slots.at((group - 1) * 2 + phase);
        if (!destination.arena) throw std::logic_error("Missing triple-phase pool registration");
        return *destination.arena;
    }

    const htp_probe::Htp& live_profile_backend() const { return htp; }

    Execution render_half_phase_to_slot(unsigned slot, Qnn_ProfileHandle_t profile = nullptr) {
        if (phase_total != 1) throw std::logic_error("Expected a half-phase graph");
        if (!slot) return phases[0].execute_with_outputs(phases[0].outputs.data(), profile);
        auto& destination = extra_output_slots.at(slot - 1);
        if (!destination.arena) throw std::logic_error("Missing half-phase pool registration");
        return phases[0].execute_with_outputs(&destination.tensor, profile);
    }

    DmaArena& half_phase_buffer(unsigned slot) const {
        if (phase_total != 1) throw std::logic_error("Expected a half-phase graph");
        if (!slot) return output(0);
        const auto& destination = extra_output_slots.at(slot - 1);
        if (!destination.arena) throw std::logic_error("Missing half-phase pool registration");
        return *destination.arena;
    }

    unsigned phase_count() const { return phase_total; }
    bool has_separate_motion() const { return !joined_motion; }
    DmaArena& input(unsigned index) const { return *buffers[input_slot(index)]; }
    DmaArena& output(unsigned index) const { return *buffers[output_slot(index)]; }
    void set_input_order(bool reversed) {
        set_input_slots(reversed ? 1 : 0, reversed ? 0 : 1);
    }
    void set_input_slots(unsigned a, unsigned b) {
        if (a == b) throw std::invalid_argument("HTP endpoints must be distinct");
        if (!catalog[detail::frame_a]->matches(*catalog[detail::frame_b]))
            throw std::runtime_error("Raw endpoint DMA allocations have different ABIs");
        const auto source_a = input_slot(a), source_b = input_slot(b);
        if (has_separate_motion())
            motion_graph.retarget_endpoints(buffers, source_a, source_b);
        for (unsigned index = 0; index < phase_total; ++index)
            phases[index].retarget_endpoints(buffers, source_a, source_b);
        if (half_in_triple.context)
            half_in_triple.retarget_endpoints(buffers, source_a, source_b);
    }

    Execution estimate() {
        if (!has_separate_motion()) throw std::logic_error("Joined motion executes with the half phase");
        return motion_graph.execute();
    }
    bool luma_history_enabled() const { return cold_graph.context != nullptr; }
    void discard_luma_history() { history.reset(); }
    std::pair<Execution, unsigned> estimate_shared(unsigned a, unsigned b, uint64_t ordinal) {
        if (!luma_history_enabled()) return {estimate(), 0};
        // Slot numbers can be recycled after bypass. The Renderer ordinal proves adjacency.
        const bool reuse = history && ordinal > history->ordinal && ordinal - history->ordinal == 1 &&
            history->source_b == a;
        const unsigned previous = reuse ? history->luma_buffer : 0;
        const unsigned next = reuse ? 1 - previous : 0;
        history.reset();
        auto& graph = reuse ? warm_graph : cold_graph;
        graph.retarget_endpoints(buffers, input_slot(a), input_slot(b));
        graph.retarget_history(buffers, previous, next);
        const auto result = graph.execute();
        history = History{ordinal, b, next};
        return {result, reuse ? 2u : 1u};
    }

    Execution render(unsigned index) {
        if (index >= phase_total)
            throw std::out_of_range("Raw HTP phase index");
        return phases[index].execute();
    }
    Execution render_midpoint() {
        if (phase_total == 1) return phases[0].execute();
        if (!half_in_triple.context) return phases[0].execute();
        return half_in_triple.execute();
    }
    Execution render_midpoint_to_group(unsigned group, Qnn_ProfileHandle_t profile = nullptr) {
        if (phase_total == 1) return render_half_phase_to_slot(group, profile);
        if (!half_in_triple.context) return render_phase_to_group(0, group, profile);
        if (group >= 3) throw std::out_of_range("Triple midpoint group out of range");
        if (group == 0) return half_in_triple.execute_with_outputs(half_in_triple.outputs.data(), profile);
        auto& destination = half_in_triple_extra_slots.at(group - 1);
        if (!destination.arena) throw std::logic_error("Missing triple midpoint pool registration");
        return half_in_triple.execute_with_outputs(&destination.tensor, profile);
    }

    // Diagnostics for the replay CLI. Motion is not a live host input.
    DmaArena& motion_buffer() const {
        if (!has_separate_motion())
            throw std::logic_error("Motion is internal to the joined graph");
        return *buffers[detail::motion];
    }
    const TensorAbi& motion_abi() const { return catalog[detail::motion].value(); }
    const TensorAbi& input_abi(unsigned index) const { return *catalog[input_slot(index)]; }
    const TensorAbi& output_abi(unsigned index) const { return *catalog[output_slot(index)]; }
    const char* motion_graph_name() const { return has_separate_motion() ? motion_graph.name : "internal"; }
    const char* phase_graph_name(unsigned index) const { return phases[output_slot(index) - detail::first_frame].name; }
    std::size_t context_count() const { return contexts.size(); }
    Clock::time_point buffer_setup_start() const { return setup_start; }
};
} // namespace odin3_direct_pair
