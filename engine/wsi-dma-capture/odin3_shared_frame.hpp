// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <limits>

namespace odin3::shared_frame {

enum class Format : uint32_t { unknown, bgra8 };
enum class ColorSpace : uint32_t { unknown, srgb_nonlinear };

struct Layout {
    uint32_t width{}, height{};
    Format format{};
    ColorSpace color_space{};
    uint32_t plane_count{}, row_stride{};
    uint64_t offset{}, modifier{}, allocation_bytes{};
};

struct AllocationIdentity {
    uint64_t device{}, inode{};
    constexpr bool operator==(const AllocationIdentity&) const = default;
};

enum class InputRejection { none, extent, interpretation, planes, modifier, offset, row_stride, allocation };

// Scheme 4 consumes a flat tensor from byte zero. The producer reports its
// real layout so differently laid out inputs can take a compatible fallback.
constexpr InputRejection packed_bgra8_sdr(const Layout& input, uint32_t width, uint32_t height) {
    if (!width || !height || width > std::numeric_limits<uint32_t>::max() / 4 ||
        input.width != width || input.height != height)
        return InputRejection::extent;
    if (input.format != Format::bgra8 || input.color_space != ColorSpace::srgb_nonlinear)
        return InputRejection::interpretation;
    if (input.plane_count != 1) return InputRejection::planes;
    if (input.modifier != 0) return InputRejection::modifier;
    if (input.offset != 0) return InputRejection::offset;
    if (input.row_stride != 4 * width) return InputRejection::row_stride;
    if (input.allocation_bytes != uint64_t{input.row_stride} * height) return InputRejection::allocation;
    return InputRejection::none;
}

constexpr const char* reason(InputRejection rejection) {
    switch (rejection) {
    case InputRejection::none: return "accepted";
    case InputRejection::extent: return "extent";
    case InputRejection::interpretation: return "format-or-color-space";
    case InputRejection::planes: return "plane-count";
    case InputRejection::modifier: return "modifier";
    case InputRejection::offset: return "offset";
    case InputRejection::row_stride: return "row-stride";
    case InputRejection::allocation: return "allocation-size";
    }
    return "unknown";
}

struct FrameKey {
    uint64_t epoch{};
    uint32_t slot{};
    uint64_t generation{};
    constexpr bool operator==(const FrameKey&) const = default;
};

struct Readers {
    bool queued{}, history{}, display{};
};

// The adapter owns the actual GPU handoff and sets these facts while holding
// its source locks. A display receipt or interpolation deadline is not a release.
class Lease {
public:
    explicit constexpr Lease(FrameKey key) : key_(key) {}

    template <class HtpFinished>
    constexpr bool can_submit_return(FrameKey frame, Readers readers, HtpFinished htp_finished) const {
        return frame == key_ && !return_submitted_ && !readers.queued && !readers.history &&
               !readers.display && htp_finished();
    }
    // Call only after a successful reverse GPU submission for this generation.
    constexpr void mark_return_submitted() { return_submitted_ = true; }
    constexpr bool can_release(FrameKey frame, bool reverse_complete) const {
        return frame == key_ && return_submitted_ && reverse_complete;
    }
    constexpr bool return_submitted() const { return return_submitted_; }

private:
    FrameKey key_;
    bool return_submitted_{};
};
}
