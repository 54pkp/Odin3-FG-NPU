// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "odin3_shared_frame.hpp"

#include <array>
#include <type_traits>

namespace odin3::shared_frame {

struct alignas(8) DmaBinding {
    Layout layout;
    AllocationIdentity identity;
};

// Versioned Open packet for the four-slot direct-present generation context.
// FDs use SCM_RIGHTS in the transport's original slot order, not this payload.
struct alignas(8) DmaSourcePool {
    uint64_t epoch{};
    std::array<DmaBinding, 4> sources{};
};
static_assert(sizeof(DmaBinding) == 64 && sizeof(DmaSourcePool) == 264);
static_assert(std::is_trivially_copyable_v<DmaSourcePool>);

}
