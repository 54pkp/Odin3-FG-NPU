// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "odin3_shared_dma_layout.hpp"

#include <span>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace odin3::shared_frame {

inline void validate_dma_source_pool(const DmaSourcePool& pool, std::span<const int> fds,
                                     uint32_t width, uint32_t height) {
    if (!pool.epoch || fds.size() < pool.sources.size())
        throw std::invalid_argument("Invalid shared DMA source pool epoch or slot count");
    for (size_t i = 0; i < pool.sources.size(); ++i) {
        const auto& source = pool.sources[i];
        const auto rejection = packed_bgra8_sdr(source.layout, width, height);
        if (rejection != InputRejection::none)
            throw std::invalid_argument(std::string("Invalid shared DMA source layout: ") + reason(rejection));
        struct stat info{};
        if (fds[i] < 0 || fstat(fds[i], &info) != 0 || info.st_size <= 0 ||
            static_cast<uint64_t>(info.st_size) != source.layout.allocation_bytes ||
            !source.identity.inode || source.identity != AllocationIdentity{static_cast<uint64_t>(info.st_dev),
                                                                           static_cast<uint64_t>(info.st_ino)})
            throw std::invalid_argument("Shared DMA source descriptor does not match its allocation");
        for (const auto& earlier : std::span(pool.sources).first(i))
            if (earlier.identity == source.identity)
                throw std::invalid_argument("Shared DMA source slots alias one allocation");
    }
}
}
