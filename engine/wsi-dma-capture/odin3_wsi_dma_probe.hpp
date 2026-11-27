/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "odin3_wsi_dma_capture.h"
#include "odin3_shared_dma.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <linux/dma-buf.h>
#include <span>
#include <spawn.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace odin3_wsi {
    namespace shared_frame = odin3::shared_frame;

    class ReadMapping {
    public:
        ReadMapping(int fd, std::size_t bytes) : fd(fd), bytes(bytes) {
            address = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
            if (address == MAP_FAILED) return;
            valid = sync(DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
        }
        ~ReadMapping() {
            if (valid) sync(DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
            if (address != MAP_FAILED) munmap(address, bytes);
        }
        ReadMapping(const ReadMapping&) = delete;
        ReadMapping& operator=(const ReadMapping&) = delete;
        const void* data() const { return valid ? address : nullptr; }

    private:
        bool sync(uint64_t flags) const {
            dma_buf_sync request{.flags = flags};
            int status;
            do {
                status = ioctl(fd, DMA_BUF_IOCTL_SYNC, &request);
            } while (status < 0 && (errno == EINTR || errno == EAGAIN));
            return status == 0;
        }
        int fd;
        std::size_t bytes;
        void* address = MAP_FAILED;
        bool valid = false;
    };

    class Probe {
    public:
        Probe() { for (auto& entry : exports) entry.fd = -1; }
        ~Probe() {
            if (end) end();
            for (const auto& entry : exports)
                if (entry.fd >= 0) close(entry.fd);
        }
        Probe(const Probe&) = delete;
        Probe& operator=(const Probe&) = delete;

        void start(uint32_t window, bool preferLinear) {
            const auto begin = reinterpret_cast<decltype(&odin3_wsi_dma_begin)>(dlsym(RTLD_DEFAULT, "odin3_wsi_dma_begin"));
            const auto finish = reinterpret_cast<decltype(&odin3_wsi_dma_end)>(dlsym(RTLD_DEFAULT, "odin3_wsi_dma_end"));
            if (begin && finish && begin(window, exports.data(), exports.size(), preferLinear)) end = finish;
            else std::clog << "MAKO Renderer: wsi-dma probe capture symbols unavailable\n";
        }

        void stop(std::size_t imageCount, uint32_t width, uint32_t height,
                  shared_frame::Format format, shared_frame::ColorSpace colorSpace) {
            if (!end) return;
            const auto seen = end();
            end = nullptr;
            this->imageCount = imageCount;
            usable = seen == imageCount && seen > 0 && seen <= exports.size();
            std::clog << "MAKO Renderer: wsi-dma enumeration count=" << imageCount << " exports=" << seen << '\n';
            for (std::size_t i = 0; i < std::min(seen, exports.size()); ++i) {
                const auto& entry = exports[i];
                struct stat info{};
                const bool identified = entry.fd >= 0 && fstat(entry.fd, &info) == 0;
                const shared_frame::Layout layout{entry.width, entry.height,
                    entry.bpp == 32 ? format : shared_frame::Format::unknown, colorSpace,
                    entry.planes, entry.stride, entry.offset, entry.modifier,
                    info.st_size > 0 ? static_cast<uint64_t>(info.st_size) : 0};
                const auto rejection = shared_frame::packed_bgra8_sdr(layout, width, height);
                const shared_frame::AllocationIdentity allocation{static_cast<uint64_t>(info.st_dev),
                                                                  static_cast<uint64_t>(info.st_ino)};
                const auto earlier = std::span(bindings).first(i);
                const bool unique = std::none_of(earlier.begin(), earlier.end(),
                                                 [&](const auto& row) { return row.identity == allocation; });
                bindings[i] = {layout, allocation};
                const bool exact = identified && allocation.inode && unique && rejection == shared_frame::InputRejection::none;
                if (exact) bytes = static_cast<std::size_t>(info.st_size);
                usable = usable && exact;
                std::clog << "MAKO Renderer: wsi-dma image index=" << i << " pixmap=" << entry.pixmap
                          << " dev=" << info.st_dev << " inode=" << info.st_ino << " bytes=" << info.st_size
                          << " width=" << entry.width << " height=" << entry.height
                          << " stride=" << entry.stride << " offset=" << entry.offset
                          << " planes=" << unsigned(entry.planes) << " modifier=" << entry.modifier
                          << " packed=" << exact << " unique=" << unique
                          << " layout=" << shared_frame::reason(rejection) << '\n';
            }
        }

        bool choose() {
            if (!usable) return false;
            ++presentCount;
            return presentCount == 180 || presentCount == 720 || presentCount == 1400;
        }

        bool borrowSources(std::span<int> descriptors) const {
            if (!usable || imageCount < 4 || descriptors.size() < imageCount) return false;
            for (std::size_t i = 0; i < descriptors.size(); ++i) {
                descriptors[i] = fcntl(exports[i < imageCount ? i : 0].fd, F_DUPFD_CLOEXEC, 0);
                if (descriptors[i] < 0) {
                    for (int& fd : descriptors.first(i)) {
                        close(fd);
                        fd = -1;
                    }
                    return false;
                }
            }
            return true;
        }

        int sourceFd(unsigned index) const {
            if (!usable || index >= imageCount) throw std::out_of_range("WSI source index");
            return exports.at(index).fd;
        }

        shared_frame::DmaSourcePool sourcePool(uint64_t epoch) const {
            shared_frame::DmaSourcePool result{epoch};
            if (!usable || imageCount < result.sources.size())
                throw std::logic_error("WSI does not have a complete four-slot source pool");
            std::copy_n(bindings.begin(), result.sources.size(), result.sources.begin());
            return result;
        }

        void compare(uint32_t index, int copiedFd) const {
            const auto& entry = exports.at(index);
            ReadMapping original(entry.fd, bytes);
            ReadMapping copy(copiedFd, bytes);
            const bool readable = original.data() && copy.data();
            const bool exact = readable && std::memcmp(original.data(), copy.data(), bytes) == 0;
            uint64_t sampleHash = 14695981039346656037ull;
            if (const auto* pixels = static_cast<const uint8_t*>(original.data())) {
                for (std::size_t i = 0; i < bytes; i += 4093)
                    sampleHash = (sampleHash ^ pixels[i]) * 1099511628211ull;
            }
            struct stat clone{};
            fstat(copiedFd, &clone);
            std::clog << "MAKO Renderer: wsi-dma pixel proof index=" << index << " pixmap=" << entry.pixmap
                      << " copy_inode=" << clone.st_ino << " sample=" << presentCount << " bytes=" << bytes
                      << " readable=" << readable << " exact=" << exact << " sample_hash=" << sampleHash << '\n';
        }

        // The caller holds the acquired WSI index until this diagnostic child
        // exits. The steady-state presentation path never runs the helper.
        void testHtp(uint32_t index, int copiedFd) const {
            const auto* helper = std::getenv("ODIN3_WSI_HTP_PROBE");
            if (!helper || helper[0] != '/' || presentCount != 1400) return;
            const int original = fcntl(exports.at(index).fd, F_DUPFD_CLOEXEC, 192);
            const int copy = fcntl(copiedFd, F_DUPFD_CLOEXEC, 192);
            posix_spawn_file_actions_t actions;
            const int init = posix_spawn_file_actions_init(&actions);
            int error = init;
            if (!error && (original < 0 || copy < 0)) error = EBADF;
            if (!error) error = posix_spawn_file_actions_adddup2(&actions, original, 180);
            if (!error) error = posix_spawn_file_actions_adddup2(&actions, copy, 181);
            char originalArg[] = "180", copyArg[] = "181";
            char* argv[]{const_cast<char*>(helper), originalArg, copyArg, nullptr};
            pid_t child{};
            if (!error) error = posix_spawn(&child, helper, &actions, nullptr, argv, environ);
            if (!init) posix_spawn_file_actions_destroy(&actions);
            if (original >= 0) close(original);
            if (copy >= 0) close(copy);
            int status = -1;
            if (!error) {
                int result;
                do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
                if (result < 0) error = errno;
            }
            std::clog << "MAKO Renderer: wsi-dma HTP check index=" << index << " pixmap=" << exports[index].pixmap
                      << " sample=" << presentCount << " spawn_or_wait_error=" << error
                      << " child_status=" << status << '\n';
        }

    private:
        std::array<Odin3WsiDmaExport, 16> exports{};
        std::array<shared_frame::DmaBinding, 16> bindings{};
        decltype(&odin3_wsi_dma_end) end{};
        bool usable{};
        std::size_t bytes{}, presentCount{}, imageCount{};
    };
}
