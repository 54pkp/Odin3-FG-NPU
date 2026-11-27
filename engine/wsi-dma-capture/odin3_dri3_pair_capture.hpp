// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "odin3_wsi_dma_probe.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <syncstream>
#include <span>
#include <vector>

namespace odin3_direct {

// All reservation and snapshot calls are made by the presenter. Source return
// consults holding() under the same jobs lock used to reserve the frame keys.
class PairCapture {
public:
    struct Endpoints {
        uint64_t ordinal;
        odin3::shared_frame::FrameKey a, b;
    };

    explicit PairCapture(const char* path) : directory(path) {}

    bool armed() const {
        std::error_code error;
        return !pending && !captured && std::filesystem::exists(directory / "capture-next", error);
    }

    void reserve(Endpoints source) { pending = source; }

    bool holding(size_t slot) const {
        return pending && (pending->a.slot == slot || pending->b.slot == slot);
    }

    std::optional<Endpoints> for_frame(uint64_t ordinal) const {
        if (pending && pending->ordinal == ordinal) return pending;
        return std::nullopt;
    }

    void release() { pending.reset(); }

    void take(Endpoints input, unsigned phase, uint64_t begin, std::array<int, 3> descriptors, size_t bytes) {
        if (!std::filesystem::remove(directory / "capture-next")) return;
        Snapshot snapshot{input, phase, begin};
        for (size_t i = 0; i < descriptors.size(); ++i) {
            struct stat attributes{};
            if (fstat(descriptors[i], &attributes) != 0 || attributes.st_size <= 0 ||
                    static_cast<uint64_t>(attributes.st_size) != bytes)
                throw std::runtime_error("Invalid presented DMA-BUF size");
            snapshot.identity[i] = {static_cast<uint64_t>(attributes.st_dev), static_cast<uint64_t>(attributes.st_ino)};
            odin3_wsi::ReadMapping image(descriptors[i], bytes);
            if (!image.data()) throw std::runtime_error("Presented DMA-BUF is not CPU readable");
            const auto* first = static_cast<const char*>(image.data());
            snapshot.frames[i].assign(first, first + bytes);
        }
        snapshot.end = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        captured.emplace(std::move(snapshot));
        // Some games exit without explicitly destroying their swapchain. Both
        // presents were already submitted, so publish the diagnostic now.
        save();
    }

    void takeTriple(Endpoints input, unsigned phase1, unsigned phase2, uint64_t begin,
            std::array<int, 4> descriptors, size_t bytes) {
        if (!std::filesystem::remove(directory / "capture-next")) return;
        TripleSnapshot snapshot{input, phase1, phase2, begin};
        for (size_t i = 0; i < descriptors.size(); ++i) {
            struct stat attributes{};
            if (fstat(descriptors[i], &attributes) != 0 || attributes.st_size <= 0 ||
                    static_cast<uint64_t>(attributes.st_size) != bytes)
                throw std::runtime_error("Invalid presented triple DMA-BUF size");
            snapshot.identity[i] = {static_cast<uint64_t>(attributes.st_dev), static_cast<uint64_t>(attributes.st_ino)};
            odin3_wsi::ReadMapping image(descriptors[i], bytes);
            if (!image.data()) throw std::runtime_error("Presented triple DMA-BUF is not CPU readable");
            const auto* first = static_cast<const char*>(image.data());
            snapshot.frames[i].assign(first, first + bytes);
        }
        snapshot.end = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        capturedTriple.emplace(std::move(snapshot));
        save();
    }

    void save() {
        if (capturedTriple && !saved) {
            const auto& shot = *capturedTriple;
            const auto path = directory / ("triple-" + std::to_string(getpid()) + "-" +
                std::to_string(shot.input.b.epoch) + "-" + std::to_string(shot.input.ordinal));
            auto temporary = path;
            temporary += ".partial";
            if (!std::filesystem::create_directory(temporary)) throw std::runtime_error("Presented triple capture already exists");
            constexpr std::array names{"a.bgra", "b.bgra", "m1.bgra", "m2.bgra"};
            try {
                for (size_t i = 0; i < names.size(); ++i) {
                    std::ofstream file(temporary / names[i], std::ios::binary);
                    const auto& frame = shot.frames[i];
                    file.write(frame.data(), static_cast<std::streamsize>(frame.size()));
                    file.close();
                    if (!file) throw std::runtime_error("Cannot save presented triple DMA-BUF pixels");
                }
                std::filesystem::rename(temporary, path);
            } catch (...) {
                std::error_code ignored;
                for (const auto* name : names)
                    std::filesystem::remove(temporary / name, ignored);
                std::filesystem::remove(temporary, ignored);
                throw;
            }
            saved = true;
            std::osyncstream(std::clog) << "Odin3 presented triple: pid=" << getpid() << " epoch=" << shot.input.b.epoch
                << " ordinal=" << shot.input.ordinal << " a=" << shot.input.a.slot << " b=" << shot.input.b.slot
                << " a_generation=" << shot.input.a.generation << " b_generation=" << shot.input.b.generation
                << " output1=" << shot.phase1 << " output2=" << shot.phase2
                << " a_dev=" << shot.identity[0].device << " a_inode=" << shot.identity[0].inode
                << " b_dev=" << shot.identity[1].device << " b_inode=" << shot.identity[1].inode
                << " m1_dev=" << shot.identity[2].device << " m1_inode=" << shot.identity[2].inode
                << " m2_dev=" << shot.identity[3].device << " m2_inode=" << shot.identity[3].inode
                << " begin_ns=" << shot.begin << " end_ns=" << shot.end << std::endl;
            return;
        }
        if (!captured || saved) return;
        const auto& shot = *captured;
        const auto path = directory / ("pair-" + std::to_string(getpid()) + "-" +
            std::to_string(shot.input.b.epoch) + "-" + std::to_string(shot.input.ordinal));
        auto temporary = path;
        temporary += ".partial";
        if (!std::filesystem::create_directory(temporary)) throw std::runtime_error("Presented capture already exists");
        constexpr std::array names{"a.bgra", "b.bgra", "m.bgra"};
        try {
            for (size_t i = 0; i < names.size(); ++i) {
                std::ofstream file(temporary / names[i], std::ios::binary);
                const auto& frame = shot.frames[i];
                file.write(frame.data(), static_cast<std::streamsize>(frame.size()));
                file.close();
                if (!file) throw std::runtime_error("Cannot save presented DMA-BUF pixels");
            }
            std::filesystem::rename(temporary, path);
        } catch (...) {
            std::error_code ignored;
            for (const auto* name : names)
                std::filesystem::remove(temporary / name, ignored);
            std::filesystem::remove(temporary, ignored);
            throw;
        }
        saved = true;
        std::osyncstream(std::clog) << "Odin3 presented pair: pid=" << getpid() << " epoch=" << shot.input.b.epoch
            << " ordinal=" << shot.input.ordinal << " a=" << shot.input.a.slot << " b=" << shot.input.b.slot
            << " a_generation=" << shot.input.a.generation << " b_generation=" << shot.input.b.generation
            << " output=" << shot.phase << " a_dev=" << shot.identity[0].device << " a_inode=" << shot.identity[0].inode
            << " b_dev=" << shot.identity[1].device << " b_inode=" << shot.identity[1].inode
            << " m_dev=" << shot.identity[2].device << " m_inode=" << shot.identity[2].inode
            << " begin_ns=" << shot.begin << " end_ns=" << shot.end << std::endl;
    }

private:
    struct Snapshot {
        Endpoints input;
        unsigned phase;
        uint64_t begin, end{};
        std::array<odin3::shared_frame::AllocationIdentity, 3> identity{};
        std::array<std::vector<char>, 3> frames{};
    };

    std::filesystem::path directory;
    std::optional<Endpoints> pending;
    struct TripleSnapshot {
        Endpoints input;
        unsigned phase1, phase2;
        uint64_t begin, end{};
        std::array<odin3::shared_frame::AllocationIdentity, 4> identity{};
        std::array<std::vector<char>, 4> frames{};
    };
    std::optional<Snapshot> captured;
    std::optional<TripleSnapshot> capturedTriple;
    bool saved = false;
};

} // namespace odin3_direct
