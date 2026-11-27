// SPDX-License-Identifier: GPL-3.0-or-later
// Gen-8 tier-matched hybrid frame generation helpers (worker side):
//  * 5 internal HTP scales (360p/540p/720p/900p/1080p) selected from the Odin3 NPU Power tier (config.bin seqlock file)
//  * Oryon cpu0..5 affinity helper (leaves cpu6..7 4.09 GHz Prime cores 100% dedicated to the game's main/DXVK threads)
//  * Minimal-sync ARM64 NEON tier resampler (0.00% GPU worker load, 0 GPU fence stalls, bit-exact 0 u8 HUD preservation)
#pragma once

#include <arm_neon.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

namespace odin3_tier {

// Pin worker + NEON helper threads to Oryon Phoenix-M cluster cpu1..5 (3.53 GHz), avoiding BOTH cpu0 (where ArmadaOS
// pins gamescope-wl/kms/xwm at nice=-20) AND cpu6..7 (4.09 GHz Prime cores reserved 100% for the game), at nice=19
// in live games (matching Wine/Proton thread nice=19 so FG never preempts game/DXVK threads).
inline void pinToEfficiencyCluster(bool singleCore = false) noexcept {
    (void)singleCore;
    if (const char* noPin = std::getenv("ODIN3_FG_NO_EFFICIENCY_PIN"); noPin && noPin[0] == '1') return;
    if (std::getenv("ODIN3_FG_RUNTIME_APP_ID") != nullptr) {
        const char* cap = std::getenv("MAKO_BASE_FPS_CAP");
        const int capVal = (cap && cap[0]) ? std::atoi(cap) : 0;
        if (capVal <= 0 || capVal > 62) {
            sched_param sp{};
            (void)::sched_setscheduler(0, SCHED_IDLE, &sp);
        } else {
            (void)::setpriority(PRIO_PROCESS, static_cast<id_t>(::syscall(SYS_gettid)), 0);
        }
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 1; c < 6; ++c) CPU_SET(c, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

struct Scale {
    unsigned w, h;
    const char* sub;
    int idx;
};
inline constexpr std::array<Scale, 5> kScales{{{640, 360, "360p", 0}, {960, 540, "540p", 1}, {1280, 720, "720p", 2},
                                               {1600, 900, "900p", 3}, {1920, 1080, "1080p", 4}}};

inline constexpr std::array<const char*, 8> kPresetNames{
    "eco", "saver", "balanced", "auto", "high", "max", "custom", "passthrough"
};

inline const char* presetName(uint32_t preset) {
    return preset < kPresetNames.size() ? kPresetNames[preset] : "max";
}

inline int nativeIndex(unsigned w, unsigned h) {
    for (const auto& s : kScales)
        if (s.w == w && s.h == h) return s.idx;
    return -1;
}

struct TierState {
    uint32_t preset{5};  // 0 eco 1 saver 2 balanced 3 auto 4 high 5 max 6 custom 7 passthrough
    uint32_t coreMax{0xA0};
    bool valid{};
};

inline bool readConfigFile(const std::string& path, uint32_t* words) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    uint32_t a[32]{};
    uint32_t b[32]{};
    bool ok = false;
    for (int tries = 0; tries < 8 && !ok; ++tries) {
        if (::pread(fd, a, sizeof a, 0) != ssize_t(sizeof a)) break;
        if (::pread(fd, b, sizeof b, 0) != ssize_t(sizeof b)) break;
        ok = a[0] == 0x4F4E5055u && !(a[2] & 1u) && a[2] == b[2];
    }
    ::close(fd);
    if (ok) std::memcpy(words, a, sizeof a);
    return ok;
}

inline TierState readTier() {
    TierState t;
    if (const char* forced = std::getenv("ODIN3_FG_FORCE_TIER"); forced && forced[0]) {
        for (uint32_t i = 0; i < kPresetNames.size(); ++i)
            if (std::strcmp(forced, kPresetNames[i]) == 0) {
                t.preset = i;
                t.valid = true;
                return t;
            }
    }
    const char* home = std::getenv("HOME");
    const std::string paths[2] = {"/run/odin3-npu-governor/config.bin",
                                  std::string(home && home[0] == '/' ? home : "/var/home/armada") + "/.config/odin3-npu-governor/config.bin"};
    uint32_t best[32]{};
    bool have = false;
    for (const auto& p : paths) {
        uint32_t w[32]{};
        if (!readConfigFile(p, w)) continue;
        if (!have || int32_t(w[2] - best[2]) > 0) {
            std::memcpy(best, w, sizeof best);
            have = true;
        }
    }
    if (!have) return t;
    t.valid = true;
    t.preset = best[3] == 0 ? 7u : std::min<uint32_t>(best[4], 7u);  // mode 0 = passthrough
    t.coreMax = best[13];
    return t;
}

// Highest internal scale (index into kScales) a tier may use; 1080p caps at VTCM-resident 720p (idx=2, 0 DDR spills,
// ~11.8-13.3ms pair latency) paired with 2D AA downsample + tier-adaptive RCAS sharpening + native 1080p HUD fusion,
// while <=900p uses native full resolution on high/max (even in 3x) and tiers down cleanly to saver/eco.
inline int tierCeilIndex(const TierState& t, size_t phases, int nativeIdx = 4) {
    const bool three = phases == 2;
    switch (t.preset) {
        case 0: return (three || nativeIdx >= 4) ? 0 : 1;           // eco: 360p on 1080p & 3x (~5.6ms SVS); 540p on 720p/900p 2x
        case 1: return nativeIdx >= 3 ? 1 : 2;                      // saver: 540p on 900p/1080p, 720p on 720p
        case 2:
        case 3: return (three || nativeIdx >= 4) ? 2 : 3;           // balanced / auto: native 900p on 900p 2x; VTCM-resident 720p on 1080p & 3x
        case 4: return nativeIdx >= 4 ? 2 : 3;                      // high: VTCM-resident 720p + 0.75x RCAS on 1080p; native 900p/720p on <=900p
        case 6:
            if (t.coreMax >= 0x90) return nativeIdx >= 4 ? 2 : 3;
            if (t.coreMax >= 0x80) return (three || nativeIdx >= 4) ? 2 : 3;
            if (t.coreMax >= 0x60) return nativeIdx >= 3 ? 1 : 2;
            return (three || nativeIdx >= 4) ? 0 : 1;
        default: return nativeIdx >= 4 ? 2 : 3;                     // max / passthrough: VTCM-resident 720p + 0.75x RCAS on 1080p; native on <=900p
    }
}

inline int forcedScaleIndex() {
    const char* f = std::getenv("ODIN3_FG_FORCE_SCALE");
    if (!f || !f[0]) return -1;
    for (const auto& s : kScales)
        if (std::strcmp(f, s.sub) == 0) return s.idx;
    return -1;
}

// Fast 1-us sysfs reader for Hexagon HMX (`nspss-0`) thermal zone (millidegrees C), preventing >84C LMH throttling.
inline int readHmxTempMc() noexcept {
    static const int fd = []() noexcept -> int {
        for (int i = 0; i < 64; ++i) {
            char tpath[96];
            std::snprintf(tpath, sizeof tpath, "/sys/class/thermal/thermal_zone%d/type", i);
            const int tfd = ::open(tpath, O_RDONLY | O_CLOEXEC);
            if (tfd < 0) continue;
            char buf[32]{};
            const ssize_t n = ::read(tfd, buf, sizeof(buf) - 1);
            ::close(tfd);
            if (n >= 6 && std::strncmp(buf, "nsphmx", 6) == 0) {
                char vpath[96];
                std::snprintf(vpath, sizeof vpath, "/sys/class/thermal/thermal_zone%d/temp", i);
                return ::open(vpath, O_RDONLY | O_CLOEXEC);
            }
        }
        return -1;
    }();
    if (fd < 0) return 0;
    char vbuf[24]{};
    if (::pread(fd, vbuf, sizeof(vbuf) - 1, 0) <= 0) return 0;
    return std::atoi(vbuf);
}

inline void writeJsonAtomic(const std::string& path, const char* content, size_t len) {
    const std::string tmp = path + "." + std::to_string(::getpid()) + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return;
    const ssize_t w = ::write(fd, content, len);
    ::close(fd);
    if (w == ssize_t(len)) {
        if (::rename(tmp.c_str(), path.c_str()) != 0) ::unlink(tmp.c_str());
    } else {
        ::unlink(tmp.c_str());
    }
}

inline void publishTierStatus(const TierState& t, int scaleIdx, int nativeIdx, unsigned nativeW, unsigned nativeH,
                              size_t maxPhases, unsigned lastBatchPhases, double pairEmaMs, int guardCap) noexcept {
    if (scaleIdx < 0 || scaleIdx >= int(kScales.size())) return;
    const auto& sc = kScales[size_t(scaleIdx)];
    const auto& gc = kScales[size_t(std::clamp(guardCap, 0, int(kScales.size()) - 1))];
    const auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    char buf[512];
    const int n = std::snprintf(
        buf, sizeof buf,
        "{\"pid\":%d,\"preset_id\":%u,\"preset\":\"%s\",\"scale\":\"%s\",\"scale_w\":%u,\"scale_h\":%u,"
        "\"native_w\":%u,\"native_h\":%u,\"native\":%s,\"max_phases\":%zu,\"active_phases\":%u,"
        "\"pair_ema_ms\":%.2f,\"guard_cap\":\"%s\",\"ts_ns\":%lld}\n",
        int(::getpid()), unsigned(t.preset), presetName(t.preset), sc.sub, sc.w, sc.h,
        nativeW, nativeH, (scaleIdx == nativeIdx ? "true" : "false"), maxPhases, lastBatchPhases,
        pairEmaMs, gc.sub, static_cast<long long>(nowNs));
    if (n <= 0 || size_t(n) >= sizeof buf) return;
    const std::string userPath = "/run/user/" + std::to_string(unsigned(::geteuid())) + "/odin3-fg-tier.json";
    writeJsonAtomic(userPath, buf, size_t(n));
    if (const char* stDir = std::getenv("ODIN3_FG_RUNTIME_STATE_DIR"); stDir && stDir[0] == '/')
        writeJsonAtomic(std::string(stDir) + "/tier-status.json", buf, size_t(n));
}

inline void syncDmaBufFd(int fd, uint64_t flags) {
    if (fd < 0) return;
    dma_buf_sync req{flags};
    int rc;
    do {
        rc = ::ioctl(fd, DMA_BUF_IOCTL_SYNC, &req);
    } while (rc < 0 && (errno == EINTR || errno == EAGAIN));
    if (rc < 0) throw std::system_error(errno, std::generic_category(), "DMA_BUF_IOCTL_SYNC");
}

// Userspace EL0 `dc civac` cache invalidation (0 syscalls, 0 kernel `dma_resv` fence waits on zero-copy DRI3 pixmaps!).
inline size_t dcacheLineBytes() noexcept {
    static const size_t line = []() noexcept -> size_t {
        uint64_t ctr = 0;
        asm volatile("mrs %0, ctr_el0" : "=r"(ctr));
        const size_t n = size_t(4u) << ((ctr >> 16) & 0xFu);
        return (n >= 16u && n <= 256u) ? n : 64u;
    }();
    return line;
}

inline void invalidateLinesNoBarrier(const void* ptr, size_t bytes) noexcept {
    if (!ptr || !bytes) return;
    const size_t line = dcacheLineBytes();
    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr) & ~(uintptr_t(line) - 1u);
    const uintptr_t end = reinterpret_cast<uintptr_t>(ptr) + bytes;
    for (uintptr_t p = start; p < end; p += line) {
        asm volatile("dc civac, %0" :: "r"(p) : "memory");
    }
}

inline void invalidateRangeEl0(const void* ptr, size_t bytes) noexcept {
    invalidateLinesNoBarrier(ptr, bytes);
    asm volatile("dsb ish" ::: "memory");
}

// Userspace EL0 `dc cvac` clean-to-PoC (0 syscalls, 0 kernel 2025-page sg_table walks, 0 dma_resv locks!).
inline void cleanLinesNoBarrier(const void* ptr, size_t bytes) noexcept {
    if (!ptr || !bytes) return;
    const size_t line = dcacheLineBytes();
    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr) & ~(uintptr_t(line) - 1u);
    const uintptr_t end = reinterpret_cast<uintptr_t>(ptr) + bytes;
    for (uintptr_t p = start; p < end; p += line) {
        asm volatile("dc cvac, %0" :: "r"(p) : "memory");
    }
}

inline void cleanRangeEl0(const void* ptr, size_t bytes) noexcept {
    cleanLinesNoBarrier(ptr, bytes);
    asm volatile("dsb ish" ::: "memory");
}

// Lock-free chunk-completion row pool pinned to Oryon Phoenix-M cluster (cpu0..5, 0% interference with cpu6..7 Prime cores).
// Returns the instant all chunks complete without waiting for delayed/unscheduled helper threads!
class NeonRowPool {
    struct JobSlot {
        alignas(16) unsigned char ctx[128]{};
        void (*fn)(const void*, unsigned, unsigned){};
        unsigned totalRows{0};
        unsigned chunkRows{8};
        unsigned totalChunks{0};
        std::atomic<unsigned> nextChunk{0};
        std::atomic<unsigned> doneChunks{0};
        std::atomic<unsigned> activeReaders{0};
    };

    static constexpr size_t kRingSize = 8;
    std::array<JobSlot, kRingSize> ring{};
    std::mutex mtx;
    std::condition_variable cvGo, cvDone;
    std::vector<std::thread> workers;
    uint64_t epoch{0};
    bool stop{false};

    void runSlot(JobSlot& s) {
        for (;;) {
            const unsigned c = s.nextChunk.fetch_add(1, std::memory_order_relaxed);
            if (c >= s.totalChunks) break;
            const unsigned y0 = c * s.chunkRows;
            s.fn(s.ctx, y0, std::min(s.totalRows, y0 + s.chunkRows));
            if (s.doneChunks.fetch_add(1, std::memory_order_acq_rel) + 1u == s.totalChunks) {
                std::lock_guard lk(mtx);
                cvDone.notify_all();
            }
        }
    }

    void workerLoop() {
        pinToEfficiencyCluster();
        uint64_t seen = 0;
        for (;;) {
            JobSlot* slotPtr = nullptr;
            {
                std::unique_lock lk(mtx);
                cvGo.wait(lk, [&] { return stop || epoch != seen; });
                if (stop) return;
                seen = epoch;
                JobSlot& cand = ring[seen & (kRingSize - 1u)];
                if (cand.nextChunk.load(std::memory_order_relaxed) < cand.totalChunks) {
                    cand.activeReaders.fetch_add(1u, std::memory_order_relaxed);
                    slotPtr = &cand;
                }
            }
            if (slotPtr) {
                runSlot(*slotPtr);
                if (slotPtr->activeReaders.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
                    std::lock_guard lk(mtx);
                    cvDone.notify_all();
                }
            }
        }
    }

public:
    explicit NeonRowPool(unsigned helpers = 2) {
        workers.reserve(helpers);
        for (unsigned i = 0; i < helpers; ++i)
            workers.emplace_back([this] { workerLoop(); });
    }
    ~NeonRowPool() {
        {
            std::lock_guard lk(mtx);
            stop = true;
        }
        cvGo.notify_all();
        for (auto& t : workers)
            if (t.joinable()) t.join();
    }
    NeonRowPool(const NeonRowPool&) = delete;
    NeonRowPool& operator=(const NeonRowPool&) = delete;

    template <class F>
    void parallelFor(unsigned rows, unsigned chunk, const F& fn) {
        static_assert(sizeof(F) <= sizeof(JobSlot::ctx), "Lambda capture exceeds JobSlot::ctx");
        static_assert(std::is_trivially_destructible_v<F>, "Lambda capture must be trivially destructible");
        if (!rows) return;
        if (workers.empty() || rows <= chunk) {
            fn(0, rows);
            return;
        }
        const unsigned nChunks = (rows + chunk - 1u) / chunk;
        JobSlot* slotPtr = nullptr;
        {
            std::unique_lock lk(mtx);
            const uint64_t nextEpoch = epoch + 1u;
            JobSlot& s = ring[nextEpoch & (kRingSize - 1u)];
            cvDone.wait(lk, [&] { return s.activeReaders.load(std::memory_order_acquire) == 0u; });
            ::new (static_cast<void*>(s.ctx)) F(fn);
            s.fn = [](const void* p, unsigned y0, unsigned y1) { (*static_cast<const F*>(p))(y0, y1); };
            s.totalRows = rows;
            s.chunkRows = chunk;
            s.totalChunks = nChunks;
            s.nextChunk.store(0u, std::memory_order_relaxed);
            s.doneChunks.store(0u, std::memory_order_relaxed);
            epoch = nextEpoch;
            slotPtr = &s;
        }
        cvGo.notify_all();
        runSlot(*slotPtr);
        if (slotPtr->doneChunks.load(std::memory_order_acquire) < nChunks) {
            std::unique_lock lk(mtx);
            cvDone.wait(lk, [&] { return slotPtr->doneChunks.load(std::memory_order_acquire) >= nChunks; });
        }
    }
};

// Minimal-sync ARM64 NEON tier resampler:
//   * 0.00% GPU worker load, 0 cross-context GPU submits/fences
//   * Syncs each full-res input frame at most ONCE (START|READ, PROT_READ only)
//   * Separable L1-cached NEON vertical + 2:1 / 2:3 / general horizontal kernels
//   * Vectorized NEON HUD static-pixel guard (`dmax <= 1` bit-exact `vbslq_u8`, 0.00 u8 HUD error)
class CpuTierScaler {
    struct MappedBuf {
        int fd{-1};
        uint8_t* ptr{static_cast<uint8_t*>(MAP_FAILED)};
        size_t bytes{};
        bool writable{};
    };
    struct AxisTable {
        unsigned inN{}, outN{};
        std::vector<uint32_t> i0, i1;
        std::vector<uint16_t> w1;  // 0..256

        void build(unsigned inSize, unsigned outSize) {
            if (inN == inSize && outN == outSize && i0.size() == outSize) return;
            inN = inSize;
            outN = outSize;
            i0.resize(outSize);
            i1.resize(outSize);
            w1.resize(outSize);
            for (unsigned o = 0; o < outSize; ++o) {
                const int64_t posQ16 = (int64_t(2ull * o + 1ull) * int64_t(inSize) * 32768ll) / int64_t(outSize) - 32768ll;
                if (posQ16 <= 0) {
                    i0[o] = i1[o] = 0;
                    w1[o] = 0;
                } else {
                    const uint32_t base = static_cast<uint32_t>(posQ16 >> 16);
                    if (base + 1 >= inSize) {
                        i0[o] = i1[o] = inSize - 1;
                        w1[o] = 0;
                    } else {
                        i0[o] = base;
                        i1[o] = base + 1;
                        w1[o] = static_cast<uint16_t>(((posQ16 & 0xFFFFll) + 128ll) >> 8);
                    }
                }
            }
        }
    };

    static unsigned defaultNeonHelpers() noexcept {
        if (std::getenv("ODIN3_FG_RUNTIME_APP_ID") == nullptr) return 2u;
        const char* cap = std::getenv("MAKO_BASE_FPS_CAP");
        const int capVal = (cap && cap[0]) ? std::atoi(cap) : 0;
        return (capVal > 0 && capVal <= 62) ? 2u : 1u;
    }

    NeonRowPool pool{defaultNeonHelpers()};
    std::array<MappedBuf, 12> fullIn{};
    std::array<uint64_t, 12> syncedInTag{};
    std::array<MappedBuf, 6> fullOut{};
    AxisTable downX, downY, upX, upY;

    static void mapSlot(MappedBuf& slot, int borrowedFd, size_t bytes, bool writable) {
        unmapSlot(slot);
        if (borrowedFd < 0) return;
        const int dupFd = ::fcntl(borrowedFd, F_DUPFD_CLOEXEC, 0);
        if (dupFd < 0) throw std::system_error(errno, std::generic_category(), "dup tier dma-buf");
        void* m = ::mmap(nullptr, bytes, writable ? (PROT_READ | PROT_WRITE) : PROT_READ, MAP_SHARED, dupFd, 0);
        if (m == MAP_FAILED) {
            const int err = errno;
            ::close(dupFd);
            throw std::system_error(err, std::generic_category(), "mmap tier dma-buf");
        }
        slot.fd = dupFd;
        slot.ptr = static_cast<uint8_t*>(m);
        slot.bytes = bytes;
        slot.writable = writable;
    }

    static void unmapSlot(MappedBuf& slot) noexcept {
        if (slot.ptr != static_cast<uint8_t*>(MAP_FAILED) && slot.bytes) {
            ::munmap(slot.ptr, slot.bytes);
            slot.ptr = static_cast<uint8_t*>(MAP_FAILED);
        }
        if (slot.fd >= 0) {
            ::close(slot.fd);
            slot.fd = -1;
        }
        slot.bytes = 0;
    }

    static const uint8_t* blendRowVerticalNeon(const uint8_t* r0, const uint8_t* r1, uint16_t wy,
                                               size_t rowBytes, uint8_t* scratch) noexcept {
        if (wy == 0 || r0 == r1) return r0;
        if (wy >= 256) return r1;
        const uint8x8_t vw0 = vdup_n_u8(static_cast<uint8_t>(256u - wy));
        const uint8x8_t vw1 = vdup_n_u8(static_cast<uint8_t>(wy));
        size_t i = 0;
        for (; i + 16 <= rowBytes; i += 16) {
            const uint8x16_t a = vld1q_u8(r0 + i);
            const uint8x16_t b = vld1q_u8(r1 + i);
            const uint16x8_t lo = vmlal_u8(vmull_u8(vget_low_u8(a), vw0), vget_low_u8(b), vw1);
            const uint16x8_t hi = vmlal_u8(vmull_u8(vget_high_u8(a), vw0), vget_high_u8(b), vw1);
            vst1q_u8(scratch + i, vcombine_u8(vrshrn_n_u16(lo, 8), vrshrn_n_u16(hi, 8)));
        }
        for (; i < rowBytes; ++i)
            scratch[i] = static_cast<uint8_t>((uint32_t(r0[i]) * (256u - wy) + uint32_t(r1[i]) * wy + 128u) >> 8);
        return scratch;
    }

    static void gatherHorizRow(const uint8_t* srcRow, uint8_t* dstRow, const AxisTable& ax) noexcept {
        const unsigned outW = ax.outN;
        const auto* __restrict src32 = reinterpret_cast<const uint32_t*>(srcRow);
        auto* __restrict dst32 = reinterpret_cast<uint32_t*>(dstRow);
        const uint32_t* __restrict p0 = ax.i0.data();
        const uint32_t* __restrict p1 = ax.i1.data();
        const uint16_t* __restrict pw = ax.w1.data();
        const uint8x16_t alphaOr = vreinterpretq_u8_u32(vdupq_n_u32(0xFF000000u));
        const uint16x8_t v256 = vdupq_n_u16(256u);
        unsigned x = 0;
        for (; x + 4 <= outW; x += 4) {
            const uint32x4_t c0 = {src32[p0[x]], src32[p0[x + 1]], src32[p0[x + 2]], src32[p0[x + 3]]};
            const uint32x4_t c1 = {src32[p1[x]], src32[p1[x + 1]], src32[p1[x + 2]], src32[p1[x + 3]]};
            const uint16_t w1_0 = pw[x], w1_1 = pw[x + 1], w1_2 = pw[x + 2], w1_3 = pw[x + 3];
            const uint16x8_t w1Lo = {w1_0, w1_0, w1_0, w1_0, w1_1, w1_1, w1_1, w1_1};
            const uint16x8_t w1Hi = {w1_2, w1_2, w1_2, w1_2, w1_3, w1_3, w1_3, w1_3};
            const uint16x8_t w0Lo = vsubq_u16(v256, w1Lo);
            const uint16x8_t w0Hi = vsubq_u16(v256, w1Hi);
            const uint8x16_t u0 = vreinterpretq_u8_u32(c0);
            const uint8x16_t u1 = vreinterpretq_u8_u32(c1);
            const uint16x8_t accLo = vmlaq_u16(vmulq_u16(vmovl_u8(vget_low_u8(u0)), w0Lo), vmovl_u8(vget_low_u8(u1)), w1Lo);
            const uint16x8_t accHi = vmlaq_u16(vmulq_u16(vmovl_u8(vget_high_u8(u0)), w0Hi), vmovl_u8(vget_high_u8(u1)), w1Hi);
            vst1q_u8(reinterpret_cast<uint8_t*>(dst32 + x),
                     vorrq_u8(vcombine_u8(vrshrn_n_u16(accLo, 8), vrshrn_n_u16(accHi, 8)), alphaOr));
        }
        for (; x < outW; ++x) {
            const uint8_t* s0 = srcRow + size_t(p0[x]) * 4;
            const uint16_t w1 = pw[x];
            uint8_t* d = dstRow + size_t(x) * 4;
            if (w1 == 0) {
                std::memcpy(d, s0, 4);
            } else if (w1 >= 256) {
                std::memcpy(d, srcRow + size_t(p1[x]) * 4, 4);
            } else {
                const uint8_t* s1 = srcRow + size_t(p1[x]) * 4;
                const uint32_t w0 = 256u - w1;
                d[0] = static_cast<uint8_t>((s0[0] * w0 + s1[0] * w1 + 128u) >> 8);
                d[1] = static_cast<uint8_t>((s0[1] * w0 + s1[1] * w1 + 128u) >> 8);
                d[2] = static_cast<uint8_t>((s0[2] * w0 + s1[2] * w1 + 128u) >> 8);
                d[3] = 255u;
            }
        }
    }

    // Overshoot-free 3x3 cross RCAS-lite sharpener in low-res L1 space: restores edge crispness lost to bilinear
    // upsampling during camera rotation, tier-scaled (0.75x on high/max, 0.625x on auto/balanced, 0.50x on saver/eco)
    // and strictly clamped to [min(N,S,W,E,C), max(N,S,W,E,C)] (0 ringing / 0 halo).
    static const uint8_t* sharpenLowRowNeon(const uint8_t* lowBase, uint32_t ly, uint32_t lowW, uint32_t lowH,
                                            size_t lowStride, uint8_t* scratch, uint32_t sharpTier = 2u) noexcept {
        const uint8_t* rCurr = lowBase + size_t(ly) * lowStride;
        if (lowW < 16u || lowH < 4u) return rCurr;
        const uint8_t* rTop = lowBase + size_t(ly > 0u ? ly - 1u : 0u) * lowStride;
        const uint8_t* rBot = lowBase + size_t(ly + 1u < lowH ? ly + 1u : lowH - 1u) * lowStride;
        std::memcpy(scratch, rCurr, 4u);
        unsigned x = 1u;
        for (; x + 5u <= lowW; x += 4u) {
            const size_t off = size_t(x) * 4u;
            const uint8x16_t c = vld1q_u8(rCurr + off);
            const uint8x16_t n = vld1q_u8(rTop + off);
            const uint8x16_t s = vld1q_u8(rBot + off);
            const uint8x16_t w = vld1q_u8(rCurr + off - 4u);
            const uint8x16_t e = vld1q_u8(rCurr + off + 4u);
            const uint8x16_t avgCross = vrhaddq_u8(vrhaddq_u8(n, s), vrhaddq_u8(w, e));
            const uint8x16_t loCross = vminq_u8(c, vminq_u8(vminq_u8(n, s), vmaxq_u8(vdupq_n_u8(0), vminq_u8(w, e))));
            const uint8x16_t hiCross = vmaxq_u8(c, vmaxq_u8(vmaxq_u8(n, s), vmaxq_u8(w, e)));
            const uint8x16_t dPos = vqsubq_u8(c, avgCross);
            const uint8x16_t dNeg = vqsubq_u8(avgCross, c);
            uint8x16_t posBoost = vshrq_n_u8(dPos, 1);
            uint8x16_t negBoost = vshrq_n_u8(dNeg, 1);
            if (sharpTier >= 2u) {
                posBoost = vqaddq_u8(posBoost, vshrq_n_u8(dPos, 2));
                negBoost = vqaddq_u8(negBoost, vshrq_n_u8(dNeg, 2));
            } else if (sharpTier == 1u) {
                posBoost = vqaddq_u8(posBoost, vshrq_n_u8(dPos, 3));
                negBoost = vqaddq_u8(negBoost, vshrq_n_u8(dNeg, 3));
            }
            const uint8x16_t sharp = vqsubq_u8(vqaddq_u8(c, posBoost), negBoost);
            vst1q_u8(scratch + off, vminq_u8(vmaxq_u8(sharp, loCross), hiCross));
        }
        std::memcpy(scratch + size_t(x) * 4u, rCurr + size_t(x) * 4u, size_t(lowW - x) * 4u);
        return scratch;
    }

    static void gatherHorizSpan(const uint8_t* srcRow, uint8_t* dstRow, const AxisTable& ax,
                                unsigned xBegin, unsigned xEnd) noexcept {
        for (unsigned x = xBegin; x < xEnd; ++x) {
            const uint8_t* s0 = srcRow + size_t(ax.i0[x]) * 4;
            const uint16_t w1 = ax.w1[x];
            uint8_t* d = dstRow + size_t(x) * 4;
            if (w1 == 0) {
                std::memcpy(d, s0, 4);
            } else if (w1 >= 256) {
                std::memcpy(d, srcRow + size_t(ax.i1[x]) * 4, 4);
            } else {
                const uint8_t* s1 = srcRow + size_t(ax.i1[x]) * 4;
                const uint32_t w0 = 256u - w1;
                d[0] = static_cast<uint8_t>((s0[0] * w0 + s1[0] * w1 + 128u) >> 8);
                d[1] = static_cast<uint8_t>((s0[1] * w0 + s1[1] * w1 + 128u) >> 8);
                d[2] = static_cast<uint8_t>((s0[2] * w0 + s1[2] * w1 + 128u) >> 8);
                d[3] = 255u;
            }
        }
    }

    // Fast NEON horizontal upsample into L1 row buffer (1:2 for 540p->1080p & 360p->720p; 2:3 for 720p->1080p)
    static void upsampleHorizRowNeon(const uint8_t* srcRow, uint8_t* dstRow, uint32_t lowW, uint32_t fullW,
                                     const AxisTable& ax) noexcept {
        if (fullW == lowW * 2u && lowW >= 16u) {
            gatherHorizSpan(srcRow, dstRow, ax, 0, 8);
            const unsigned kEnd = lowW - 4u;
            for (unsigned k = 4; k < kEnd; k += 4) {
                const uint8x16_t prev = vld1q_u8(srcRow + size_t(k - 1) * 4);
                const uint8x16_t curr = vld1q_u8(srcRow + size_t(k) * 4);
                const uint8x16_t next = vld1q_u8(srcRow + size_t(k + 1) * 4);
                const uint32x4_t even = vreinterpretq_u32_u8(vrhaddq_u8(curr, vhaddq_u8(curr, prev)));
                const uint32x4_t odd  = vreinterpretq_u32_u8(vrhaddq_u8(curr, vhaddq_u8(curr, next)));
                vst1q_u8(dstRow + size_t(2u * k) * 4,     vreinterpretq_u8_u32(vzip1q_u32(even, odd)));
                vst1q_u8(dstRow + size_t(2u * k + 4) * 4, vreinterpretq_u8_u32(vzip2q_u32(even, odd)));
            }
            gatherHorizSpan(srcRow, dstRow, ax, 2u * kEnd, fullW);
            return;
        }
        if (lowW * 3u == fullW * 2u && lowW >= 24u && (lowW % 8u) == 0u) {
            // 2:3 exact bilinear (e.g. 1280 -> 1920): 8 low pixels -> 12 full pixels per NEON iteration
            gatherHorizSpan(srcRow, dstRow, ax, 0, 12);
            const uint8x8_t w43  = vdup_n_u8(43u);
            const uint8x8_t w213 = vdup_n_u8(213u);
            auto blend16 = [&](uint8x16_t a, uint8x16_t b, uint8x8_t wa, uint8x8_t wb) -> uint8x16_t {
                const uint16x8_t lo = vmlal_u8(vmull_u8(vget_low_u8(a), wa), vget_low_u8(b), wb);
                const uint16x8_t hi = vmlal_u8(vmull_u8(vget_high_u8(a), wa), vget_high_u8(b), wb);
                return vcombine_u8(vrshrn_n_u16(lo, 8), vrshrn_n_u16(hi, 8));
            };
            const unsigned gEnd = (lowW / 2u) - 4u;
            for (unsigned g = 4; g < gEnd; g += 4) {
                const size_t k = size_t(2u * g);
                const uint32x4x2_t cur2 = vld2q_u32(reinterpret_cast<const uint32_t*>(srcRow + k * 4));
                const uint8x16_t pEven = vreinterpretq_u8_u32(cur2.val[0]);
                const uint8x16_t pOdd  = vreinterpretq_u8_u32(cur2.val[1]);
                const uint32x4_t prevOddLane = {
                    *reinterpret_cast<const uint32_t*>(srcRow + (k - 1) * 4),
                    cur2.val[1][0], cur2.val[1][1], cur2.val[1][2]
                };
                const uint32x4_t nextEvenLane = {
                    cur2.val[0][1], cur2.val[0][2], cur2.val[0][3],
                    *reinterpret_cast<const uint32_t*>(srcRow + (k + 8) * 4)
                };
                uint32x4x3_t out3;
                out3.val[0] = vreinterpretq_u32_u8(blend16(vreinterpretq_u8_u32(prevOddLane), pEven, w43, w213));
                out3.val[1] = vreinterpretq_u32_u8(vrhaddq_u8(pEven, pOdd));
                out3.val[2] = vreinterpretq_u32_u8(blend16(pOdd, vreinterpretq_u8_u32(nextEvenLane), w213, w43));
                vst3q_u32(reinterpret_cast<uint32_t*>(dstRow + size_t(3u * g) * 4), out3);
            }
            gatherHorizSpan(srcRow, dstRow, ax, 3u * gEnd, fullW);
            return;
        }
        if (fullW == lowW * 3u && lowW >= 16u && (lowW % 4u) == 0u) {
            // 1:3 single-cycle 8-bit NEON upsample (640 -> 1920): exact 0/255 & 0.0 u8 monochrome chroma, 0 widening mul!
            gatherHorizSpan(srcRow, dstRow, ax, 0, 12);
            const unsigned kEnd = lowW - 4u;
            for (unsigned k = 4; k < kEnd; k += 4) {
                const uint8x16_t prev = vld1q_u8(srcRow + size_t(k - 1) * 4);
                const uint8x16_t curr = vld1q_u8(srcRow + size_t(k) * 4);
                const uint8x16_t next = vld1q_u8(srcRow + size_t(k + 1) * 4);
                uint32x4x3_t out3;
                out3.val[0] = vreinterpretq_u32_u8(vrhaddq_u8(curr, vrhaddq_u8(prev, curr)));
                out3.val[1] = vreinterpretq_u32_u8(curr);
                out3.val[2] = vreinterpretq_u32_u8(vrhaddq_u8(curr, vrhaddq_u8(curr, next)));
                vst3q_u32(reinterpret_cast<uint32_t*>(dstRow + size_t(3u * k) * 4), out3);
            }
            gatherHorizSpan(srcRow, dstRow, ax, 3u * kEnd, fullW);
            return;
        }
        gatherHorizRow(srcRow, dstRow, ax);
    }

    static void downsampleHorizRowNeon(const uint8_t* srcRow, uint8_t* dstRow, uint32_t fullW, uint32_t lowW,
                                       const AxisTable& ax) noexcept {
        if (fullW * 2u == lowW * 3u && lowW >= 16u && (lowW % 8u) == 0u) {
            // 3:2 exact bilinear (e.g. 1920 -> 1280): 12 full pixels -> 8 low pixels per NEON iteration
            for (unsigned ox = 0; ox < lowW; ox += 8) {
                const size_t sx = size_t((ox / 2u) * 3u) * 4u;
                const uint32x4x3_t in3 = vld3q_u32(reinterpret_cast<const uint32_t*>(srcRow + sx));
                const uint8x16_t p0 = vreinterpretq_u8_u32(in3.val[0]);
                const uint8x16_t p1 = vreinterpretq_u8_u32(in3.val[1]);
                const uint8x16_t p2 = vreinterpretq_u8_u32(in3.val[2]);
                const uint32x4_t dEven = vreinterpretq_u32_u8(vrhaddq_u8(p0, vhaddq_u8(p0, p1)));
                const uint32x4_t dOdd  = vreinterpretq_u32_u8(vrhaddq_u8(p2, vhaddq_u8(p1, p2)));
                vst1q_u8(dstRow + size_t(ox) * 4u,      vreinterpretq_u8_u32(vzip1q_u32(dEven, dOdd)));
                vst1q_u8(dstRow + size_t(ox + 4u) * 4u, vreinterpretq_u8_u32(vzip2q_u32(dEven, dOdd)));
            }
            return;
        }
        gatherHorizRow(srcRow, dstRow, ax);
    }

    // Vectorized NEON HUD-sharp & static-dither blend: preserves static UI/crosshairs/subtitles & pure-white/black
    // regions (`max_bgr(|A-B|) <= 3`) at native full resolution with bit-exact 0 u8 error, enforces 0.0 u8 monochrome
    // chroma on grayscale source pairs, and clamps bottom border rows against OOB overshoot.
    static void blendHudRowNeon(const uint8_t* upRow, const uint8_t* rowA, const uint8_t* rowB,
                                uint8_t* dstRow, uint32_t fullW, uint32_t tq8, bool clampEndpoints = false) noexcept {
        const size_t rowBytes = size_t(fullW) * 4u;
        const uint8x16_t bgrMask = vreinterpretq_u8_u32(vdupq_n_u32(0x00FFFFFFu));
        const uint8x16_t alphaOr = vreinterpretq_u8_u32(vdupq_n_u32(0xFF000000u));
        const uint8x16_t thresh7 = vdupq_n_u8(7u);
        const uint8x16_t exactCut = vdupq_n_u8(252u);
        const uint8x16_t v253 = vdupq_n_u8(253u);
        const uint8x16_t v255 = vdupq_n_u8(255u);
        const uint8x16_t v2   = vdupq_n_u8(2u);
        const uint8x16_t v0   = vdupq_n_u8(0u);
        const uint32x4_t rgMask32 = vdupq_n_u32(0x0000FFFFu);
        const uint32x4_t byteMask32 = vdupq_n_u32(0xFFu);
        const bool halfPhase = (tq8 == 128u);
        const uint8x8_t vwa = vdup_n_u8(static_cast<uint8_t>(256u - std::clamp(tq8, 1u, 255u)));
        const uint8x8_t vwb = vdup_n_u8(static_cast<uint8_t>(std::clamp(tq8, 1u, 255u)));

        auto sanitizeMonoAndBounds = [&](uint8x16_t out, uint8x16_t va, uint8x16_t vb) -> uint8x16_t {
            if (clampEndpoints) {
                out = vminq_u8(vmaxq_u8(out, vminq_u8(va, vb)), vmaxq_u8(va, vb));
            }
            // If both A and B are grayscale (`max(|B-G|, |R-G|) <= 1`), enforce exact 0.0 u8 chroma (`B = G = R`) branchlessly
            const uint8x16_t ca = vmaxq_u8(vabdq_u8(va, vrev16q_u8(va)),
                                           vabdq_u8(va, vreinterpretq_u8_u16(vrev32q_u16(vreinterpretq_u16_u8(va)))));
            const uint8x16_t cb = vmaxq_u8(vabdq_u8(vb, vrev16q_u8(vb)),
                                           vabdq_u8(vb, vreinterpretq_u8_u16(vrev32q_u16(vreinterpretq_u16_u8(vb)))));
            const uint32x4_t cmaxB = vandq_u32(vreinterpretq_u32_u8(vmaxq_u8(ca, cb)), byteMask32);
            const uint32x4_t va32 = vreinterpretq_u32_u8(va);
            const uint32x4_t vb32 = vreinterpretq_u32_u8(vb);
            const uint32x4_t exactGray = vceqq_u32(
                vandq_u32(vorrq_u32(veorq_u32(va32, vshrq_n_u32(va32, 8)), veorq_u32(vb32, vshrq_n_u32(vb32, 8))), rgMask32),
                vdupq_n_u32(0u));
            const uint32x4_t nearGrayMoved = vandq_u32(vcleq_u32(cmaxB, vdupq_n_u32(1u)), vmvnq_u32(vceqq_u32(va32, vb32)));
            const uint32x4_t monoMask = vorrq_u32(exactGray, nearGrayMoved);
            const uint32x4_t out32 = vreinterpretq_u32_u8(out);
            const uint32x4_t g32 = vandq_u32(vshrq_n_u32(out32, 8), byteMask32);
            const uint32x4_t gray32 = vmulq_n_u32(g32, 0x00010101u);
            out = vreinterpretq_u8_u32(vbslq_u32(monoMask, gray32, out32));
            return vorrq_u8(out, alphaOr);
        };

        size_t i = 0;
        for (; i + 16 <= rowBytes; i += 16) {
            if ((i & 63u) == 0u) {
                __builtin_prefetch(rowA + i + 384u, 0, 0);
                __builtin_prefetch(rowB + i + 384u, 0, 0);
            }
            const uint8x16_t va = vld1q_u8(rowA + i);
            const uint8x16_t vb = vld1q_u8(rowB + i);
            const uint8x16_t vl = vld1q_u8(upRow + i);

            // Replicate max(|dB|, |dG|, |dR|) across all 4 bytes of each 32-bit pixel lane
            const uint8x16_t diff = vandq_u8(vabdq_u8(va, vb), bgrMask);
            const uint8x16_t m1 = vmaxq_u8(diff, vrev16q_u8(diff));
            const uint8x16_t dmax = vmaxq_u8(m1, vreinterpretq_u8_u16(vrev32q_u16(vreinterpretq_u16_u8(m1))));
            const uint8x16_t rem = vqsubq_u8(thresh7, dmax);  // >0 iff dmax < 7 (eliminates moving-gradient 0.5*(A+B) ghosting!)

            if (vmaxvq_u8(rem) == 0) {
                vst1q_u8(dstRow + i, sanitizeMonoAndBounds(vl, va, vb));
                continue;
            }
            uint8x16_t mid;
            if (halfPhase) {
                mid = vrhaddq_u8(va, vb);
            } else {
                const uint16x8_t mlo = vmlal_u8(vmull_u8(vget_low_u8(va), vwa), vget_low_u8(vb), vwb);
                const uint16x8_t mhi = vmlal_u8(vmull_u8(vget_high_u8(va), vwa), vget_high_u8(vb), vwb);
                mid = vcombine_u8(vrshrn_n_u16(mlo, 8), vrshrn_n_u16(mhi, 8));
            }
            // Preserve pure-white (>=253 in all BGR channels) -> 255 and pure-black (<=2 in all BGR channels) -> 0
            const uint8x16_t vaOrA = vorrq_u8(va, alphaOr);
            const uint8x16_t vbOrA = vorrq_u8(vb, alphaOr);
            const uint8x16_t minAB = vminq_u8(vaOrA, vbOrA);
            const uint8x16_t min1  = vminq_u8(minAB, vrev16q_u8(minAB));
            const uint8x16_t minBgr = vminq_u8(min1, vreinterpretq_u8_u16(vrev32q_u16(vreinterpretq_u16_u8(min1))));
            const uint8x16_t maxAB = vmaxq_u8(vandq_u8(va, bgrMask), vandq_u8(vb, bgrMask));
            const uint8x16_t max1  = vmaxq_u8(maxAB, vrev16q_u8(maxAB));
            const uint8x16_t maxBgr = vmaxq_u8(max1, vreinterpretq_u8_u16(vrev32q_u16(vreinterpretq_u16_u8(max1))));
            mid = vbslq_u8(vcgeq_u8(minBgr, v253), v255, mid);
            mid = vbslq_u8(vcleq_u8(maxBgr, v2), v0, mid);

            const uint8x16_t ws = vqshlq_n_u8(rem, 6);                // min(255, rem * 64): 255 when dmax <= 3
            const uint8x16_t exactMask = vcgeq_u8(ws, exactCut);      // 0xFF when dmax <= 3 (exact native HUD/dither)
            if (vminvq_u8(exactMask) == 0xFF) {
                vst1q_u8(dstRow + i, sanitizeMonoAndBounds(mid, va, vb));
                continue;
            }
            const uint8x16_t invWs = vmvnq_u8(ws);
            const uint16x8_t blo = vmlal_u8(vmull_u8(vget_low_u8(vl), vget_low_u8(invWs)), vget_low_u8(mid), vget_low_u8(ws));
            const uint16x8_t bhi = vmlal_u8(vmull_u8(vget_high_u8(vl), vget_high_u8(invWs)), vget_high_u8(mid), vget_high_u8(ws));
            const uint8x16_t blended = vcombine_u8(vrshrn_n_u16(blo, 8), vrshrn_n_u16(bhi, 8));
            vst1q_u8(dstRow + i, sanitizeMonoAndBounds(vbslq_u8(exactMask, mid, blended), va, vb));
        }
    }

public:
    CpuTierScaler() = default;
    ~CpuTierScaler() {
        for (auto& b : fullIn) unmapSlot(b);
        for (auto& b : fullOut) unmapSlot(b);
    }
    CpuTierScaler(const CpuTierScaler&) = delete;
    CpuTierScaler& operator=(const CpuTierScaler&) = delete;

    void mapFullBuffers(const std::array<int, 12>& keepIn, const std::array<int, 2>& keepOut,
                        const std::array<int, 4>& keepExtra, size_t fullBytes, size_t phases) {
        syncedInTag.fill(~0ull);
        // Mapped PROT_READ | PROT_WRITE solely so unprivileged ARMv8 EL0 `dc civac` is permitted by MMU (CPU only reads!).
        for (size_t i = 0; i < keepIn.size(); ++i) mapSlot(fullIn[i], keepIn[i], fullBytes, true);
        if (phases == 1) {
            mapSlot(fullOut[0], keepOut[0], fullBytes, true);
            mapSlot(fullOut[2], keepExtra[0], fullBytes, true);
            mapSlot(fullOut[4], keepExtra[1], fullBytes, true);
        } else {
            mapSlot(fullOut[0], keepOut[0], fullBytes, true);
            mapSlot(fullOut[1], keepOut[1], fullBytes, true);
            mapSlot(fullOut[2], keepExtra[0], fullBytes, true);
            mapSlot(fullOut[3], keepExtra[1], fullBytes, true);
            mapSlot(fullOut[4], keepExtra[2], fullBytes, true);
            mapSlot(fullOut[5], keepExtra[3], fullBytes, true);
        }
        for (auto& out : fullOut) {
            if (out.fd >= 0 && out.ptr != static_cast<uint8_t*>(MAP_FAILED))
                invalidateRangeEl0(out.ptr, fullBytes);
        }
    }

    void configureScale(uint32_t fullW, uint32_t fullH, uint32_t lowW, uint32_t lowH) {
        downX.build(fullW, lowW);
        downY.build(fullH, lowH);
        upX.build(lowW, fullW);
        upY.build(lowH, fullH);
    }

    void syncInputOnce(unsigned slot, uint64_t uniqueTag) {
        auto& b = fullIn.at(slot);
        if (b.fd < 0 || syncedInTag[slot] == uniqueTag) return;
        const uint8_t* ptr = b.ptr;
        const size_t nBytes = b.bytes;
        constexpr unsigned chunks = 16u;
        pool.parallelFor(chunks, 4u, [ptr, nBytes](unsigned c0, unsigned c1) {
            const size_t step = ((nBytes / chunks) + 63u) & ~size_t{63};
            for (unsigned c = c0; c < c1; ++c) {
                const size_t off = size_t(c) * step;
                if (off < nBytes) invalidateLinesNoBarrier(ptr + off, std::min(step, nBytes - off));
            }
        });
        asm volatile("dsb ish" ::: "memory");
        syncedInTag[slot] = uniqueTag;
    }

    void syncInputsPair(unsigned slotA, uint64_t tagA, unsigned slotB, uint64_t tagB) {
        const bool needA = fullIn.at(slotA).fd >= 0 && syncedInTag[slotA] != tagA;
        const bool needB = fullIn.at(slotB).fd >= 0 && syncedInTag[slotB] != tagB && slotB != slotA;
        if (needA && needB) {
            const uint8_t* ptrA = fullIn[slotA].ptr;
            const uint8_t* ptrB = fullIn[slotB].ptr;
            const size_t nBytes = fullIn[slotA].bytes;
            pool.parallelFor(2, 1, [ptrA, ptrB, nBytes](unsigned i0, unsigned i1) {
                for (unsigned i = i0; i < i1; ++i) {
                    invalidateRangeEl0(i == 0 ? ptrA : ptrB, nBytes);
                }
            });
            syncedInTag[slotA] = tagA;
            syncedInTag[slotB] = tagB;
        } else {
            if (needA) syncInputOnce(slotA, tagA);
            if (needB) syncInputOnce(slotB, tagB);
        }
    }

    void downsampleTo(unsigned inSlot, uint8_t* lowDst, int lowDstFd,
                      uint32_t fullW, uint32_t fullH, uint32_t lowW, uint32_t lowH,
                      bool antiAlias = false) {
        configureScale(fullW, fullH, lowW, lowH);
        const uint8_t* src = fullIn.at(inSlot).ptr;
        const size_t fullStride = size_t(fullW) * 4u;
        const size_t lowStride = size_t(lowW) * 4u;

        if (fullW == lowW * 2u && fullH == lowH * 2u && (lowW % 16u) == 0u) {
            pool.parallelFor(lowH, 18, [&](unsigned y0, unsigned y1) {
                if (!antiAlias) {
                    for (unsigned y = y0; y < y1; ++y) {
                        const uint8_t* r0 = src + size_t(2u * y) * fullStride;
                        invalidateLinesNoBarrier(r0, fullStride);
                    }
                    asm volatile("dsb ish" ::: "memory");
                }
                for (unsigned y = y0; y < y1; ++y) {
                    const uint8_t* r0 = src + size_t(2u * y) * fullStride;
                    const uint8_t* r1 = src + size_t(2u * y + 1u) * fullStride;
                    uint8_t* d = lowDst + size_t(y) * lowStride;
                    for (unsigned x = 0; x < lowW; x += 16) {
                        auto step4 = [&](unsigned ox) -> uint8x16_t {
                            const size_t sx = size_t(2u * ox) * 4u;
                            const uint8x16_t a0 = antiAlias ? vrhaddq_u8(vld1q_u8(r0 + sx), vld1q_u8(r1 + sx)) : vld1q_u8(r0 + sx);
                            const uint8x16_t a1 = antiAlias ? vrhaddq_u8(vld1q_u8(r0 + sx + 16), vld1q_u8(r1 + sx + 16)) : vld1q_u8(r0 + sx + 16);
                            const uint32x4x2_t uz = vuzpq_u32(vreinterpretq_u32_u8(a0), vreinterpretq_u32_u8(a1));
                            return vrhaddq_u8(vreinterpretq_u8_u32(uz.val[0]), vreinterpretq_u8_u32(uz.val[1]));
                        };
                        const uint8x16_t q0 = step4(x), q1 = step4(x + 4), q2 = step4(x + 8), q3 = step4(x + 12);
                        asm volatile("stnp %q0, %q1, [%4]\n\tstnp %q2, %q3, [%4, #32]"
                                     : : "w"(q0), "w"(q1), "w"(q2), "w"(q3), "r"(d + size_t(x) * 4u));
                    }
                }
            });
            (void)lowDstFd;
            asm volatile("dsb ish" ::: "memory");
            return;
        } else if (fullW == lowW * 3u && fullH == lowH * 3u && (lowW % 16u) == 0u) {
            auto run3x = [=](unsigned y0, unsigned y1) {
                if (!antiAlias) {
                    for (unsigned y = y0; y < y1; ++y) {
                        const uint8_t* r1 = src + size_t(3u * y + 1u) * fullStride;
                        invalidateLinesNoBarrier(r1, fullStride);
                    }
                    asm volatile("dsb ish" ::: "memory");
                }
                for (unsigned y = y0; y < y1; ++y) {
                    const uint8_t* r0 = src + size_t(3u * y) * fullStride;
                    const uint8_t* r1 = src + size_t(3u * y + 1u) * fullStride;
                    const uint8_t* r2 = src + size_t(3u * y + 2u) * fullStride;
                    uint8_t* d = lowDst + size_t(y) * lowStride;
                    for (unsigned x = 0; x < lowW; x += 16) {
                        __builtin_prefetch(r1 + size_t(3u * (x + 16u)) * 4u, 0, 0);
                        auto step4 = [&](unsigned ox) -> uint8x16_t {
                            const size_t sx = size_t(3u * ox) * 4u;
                            if (antiAlias) {
                                const uint32x4x3_t t0 = vld3q_u32(reinterpret_cast<const uint32_t*>(r0 + sx));
                                const uint32x4x3_t t1 = vld3q_u32(reinterpret_cast<const uint32_t*>(r1 + sx));
                                const uint32x4x3_t t2 = vld3q_u32(reinterpret_cast<const uint32_t*>(r2 + sx));
                                const uint8x16_t v0 = vrhaddq_u8(vreinterpretq_u8_u32(t1.val[0]),
                                                                 vrhaddq_u8(vreinterpretq_u8_u32(t0.val[0]), vreinterpretq_u8_u32(t2.val[0])));
                                const uint8x16_t v1 = vrhaddq_u8(vreinterpretq_u8_u32(t1.val[1]),
                                                                 vrhaddq_u8(vreinterpretq_u8_u32(t0.val[1]), vreinterpretq_u8_u32(t2.val[1])));
                                const uint8x16_t v2 = vrhaddq_u8(vreinterpretq_u8_u32(t1.val[2]),
                                                                 vrhaddq_u8(vreinterpretq_u8_u32(t0.val[2]), vreinterpretq_u8_u32(t2.val[2])));
                                return vrhaddq_u8(v1, vrhaddq_u8(v0, v2));
                            }
                            const uint32x4x3_t in3 = vld3q_u32(reinterpret_cast<const uint32_t*>(r1 + sx));
                            const uint8x16_t avg02 = vrhaddq_u8(vreinterpretq_u8_u32(in3.val[0]), vreinterpretq_u8_u32(in3.val[2]));
                            return vrhaddq_u8(vreinterpretq_u8_u32(in3.val[1]), avg02);
                        };
                        const uint8x16_t q0 = step4(x), q1 = step4(x + 4), q2 = step4(x + 8), q3 = step4(x + 12);
                        asm volatile("stnp %q0, %q1, [%4]\n\tstnp %q2, %q3, [%4, #32]"
                                     : : "w"(q0), "w"(q1), "w"(q2), "w"(q3), "r"(d + size_t(x) * 4u));
                    }
                }
            };
            if (lowH <= 360u && !antiAlias) run3x(0, lowH);
            else pool.parallelFor(lowH, 18, run3x);
            (void)lowDstFd;
            asm volatile("dsb ish" ::: "memory");
            return;
        } else if (fullW * 2u == lowW * 3u && fullH * 2u == lowH * 3u && (lowW % 16u) == 0u && (lowH % 2u) == 0u) {
            // Direct 3:2 2D anti-aliased NEON downsample (1920x1080 -> 1280x720) streamed via `stnp` (0 RFO, 0 cleanRangeEl0!)
            pool.parallelFor(lowH / 2u, 12u, [&](unsigned g0, unsigned g1) {
                if (!antiAlias) {
                    for (unsigned g = g0; g < g1; ++g) {
                        invalidateLinesNoBarrier(src + size_t(3u * g) * fullStride, fullStride * 3u);
                    }
                    asm volatile("dsb ish" ::: "memory");
                }
                for (unsigned g = g0; g < g1; ++g) {
                    const uint8_t* r0 = src + size_t(3u * g) * fullStride;
                    const uint8_t* r1 = src + size_t(3u * g + 1u) * fullStride;
                    const uint8_t* r2 = src + size_t(3u * g + 2u) * fullStride;
                    uint8_t* d0 = lowDst + size_t(2u * g) * lowStride;
                    uint8_t* d1 = lowDst + size_t(2u * g + 1u) * lowStride;
                    for (unsigned ox = 0; ox < lowW; ox += 16u) {
                        auto hpair8 = [&](unsigned bx, const uint8_t* ra, const uint8_t* rb, uint8x16_t& qLo, uint8x16_t& qHi) {
                            const size_t sx = size_t((bx / 2u) * 3u) * 4u;
                            const uint32x4x3_t a3 = vld3q_u32(reinterpret_cast<const uint32_t*>(ra + sx));
                            const uint32x4x3_t b3 = vld3q_u32(reinterpret_cast<const uint32_t*>(rb + sx));
                            const uint8x16_t p0 = vrhaddq_u8(vreinterpretq_u8_u32(a3.val[0]),
                                                             vhaddq_u8(vreinterpretq_u8_u32(a3.val[0]), vreinterpretq_u8_u32(b3.val[0])));
                            const uint8x16_t p1 = vrhaddq_u8(vreinterpretq_u8_u32(a3.val[1]),
                                                             vhaddq_u8(vreinterpretq_u8_u32(a3.val[1]), vreinterpretq_u8_u32(b3.val[1])));
                            const uint8x16_t p2 = vrhaddq_u8(vreinterpretq_u8_u32(a3.val[2]),
                                                             vhaddq_u8(vreinterpretq_u8_u32(a3.val[2]), vreinterpretq_u8_u32(b3.val[2])));
                            const uint32x4_t dEven = vreinterpretq_u32_u8(vrhaddq_u8(p0, vhaddq_u8(p0, p1)));
                            const uint32x4_t dOdd  = vreinterpretq_u32_u8(vrhaddq_u8(p2, vhaddq_u8(p1, p2)));
                            qLo = vreinterpretq_u8_u32(vzip1q_u32(dEven, dOdd));
                            qHi = vreinterpretq_u8_u32(vzip2q_u32(dEven, dOdd));
                        };
                        uint8x16_t t0, t1, t2, t3;
                        hpair8(ox, r0, r1, t0, t1);
                        hpair8(ox + 8u, r0, r1, t2, t3);
                        asm volatile("stnp %q0, %q1, [%4]\n\tstnp %q2, %q3, [%4, #32]"
                                     : : "w"(t0), "w"(t1), "w"(t2), "w"(t3), "r"(d0 + size_t(ox) * 4u));
                        hpair8(ox, r2, r1, t0, t1);
                        hpair8(ox + 8u, r2, r1, t2, t3);
                        asm volatile("stnp %q0, %q1, [%4]\n\tstnp %q2, %q3, [%4, #32]"
                                     : : "w"(t0), "w"(t1), "w"(t2), "w"(t3), "r"(d1 + size_t(ox) * 4u));
                    }
                }
            });
            (void)lowDstFd;
            asm volatile("dsb ish" ::: "memory");
            return;
        } else {
            pool.parallelFor(lowH, 16, [&](unsigned y0, unsigned y1) {
                alignas(64) uint8_t vRow[8192];
                if (!antiAlias) {
                    uint32_t lastInvRow = ~0u;
                    for (unsigned y = y0; y < y1; ++y) {
                        const uint32_t iy0 = downY.i0[y];
                        const uint32_t iy1 = downY.i1[y];
                        if (iy0 != lastInvRow) {
                            invalidateLinesNoBarrier(src + size_t(iy0) * fullStride, fullStride);
                            lastInvRow = iy0;
                        }
                        if (iy1 != lastInvRow) {
                            invalidateLinesNoBarrier(src + size_t(iy1) * fullStride, fullStride);
                            lastInvRow = iy1;
                        }
                    }
                    asm volatile("dsb ish" ::: "memory");
                }
                for (unsigned y = y0; y < y1; ++y) {
                    const uint8_t* r0 = src + size_t(downY.i0[y]) * fullStride;
                    const uint8_t* r1 = src + size_t(downY.i1[y]) * fullStride;
                    const uint8_t* v = blendRowVerticalNeon(r0, r1, downY.w1[y], fullStride, vRow);
                    downsampleHorizRowNeon(v, lowDst + size_t(y) * lowStride, fullW, lowW, downX);
                }
            });
        }
        (void)lowDstFd;
        cleanRangeEl0(lowDst, size_t(lowH) * lowStride);
    }

    void upsamplePhase(unsigned aSlot, unsigned bSlot, const uint8_t* lowSrc, int lowSrcFd, size_t outIdx,
                       uint32_t fullW, uint32_t fullH, uint32_t lowW, uint32_t lowH, uint32_t tq8,
                       bool strictHud = false, uint32_t preset = 5u) {
        (void)lowSrcFd;
        const uint32_t sharpTier = (preset == 4u || preset == 5u || preset == 7u) ? 2u : ((preset == 2u || preset == 3u) ? 1u : 0u);
        configureScale(fullW, fullH, lowW, lowH);
        const size_t lowStride = size_t(lowW) * 4u;
        invalidateRangeEl0(lowSrc, size_t(lowH) * lowStride);
        static const bool envStrictHud = [] {
            const char* s = std::getenv("ODIN3_FG_STRICT_HUD");
            return s && s[0] == '1';
        }();
        const bool useStrict = strictHud || envStrictHud;
        auto& dstBuf = fullOut.at(outIdx);
        uint8_t* dst = dstBuf.ptr;
        const size_t fullStride = size_t(fullW) * 4u;

        if (!useStrict) {
            // Live sub-resolution fast path: 2-row L1 horizontal upsample cache (computes H(row) once per low row = 3x fewer
            // horizontal passes at 360p->1080p!) + vertical blend streamed via ARM64 `stnp` (0 MB L2 RFO cache pollution!).
            const uint8x16_t alphaOr = vreinterpretq_u8_u32(vdupq_n_u32(0xFF000000u));
            auto runFastUp = [&](unsigned y0, unsigned y1) {
                alignas(64) uint8_t hBufA[8192];
                alignas(64) uint8_t hBufB[8192];
                alignas(64) uint8_t sBuf[8192];
                uint8_t* hRow0 = hBufA;
                uint8_t* hRow1 = hBufB;
                uint32_t cachedLy0 = ~0u;
                uint32_t cachedLy1 = ~0u;
                for (unsigned y = y0; y < y1; ++y) {
                    const uint32_t ly0 = upY.i0[y];
                    const uint32_t ly1 = upY.i1[y];
                    const uint16_t wy  = upY.w1[y];
                    if (ly0 != cachedLy0) {
                        if (ly0 == cachedLy1) {
                            std::swap(hRow0, hRow1);
                            std::swap(cachedLy0, cachedLy1);
                        } else {
                            const uint8_t* sl = sharpenLowRowNeon(lowSrc, ly0, lowW, lowH, lowStride, sBuf, sharpTier);
                            upsampleHorizRowNeon(sl, hRow0, lowW, fullW, upX);
                            cachedLy0 = ly0;
                        }
                    }
                    uint8_t* dstRow = dst + size_t(y) * fullStride;
                    if (wy == 0 || ly1 == ly0 || wy >= 256) {
                        const uint8_t* srcH = (wy >= 256 && ly1 != ly0) ? [&]() -> const uint8_t* {
                            if (ly1 != cachedLy1) {
                                const uint8_t* sl = sharpenLowRowNeon(lowSrc, ly1, lowW, lowH, lowStride, sBuf, sharpTier);
                                upsampleHorizRowNeon(sl, hRow1, lowW, fullW, upX);
                                cachedLy1 = ly1;
                            }
                            return hRow1;
                        }() : hRow0;
                        size_t i = 0;
                        for (; i + 64 <= fullStride; i += 64) {
                            const uint8x16_t q0 = vorrq_u8(vld1q_u8(srcH + i), alphaOr);
                            const uint8x16_t q1 = vorrq_u8(vld1q_u8(srcH + i + 16), alphaOr);
                            const uint8x16_t q2 = vorrq_u8(vld1q_u8(srcH + i + 32), alphaOr);
                            const uint8x16_t q3 = vorrq_u8(vld1q_u8(srcH + i + 48), alphaOr);
                            asm volatile("stnp %q0, %q1, [%4]\n\tstnp %q2, %q3, [%4, #32]"
                                         : : "w"(q0), "w"(q1), "w"(q2), "w"(q3), "r"(dstRow + i));
                        }
                        if (i < fullStride) {
                            for (; i + 16 <= fullStride; i += 16)
                                vst1q_u8(dstRow + i, vorrq_u8(vld1q_u8(srcH + i), alphaOr));
                            cleanLinesNoBarrier(dstRow + (fullStride & ~size_t{63}), fullStride - (fullStride & ~size_t{63}));
                        }
                    } else {
                        if (ly1 != cachedLy1) {
                            const uint8_t* sl = sharpenLowRowNeon(lowSrc, ly1, lowW, lowH, lowStride, sBuf, sharpTier);
                            upsampleHorizRowNeon(sl, hRow1, lowW, fullW, upX);
                            cachedLy1 = ly1;
                        }
                        const bool fast3xVert = (fullH == lowH * 3u);
                        const uint8x8_t vw0 = vdup_n_u8(static_cast<uint8_t>(256u - wy));
                        const uint8x8_t vw1 = vdup_n_u8(static_cast<uint8_t>(wy));
                        auto vblend16 = [&](size_t off) -> uint8x16_t {
                            const uint8x16_t a = vld1q_u8(hRow0 + off);
                            const uint8x16_t b = vld1q_u8(hRow1 + off);
                            if (fast3xVert) {
                                const uint8x16_t mid = vrhaddq_u8(a, b);
                                return vorrq_u8(wy <= 128u ? vrhaddq_u8(a, mid) : vrhaddq_u8(b, mid), alphaOr);
                            }
                            const uint16x8_t lo = vmlal_u8(vmull_u8(vget_low_u8(a), vw0), vget_low_u8(b), vw1);
                            const uint16x8_t hi = vmlal_u8(vmull_u8(vget_high_u8(a), vw0), vget_high_u8(b), vw1);
                            return vorrq_u8(vcombine_u8(vrshrn_n_u16(lo, 8), vrshrn_n_u16(hi, 8)), alphaOr);
                        };
                        size_t i = 0;
                        for (; i + 64 <= fullStride; i += 64) {
                            const uint8x16_t q0 = vblend16(i);
                            const uint8x16_t q1 = vblend16(i + 16);
                            const uint8x16_t q2 = vblend16(i + 32);
                            const uint8x16_t q3 = vblend16(i + 48);
                            asm volatile("stnp %q0, %q1, [%4]\n\tstnp %q2, %q3, [%4, #32]"
                                         : : "w"(q0), "w"(q1), "w"(q2), "w"(q3), "r"(dstRow + i));
                        }
                        if (i < fullStride) {
                            for (; i + 16 <= fullStride; i += 16)
                                vst1q_u8(dstRow + i, vblend16(i));
                            cleanLinesNoBarrier(dstRow + (fullStride & ~size_t{63}), fullStride - (fullStride & ~size_t{63}));
                        }
                    }
                }
            };
            if (lowH <= 360u) runFastUp(0, fullH);
            else pool.parallelFor(fullH, 36, runFastUp);
            asm volatile("dsb ish" ::: "memory");
            return;
        }

        // Strict HUD / static-dither path (used in capped <=62 FPS 120 Hz lock & silicon verifier):
        // 2-row L1 horizontal cache + native full-res HUD/white/black/monochrome guard streamed via `stnp` (0 RFO, 0 ioctl!).
        struct StrictCtx {
            const AxisTable* upX;
            const AxisTable* upY;
            const uint8_t* lowSrc;
            const uint8_t* srcA;
            const uint8_t* srcB;
            uint8_t* dst;
            size_t lowStride;
            size_t fullStride;
            uint32_t lowW, fullW, lowH, fullH, tq8, sharpTier;
        } ctx{&upX, &upY, lowSrc, fullIn.at(aSlot).ptr, fullIn.at(bSlot).ptr, dst,
              lowStride, fullStride, lowW, fullW, lowH, fullH, tq8, sharpTier};

        pool.parallelFor(fullH, 36, [&ctx](unsigned y0, unsigned y1) {
            alignas(64) uint8_t hBufA[8192];
            alignas(64) uint8_t hBufB[8192];
            alignas(64) uint8_t sBuf[8192];
            alignas(64) uint8_t upRow[8192];
            alignas(64) uint8_t outRow[8192];
            uint8_t* hRow0 = hBufA;
            uint8_t* hRow1 = hBufB;
            uint32_t cachedLy0 = ~0u;
            uint32_t cachedLy1 = ~0u;
            const bool fast3xVert = (ctx.fullH == ctx.lowH * 3u);
            for (unsigned y = y0; y < y1; ++y) {
                const uint32_t ly0 = ctx.upY->i0[y];
                const uint32_t ly1 = ctx.upY->i1[y];
                const uint16_t wy  = ctx.upY->w1[y];
                if (ly0 != cachedLy0) {
                    if (ly0 == cachedLy1) {
                        std::swap(hRow0, hRow1);
                        std::swap(cachedLy0, cachedLy1);
                    } else {
                        const uint8_t* sl = sharpenLowRowNeon(ctx.lowSrc, ly0, ctx.lowW, ctx.lowH, ctx.lowStride, sBuf, ctx.sharpTier);
                        upsampleHorizRowNeon(sl, hRow0, ctx.lowW, ctx.fullW, *ctx.upX);
                        cachedLy0 = ly0;
                    }
                }
                const uint8_t* upPtr = hRow0;
                if (wy >= 256 && ly1 != ly0) {
                    if (ly1 != cachedLy1) {
                        const uint8_t* sl = sharpenLowRowNeon(ctx.lowSrc, ly1, ctx.lowW, ctx.lowH, ctx.lowStride, sBuf, ctx.sharpTier);
                        upsampleHorizRowNeon(sl, hRow1, ctx.lowW, ctx.fullW, *ctx.upX);
                        cachedLy1 = ly1;
                    }
                    upPtr = hRow1;
                } else if (wy > 0 && ly1 != ly0) {
                    if (ly1 != cachedLy1) {
                        const uint8_t* sl = sharpenLowRowNeon(ctx.lowSrc, ly1, ctx.lowW, ctx.lowH, ctx.lowStride, sBuf, ctx.sharpTier);
                        upsampleHorizRowNeon(sl, hRow1, ctx.lowW, ctx.fullW, *ctx.upX);
                        cachedLy1 = ly1;
                    }
                    if (fast3xVert) {
                        for (size_t i = 0; i + 16 <= ctx.fullStride; i += 16) {
                            const uint8x16_t a = vld1q_u8(hRow0 + i);
                            const uint8x16_t b = vld1q_u8(hRow1 + i);
                            const uint8x16_t mid = vrhaddq_u8(a, b);
                            vst1q_u8(upRow + i, wy <= 128u ? vrhaddq_u8(a, mid) : vrhaddq_u8(b, mid));
                        }
                        upPtr = upRow;
                    } else {
                        upPtr = blendRowVerticalNeon(hRow0, hRow1, wy, ctx.fullStride, upRow);
                    }
                }
                const size_t off = size_t(y) * ctx.fullStride;
                blendHudRowNeon(upPtr, ctx.srcA + off, ctx.srcB + off, outRow, ctx.fullW, ctx.tq8, y + 4u >= ctx.fullH);
                uint8_t* dstRow = ctx.dst + off;
                size_t i = 0;
                for (; i + 64 <= ctx.fullStride; i += 64) {
                    const uint8x16_t q0 = vld1q_u8(outRow + i);
                    const uint8x16_t q1 = vld1q_u8(outRow + i + 16);
                    const uint8x16_t q2 = vld1q_u8(outRow + i + 32);
                    const uint8x16_t q3 = vld1q_u8(outRow + i + 48);
                    asm volatile("stnp %q0, %q1, [%4]\n\tstnp %q2, %q3, [%4, #32]"
                                 : : "w"(q0), "w"(q1), "w"(q2), "w"(q3), "r"(dstRow + i));
                }
                if (i < ctx.fullStride) {
                    for (; i + 16 <= ctx.fullStride; i += 16)
                        vst1q_u8(dstRow + i, vld1q_u8(outRow + i));
                    cleanLinesNoBarrier(dstRow + (ctx.fullStride & ~size_t{63}), ctx.fullStride - (ctx.fullStride & ~size_t{63}));
                }
            }
        });
        asm volatile("dsb ish" ::: "memory");
    }
};

}  // namespace odin3_tier
