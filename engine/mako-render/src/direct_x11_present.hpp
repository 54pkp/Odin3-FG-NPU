/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
#include <span>
#include <vector>
#include <xcb/dri3.h>
#include <xcb/present.h>
#include <xcb/xcb.h>
#include <xcb/xcbext.h>

namespace odin3_direct {
using Clock = std::chrono::steady_clock;
inline uint64_t ns(Clock::time_point time = Clock::now()) noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

class Dri3 {
public:
    static constexpr unsigned phaseSlots = 3;
    static constexpr unsigned triplePhaseSlots = 6;
    static constexpr unsigned maxPhaseSlots = 6;
    static constexpr unsigned originalSlots = 12;
    static constexpr unsigned slots = maxPhaseSlots + originalSlots;
    static constexpr unsigned credit = 6;
    enum class Role { Generated, Original };
private:
    struct Sync {
        xcb_dri3_syncobj_t acquire{}, release{};
        uint32_t acquireHandle{}, releaseHandle{};
        uint64_t point{}, lastPendingQueryBegin{};
        uint32_t serial{};
        uint64_t device{}, inode{}, watchFrame{};
        unsigned watchPhase{};
        bool watchFlushed{};
    };
    struct Record {
        uint64_t frame, submit, target, complete{}, completeObserved{}, release{};
        uint64_t msc{};
        uint32_t serial;
        unsigned slot, phase, mode{255};
        uint64_t phaseCutoff{}, creditCutoff{};
    };

    struct ReleaseQueryRecord {
        Record delivery;
        uint64_t lastPendingQueryBegin, firstReadyQueryBegin;
    };

    template<class F>
    static F symbol(void* lib, const char* name) {
        const auto func = reinterpret_cast<F>(dlsym(lib, name));
        if (!func) throw std::runtime_error(name);
        return func;
    }
    static void* library(const char* name) {
        auto* lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw std::runtime_error(name);
        return lib;
    }
    struct Dispatch {
        void* xcb = library("libxcb.so.1");
        void* dri = library("libxcb-dri3.so.0");
        void* present = library("libxcb-present.so.0");
        void* drm = library("libdrm.so.2");
#define ODIN_XCB(member, lib, function) decltype(&function) member = symbol<decltype(&function)>(lib, #function)
        ODIN_XCB(connect, xcb, xcb_connect);
        ODIN_XCB(disconnect, xcb, xcb_disconnect);
        ODIN_XCB(hasError, xcb, xcb_connection_has_error);
        ODIN_XCB(getGeometry, xcb, xcb_get_geometry);
        ODIN_XCB(geometryReply, xcb, xcb_get_geometry_reply);
        ODIN_XCB(newId, xcb, xcb_generate_id);
        ODIN_XCB(check, xcb, xcb_request_check);
        ODIN_XCB(flush, xcb, xcb_flush);
        ODIN_XCB(getFd, xcb, xcb_get_file_descriptor);
        ODIN_XCB(special, xcb, xcb_register_for_special_xge);
        ODIN_XCB(pollSpecial, xcb, xcb_poll_for_special_event);
        ODIN_XCB(pollEvent, xcb, xcb_poll_for_event);
        ODIN_XCB(importPixmap, dri, xcb_dri3_pixmap_from_buffers_checked);
        ODIN_XCB(importSync, dri, xcb_dri3_import_syncobj_checked);
        ODIN_XCB(selectPresent, present, xcb_present_select_input_checked);
        ODIN_XCB(capabilities, present, xcb_present_query_capabilities);
        ODIN_XCB(capabilitiesReply, present, xcb_present_query_capabilities_reply);
        ODIN_XCB(presentSynced, present, xcb_present_pixmap_synced);
#undef ODIN_XCB
        xcb_extension_t* extension = symbol<xcb_extension_t*>(present, "xcb_present_id");
        using SyncCreate = int (*)(int, uint32_t, uint32_t*);
        using SyncDestroy = int (*)(int, uint32_t);
        using SyncExport = int (*)(int, uint32_t, int*);
        using SyncSignal = int (*)(int, const uint32_t*, uint64_t*, uint32_t);
        using SyncQuery = int (*)(int, uint32_t*, uint64_t*, uint32_t);
        SyncCreate create = symbol<SyncCreate>(drm, "drmSyncobjCreate");
        SyncDestroy destroy = symbol<SyncDestroy>(drm, "drmSyncobjDestroy");
        SyncExport exportFd = symbol<SyncExport>(drm, "drmSyncobjHandleToFD");
        SyncSignal signal = symbol<SyncSignal>(drm, "drmSyncobjTimelineSignal");
        SyncQuery query = symbol<SyncQuery>(drm, "drmSyncobjQuery");
    };
    static Dispatch& dispatch() {
        static Dispatch api;
        return api;
    }

    Dispatch& api = dispatch();
    xcb_connection_t* connection{};
    xcb_special_event_t* events{};
    xcb_window_t drawable{};
    uint32_t eventStamp{};
    int drm{-1};
    int releaseWatch{-1};
    std::string releaseWatchPath;
    uint64_t releaseWatchContext{};
    unsigned activePhaseSlots{phaseSlots};
    unsigned activeSlots{phaseSlots + originalSlots};
    bool recordDelivery{};
    std::chrono::nanoseconds vblankPeriod{8'326'000};
    std::array<xcb_pixmap_t, slots> pixmaps{};
    std::array<Sync, slots> sync{};
    std::array<uint32_t, credit> inflight{};
    uint32_t count{}, serial{};
    uint64_t lastCompleteMsc{}, lastTargetMsc{};
    Clock::time_point lastFlip{}, lastCompleteTime{};
    unsigned peak{}, nonFlip{};
    std::vector<Record> records;
    static constexpr std::size_t capacity = 24000;

    struct JournalHeader {
        std::array<char, 8> magic;
        uint32_t version, headerBytes, recordBytes, maxRecords;
        uint32_t count, dropped, pid, window;
        uint64_t context;
    };
    static_assert(sizeof(JournalHeader) == 48 && sizeof(Record) == 88 && sizeof(ReleaseQueryRecord) == 104);
    JournalHeader* journal{};
    std::size_t journalBytes{}, journalRecordBytes{sizeof(Record)};

    void openJournal(uint64_t context) {
        const auto* directory = std::getenv("ODIN3_DRI3_DELIVERY_DIR");
        if (!directory || directory[0] != '/') return;
        const auto pid = static_cast<uint32_t>(getpid());
        const auto path = std::string(directory) + "/dri3-" + std::to_string(pid) + "-" +
            std::to_string(context) + "-" + std::to_string(drawable) + ".bin";
        const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) fail("Create DRI3 delivery journal");
        const auto* trace = std::getenv("ODIN3_DRI3_TRACE_RELEASE_QUERIES");
        const bool traceQueries = trace && std::strcmp(trace, "1") == 0;
        journalRecordBytes = traceQueries ? sizeof(ReleaseQueryRecord) : sizeof(Record);
        journalBytes = sizeof(JournalHeader) + capacity * journalRecordBytes;
        void* mapping = MAP_FAILED;
        if (ftruncate(fd, journalBytes) == 0)
            mapping = mmap(nullptr, journalBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (mapping == MAP_FAILED) fail("Map DRI3 delivery journal");
        const JournalHeader header{{'O','D','R','I','3','L','E','D'}, traceQueries ? 2u : 1u,
            sizeof(JournalHeader), static_cast<uint32_t>(journalRecordBytes), capacity, 0, 0, pid, drawable, context};
        journal = static_cast<JournalHeader*>(std::memcpy(mapping, &header, sizeof(header)));
    }

    unsigned char* journalRow(uint32_t id) noexcept {
        return reinterpret_cast<unsigned char*>(journal) + sizeof(JournalHeader) + (id - 1) * journalRecordBytes;
    }

    void mirror(const Record& record) noexcept {
        if (journal) std::memcpy(journalRow(record.serial), &record, sizeof(record));
    }

    [[noreturn]] static void fail(const char* what) { throw std::runtime_error(what); }

    void checked(xcb_void_cookie_t cookie, const char* what) {
        auto* error = api.check(connection, cookie);
        if (!error) return;
        const auto code = error->error_code;
        free(error);
        throw std::runtime_error(std::string(what) + " X11 error " + std::to_string(code));
    }

    xcb_dri3_syncobj_t importSync(uint32_t& handle) {
        int fd = -1;
        if (api.create(drm, 0, &handle) || api.exportFd(drm, handle, &fd)) fail("Export DRM timeline");
        const auto id = api.newId(connection);
        checked(api.importSync(connection, id, drawable, fd), "Import DRM timeline");
        return id;
    }

    bool sendReleaseWatch(const char* message, int size, int fd = -1) noexcept {
        if (size <= 0 || size >= 256) return false;
        iovec payload{const_cast<char*>(message), static_cast<size_t>(size)};
        msghdr packet{};
        packet.msg_iov = &payload;
        packet.msg_iovlen = 1;
        alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(sizeof(int))> ancillary{};
        if (fd >= 0) {
            packet.msg_control = ancillary.data();
            packet.msg_controllen = ancillary.size();
            auto* rights = CMSG_FIRSTHDR(&packet);
            rights->cmsg_level = SOL_SOCKET;
            rights->cmsg_type = SCM_RIGHTS;
            rights->cmsg_len = CMSG_LEN(sizeof(int));
            std::memcpy(CMSG_DATA(rights), &fd, sizeof(fd));
        }
        return sendmsg(releaseWatch, &packet, MSG_DONTWAIT | MSG_NOSIGNAL) == size;
    }

    void exportPendingReleases(uint64_t context) noexcept {
        if (releaseWatchPath.empty() || releaseWatch >= 0) return;
        sockaddr_un destination{};
        const auto length = releaseWatchPath.size();
        if (length >= sizeof(destination.sun_path)) {
            std::fprintf(stderr, "MAKO Renderer: DRI3 release socket path is too long\n");
            return;
        }
        destination.sun_family = AF_UNIX;
        std::memcpy(destination.sun_path, releaseWatchPath.c_str(), length + 1);
        releaseWatch = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (releaseWatch < 0 || ::connect(releaseWatch, reinterpret_cast<const sockaddr*>(&destination),
                static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + length + 1)) != 0) {
            const auto error = errno;
            if (releaseWatch >= 0) ::close(std::exchange(releaseWatch, -1));
            std::fprintf(stderr, "MAKO Renderer: DRI3 release observer unavailable errno=%d\n", error);
            return;
        }
        releaseWatchContext = context;
        unsigned exported = 0;
        for (unsigned index = 0; index < activeSlots; ++index) {
            const auto& slot = sync[index];
            if (!slot.serial) continue;
            int descriptor = -1;
            if (!slot.watchFlushed || !slot.inode || api.exportFd(drm, slot.releaseHandle, &descriptor) || descriptor < 0) {
                std::fprintf(stderr, "MAKO Renderer: DRI3 release export unavailable context=%ju serial=%u\n",
                             uintmax_t(context), slot.serial);
                if (descriptor >= 0) ::close(descriptor);
                continue;
            }
            char packet[256];
            const auto size = std::snprintf(packet, sizeof(packet), "watch %d %ju %u %u %u %ju %u %ju %ju %ju %ju",
                getpid(), uintmax_t(context), drawable, slot.serial, index, uintmax_t(slot.watchFrame), slot.watchPhase,
                uintmax_t(slot.point), uintmax_t(slot.device), uintmax_t(slot.inode), uintmax_t(ns()));
            if (sendReleaseWatch(packet, size, descriptor)) ++exported;
            else std::fprintf(stderr, "MAKO Renderer: DRI3 release export undelivered context=%ju serial=%u\n",
                              uintmax_t(context), slot.serial);
            ::close(descriptor);
        }
        std::fprintf(stderr, "MAKO Renderer: direct-dri3-release-export context=%ju count=%u\n",
                     uintmax_t(context), exported);
    }

    void close() noexcept {
        if (journal) munmap(std::exchange(journal, nullptr), journalBytes);
        if (connection) api.disconnect(std::exchange(connection, nullptr));
        if (drm < 0) return;
        for (auto& slot : sync) {
            if (slot.acquireHandle) api.destroy(drm, slot.acquireHandle);
            if (slot.releaseHandle) api.destroy(drm, slot.releaseHandle);
        }
        ::close(std::exchange(drm, -1));
        if (releaseWatch >= 0) {
            char packet[256];
            const auto size = std::snprintf(packet, sizeof(packet), "closed %d %ju %u %ju",
                getpid(), uintmax_t(releaseWatchContext), drawable, uintmax_t(ns()));
            if (!sendReleaseWatch(packet, size))
                std::fprintf(stderr, "MAKO Renderer: DRI3 release close marker undelivered context=%ju\n",
                             uintmax_t(releaseWatchContext));
            ::close(std::exchange(releaseWatch, -1));
        }
    }

    Record* find(uint32_t id) {
        return id && id <= records.size() ? &records[id-1] : nullptr;
    }

    void completed(const xcb_present_complete_notify_event_t& evt) {
        if (evt.window != drawable || evt.kind != XCB_PRESENT_COMPLETE_KIND_PIXMAP ||
            !count || inflight[0] != evt.serial) fail("Out-of-order DRI3 Complete");
        if (auto* record = find(evt.serial)) {
            record->complete = evt.ust*1000;
            record->completeObserved = ns();
            record->msc = evt.msc;
            record->mode = evt.mode;
            mirror(*record);
        }
        if (evt.mode != XCB_PRESENT_COMPLETE_MODE_FLIP) ++nonFlip;
        if (evt.ust) lastCompleteTime = Clock::time_point(std::chrono::microseconds(evt.ust));
        else if (evt.msc) lastCompleteTime = Clock::now();
        if (evt.mode == XCB_PRESENT_COMPLETE_MODE_FLIP && evt.ust) lastFlip = lastCompleteTime;
        lastCompleteMsc = evt.msc;
        for (unsigned i = 1; i < count; ++i) inflight[i-1] = inflight[i];
        --count;
    }

    void eventsReady() {
        while (auto* event = api.pollSpecial(connection, events)) {
            const auto type = reinterpret_cast<xcb_ge_generic_event_t*>(event)->event_type;
            if (type == XCB_PRESENT_COMPLETE_NOTIFY) {
                const auto body = *reinterpret_cast<xcb_present_complete_notify_event_t*>(event);
                free(event);
                completed(body);
            } else {
                free(event);
            }
        }
    }

    void releasesReady() {
        std::array<uint32_t, slots> handles{};
        std::array<uint64_t, slots> points{};
        std::array<unsigned, slots> indexes{};
        unsigned pending = 0;
        for (unsigned i = 0; i < activeSlots; ++i) {
            if (!sync[i].serial) continue;
            handles[pending] = sync[i].releaseHandle;
            indexes[pending++] = i;
        }
        if (!pending) return;
        const bool traceQueries = journal && journalRecordBytes == sizeof(ReleaseQueryRecord);
        const auto queryBegin = traceQueries ? ns() : 0;
        if (api.query(drm, handles.data(), points.data(), pending)) fail("Read DRM release points");
        const auto at = ns();
        for (unsigned i = 0; i < pending; ++i) {
            auto& slot = sync[indexes[i]];
            if (points[i] < slot.point) {
                if (traceQueries) slot.lastPendingQueryBegin = queryBegin;
                continue;
            }
            if (auto* record = find(slot.serial)) {
                if (traceQueries) {
                    // The point was still pending during the preceding negative query.
                    // The start of that query gives a conservative lower time bound.
                    const std::array bounds{slot.lastPendingQueryBegin, queryBegin};
                    std::memcpy(journalRow(slot.serial) + sizeof(Record), bounds.data(), sizeof(bounds));
                }
                record->release = at;
                mirror(*record);
            }
            slot.serial = 0;
        }
    }

public:
    Dri3(xcb_window_t window, std::span<const int> buffers, uint16_t width, uint16_t height, uint64_t context, unsigned phasePoolCount = phaseSlots) : drawable(window), activePhaseSlots(phasePoolCount), activeSlots(phasePoolCount + originalSlots) {
        if ((phasePoolCount != phaseSlots && phasePoolCount != triplePhaseSlots) || buffers.size() != activeSlots)
            fail("Unexpected DRI3 buffer pool size");
        const auto* verbose = std::getenv("ODIN3_FG_VERBOSE_LOG");
        const auto* journalEnv = std::getenv("ODIN3_DRI3_DELIVERY_DIR");
        recordDelivery = (verbose && std::strcmp(verbose, "1") == 0) || (journalEnv && journalEnv[0] == '/');
        if (const auto* hz = std::getenv("ODIN3_HTP_INDEPENDENT_OUTPUT_HZ"); hz && std::strcmp(hz, "60") == 0)
            vblankPeriod = std::chrono::nanoseconds{16'666'667};
        if (const auto* watcher = std::getenv("ODIN3_DRI3_RELEASE_SOCKET"); watcher && watcher[0] == '/')
            releaseWatchPath = watcher;
        try {
            if (recordDelivery) records.reserve(capacity);
            connection = api.connect(nullptr, nullptr);
            if (!connection || api.hasError(connection)) fail("Connect to game Xwayland");
            auto* geo = api.geometryReply(connection, api.getGeometry(connection, drawable), nullptr);
            if (!geo) fail("Read game drawable geometry");
            const bool compatible = geo->width == width && geo->height == height && geo->depth == 24;
            free(geo);
            if (!compatible) fail("DRI3 game drawable geometry does not match the depth-24 shared images");
            auto* caps = api.capabilitiesReply(connection, api.capabilities(connection, drawable), nullptr);
            if (!caps) fail("Read game Present capabilities");
            const auto capability = caps->capabilities;
            free(caps);
            if (!(capability & XCB_PRESENT_CAPABILITY_SYNCOBJ)) fail("Game drawable lacks DRM syncobj support");
            drm = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
            if (drm < 0) fail("Open active DRM render node");
            const auto event = api.newId(connection);
            events = api.special(connection, api.extension, event, &eventStamp);
            if (!events) fail("Subscribe to direct game events");
            checked(api.selectPresent(connection, event, drawable, XCB_PRESENT_EVENT_MASK_COMPLETE_NOTIFY),
                "Select game Present events");
            for (unsigned i = 0; i < activeSlots; ++i) {
                if (!releaseWatchPath.empty()) {
                    struct stat info{};
                    if (fstat(buffers[i], &info) == 0) {
                        sync[i].device = static_cast<uint64_t>(info.st_dev);
                        sync[i].inode = static_cast<uint64_t>(info.st_ino);
                    }
                }
                const int fd = fcntl(buffers[i], F_DUPFD_CLOEXEC, 0);
                if (fd < 0) fail("Duplicate game DMA-BUF");
                pixmaps[i] = api.newId(connection);
                checked(api.importPixmap(connection, pixmaps[i], drawable, 1, width, height, uint32_t{width} * 4, 0,
                    0, 0, 0, 0, 0, 0, 24, 32, 0, &fd), "Import direct game pixmap");
                sync[i].acquire = importSync(sync[i].acquireHandle);
                sync[i].release = importSync(sync[i].releaseHandle);
            }
            openJournal(context);
            std::clog << "MAKO Renderer: direct-dri3 initialized window=" << drawable
                << " capabilities=" << capability << " phase_slots=" << activePhaseSlots << " original_slots=" << originalSlots
                << " present_depth=" << credit << '\n';
        } catch (...) {
            close();
            throw;
        }
    }
    Dri3(const Dri3&) = delete;
    Dri3& operator=(const Dri3&) = delete;
    ~Dri3() { close(); }

    unsigned phaseSlotCount() const { return activePhaseSlots; }
    bool available(unsigned slot) const { return slot < activeSlots && !sync.at(slot).serial; }
    bool hasCredit() const { return count < credit; }
    uint32_t pendingCount() const { return count; }
    bool hasPendingRetirements() const {
        // The current scanout pixmap can legally stay owned until it is replaced.
        return count || std::any_of(sync.begin(), sync.begin() + activeSlots,
            [&](const Sync& slot) { return slot.serial && slot.serial != serial; });
    }
    uint64_t nextTargetMsc() const {
        if (!lastCompleteMsc) return 0;
        if (!count && lastCompleteTime != Clock::time_point{} &&
                Clock::now() - lastCompleteTime > std::chrono::milliseconds(120)) {
            return 0;
        }
        return std::max(lastCompleteMsc, lastTargetMsc) + 1;
    }

    std::chrono::nanoseconds period() const { return vblankPeriod; }
    bool anchored() const {
        return lastCompleteTime != Clock::time_point{} && lastCompleteMsc &&
               Clock::now() - lastCompleteTime < std::chrono::seconds(2);
    }
    Clock::time_point anchorTime() const { return lastCompleteTime; }
    Clock::time_point lastFlipTime() const { return lastFlip; }
    uint64_t anchorMsc() const { return lastCompleteMsc; }

    uint64_t flipAnchorMsc() const {
        return lastFlip != Clock::time_point{} ? lastCompleteMsc : 0;
    }

    std::optional<Clock::time_point> predictTarget(uint64_t target, std::chrono::nanoseconds period,
            Clock::time_point now) const {
        if (period.count() <= 0 || lastFlip == Clock::time_point{} || lastFlip > now ||
                now - lastFlip > 8 * period || target <= lastCompleteMsc || target - lastCompleteMsc > credit + 2)
            return std::nullopt;
        return lastFlip + static_cast<int64_t>(target - lastCompleteMsc) * period;
    }

    void poll() {
        eventsReady();
        while (auto* event = api.pollEvent(connection)) {
            const bool error = event->response_type == 0;
            free(event);
            if (error) fail("Direct game X11 request failed");
        }
        eventsReady();
        if (api.hasError(connection)) fail("Lost direct game X11 connection");
        releasesReady();
    }
    void waitUntil(Clock::time_point deadline) {
        const auto now = Clock::now();
        if (now >= deadline) return;
        const auto duration = std::min(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline-now),
                                      std::chrono::nanoseconds{1200000});
        const timespec timeout{0, duration.count()};
        pollfd fd{api.getFd(connection), POLLIN, 0};
        if (ppoll(&fd, 1, &timeout, nullptr) < 0 && errno != EINTR) fail("Poll direct game events");
    }
    uint64_t submit(unsigned slot, uint64_t frame, Role role, uint64_t phaseCutoff = 0, uint64_t creditCutoff = 0, unsigned phaseNumber = 1,
            uint64_t explicitTarget = 0, bool asyncMode = false) {
        if (!available(slot) || !hasCredit() || serial == UINT32_MAX - 1) fail("DRI3 slot submitted without credit");
        auto& entry = sync.at(slot);
        uint64_t point = ++entry.point;
        if (api.signal(drm, &entry.acquireHandle, &point, 1)) fail("Publish direct game acquire point");
        const uint64_t target = asyncMode ? 0ull : (explicitTarget ? std::max<uint64_t>(explicitTarget, lastTargetMsc + 1) : nextTargetMsc());
        lastTargetMsc = target;
        ++serial;
        const auto time = recordDelivery ? ns() : 0;
        const unsigned phase = role == Role::Generated ? phaseNumber : 0;
        if (recordDelivery && records.size() < capacity) {
            records.push_back(Record{frame, time, target, 0, 0, 0, 0, serial, slot, phase});
            records.back().phaseCutoff = phaseCutoff;
            records.back().creditCutoff = creditCutoff;
        }
        entry.serial = serial;
        entry.lastPendingQueryBegin = 0;
        if (!releaseWatchPath.empty()) {
            entry.watchFlushed = false;
            entry.watchFrame = frame;
            entry.watchPhase = phase;
        }
        inflight[count++] = serial;
        peak = std::max(peak, count);
        const uint32_t presOpts = asyncMode ? uint32_t(XCB_PRESENT_OPTION_ASYNC) : uint32_t(XCB_PRESENT_OPTION_SUBOPTIMAL);
        api.presentSynced(connection, drawable, pixmaps[slot], serial, 0, 0, 0, 0, 0, entry.acquire,
            entry.release, point, point, presOpts, target, 0, 0, 0, nullptr);
        if (api.flush(connection) <= 0) fail("Flush direct game Present");
        if (!releaseWatchPath.empty()) entry.watchFlushed = true;
        if (journal) {
            if (auto* record = find(serial)) {
                mirror(*record);
                std::atomic_ref(journal->count).store(serial, std::memory_order_release);
            } else {
                ++journal->dropped;
            }
        }
        return target;
    }
    void finish(uint64_t context) noexcept {
        try {
            const auto limit = Clock::now() + std::chrono::milliseconds(40);
            do {
                poll();
                if (!hasPendingRetirements()) break;
                waitUntil(limit);
            } while (Clock::now() < limit);
        } catch (...) {
            // Window destruction can end the event stream before all
            // release points have been observed locally.
        }
        exportPendingReleases(context);
        if (const auto* verbose = std::getenv("ODIN3_FG_VERBOSE_LOG"); verbose && std::strcmp(verbose, "1") == 0) {
            for (const auto& row : records)
                std::clog << "MAKO Renderer: direct-dri3-delivery context=" << context << " frame=" << row.frame
                    << " phase=" << row.phase << " serial=" << row.serial << " slot=" << row.slot << " submit_ns=" << row.submit
                    << " target_msc=" << row.target << " complete_ns=" << row.complete << " observed_ns=" << row.completeObserved
                    << " release_ns=" << row.release << " msc=" << row.msc << " mode=" << row.mode
                    << " phase_cutoff_ns=" << row.phaseCutoff << " credit_cutoff_ns=" << row.creditCutoff << '\n';
        }
        for (unsigned i = 0; i < activeSlots; ++i) {
            if (sync[i].serial)
                std::clog << "MAKO Renderer: direct-dri3-retirement context=" << context
                    << " active_serial=" << sync[i].serial << " slot=" << i << '\n';
        }
        std::clog << "MAKO Renderer: direct-dri3-summary context=" << context << " submitted=" << serial
            << " logged=" << records.size() << " pending=" << count << " peak=" << peak << " non_flip=" << nonFlip << '\n';
    }
};
}
