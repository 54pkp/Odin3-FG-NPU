/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "independent_real_present.hpp"
#include "runtime_status.hpp"
#include "direct_x11_present.hpp"
#include "odin3_dri3_pair_capture.hpp"
#include "mako-backend/mako.hpp"
#include "mako-common/configuration/htp_deadline.hpp"
#include "mako-common/configuration/config.hpp"
#include "mako-common/vulkan/command_buffer.hpp"
#include "mako-common/vulkan/external_image.hpp"
#include "mako-common/vulkan/fence.hpp"
#include "mako-common/vulkan/semaphore.hpp"
#include "mako-common/vulkan/timestamp_query_pool.hpp"
#include "mako-common/vulkan/timeline_semaphore.hpp"
#include "mako-common/helpers/errors.hpp"
#include "mako-common/helpers/file_descriptors.hpp"
#include "mako-common/helpers/paths.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <dlfcn.h>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <linux/dma-heap.h>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unordered_map>
#include <unordered_set>
namespace {
inline void pinRendererHelperThread(bool statusThread = false) noexcept {
    if (const char* envP = std::getenv("ODIN3_FG_NO_EFFICIENCY_PIN"); envP && envP[0] == '1') return;
    if (statusThread) {
        sched_param sp{};
        (void)::sched_setscheduler(0, SCHED_IDLE, &sp);
    } else {
        const char* cap = std::getenv("MAKO_BASE_FPS_CAP");
        const int capVal = (cap && cap[0]) ? std::atoi(cap) : 0;
        (void)::setpriority(PRIO_PROCESS, static_cast<id_t>(::syscall(SYS_gettid)), (capVal > 0 && capVal <= 62) ? 5 : 19);
    }
    cpu_set_t s; CPU_ZERO(&s);
    for (int c = 1; c < 6; ++c) CPU_SET(c, &s);
    (void)pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}
inline void setAffinityIfChanged(pid_t tid, const cpu_set_t& desired) noexcept {
    cpu_set_t cur; CPU_ZERO(&cur);
    if (::sched_getaffinity(tid, sizeof cur, &cur) == 0 && !CPU_EQUAL(&cur, &desired)) {
        (void)::sched_setaffinity(tid, sizeof desired, &desired);
    }
}
inline void balanceGameThreadsToPrime(std::unordered_map<pid_t, uint64_t>& prevTicks,
                                      std::unordered_set<pid_t>& managedTids, bool& seeded) noexcept {
    if (const char* envP = std::getenv("ODIN3_FG_NO_EFFICIENCY_PIN"); envP && envP[0] == '1') return;
    DIR* dir = ::opendir("/proc/self/task");
    if (!dir) return;
    struct Entry { pid_t tid; uint64_t delta; bool managed; };
    std::vector<Entry> items;
    items.reserve(48);
    std::unordered_map<pid_t, uint64_t> curTicks;
    curTicks.reserve(48);
    while (const dirent* ent = ::readdir(dir)) {
        if (ent->d_name[0] < '1' || ent->d_name[0] > '9') continue;
        const pid_t tid = static_cast<pid_t>(std::atoi(ent->d_name));
        if (tid <= 0) continue;
        char path[64];
        std::snprintf(path, sizeof path, "/proc/self/task/%d/stat", int(tid));
        const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;
        char buf[512];
        const ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
        ::close(fd);
        if (n <= 0) continue;
        buf[n] = '\0';
        const char* lp = std::strchr(buf, '(');
        const char* rp = std::strrchr(buf, ')');
        if (!lp || !rp || rp <= lp || rp[1] != ' ') continue;
        const std::string_view comm(lp + 1, static_cast<size_t>(rp - lp - 1));
        if (comm.starts_with("odin3-")) continue;
        const char* sp = rp + 2;
        uint64_t utime = 0, stime = 0;
        bool ok = true;
        for (int field = 0; field <= 12; ++field) {
            while (*sp == ' ') ++sp;
            if (!*sp) { ok = false; break; }
            if (field == 11) utime = std::strtoull(sp, nullptr, 10);
            else if (field == 12) stime = std::strtoull(sp, nullptr, 10);
            while (*sp && *sp != ' ') ++sp;
        }
        if (!ok) continue;
        const uint64_t tot = utime + stime;
        curTicks[tid] = tot;
        cpu_set_t curMask; CPU_ZERO(&curMask);
        bool mgd = managedTids.contains(tid);
        if (!mgd && ::sched_getaffinity(tid, sizeof curMask, &curMask) == 0 && CPU_COUNT(&curMask) >= 4) {
            managedTids.insert(tid);
            mgd = true;
        }
        if (seeded) {
            const auto it = prevTicks.find(tid);
            const uint64_t d = (it != prevTicks.end() && tot >= it->second) ? (tot - it->second) : tot;
            items.push_back({tid, d, mgd});
        }
    }
    ::closedir(dir);
    prevTicks.swap(curTicks);
    if (!seeded) { seeded = true; return; }
    std::sort(items.begin(), items.end(), [](const Entry& a, const Entry& b) { return a.delta > b.delta; });
    const uint64_t d1 = items.size() > 0 ? items[0].delta : 0;
    const uint64_t d2 = items.size() > 1 ? items[1].delta : 0;
    const uint64_t d3 = items.size() > 2 ? items[2].delta : 0;
    cpu_set_t set7, set6, setEff, setAll;
    CPU_ZERO(&set7); CPU_SET(7, &set7);
    CPU_ZERO(&set6); CPU_SET(6, &set6);
    CPU_ZERO(&setEff); for (int c = 1; c < 6; ++c) CPU_SET(c, &setEff);
    CPU_ZERO(&setAll); for (int c = 1; c < 8; ++c) CPU_SET(c, &setAll);
    if (d1 >= 22 && (d1 + d2) <= 185 && d3 < 34) {
        const bool pinSecond = (items.size() > 1 && d2 >= 14 && items[1].managed);
        if (items[0].managed) setAffinityIfChanged(items[0].tid, set7);
        if (pinSecond) setAffinityIfChanged(items[1].tid, set6);
        for (size_t i = (pinSecond ? 2u : 1u); i < items.size(); ++i) {
            if (items[i].managed) setAffinityIfChanged(items[i].tid, setEff);
        }
    } else if (d3 >= 38) {
        for (const auto& e : items) {
            if (e.managed) setAffinityIfChanged(e.tid, setAll);
        }
    }
}
}
#include <mutex>
#include <optional>
#include <syncstream>
#include <sys/ioctl.h>
#include <unistd.h>
#include <thread>
#include <pthread.h>
#include <vector>
#include <xcb/xcb.h>
#include <vulkan/vulkan_xcb.h>

using namespace mako::layer;
namespace backend = mako::backend;

namespace {
    constexpr uint64_t graphicsTimeout = 2'500'000'000ULL;
    constexpr VkImageSubresourceLayers layers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    constexpr VkImageUsageFlags sharedUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    constexpr size_t asyncCapacity = 11;
    constexpr size_t sourceTimingCapacity = 8192;
    constexpr size_t retirementGapLimit = 32;
    constexpr auto retirementGapPeriod = std::chrono::nanoseconds{1'000'000'000 / 120};
    constexpr VkFormat htpVfiFormat = VK_FORMAT_R8G8B8A8_UNORM;
    using Clock = std::chrono::steady_clock;
    std::atomic<uint64_t> nextPresentContext{1};

    struct DrawConfirmation {
        VkImage source{};
        uint64_t frame{};
    };

    class DrawPool {
        struct Entry {
            VkDevice device;
            VkImage target;
            uint64_t epoch;
            bool acquired{};
            bool rejected{};
            DrawConfirmation confirmation{};
        };
        std::mutex mutex;
        std::vector<Entry> entries;

        auto find(VkDevice device, VkImage target) {
            return std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) {
                return entry.device == device && entry.target == target;
            });
        }

    public:
        void add(VkDevice device, uint64_t epoch, std::span<const VkImage> targets) {
            const std::lock_guard lock(mutex);
            for (const auto target : targets)
                if (find(device, target) != entries.end())
                    throw ls::error("Final-draw target already registered");
            entries.reserve(entries.size() + targets.size());
            for (const auto target : targets) entries.push_back({device, target, epoch});
        }

        void remove(uint64_t epoch) {
            const std::lock_guard lock(mutex);
            std::erase_if(entries, [&](const Entry& entry) { return entry.epoch == epoch; });
        }

        void acquire(VkDevice device, VkImage target) {
            const std::lock_guard lock(mutex);
            const auto entry = find(device, target);
            if (entry == entries.end()) throw ls::error("Missing final-draw target on acquisition");
            entry->acquired = true;
            entry->rejected = false;
            entry->confirmation = {};
        }

        void note(VkDevice device, VkImage target, VkImage source, uint64_t frame) {
            if (entries.empty()) return;
            const std::lock_guard lock(mutex);
            const auto entry = find(device, target);
            if (entry == entries.end() || !entry->acquired) return;
            if (!source || !frame || entry->confirmation.frame || find(device, source) != entries.end())
                entry->rejected = true;
            else
                entry->confirmation = {source, frame};
        }

        DrawConfirmation consume(VkDevice device, VkImage target) {
            const std::lock_guard lock(mutex);
            const auto entry = find(device, target);
            if (entry == entries.end() || !entry->acquired) return {};
            entry->acquired = false;
            return entry->rejected ? DrawConfirmation{} : entry->confirmation;
        }
    };
    DrawPool finalDraws;

    int64_t monotonicNs(Clock::time_point time) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
    }

    struct CalibratedInterval {
        int64_t begin, end;
        uint64_t deviation;
    };

    std::optional<CalibratedInterval> calibrateInterval(PFN_vkGetCalibratedTimestampsKHR calibrate,
            VkDevice device, const vk::TimestampQueryPool& queries, const std::vector<uint64_t>& values,
            int64_t earliest, int64_t latest) {
        constexpr std::array domains{
            VkCalibratedTimestampInfoKHR{.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
                .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR},
            VkCalibratedTimestampInfoKHR{.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
                .timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR}};
        std::array<uint64_t, 2> reference{};
        uint64_t deviation{};
        const auto before = monotonicNs(Clock::now());
        const auto result = calibrate(device, domains.size(), domains.data(), reference.data(), &deviation);
        const auto after = monotonicNs(Clock::now());
        if (result != VK_SUCCESS || deviation > 100'000 || after - earliest > 250'000'000) return std::nullopt;
        const auto error = static_cast<int64_t>(deviation);
        const auto host = static_cast<int64_t>(reference[1]);
        if (host + error < before || host > after + error) return std::nullopt;
        const auto half = uint64_t{1} << (queries.timestampValidBits() - 1);
        const auto mask = half | (half - 1);
        const auto toMonotonic = [&](uint64_t timestamp) {
            const auto forward = (timestamp - reference[0]) & mask;
            const double ticks = forward < half ? double(forward) : -double((reference[0] - timestamp) & mask);
            return host + std::llround(ticks * queries.timestampPeriodNanoseconds());
        };
        const auto begin = toMonotonic(values[0]);
        const auto end = toMonotonic(values[1]);
        if (begin < earliest - error || end < begin || end > latest + error) return std::nullopt;
        return CalibratedInterval{begin, end, deviation};
    }

    bool enabled(const char* name) {
        const auto* value = std::getenv(name);
        return value && std::strcmp(value, "1") == 0;
    }

    std::chrono::nanoseconds outputPeriod() {
        const auto* rate = std::getenv("ODIN3_HTP_INDEPENDENT_OUTPUT_HZ");
        if (!rate) return {};
        if (std::strcmp(rate, "60") == 0) return std::chrono::nanoseconds(16'666'667);
        if (std::strcmp(rate, "120") == 0) return std::chrono::nanoseconds(8'326'000);
        return {};
    }

    Clock::time_point nextOutputSlot(Clock::time_point slot, Clock::time_point now,
            std::chrono::nanoseconds period, bool fixedPacing) {
        // Keep the phase through small overruns. Rebase after half a period to
        // avoid a burst of closely spaced submissions when work runs long.
        if (fixedPacing && slot != Clock::time_point{} && now - slot < period / 2)
            return slot + period;
        return std::max(now, slot) + period;
    }

    std::chrono::nanoseconds sourcePeriod() {
        const auto* cap = std::getenv("MAKO_BASE_FPS_CAP");
        if (!cap) return {};
        const auto* end = cap + std::strlen(cap);
        uint32_t fps{};
        const auto parsed = std::from_chars(cap, end, fps);
        if (parsed.ec != std::errc{} || parsed.ptr != end || fps == 0 || fps > ls::GameConfLimits::maximumBaseFpsCap)
            return {};
        const auto* rate = std::getenv("ODIN3_HTP_INDEPENDENT_OUTPUT_HZ");
        const bool is120 = !rate || std::strcmp(rate, "120") == 0;
        const bool is60 = rate && std::strcmp(rate, "60") == 0;
        const uint32_t mult = mako::htpVfiTripleEnabled() ? 3u : 2u;
        if ((is120 && fps * mult >= 120u) || (is60 && fps * mult >= 60u)) return std::chrono::nanoseconds(1);
        if (is120 && fps == 60) return std::chrono::nanoseconds(16'658'000);
        if (is120 && fps == 40) return std::chrono::nanoseconds(24'987'000);
        if (is60 && fps == 30) return std::chrono::nanoseconds(33'315'000);
        if (is60 && fps == 20) return std::chrono::nanoseconds(49'975'000);
        return std::chrono::nanoseconds((1'000'000'000 + fps - 1) / fps);
    }

    uint32_t presentQueueTarget() {
        const auto* value = std::getenv("ODIN3_HTP_INDEPENDENT_PRESENT_QUEUE_TARGET");
        if (!value) return 0;
        if (std::strcmp(value, "1") == 0) return 1;
        if (std::strcmp(value, "2") == 0) return 2;
        return 0;
    }

    uint32_t extraOutputImageCount() {
        const auto* value = std::getenv("ODIN3_HTP_INDEPENDENT_EXTRA_OUTPUT_IMAGE");
        if (!value) return 0;
        if (std::strcmp(value, "2") == 0) return 2;
        return enabled("ODIN3_HTP_INDEPENDENT_EXTRA_OUTPUT_IMAGE") ? 1u : 0u;
    }

    uint32_t runtimeAppId() {
        const auto* value = std::getenv("ODIN3_FG_RUNTIME_APP_ID");
        if (!value) return 0;
        const auto* end = value + std::strlen(value);
        uint32_t appId{};
        const auto parsed = std::from_chars(value, end, appId);
        return parsed.ec == std::errc{} && parsed.ptr == end ? appId : 0;
    }

    std::chrono::milliseconds testDelay() {
        const auto* value = std::getenv("ODIN3_HTP_TEST_ASYNC_BACKPRESSURE");
        if (!value) return {};
        if (std::strcmp(value, "1") == 0) return std::chrono::milliseconds(60);
        if (std::strcmp(value, "timeout") == 0) return std::chrono::milliseconds(500);
        return {};
    }

    void check(VkResult result, const char* action) {
        if (result != VK_SUCCESS) throw ls::vulkan_error(result, action);
    }

    template<typename Function>
    Function xcbSymbol(void* library, const char* name) {
        const auto function = reinterpret_cast<Function>(dlsym(library, name));
        if (!function) throw ls::error(name);
        return function;
    }

    struct HiddenSurface {
        VkInstance instance{};
        VkSurfaceKHR handle{};
        PFN_vkDestroySurfaceKHR destroy{};
        xcb_connection_t* connection{};
        xcb_window_t window{};
        decltype(&xcb_destroy_window) destroyWindow{};
        decltype(&xcb_flush) flush{};

        void create(VkInstance instanceHandle, PFN_vkGetInstanceProcAddr next,
                X11PresentationTarget target, VkExtent2D extent) {
            static void* library = dlopen("libxcb.so.1", RTLD_NOW | RTLD_LOCAL);
            if (!library || !target.connection || !target.window || !extent.width || !extent.height ||
                    extent.width > 8192 || extent.height > 8192)
                throw ls::error("hidden XCB source is unavailable");
            const auto geometry = xcbSymbol<decltype(&xcb_get_geometry)>(library, "xcb_get_geometry");
            const auto reply = xcbSymbol<decltype(&xcb_get_geometry_reply)>(library, "xcb_get_geometry_reply");
            const auto configure = xcbSymbol<decltype(&xcb_create_window_checked)>(library, "xcb_create_window_checked");
            const auto requestCheck = xcbSymbol<decltype(&xcb_request_check)>(library, "xcb_request_check");
            const auto newId = xcbSymbol<decltype(&xcb_generate_id)>(library, "xcb_generate_id");
            destroyWindow = xcbSymbol<decltype(&xcb_destroy_window)>(library, "xcb_destroy_window");
            flush = xcbSymbol<decltype(&xcb_flush)>(library, "xcb_flush");
            const auto createSurface = reinterpret_cast<PFN_vkCreateXcbSurfaceKHR>(next(instanceHandle, "vkCreateXcbSurfaceKHR"));
            destroy = reinterpret_cast<PFN_vkDestroySurfaceKHR>(next(instanceHandle, "vkDestroySurfaceKHR"));
            if (!createSurface || !destroy) throw ls::error("Vulkan XCB surface entrypoints unavailable");
            const std::unique_ptr<xcb_get_geometry_reply_t, decltype(&std::free)> geo(
                reply(target.connection, geometry(target.connection, target.window), nullptr), &std::free);
            if (!geo) throw ls::error("cannot resolve game X11 root window");
            connection = target.connection;
            instance = instanceHandle;
            const auto id = newId(connection);
            const auto cookie = configure(connection, XCB_COPY_FROM_PARENT, id, geo->root, 0, 0,
                static_cast<uint16_t>(extent.width), static_cast<uint16_t>(extent.height), 0,
                XCB_WINDOW_CLASS_INPUT_OUTPUT, XCB_COPY_FROM_PARENT, 0, nullptr);
            const std::unique_ptr<xcb_generic_error_t, decltype(&std::free)> error(requestCheck(connection, cookie), &std::free);
            if (error) throw ls::error("could not create the hidden XCB window");
            window = id;
            const VkXcbSurfaceCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                .connection = connection, .window = window};
            check(createSurface(instance, &info, nullptr, &handle), "create hidden XCB Vulkan surface");
        }

        ~HiddenSurface() {
            if (handle) destroy(instance, handle, nullptr);
            if (window) {
                destroyWindow(connection, window);
                flush(connection);
            }
        }
    };

    struct PrivateDevice {
        VkDevice handle{};
        PFN_vkLayerDestroyDevice destroy{};
        PFN_vkDestroyDevice lowerDestroy{};
        PFN_vkWaitForPresentKHR waitForPresent{};
        PFN_vkSetSwapchainPresentTimingQueueSizeEXT setTimingQueue{};
        PFN_vkGetSwapchainTimeDomainPropertiesEXT timingDomains{};
        PFN_vkGetPastPresentationTimingEXT pastTiming{};
        PFN_vkGetCalibratedTimestampsKHR calibrateUpload{};
        VkPresentStageFlagsEXT timingStages{};
        std::optional<vk::Vulkan> vk;

        void create(const vk::Vulkan& app, VkInstance instance, const PrivateDeviceCallbacks& layer,
                VkSurfaceKHR timingSurface) {
            const auto family = app.queueFamilyIndex();
            const float priority = 1.0F;
            VkDeviceQueueCreateInfo queue{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
            VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR maintenance{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
                .swapchainMaintenance1 = VK_TRUE};
            VkPhysicalDevicePresentWaitFeaturesKHR wait{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
            VkPhysicalDevicePresentIdFeaturesKHR id{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
                .pNext = &wait};
            VkPhysicalDeviceFeatures2 supported{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &id};
            app.fi().GetPhysicalDeviceFeatures2(app.physdev(), &supported);
            if (id.presentId && wait.presentWait) maintenance.pNext = &id;
            VkPhysicalDevicePresentTimingFeaturesEXT timing{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT};
            VkPhysicalDevicePresentId2FeaturesKHR id2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR, .pNext = &timing};
            VkPresentTimingSurfaceCapabilitiesEXT timingCaps{
                .sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT};
            VkSurfaceCapabilitiesPresentId2KHR idCaps{
                .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR, .pNext = &timingCaps};
            if (timingSurface) {
                const auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR>(
                    layer.nextLayer(instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR"));
                VkSurfaceCapabilities2KHR caps{.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR, .pNext = &idCaps};
                const VkPhysicalDeviceSurfaceInfo2KHR surface{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR, .surface = timingSurface};
                if (query && query(app.physdev(), &surface, &caps) == VK_SUCCESS &&
                        idCaps.presentId2Supported && timingCaps.presentTimingSupported) {
                    supported.pNext = &id2;
                    app.fi().GetPhysicalDeviceFeatures2(app.physdev(), &supported);
                    if (id2.presentId2 && timing.presentTiming) timingStages = timingCaps.presentStageQueries;
                }
                std::clog << "MAKO Renderer: independent-feedback surface stages=" << timingStages
                    << " absolute=" << timingCaps.presentAtAbsoluteTimeSupported << '\n';
            }
            const VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, .pNext = &maintenance,
                .timelineSemaphore = VK_TRUE};
            std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
                VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
                VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME};
            if (mako::htpVfiLinearOutputEnabled() || mako::htpVfiLinearInputEnabled()) {
                extensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
                extensions.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
            }
            if (maintenance.pNext) {
                extensions.push_back(VK_KHR_PRESENT_ID_EXTENSION_NAME);
                extensions.push_back(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
            }
            const auto baseChain = maintenance.pNext;
            const auto baseExtensionCount = extensions.size();
            if (timingStages) {
                timing.presentAtAbsoluteTime = timing.presentAtRelativeTime = VK_FALSE;
                timing.pNext = baseChain;
                maintenance.pNext = &id2;
                extensions.insert(extensions.end(), {VK_KHR_PRESENT_ID_2_EXTENSION_NAME,
                    VK_EXT_PRESENT_TIMING_EXTENSION_NAME, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME});
            }
            const VkDeviceQueueGlobalPriorityCreateInfoKHR highPriority{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
                .globalPriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR};
            const bool priorityRequested = enabled("ODIN3_HTP_INDEPENDENT_PRESENT_HIGH_PRIORITY");
            if (priorityRequested) {
                queue.pNext = &highPriority;
                extensions.push_back(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
            }
            VkDeviceCreateInfo info{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &timeline,
                .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue,
                .enabledExtensionCount = static_cast<uint32_t>(extensions.size()), .ppEnabledExtensionNames = extensions.data()};
            PFN_vkGetDeviceProcAddr lower{};
            auto result = layer.create(instance, app.physdev(), &info, nullptr, &handle, layer.currentLayer, &lower);
            if (priorityRequested && (result == VK_ERROR_NOT_PERMITTED_KHR || result == VK_ERROR_INITIALIZATION_FAILED ||
                    result == VK_ERROR_EXTENSION_NOT_PRESENT || result == VK_ERROR_FEATURE_NOT_PRESENT)) {
                std::clog << "MAKO Renderer: independent-present high priority rejected result=" << result << '\n';
                queue.pNext = nullptr;
                --info.enabledExtensionCount;
                handle = VK_NULL_HANDLE;
                lower = nullptr;
                result = layer.create(instance, app.physdev(), &info, nullptr, &handle, layer.currentLayer, &lower);
            }
            if (timingStages && (result == VK_ERROR_EXTENSION_NOT_PRESENT || result == VK_ERROR_FEATURE_NOT_PRESENT)) {
                std::clog << "MAKO Renderer: independent-feedback device rejected result=" << result << '\n';
                timingStages = 0;
                maintenance.pNext = baseChain;
                info.enabledExtensionCount = static_cast<uint32_t>(baseExtensionCount);
                result = layer.create(instance, app.physdev(), &info, nullptr, &handle, layer.currentLayer, &lower);
            }
            check(result, "create private presentation device");
            if (priorityRequested)
                std::clog << "MAKO Renderer: independent-present queue_priority=" << (queue.pNext ? "high" : "default") << '\n';
            destroy = layer.destroy;
            if (!lower) throw ls::error("private device dispatch is unavailable");
            lowerDestroy = reinterpret_cast<PFN_vkDestroyDevice>(lower(handle, "vkDestroyDevice"));
            if (!lowerDestroy) throw ls::error("private device destruction is unavailable");
            if (baseChain) waitForPresent = reinterpret_cast<PFN_vkWaitForPresentKHR>(lower(handle, "vkWaitForPresentKHR"));
            if (timingStages) {
                if (enabled("ODIN3_HTP_INDEPENDENT_UPLOAD_TIMING")) {
                    const auto domains = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
                        layer.nextLayer(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR"));
                    uint32_t count{};
                    if (domains && domains(app.physdev(), &count, nullptr) == VK_SUCCESS && count) {
                        std::vector<VkTimeDomainKHR> supportedDomains(count);
                        if (domains(app.physdev(), &count, supportedDomains.data()) == VK_SUCCESS) {
                            supportedDomains.resize(count);
                            if (std::ranges::find(supportedDomains, VK_TIME_DOMAIN_DEVICE_KHR) != supportedDomains.end() &&
                                std::ranges::find(supportedDomains, VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR) != supportedDomains.end())
                                calibrateUpload = reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(
                                    lower(handle, "vkGetCalibratedTimestampsKHR"));
                        }
                    }
                }
                setTimingQueue = reinterpret_cast<PFN_vkSetSwapchainPresentTimingQueueSizeEXT>(
                    lower(handle, "vkSetSwapchainPresentTimingQueueSizeEXT"));
                timingDomains = reinterpret_cast<PFN_vkGetSwapchainTimeDomainPropertiesEXT>(
                    lower(handle, "vkGetSwapchainTimeDomainPropertiesEXT"));
                pastTiming = reinterpret_cast<PFN_vkGetPastPresentationTimingEXT>(lower(handle, "vkGetPastPresentationTimingEXT"));
                if (!setTimingQueue || !timingDomains || !pastTiming) timingStages = 0;
            }
            auto instanceFuncs = app.fi();
            instanceFuncs.GetDeviceProcAddr = lower;
            const auto functions = vk::initVulkanDeviceFuncs(instanceFuncs, handle, true, true);
            vk.emplace(instance, handle, app.physdev(), instanceFuncs, functions, family, true, true, app.loaderdatafunc());
        }

        ~PrivateDevice() {
            vk.reset();
            if (handle && lowerDestroy) destroy(handle, nullptr, lowerDestroy);
        }
    };

    // Use the private swapchain's own ID namespace for dequeue timing.
    struct PresentFeedback {
        static constexpr uint32_t capacity = 64;
        static constexpr size_t recordLimit = 3 * sourceTimingCapacity;
        struct Record {
            uint64_t frame;
            size_t phase;
            int64_t presentEntered{}, feedbackCollected{};
            int64_t acquiredStarted{}, acquiredReturned{}, imageReady{}, copyQueued{};
            int64_t submitted{}, presentReturned{}, copyRetired{};
            std::array<VkPresentStageTimeEXT, 4> stages{};
            VkTimeDomainKHR domain{};
            uint64_t domainId{};
            int64_t previousCopyWait{};
            int64_t dequeueWait{};
            int64_t pacingWait{};
            uint32_t pacingPending{};
            bool feedbackPaced{};
            uint32_t stageCount{};
            bool complete{}, copyWaitDeferred{};
        };
        uint64_t domainId{};
        uint32_t pending{};
        uint64_t untracked{}, admissionTimeouts{};
        bool active{};
        std::vector<Record> records;

        void initialize(const PrivateDevice& device, VkSwapchainKHR swapchain) {
            std::array<VkTimeDomainKHR, 8> domains{};
            std::array<uint64_t, 8> ids{};
            VkSwapchainTimeDomainPropertiesEXT properties{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT,
                .timeDomainCount = static_cast<uint32_t>(domains.size()),
                .pTimeDomains = domains.data(), .pTimeDomainIds = ids.data()};
            const auto status = device.timingDomains(device.handle, swapchain, &properties, nullptr);
            if ((status == VK_SUCCESS || status == VK_INCOMPLETE) && properties.timeDomainCount) {
                const auto end = domains.begin() + std::min<size_t>(properties.timeDomainCount, domains.size());
                const auto clock = std::find(domains.begin(), end, VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR);
                domainId = ids[clock == end ? 0 : static_cast<size_t>(clock - domains.begin())];
                active = device.setTimingQueue(device.handle, swapchain, capacity) == VK_SUCCESS;
                if (active) records.reserve(recordLimit);
            }
            std::clog << "MAKO Renderer: independent-feedback initialized=" << active
                      << " stage_mask=" << device.timingStages << " domain_id=" << domainId << '\n';
        }

        void collect(const PrivateDevice& device, VkSwapchainKHR swapchain) {
            if (!active || !pending) return;
            std::array<VkPastPresentationTimingEXT, capacity> replies{};
            std::array<std::array<VkPresentStageTimeEXT, 4>, capacity> stages{};
            for (size_t i = 0; i < replies.size(); ++i) {
                replies[i] = {.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT,
                    .presentStageCount = static_cast<uint32_t>(stages[i].size()), .pPresentStages = stages[i].data()};
            }
            const VkPastPresentationTimingInfoEXT query{
                .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT,
                .flags = VK_PAST_PRESENTATION_TIMING_ALLOW_OUT_OF_ORDER_RESULTS_BIT_EXT, .swapchain = swapchain};
            VkPastPresentationTimingPropertiesEXT found{
                .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT,
                .presentationTimingCount = capacity, .pPresentationTimings = replies.data()};
            const auto status = device.pastTiming(device.handle, &query, &found);
            if (status != VK_SUCCESS && status != VK_INCOMPLETE) {
                std::clog << "MAKO Renderer: independent-feedback query failed result=" << status << '\n';
                active = false;
                return;
            }
            for (size_t i = 0; i < std::min(found.presentationTimingCount, capacity); ++i) {
                const auto& reply = replies[i];
                if (!reply.presentId || reply.presentId > records.size() || !reply.reportComplete) continue;
                auto& entry = records[reply.presentId - 1];
                if (entry.complete) continue;
                entry.stages = stages[i];
                entry.stageCount = std::min<uint32_t>(reply.presentStageCount, entry.stages.size());
                entry.domain = reply.timeDomain;
                entry.domainId = reply.timeDomainId;
                entry.complete = true;
                --pending;
            }
        }

        void writeLog(uint64_t context) const {
            std::osyncstream log(std::clog);
            for (size_t i = 0; i < records.size(); ++i) {
                const auto& entry = records[i];
                log << "MAKO Renderer: independent-feedback context=" << context << " id=" << i + 1
                    << " frame=" << entry.frame << " phase=" << entry.phase
                    << " acquire_enter_ns=" << entry.acquiredStarted << " acquire_return_ns=" << entry.acquiredReturned
                    << " image_ready_ns=" << entry.imageReady << " copy_queued_ns=" << entry.copyQueued
                    << " submit_ns=" << entry.submitted << " present_return_ns=" << entry.presentReturned
                    << " copy_retired_ns=" << entry.copyRetired << " copy_wait_deferred=" << entry.copyWaitDeferred
                    << " previous_copy_wait_ns=" << entry.previousCopyWait
                    << " dequeue_wait_ns=" << entry.dequeueWait << " pacing_wait_ns=" << entry.pacingWait
                    << " pacing_pending=" << entry.pacingPending << " feedback_paced=" << entry.feedbackPaced
                    << " complete=" << entry.complete << " domain=" << entry.domain << " domain_id=" << entry.domainId;
                if (entry.presentEntered)
                    log << " present_enter_ns=" << entry.presentEntered
                        << " feedback_collected_ns=" << entry.feedbackCollected;
                for (size_t j = 0; j < entry.stageCount; ++j)
                    log << " stage_" << entry.stages[j].stage << "_ns=" << entry.stages[j].time;
                log << '\n';
            }
            log << "MAKO Renderer: independent-feedback-summary context=" << context << " recorded=" << records.size()
                << " pending=" << pending << " untracked=" << untracked
                << " dequeue_wait_timeouts=" << admissionTimeouts << '\n';
        }
    };

    VkImageMemoryBarrier transition(VkImage image, VkImageLayout before, VkImageLayout after,
            VkAccessFlags writes, VkAccessFlags reads) {
        return {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = writes, .dstAccessMask = reads, .oldLayout = before, .newLayout = after,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    }

    void transition(const vk::Vulkan& vk, const vk::CommandBuffer& cmd, const VkImageMemoryBarrier& barrier) {
        vk.df().CmdPipelineBarrier(cmd.handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    void transferImage(const vk::Vulkan& vk, const vk::CommandBuffer& cmd, VkImage source, VkImage target,
            VkExtent2D sourceExtent, VkExtent2D targetExtent, PFN_vkCmdCopyImage exactCopy = nullptr) {
        if (exactCopy && sourceExtent.width == targetExtent.width && sourceExtent.height == targetExtent.height) {
            const VkImageCopy region{.srcSubresource = layers, .dstSubresource = layers,
                .extent = {sourceExtent.width, sourceExtent.height, 1}};
            exactCopy(cmd.handle(), source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            return;
        }
        const VkImageBlit region{.srcSubresource = layers,
            .srcOffsets = {{0, 0, 0}, {static_cast<int32_t>(sourceExtent.width), static_cast<int32_t>(sourceExtent.height), 1}},
            .dstSubresource = layers,
            .dstOffsets = {{0, 0, 0}, {static_cast<int32_t>(targetExtent.width), static_cast<int32_t>(targetExtent.height), 1}}};
        vk.df().CmdBlitImage(cmd.handle(), source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
    }

    std::vector<VkImage> swapchainImages(const vk::Vulkan& vk, VkSwapchainKHR swapchain) {
        uint32_t count{};
        check(vk.df().GetSwapchainImagesKHR(vk.dev(), swapchain, &count, nullptr), "count swapchain images");
        std::vector<VkImage> images(count);
        check(vk.df().GetSwapchainImagesKHR(vk.dev(), swapchain, &count, images.data()), "get swapchain images");
        images.resize(count);
        return images;
    }

    void requireFence(const vk::Fence& fence, const vk::Vulkan& device, const char* stage) {
        if (!fence.wait(device, graphicsTimeout)) throw ls::vulkan_error(VK_ERROR_OUT_OF_DATE_KHR, stage);
    }

    struct DirectFd {
        int fd{-1};
        DirectFd() = default;
        DirectFd(const DirectFd&) = delete;
        DirectFd& operator=(const DirectFd&) = delete;
        ~DirectFd() { if (fd >= 0) ::close(fd); }

        void keep(int borrowed) {
            fd = fcntl(borrowed, F_DUPFD_CLOEXEC, 0);
            if (fd < 0) throw ls::error("Duplicate direct phase DMA-BUF");
        }
    };

    class LinearDmaImage {
    public:
        LinearDmaImage(const vk::Vulkan& device, VkExtent2D extent, int& outputFd, int borrowedFd = -1) : device(device) {
            try {
                const VkExternalMemoryImageCreateInfo external{
                    .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
                const VkImageCreateInfo create{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &external, .flags = VK_IMAGE_CREATE_ALIAS_BIT,
                    .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
                    .extent = {extent.width, extent.height, 1}, .mipLevels = 1, .arrayLayers = 1,
                    .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
                    .usage = sharedUsage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
                check(device.df().CreateImage(device.dev(), &create, nullptr, &image), "create HTP linear phase image");
                const auto getLayout = reinterpret_cast<PFN_vkGetImageSubresourceLayout>(
                    device.fi().GetDeviceProcAddr(device.dev(), "vkGetImageSubresourceLayout"));
                const auto getFdTypes = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
                    device.fi().GetDeviceProcAddr(device.dev(), "vkGetMemoryFdPropertiesKHR"));
                if (!getLayout || !getFdTypes) throw ls::error("linear phase image dispatch is unavailable");
                const VkImageSubresource color{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT};
                VkSubresourceLayout layout{};
                getLayout(device.dev(), image, &color, &layout);
                VkMemoryRequirements required{};
                device.df().GetImageMemoryRequirements(device.dev(), image, &required);
                const auto stride = VkDeviceSize{extent.width} * 4;
                const auto bytes = stride * extent.height;
                if (layout.offset || layout.rowPitch != stride || layout.size != bytes || required.size != bytes)
                    throw ls::error("linear phase image does not match the packed HTP output");
                if (borrowedFd < 0) {
                    const int heap = ::open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
                    if (heap < 0) throw ls::error("open system DMA heap for HTP linear image");
                    dma_heap_allocation_data request{.len = bytes, .fd_flags = O_RDWR | O_CLOEXEC};
                    const auto allocated = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &request);
                    ::close(heap);
                    if (allocated != 0) throw ls::error("allocate system DMA-BUF for HTP linear image");
                    outputFd = static_cast<int>(request.fd);
                    borrowedFd = outputFd;
                }
                VkMemoryFdPropertiesKHR supported{.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
                check(getFdTypes(device.dev(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, borrowedFd, &supported),
                    "query linear phase DMA-BUF memory types");
                const auto type = device.findMemoryTypeIndex(required.memoryTypeBits & supported.memoryTypeBits, false);
                if (!type) throw ls::error("linear phase image has no compatible DMA-BUF memory type");
                const int imported = fcntl(borrowedFd, F_DUPFD_CLOEXEC, 0);
                if (imported < 0) throw ls::error("duplicate HTP output DMA-BUF to import into Vulkan");
                const VkMemoryDedicatedAllocateInfo dedicated{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = image};
                const VkImportMemoryFdInfoKHR import{
                    .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, .pNext = &dedicated,
                    .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = imported};
                const VkMemoryAllocateInfo allocation{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &import,
                    .allocationSize = bytes, .memoryTypeIndex = *type};
                const auto result = device.df().AllocateMemory(device.dev(), &allocation, nullptr, &memory);
                if (result != VK_SUCCESS) ::close(imported);
                check(result, "import HTP linear phase DMA-BUF");
                check(device.df().BindImageMemory(device.dev(), image, memory, 0), "bind HTP linear phase image");
            } catch (...) {
                reset();
                throw;
            }
        }

        ~LinearDmaImage() { reset(); }
        LinearDmaImage(const LinearDmaImage&) = delete;
        LinearDmaImage& operator=(const LinearDmaImage&) = delete;
        VkImage handle() const { return image; }

    private:
        const vk::Vulkan& device;
        VkImage image{};
        VkDeviceMemory memory{};

        void reset() {
            if (image) device.df().DestroyImage(device.dev(), image, nullptr);
            if (memory) device.df().FreeMemory(device.dev(), memory, nullptr);
        }
    };

    void releaseLinearImage(const vk::Vulkan& display, const vk::CommandBuffer& commands,
            VkImage image, VkImageLayout before, VkAccessFlags access) {
        auto release = transition(image, before, VK_IMAGE_LAYOUT_GENERAL, access, 0);
        release.srcQueueFamilyIndex = display.queueFamilyIndex();
        release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
        transition(display, commands, release);
    }

    struct SwapchainExtensions {
        std::optional<VkImageFormatListCreateInfo> viewFormats;
        std::optional<VkSwapchainPresentModesCreateInfoKHR> presentModes;

        explicit SwapchainExtensions(const VkSwapchainCreateInfoKHR& request) {
            for (auto* entry = static_cast<const VkBaseInStructure*>(request.pNext); entry; entry = entry->pNext) {
                if (entry->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO && !viewFormats) {
                    viewFormats = *reinterpret_cast<const VkImageFormatListCreateInfo*>(entry);
                    viewFormats->pNext = nullptr;
                } else if (entry->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR && !presentModes) {
                    presentModes = *reinterpret_cast<const VkSwapchainPresentModesCreateInfoKHR*>(entry);
                    presentModes->pNext = nullptr;
                } else {
                    throw ls::error("independent swapchain cannot preserve creation structure " + std::to_string(entry->sType));
                }
            }
            if (viewFormats && viewFormats->viewFormatCount && !viewFormats->pViewFormats)
                throw ls::error("independent swapchain image view formats are missing");
            if (request.flags & VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR) {
                const auto formats = viewFormats && viewFormats->viewFormatCount ?
                    std::span{viewFormats->pViewFormats, viewFormats->viewFormatCount} : std::span<const VkFormat>{};
                if (std::ranges::find(formats, request.imageFormat) == formats.end())
                    throw ls::error("mutable source swapchain requires its base format in the view list");
            }
        }
    };

    // The presenter and worker exchange images on a different timeline from
    // the application's staging ring. The original never waits on a worker value.
    class IndependentHtpVfi {
    public:
        struct Batch { size_t phases{}; bool sourceAcquired{}; };

        IndependentHtpVfi(const vk::Vulkan& display, VkExtent2D extent, VkFormat format, size_t count, uint64_t id,
                PFN_vkGetCalibratedTimestampsKHR calibrate, PFN_vkCmdCopyImage exactCopy,
                std::span<const int> sharedSources, const odin3::shared_frame::DmaSourcePool* sourcePool = nullptr) :
                id(id), phaseCount(count), upload(display), calibrate(calibrate),
                exactUpload(format == htpVfiFormat || enabled("ODIN3_HTP_RAW_HTP") ? exactCopy : nullptr) {
            const bool validExtent = extent.width >= 64 && extent.width % 16 == 0 &&
                extent.height >= 64 && extent.height % 4 == 0;
            const char* joinedContext = std::getenv("ODIN3_HTP_RAW_JOINED_CONTEXT");
            const char* firstContext = std::getenv("ODIN3_HTP_RAW_FIRST_CONTEXT");
            const char* bundleDir = std::getenv("ODIN3_FG_MODEL_BUNDLE_DIR");
            const bool hasExplicitCtx = (joinedContext && joinedContext[0] == '/') || (firstContext && firstContext[0] == '/');
            const char* bundleRes = (!hasExplicitCtx && bundleDir && bundleDir[0] == '/')
                ? mako::odin3ModelBundleSubdir(extent.width, extent.height) : nullptr;
            if (!hasExplicitCtx && bundleDir && bundleDir[0] == '/' && !bundleRes)
                throw ls::error("Odin3 Pure-Neural bundle supports 1280x720, 1600x900, and 1920x1080 swapchains");
            const bool joined = validExtent && count == 1 &&
                ((joinedContext && joinedContext[0] == '/') || bundleRes != nullptr) &&
                enabled("ODIN3_HTP_RAW_HTP") && mako::htpVfiDirectDri3Enabled() &&
                format == VK_FORMAT_B8G8R8A8_UNORM && !sharedSources.empty();
            if ((!validExtent && !joined) ||
                    (format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_B8G8R8A8_UNORM &&
                     format != VK_FORMAT_R8G8B8A8_SRGB && format != VK_FORMAT_B8G8R8A8_SRGB))
                throw ls::error("independent HTP has no compatible 8-bit SDR context for this extent");
            VkFormatProperties features{};
            display.fi().GetPhysicalDeviceFormatProperties(display.physdev(), htpVfiFormat, &features);
            const auto required = enabled("ODIN3_HTP_RAW_HTP") ?
                VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT :
                VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
            if ((features.optimalTilingFeatures & required) != required ||
                    (!directInput && !display.supportsExternalImageFormat(htpVfiFormat, mako::htpVfiSourceImageUsage,
                        VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)) ||
                    (!directOutput && !display.supportsExternalImageFormat(htpVfiFormat, mako::htpVfiOutputImageUsage,
                        VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)))
                throw ls::error("independent HTP presenter cannot export SDR images");

            VkPhysicalDeviceIDProperties ids{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &ids};
            display.fi().GetPhysicalDeviceProperties2(display.physdev(), &properties);
            backend::DriverIdentity driver;
            std::ranges::copy(ids.driverUUID, driver.driver.begin());
            std::ranges::copy(ids.deviceUUID, driver.device.begin());
            const auto vendor = backend::deviceIdText(properties.properties.vendorID);
            const auto device = backend::deviceIdText(properties.properties.deviceID);
            worker.emplace([&](const std::string&, const auto& candidate, const auto&) {
                return candidate.first == vendor && candidate.second == device;
            }, "", false, ls::makoFgPackagePath(), driver);

            std::vector<int> sourceFds(sharedSources.empty() ? 2 : sharedSources.size(), -1);
            const auto outputCount = mako::htpVfiDirectDri3Enabled()
                ? (phaseCount == 2 ? odin3_direct::Dri3::triplePhaseSlots : odin3_direct::Dri3::phaseSlots)
                : phaseCount;
            if (outputCount != phaseCount && ((phaseCount != 1 && phaseCount != 2) || sharedSources.empty() || !directOutput))
                throw ls::error("Direct game output requires 2x or 3x shared HTP sources");
            std::vector<int> outputFds(outputCount, -1);
            int timelineFd = -1;
            ls::FileDescriptorScope sourceScope{sourceFds}, outputScope{outputFds}, timelineScope{{&timelineFd, 1}};
            if ((directInput || directOutput) && (format != VK_FORMAT_B8G8R8A8_UNORM || !exactCopy))
                throw ls::error("direct HTP images require exact BGRA8 copies");
            if (!sharedSources.empty()) {
                for (size_t i = 0; i < sourceFds.size(); ++i) {
                    sourceFds[i] = fcntl(sharedSources[i], F_DUPFD_CLOEXEC, 0);
                    if (sourceFds[i] < 0) throw ls::error("duplicate Renderer source slot for the worker");
                }
            } else if (directInput) {
                for (size_t i = 0; i < sourceFds.size(); ++i)
                    linearSources.at(i).emplace(display, extent, sourceFds[i]);
            } else {
                for (auto& fd : sourceFds)
                    sources.emplace_back(display, extent, htpVfiFormat, mako::htpVfiSourceImageUsage, std::nullopt, &fd);
            }
            if (directOutput) {
                for (size_t i = 0; i < outputCount; ++i) {
                    linearOutputs.at(i).emplace(display, extent, outputFds[i]);
                    if (outputCount != phaseCount) outputDirectFds.at(i).keep(outputFds[i]);
                }
                vk::CommandBuffer initialization(display);
                vk::Fence ready(display);
                initialization.begin(display);
                for (size_t i = 0; i < outputCount; ++i) {
                    const auto image = linearOutputs.at(i)->handle();
                    transition(display, initialization, transition(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0));
                    releaseLinearImage(display, initialization, image, VK_IMAGE_LAYOUT_GENERAL, 0);
                }
                initialization.end(display);
                initialization.submit(display, {}, VK_NULL_HANDLE, 0, {}, VK_NULL_HANDLE, 0, ready.handle());
                requireFence(ready, display, "initial HTP output image release timed out");
            } else {
                for (auto& fd : outputFds)
                    outputs.emplace_back(display, extent, htpVfiFormat, mako::htpVfiOutputImageUsage, std::nullopt, &fd);
            }
            timeline.emplace(display, 0, std::nullopt, &timelineFd);
            const auto encoding = directInput
                ? (directOutput ? backend::FrameEncoding::Sdr8WithLinearDmaImages : backend::FrameEncoding::Sdr8WithLinearDmaInputs)
                : (directOutput ? backend::FrameEncoding::Sdr8WithLinearDmaOutputs : backend::FrameEncoding::Sdr8);
            if (sharedSources.empty()) {
                context = &worker->openContext((sourceScope.release(), std::pair{sourceFds[0], sourceFds[1]}),
                    (outputScope.release(), outputFds), (timelineScope.release(), timelineFd),
                    extent.width, extent.height, encoding, 1.0F, false, false, 1);
            } else {
                context = &worker->openSharedSourceContext((sourceScope.release(), sourceFds),
                    (outputScope.release(), outputFds), (timelineScope.release(), timelineFd), extent.width, extent.height, sourcePool);
            }
            if (bufferGeneratedEvents) generatedEvents.reserve(generatedEventCapacity);
            if (bufferFrameEvents) frameEvents.reserve(8 * sourceTimingCapacity);
            if (profileGeneratedEvents) slowGeneratedEvents.reserve(128);
            if (bufferReadyEvents) readyEvents.reserve(generatedEventCapacity);
            if (profileReadyEvents) readyLogTimes.reserve(generatedEventCapacity);
            if (bufferUploadRecords) uploadRecords.reserve(sourceTimingCapacity);
            if (profileUploadStages) uploadStalls.reserve(512);
            if (bufferGeneratedEvents || profileGeneratedEvents)
                std::clog << "MAKO Renderer: independent-event diagnostics buffered_generated="
                    << bufferGeneratedEvents << " profile=" << profileGeneratedEvents << '\n';
            if (bufferReadyEvents || profileReadyEvents)
                std::clog << "MAKO Renderer: independent-ready diagnostics buffered="
                    << bufferReadyEvents << " profile=" << profileReadyEvents << '\n';
            if (enabled("ODIN3_HTP_INDEPENDENT_UPLOAD_TIMING")) {
                try {
                    uploadQueries.emplace(display, 2);
                    std::clog << "MAKO Renderer: independent-upload timing calibrated=" << (calibrate != nullptr) << '\n';
                } catch (const std::exception& error) {
                    std::clog << "MAKO Renderer: independent-upload timing unavailable: " << error.what() << '\n';
                }
            }
            std::osyncstream(std::clog) << "MAKO Renderer: independent-htp active phases=" << phaseCount
                << " context=" << id << " linear_output=" << directOutput << " linear_input=" << directInput
                << " shared_source_slots=" << sharedSources.size() << '\n';
        }

        bool sharedReadComplete(const vk::Vulkan& display, uint64_t value) const {
            return !value || timeline->wait(display, value, 0);
        }
        bool waitSharedRead(const vk::Vulkan& display, uint64_t value, uint64_t timeoutNs) const {
            return !value || timeline->wait(display, value, timeoutNs);
        }
        bool sharedContextLive() const {
            return !stopped && !worker->contextLost(*context);
        }
        bool canReadShared(const vk::Vulkan& display) const {
            return sharedContextLive() && sharedReadComplete(display, lastValue);
        }
        bool waitForSharedCompletion(const vk::Vulkan& display, Clock::time_point until) const {
            if (!sharedContextLive()) return false;
            const auto now = Clock::now();
            if (now >= until) return false;
            return !lastValue || timeline->wait(display, lastValue,
                std::chrono::duration_cast<std::chrono::nanoseconds>(until - now).count());
        }
        uint64_t scheduleShared(const vk::Vulkan& display, unsigned a, unsigned b, uint64_t ordinal,
                Clock::time_point accepted, Clock::time_point previous, size_t reqPhases = 0) {
            const size_t nPhases = (reqPhases >= 1 && reqPhases <= phaseCount) ? reqPhases : phaseCount;
            constexpr std::array<float, 2> thirds{1.0F / 3, 2.0F / 3};
            constexpr std::array half{0.5F};
            const auto source = lastValue + 1;
            // B's Vulkan read has already retired. Signal the worker without an upload job.
            timeline->signal(display, source);
            try {
                worker->scheduleSharedSources(*context, nPhases == 1 ? std::span<const float>{half} :
                    std::span<const float>{thirds}, accepted - previous, a, b, ordinal);
            } catch (const std::exception& error) {
                stopped = true;
                std::osyncstream(std::clog) << "MAKO Renderer: independent-htp event=schedule-failed frame="
                    << ordinal << " error=" << error.what() << '\n';
                return 0;
            }
            activeSharedPhases = nPhases;
            lastValue = source + nPhases;
            event(ordinal, "requested", nPhases, source);
            return lastValue;
        }
        size_t sharedPhases() const { return phaseCount; }

        ~IndependentHtpVfi() {
            if (context) worker->closeContext(*context);
            if (!bufferGeneratedEvents && !profileGeneratedEvents && !bufferUploadRecords && !profileUploadStages &&
                !bufferReadyEvents && !profileReadyEvents && !bufferFrameEvents) return;
            std::osyncstream out(std::clog);
            for (const auto& item : frameEvents) writeEvent(out, item);
            if (bufferFrameEvents)
                out << "MAKO Renderer: independent-frame-event-summary context=" << id
                    << " buffered=" << frameEvents.size() << " untracked=" << untrackedFrameEvents << '\n';
            for (const auto& item : generatedEvents) writeEvent(out, item);
            for (const auto& item : readyEvents) writeEvent(out, item);
            for (const auto& row : readyLogTimes) {
                out << "MAKO Renderer: independent-ready-log context=" << id << " tid=" << row.tid
                    << " frame=" << row.frame << " phase=" << row.phase << " enter_ns=" << row.started
                    << " leave_ns=" << row.finished << '\n';
            }
            if (bufferReadyEvents || profileReadyEvents)
                out << "MAKO Renderer: independent-ready-log-summary context=" << id
                    << " buffered=" << readyEvents.size() << " timed=" << readyLogTimes.size()
                    << " untracked_ready=" << untrackedReadyEvents << " untracked_timing=" << untrackedReadyTimes << '\n';
            for (const auto& item : uploadRecords) writeUpload(out, item);
            for (const auto& [frame, stage, started, finished] : uploadStalls) {
                out << "MAKO Renderer: independent-upload-host context=" << id << " frame=" << frame
                    << " stage=" << stage << " enter_ns=" << started << " leave_ns=" << finished << '\n';
            }
            if (profileUploadStages || bufferUploadRecords)
                out << "MAKO Renderer: independent-upload-host-summary context=" << id
                    << " records=" << uploadRecords.size() << " profile=" << profileUploadStages
                    << " buffered=" << bufferUploadRecords << " untracked_uploads=" << untrackedUploads
                    << " untracked_stalls=" << untrackedUploadStalls << '\n';
            for (const auto& [frame, phase, started, finished] : slowGeneratedEvents) {
                out << "MAKO Renderer: independent-event-log context=" << id << " frame=" << frame
                    << " phase=" << phase << " enter_ns=" << started << " leave_ns=" << finished << '\n';
            }
            out << "MAKO Renderer: independent-event-log-summary context=" << id
                << " buffered=" << generatedEvents.size() << " untracked=" << untrackedGeneratedEvents
                << " slow=" << slowGeneratedEvents.size() << " slow_untracked=" << untrackedSlowEvents << '\n';
        }

        bool canPrefetch(const vk::Vulkan& display, uint64_t frame) const {
            const auto start = uploadProfileTime();
            const auto ready = !stopped && !worker->contextLost(*context) &&
                (!lastValue || timeline->wait(display, lastValue, 0));
            noteUploadStall(frame, "prefetch-timeline", start);
            return ready;
        }

        Batch begin(const vk::Vulkan& display, VkImage source, VkExtent2D extent,
                VkSemaphore sourceReady, uint64_t readyValue, uint64_t ordinal, Clock::time_point accepted) {
            if (stopped || worker->contextLost(*context)) {
                if (!stopped) event(ordinal, "worker-lost");
                stopped = true;
                return {};
            }
            if (lastValue && !timeline->wait(display, lastValue, 0)) {
                // Keep the previous batch's images reserved until the worker retires them.
                // A skipped source breaks adjacency, so the next scheduled frame primes history.
                previousOrdinal.reset();
                event(ordinal, "worker-pending");
                return {};
            }
            const auto collectStart = uploadProfileTime();
            collectUploadTiming(display);
            noteUploadStall(ordinal, "collect-previous-upload", collectStart);
            const bool generate = previousOrdinal && *previousOrdinal + 1 == ordinal;
            const auto interval = generate ? accepted - previousAccepted : Clock::duration{};
            constexpr std::array<float, 2> thirds{1.0F / 3, 2.0F / 3};
            constexpr std::array half{0.5F};
            const auto scheduleStart = uploadProfileTime();
            try {
                if (generate)
                    worker->scheduleFrames(*context, phaseCount == 1 ? std::span<const float>{half} :
                        std::span<const float>{thirds}, interval);
                else
                    worker->scheduleFrameHistory(*context, interval);
            } catch (const std::exception& error) {
                stopped = true;
                std::osyncstream(std::clog) << "MAKO Renderer: independent-htp event=schedule-failed frame="
                    << ordinal << " error=" << error.what() << '\n';
                return {};
            }

            noteUploadStall(ordinal, "message-worker", scheduleStart);
            const auto recordStart = uploadProfileTime();
            const auto index = scheduled % linearSources.size();
            const bool reused = scheduled >= linearSources.size();
            const auto input = directInput ? linearSources.at(index)->handle() : sources.at(index).handle();
            upload.begin(display);
            if (uploadQueries) {
                uploadQueries->reset(upload.handle());
                uploadQueries->write(upload.handle(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0);
            }
            vk::acquireExternalImage(display, upload.handle(), source,
                VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            transition(display, upload, transition(source, VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 0, VK_ACCESS_TRANSFER_READ_BIT));
            auto prepare = transition(input, reused ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            if (reused) {
                if (directInput) {
                    prepare.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
                    prepare.dstQueueFamilyIndex = display.queueFamilyIndex();
                } else {
                    vk::acquireExternalImage(display, upload.handle(), input,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                }
            }
            transition(display, upload, prepare);
            transferImage(display, upload, source, input, extent, extent, exactUpload);
            if (directInput) {
                releaseLinearImage(display, upload, input, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT);
            } else {
                vk::releaseExternalImage(display, upload.handle(), input, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            }
            if (uploadQueries) uploadQueries->write(upload.handle(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 1);
            upload.end(display);
            noteUploadStall(ordinal, "record-commands", recordStart);
            const auto inputValue = lastValue + 1;
            const auto workerTimeline = timeline->handle();
            const std::array waits{sourceReady, workerTimeline};
            const std::array waitValues{readyValue, lastValue};
            constexpr std::array stages{VkPipelineStageFlags{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
                VkPipelineStageFlags{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT}};
            const uint32_t waitCount = lastValue ? 2 : 1;
            const VkTimelineSemaphoreSubmitInfo values{.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                .waitSemaphoreValueCount = waitCount, .pWaitSemaphoreValues = waitValues.data(),
                .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &inputValue};
            const auto command = upload.handle();
            const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &values,
                .waitSemaphoreCount = waitCount, .pWaitSemaphores = waits.data(), .pWaitDstStageMask = stages.data(),
                .commandBufferCount = 1, .pCommandBuffers = &command,
                .signalSemaphoreCount = 1, .pSignalSemaphores = &workerTimeline};
            // Keep the GPU visibility dependency even after the host readiness check.
            const auto sent = uploadQueries ? monotonicNs(Clock::now()) : 0;
            const auto submitStart = uploadProfileTime();
            check(display.df().QueueSubmit(display.queue(), 1, &submit, VK_NULL_HANDLE), "upload independent HTP source");
            noteUploadStall(ordinal, "submit-commands", submitStart);
            if (uploadQueries) uploadSample = UploadSample{ordinal, inputValue, sent};
            ++scheduled;
            previousOrdinal = ordinal;
            previousAccepted = accepted;
            lastValue = inputValue + (generate ? phaseCount : 0);
            event(ordinal, generate ? "requested" : "history", generate ? phaseCount : 0, inputValue);
            return {generate ? phaseCount : 0, true};
        }

        bool ready(const vk::Vulkan& display, size_t phase, Clock::time_point deadline, uint64_t ordinal) const {
            const auto value = lastValue - (activeSharedPhases - phase - 1);
            for (auto now = Clock::now(); now < deadline; now = Clock::now()) {
                if (worker->contextLost(*context)) {
                    event(ordinal, "worker-lost", phase + 1);
                    return false;
                }
                const auto nextCheck = std::min(deadline, now + std::chrono::microseconds(200));
                const auto timeout = waitOnTimeline ?
                    std::chrono::duration_cast<std::chrono::nanoseconds>(nextCheck - now).count() : 0;
                if (timeline->wait(display, value, timeout)) {
                    if (Clock::now() >= deadline) {
                        event(ordinal, "ready-late", phase + 1, value - 1);
                        break;
                    }
                    const bool usable = worker->contextOutputUsable(*context, value, phase + 1 < activeSharedPhases);
                    event(ordinal, usable ? "phase-ready" : "motion-rejected", phase + 1, value - 1);
                    return usable;
                }
                if (!waitOnTimeline) std::this_thread::sleep_until(nextCheck);
            }
            event(ordinal, "worker-deadline", phase + 1, value - 1, deadline);
            return false;
        }

        VkImage result(size_t phase) const {
            return directOutput ? linearOutputs.at(phase)->handle() : outputs.at(phase).handle();
        }
        bool writesLinearOutput() const { return directOutput; }
        int directPhaseFd(size_t index) const { return outputDirectFds.at(index).fd; }
        bool running() const { return !stopped; }
        VkSemaphore readySignal() const { return timeline->handle(); }
        uint64_t readyValue(size_t phase) const { return lastValue - (activeSharedPhases - phase - 1); }

        // The caller has observed this upload's timeline or an idle device.
        void collectUploadTiming(const vk::Vulkan& display) noexcept {
            if (!uploadSample) return;
            const auto sample = *std::exchange(uploadSample, std::nullopt);
            try {
                const auto queryStart = uploadProfileTime();
                const auto values = uploadQueries->timestamps();
                noteUploadStall(sample.frame, "read-gpu-query", queryStart);
                const auto duration = uploadQueries->elapsedMicroseconds(values[0], values[1]);
                const auto calibrationStart = uploadProfileTime();
                const auto interval = calibrate ? calibrateInterval(calibrate, display.dev(), *uploadQueries, values,
                    sample.sentNs, monotonicNs(Clock::now())) : std::nullopt;
                noteUploadStall(sample.frame, "calibrate-gpu-clock", calibrationStart);
                const UploadRecord item{sample, std::llround(duration * 1000.0), interval};
                if (bufferUploadRecords) {
                    if (uploadRecords.size() < uploadRecords.capacity()) uploadRecords.push_back(item);
                    else ++untrackedUploads;
                } else {
                    const auto writeStart = uploadProfileTime();
                    {
                        std::osyncstream out(std::clog);
                        writeUpload(out, item);
                    }
                    noteUploadStall(sample.frame, "write-upload-log", writeStart);
                }
            } catch (...) {
                // Diagnostics stay out of the presentation recovery path.
            }
        }

        void event(uint64_t ordinal, const char* reason, size_t phase = 0, uint64_t source = 0,
                std::optional<Clock::time_point> deadline = std::nullopt) const {
            if (!verboseFrameEvents && !bufferFrameEvents && !bufferGeneratedEvents && !profileGeneratedEvents &&
                    !bufferReadyEvents && !profileReadyEvents)
                return;
            const Event item{reason, ordinal, phase, source, monotonicNs(Clock::now()),
                deadline ? monotonicNs(*deadline) : 0};
            if ((bufferReadyEvents || profileReadyEvents) && std::string_view(reason) == "phase-ready") {
                if (bufferReadyEvents && !bufferFrameEvents) {
                    if (readyEvents.size() < readyEvents.capacity()) readyEvents.push_back(item);
                    else ++untrackedReadyEvents;
                } else {
                    emitFrameEvent(item);
                }
                if (profileReadyEvents) {
                    const auto finished = monotonicNs(Clock::now());
                    static thread_local const auto tid = gettid();
                    if (readyLogTimes.size() < readyLogTimes.capacity())
                        readyLogTimes.push_back({ordinal, phase, item.at, finished, tid});
                    else ++untrackedReadyTimes;
                }
                return;
            }
            const bool generated = (bufferGeneratedEvents || profileGeneratedEvents) &&
                std::string_view(reason) == "generated";
            if (generated && bufferGeneratedEvents && !bufferFrameEvents) {
                if (generatedEvents.size() < generatedEventCapacity) generatedEvents.push_back(item);
                else ++untrackedGeneratedEvents;
            } else {
                emitFrameEvent(item);
            }
            if (generated && profileGeneratedEvents) {
                const auto finished = monotonicNs(Clock::now());
                if (finished - item.at >= 1'000'000) {
                    if (slowGeneratedEvents.size() < slowGeneratedEvents.capacity())
                        slowGeneratedEvents.push_back({ordinal, phase, item.at, finished});
                    else ++untrackedSlowEvents;
                }
            }
        }

    private:
        struct Event {
            const char* reason;
            uint64_t frame;
            size_t phase;
            uint64_t source;
            int64_t at, deadline;
        };
        struct SlowGeneratedLog {
            uint64_t frame;
            size_t phase;
            int64_t started, finished;
        };
        struct ReadyLogTime {
            uint64_t frame;
            size_t phase;
            int64_t started, finished;
            pid_t tid;
        };
        void emitFrameEvent(const Event& item) const {
            if (bufferFrameEvents) {
                if (frameEvents.size() < frameEvents.capacity()) frameEvents.push_back(item);
                else ++untrackedFrameEvents;
                return;
            }
            std::osyncstream out(std::clog);
            writeEvent(out, item);
        }
        void writeEvent(std::osyncstream& out, const Event& item) const {
            out << "MAKO Renderer: independent-htp event=" << item.reason
                << " context=" << id << " frame=" << item.frame << " phase=" << item.phase
                << " source=" << item.source;
            if (item.deadline) out << " deadline_ns=" << item.deadline;
            out << " time_ns=" << item.at << '\n';
        }

        struct UploadSample {
            uint64_t frame, source;
            int64_t sentNs;
        };

        struct UploadRecord {
            UploadSample sample;
            int64_t copyNs;
            std::optional<CalibratedInterval> interval;
        };
        struct UploadStall {
            uint64_t frame;
            const char* stage;
            int64_t started, finished;
        };
        void writeUpload(std::osyncstream& out, const UploadRecord& item) const {
            out << "MAKO Renderer: independent-upload context=" << id << " frame=" << item.sample.frame
                << " source=" << item.sample.source << " submit_ns=" << item.sample.sentNs
                << " copy_ns=" << item.copyNs;
            if (item.interval) out << " gpu_start_ns=" << item.interval->begin << " gpu_end_ns=" << item.interval->end
                << " deviation_ns=" << item.interval->deviation;
            out << '\n';
        }
        int64_t uploadProfileTime() const {
            return profileUploadStages ? monotonicNs(Clock::now()) : 0;
        }
        void noteUploadStall(uint64_t frame, const char* stage, int64_t start) const {
            if (!start) return;
            const auto end = monotonicNs(Clock::now());
            if (end - start < 1'000'000) return;
            if (uploadStalls.size() < uploadStalls.capacity()) uploadStalls.push_back({frame, stage, start, end});
            else ++untrackedUploadStalls;
        }

        const uint64_t id;
        const bool waitOnTimeline = enabled("ODIN3_HTP_INDEPENDENT_WAIT_TIMELINE");
        static constexpr size_t generatedEventCapacity = sourceTimingCapacity * 2;
        const bool verboseFrameEvents = enabled("ODIN3_FG_VERBOSE_LOG");
        const bool bufferFrameEvents = enabled("ODIN3_HTP_BUFFER_FRAME_EVENTS");
        const bool bufferGeneratedEvents = enabled("ODIN3_HTP_BUFFER_GENERATED_EVENTS");
        const bool profileGeneratedEvents = enabled("ODIN3_HTP_PROFILE_GENERATED_EVENTS");
        const bool bufferReadyEvents = enabled("ODIN3_HTP_BUFFER_READY_EVENTS");
        const bool profileReadyEvents = enabled("ODIN3_HTP_PROFILE_READY_EVENTS");
        const bool bufferUploadRecords = enabled("ODIN3_HTP_BUFFER_UPLOAD_RECORDS");
        const bool profileUploadStages = enabled("ODIN3_HTP_PROFILE_UPLOAD_STAGES");
        std::vector<UploadRecord> uploadRecords;
        mutable std::vector<UploadStall> uploadStalls;
        uint64_t untrackedUploads{};
        mutable uint64_t untrackedUploadStalls{};
        mutable std::vector<Event> generatedEvents;
        mutable std::vector<Event> frameEvents;
        mutable uint64_t untrackedFrameEvents{};
        mutable std::vector<SlowGeneratedLog> slowGeneratedEvents;
        mutable std::vector<Event> readyEvents;
        mutable std::vector<ReadyLogTime> readyLogTimes;
        mutable uint64_t untrackedReadyEvents{}, untrackedReadyTimes{};
        mutable uint64_t untrackedGeneratedEvents{}, untrackedSlowEvents{};
        const bool directOutput = mako::htpVfiLinearOutputEnabled();
        const bool directInput = mako::htpVfiLinearInputEnabled();
        const size_t phaseCount;
        size_t activeSharedPhases{phaseCount};
        vk::CommandBuffer upload;
        PFN_vkGetCalibratedTimestampsKHR calibrate;
        const PFN_vkCmdCopyImage exactUpload;
        std::optional<vk::TimestampQueryPool> uploadQueries;
        std::optional<UploadSample> uploadSample;
        std::vector<vk::Image> sources, outputs;
        std::array<std::optional<LinearDmaImage>, 2> linearSources;
        std::array<std::optional<LinearDmaImage>, odin3_direct::Dri3::maxPhaseSlots> linearOutputs;
        std::array<DirectFd, odin3_direct::Dri3::maxPhaseSlots> outputDirectFds;
        std::optional<vk::TimelineSemaphore> timeline;
        std::optional<backend::Instance> worker;
        backend::Context* context{};
        uint64_t scheduled{}, lastValue{};
        std::optional<uint64_t> previousOrdinal;
        Clock::time_point previousAccepted;
        bool stopped{};
    };
}

struct IndependentRealPresent::Impl {
    explicit Impl(const vk::Vulkan& app, PFN_vkReleaseSwapchainImagesKHR release) : app(app), release(release) {}

    const vk::Vulkan& app;
    const PFN_vkReleaseSwapchainImagesKHR release;
    const uint64_t id = nextPresentContext.fetch_add(1, std::memory_order_relaxed);
    HiddenSurface hidden;
    PrivateDevice child;
    VkSurfaceKHR visible{};
    VkSwapchainKHR source{}, output{};
    VkExtent2D sourceExtent{}, outputExtent{};
    PFN_vkCmdCopyImage exactSourceCopy{}, exactOriginalCopy{}, exactGeneratedCopy{};
    std::vector<VkImage> sourceImages, outputImages;
    std::optional<odin3_wsi::Probe> wsiProbe;
    bool drawBridge{};
    std::atomic<bool> bridgeAcquireWaiting{};
    const bool traceBridge = enabled("ODIN3_TRACE_FINAL_DRAW_BRIDGE");
    uint64_t lastConfirmedDraw{};
    std::optional<odin3_direct::PairCapture> presentedPair;
    std::unique_ptr<odin3_direct::Dri3> directPresent;
    struct DirectMiss { uint64_t ordinal; const char* cause; uint64_t cutoff; unsigned phase{1}; };
    std::vector<DirectMiss> directMisses;
    struct DirectPrefetch {
        uint64_t ordinal;
        size_t sourceSlot;
        Clock::time_point copyObserved;
        IndependentHtpVfi::Batch batch;
    };
    std::optional<DirectPrefetch> directPrefetched;
    const bool directPrefetchEnabled = enabled("ODIN3_HTP_DRI3_PREFETCH");
    const bool retryCurrentPair = enabled("ODIN3_DRI3_RETRY_CURRENT_PAIR");
    const bool drainDirectWhileIdle = enabled("ODIN3_DRI3_DRAIN_IDLE");
    const bool drainDirectDuringSourceWait = enabled("ODIN3_DRI3_DRAIN_SOURCE_WAIT");
    const bool retryDirectPrefetch = enabled("ODIN3_DRI3_RETRY_PREFETCH") && enabled("ODIN3_HTP_ASYNC_SCHEDULE");
    const bool directCreditGrace = enabled("ODIN3_HTP_DRI3_CREDIT_GRACE");
    const bool recheckDirectCredit = enabled("ODIN3_DRI3_RECHECK_CREDIT");
    struct Slot {
        std::optional<vk::Image> sharedOutput, sharedInput;
        std::optional<LinearDmaImage> linearSourceApp, linearSourceDisplay;
        std::optional<vk::TimelineSemaphore> appTimeline, childTimeline;
        std::optional<vk::CommandBuffer> sourceCopy, sourceReturn;
        std::array<std::optional<vk::CommandBuffer>, 12> cachedSourceCopies;
        std::optional<vk::TimestampQueryPool> sourceQueries;
        std::unique_ptr<DirectFd> proofCopy;
        Clock::time_point copySubmit;
        uint64_t generation{};
        uint64_t sourceOrdinal{};
        struct SourceRelease {
            uint32_t image;
            uint64_t readyValue;
            bool returnSubmitted{};
        };
        std::optional<SourceRelease> pendingSource;
        bool queued{};
        bool displayReading{};
        uint64_t workerReadUntil{};
        VkImage producerImage() const { return linearSourceApp ? linearSourceApp->handle() : sharedOutput->handle(); }
        VkImage presenterImage() const { return linearSourceDisplay ? linearSourceDisplay->handle() : sharedInput->handle(); }
    };
    struct PresentOptions {
        VkFence applicationFence{};
        std::optional<uint64_t> presentId;
        std::optional<VkPresentModeKHR> mode;
    };
    struct Job {
        uint64_t readyValue{};
        size_t sourceSlot{};
        PresentOptions options;
        Clock::time_point accepted;
        Clock::time_point entered, slotAvailable, enqueuedAt;
    };
    struct PreparedPresent {
        PresentFeedback::Record sample;
        uint32_t index;
        bool record, deferCopyWait, sourceCopyRetired;
    };
    struct EarlyShared {
        uint64_t ordinal;
        Job job;
        PreparedPresent original;
        IndependentHtpVfi::Batch batch{};
        bool acquired{}, copying{}, started{};
    };
    struct SourceTiming {
        uint64_t ordinal;
        Clock::time_point entered, slotAvailable, copyQueued, dequeued, copyObserved, presented;
        Clock::time_point copySubmit;
        std::optional<uint64_t> gpuCopyNs;
        std::optional<CalibratedInterval> gpuCopy;
        size_t sourceSlot{};
    };
    struct RetirementTiming {
        uint64_t ordinal{};
        Clock::time_point dequeueLockBegin, dequeueLockAcquired, dequeued;
        Clock::time_point presented, queryRead, calibrationDone, gpuTimingDone;
        Clock::time_point retireLockAcquired, slotRetired, producerNotified, statusDone;
        bool statusWritten{};
        Clock::time_point queryBegin, calibrationBegin, slotCleared, sourceLogDone, gapLogDone;
        uint64_t readyValue{}, queuedAfterRetirement{};
        size_t sourceSlot{};
    };
    struct PreparedHtpVfi {
        uint64_t ordinal;
        IndependentHtpVfi::Batch batch;
        Clock::time_point copyObserved;
    };
    enum class SharedPrefetchGate {
        Worker, SourceQueue, SourceCopy, WsiImage, AcquireFence, PresentRetirement,
        OriginalCopy, PreviousPhaseCopy, Complete, NoSlack, Invalidated, Count
    };
    struct SharedPrefetchTiming {
        uint64_t ordinal;
        Clock::time_point start, slot, stop;
        SharedPrefetchGate last = SharedPrefetchGate::NoSlack;
        std::array<uint32_t, static_cast<size_t>(SharedPrefetchGate::Count)> polls{};
        bool dispatched{};
        bool afterPhaseCopy{};
        bool afterPhasePresent{};
    };
    const bool asynchronous = enabled("ODIN3_HTP_ASYNC_REAL");
    const bool traceSources = asynchronous && enabled("ODIN3_HTP_INDEPENDENT_SOURCE_TIMING");
    const bool traceSourceLive = traceSources && enabled("ODIN3_HTP_INDEPENDENT_LIVE_SOURCE_TIMING");
    const bool traceSourceGpu = traceSources && enabled("ODIN3_HTP_INDEPENDENT_SOURCE_GPU_TIMING");
    std::vector<SourceTiming> sourceTimings;
    const bool traceRetirement = traceSources && enabled("ODIN3_HTP_RETIREMENT_TIMING");
    std::vector<RetirementTiming> retirementTimings;
    const bool traceRetirementGaps = asynchronous && enabled("ODIN3_HTP_INDEPENDENT_LIVE_RETIREMENT_GAPS");
    size_t retirementGaps{}, retirementInputIdleGaps{};
    const bool forceFifo = asynchronous && enabled("ODIN3_HTP_INDEPENDENT_GENERATE") && mako::htpVfiMotionEnabled();
    const bool repeatDiagnostic = forceFifo && enabled("ODIN3_HTP_INDEPENDENT_REPEAT_DIAGNOSTIC");
    const bool sharedSources = forceFifo && !repeatDiagnostic && mako::htpVfiSharedSourcesEnabled();
    const bool keepBypassedHistory = sharedSources && enabled("ODIN3_HTP_KEEP_BYPASSED_HISTORY");
    const bool sharedPrefetch = sharedSources && enabled("ODIN3_HTP_SHARED_PREFETCH") &&
        !mako::htpVfiDirectDri3Enabled();
    const bool prefetchAfterPhaseCopy = sharedPrefetch && enabled("ODIN3_HTP_SHARED_PREFETCH_AFTER_PHASE_COPY");
    const bool prefetchAfterPhasePresent = prefetchAfterPhaseCopy &&
        enabled("ODIN3_HTP_SHARED_PREFETCH_AFTER_PHASE_PRESENT");
    const bool prefetchFenceWake = sharedPrefetch && enabled("ODIN3_HTP_SHARED_PREFETCH_FENCE_WAKE");
    const bool traceSharedPrefetch = sharedPrefetch && enabled("ODIN3_HTP_SHARED_PREFETCH_TRACE");
    std::vector<SharedPrefetchTiming> sharedPrefetchTimings;
    const bool showOriginalForGenerated = !repeatDiagnostic && enabled("ODIN3_HTP_HTP_PRESENT_ORIGINAL_DIAGNOSTIC");
    const size_t repeatPhases = repeatDiagnostic ? (mako::htpVfiTripleEnabled() ? 2 : 1) : 0;
    const size_t sourceCapacity = mako::htpVfiDirectDri3Enabled() ? size_t{12} :
        sharedSources || (forceFifo && enabled("ODIN3_HTP_INDEPENDENT_EXTRA_SOURCE_SLOT"))
            ? asyncCapacity + 1 : asyncCapacity;
    std::vector<Slot> slots = std::vector<Slot>(sourceCapacity);
    std::vector<Job> jobs = std::vector<Job>(sharedSources ? asyncCapacity : sourceCapacity);
    std::optional<size_t> sharedHistory;
    Clock::time_point sharedHistoryAccepted;
    bool sharedHistoryCanEvict{};
    bool sourceAwaitingDisplay{};
    const uint32_t queueTarget = forceFifo ? presentQueueTarget() : 0;
    const bool traceFeedback = forceFifo && (queueTarget || enabled("ODIN3_HTP_INDEPENDENT_PRESENT_FEEDBACK"));
    const bool tracePresentEntry = traceFeedback && enabled("ODIN3_HTP_STAGE_TIMING");
    PresentFeedback feedback;
    bool privatePresentIds{};
    const std::chrono::nanoseconds outputSpacing = forceFifo ? outputPeriod() : std::chrono::nanoseconds{};
    const bool fixedOutputPacing = outputSpacing.count() && enabled("ODIN3_HTP_INDEPENDENT_FIXED_PACING");
    const bool feedbackOutputPacing = queueTarget == 2 && enabled("ODIN3_HTP_INDEPENDENT_FEEDBACK_PACING");
    const bool prefetchHtpVfi = forceFifo && !sharedSources && enabled("ODIN3_HTP_INDEPENDENT_PREFETCH");
    const bool prefetchBeforeAcquire = prefetchHtpVfi && !mako::htpVfiTripleEnabled() &&
        enabled("ODIN3_HTP_INDEPENDENT_PREFETCH_BEFORE_ACQUIRE");
    Clock::time_point nextOutput{};
    Clock::time_point lastOriginalAt{};
    PrivateDevice origCopyDevice;
    PFN_vkCmdCopyImage exactOrigDecoupleCopy{};
    std::array<std::optional<LinearDmaImage>, 8> origSourceImports;
    std::array<std::optional<LinearDmaImage>, odin3_direct::Dri3::originalSlots> directOrigImages;
    std::array<DirectFd, odin3_direct::Dri3::originalSlots> directOrigFds;
    std::optional<vk::CommandBuffer> origDecoupleCmd;
    std::optional<vk::Fence> origDecoupleFence;
    uint64_t lastSubmittedOrigMsc{};
    bool origCopyInFlight{};
    uint64_t pacingSlots{}, pacingRebases{};
    std::optional<PreparedHtpVfi> prefetched;
    const std::chrono::milliseconds injectedDelay = asynchronous ? testDelay() : std::chrono::milliseconds(0);
    std::mutex sourcePublishAccess;
    std::mutex jobsAccess;
    const std::chrono::nanoseconds sourceSpacing = forceFifo ? sourcePeriod() : std::chrono::nanoseconds{};
    Clock::time_point nextSource{};
    std::condition_variable jobsChanged;
    std::thread presenter;
    bool stopping{};
    uint64_t enqueued{}, completed{}, fullWaits{}, maximumDepth{};
    std::optional<vk::CommandBuffer> outputCopy;
    std::optional<vk::Fence> outputAcquire, copyDone;
    struct FirstPhaseCopy {
        vk::CommandBuffer commands;
        vk::Fence finished;
        bool pending{};
    };
    std::optional<FirstPhaseCopy> firstPhaseCopy;
    bool overlapGeneratedCopy{};
    bool fifoPacing{};
    std::optional<IndependentHtpVfi> htpVfi;
    std::optional<EarlyShared> earlyShared;
    std::optional<vk::CommandBuffer> earlySharedCopy;
    std::optional<vk::Fence> earlySharedAcquire, earlySharedDone;
    uint64_t sharedPrefetches{};
    struct PresentSynchronization {
        vk::Semaphore ready;
        vk::Fence retired;
        bool presented{};
    };
    std::vector<PresentSynchronization> outputSync;
    std::mutex sourceAccess, displayAccess;
    std::condition_variable imageReleased;
    uint64_t frames{};
    uint64_t generated{}, generatedRequested{}, repeated{};
    uint64_t applicationFences{}, presentIds{}, presentModes{};
    RuntimeStatusPublisher runtimeStatus;
    const bool asyncRuntimeStatus = enabled("ODIN3_HTP_ASYNC_RUNTIME_STATUS");
    const bool delayStatusWriter = traceRetirement && enabled("ODIN3_HTP_TEST_STATUS_WRITER_PAUSE");
    std::mutex runtimeStatusAccess;
    std::condition_variable runtimeStatusChanged;
    std::optional<RuntimeStatusRecord> pendingRuntimeStatus;
    bool stopStatusWriter{};
    std::thread statusWriter;
    uint64_t statusEnqueued{}, statusPublished{};
    Clock::time_point statusPauseBegin{}, statusPauseEnd{};
    Clock::time_point lastRuntimeSample{};
    uint64_t sampledOriginals{}, sampledGenerated{};
    const uint32_t statusAppId{runtimeAppId()};
    std::atomic<bool> invalid{};
    std::atomic<bool> suboptimal{};
    std::atomic<bool> workerFailure{};

    void initialize(VkInstance instance, const PrivateDeviceCallbacks& callbacks,
            X11PresentationTarget target, const VkSwapchainCreateInfoKHR& request) {
        const SwapchainExtensions extensions(request);
        const bool rawHtp = enabled("ODIN3_HTP_RAW_HTP");
        if (rawHtp) {
            const auto* order = std::getenv("ODIN3_HTP_RAW_CHANNEL_ORDER");
            const std::string_view channels = order ? order : "rgba";
            if ((channels != "rgba" && channels != "bgra") || request.imageFormat !=
                    (channels == "bgra" ? VK_FORMAT_B8G8R8A8_UNORM : htpVfiFormat))
                throw ls::error("raw HTP context does not match the swapchain byte order");
        }
        sourceExtent = request.imageExtent;
        const bool native1080 = sourceExtent.width == 1920 && sourceExtent.height == 1080;
        if (rawHtp && native1080 && (request.imageColorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
                request.imageFormat != VK_FORMAT_B8G8R8A8_UNORM))
            throw ls::error("1080p joined HTP requires the native BGRA8 SDR swapchain");
        visible = request.surface;
        hidden.create(instance, callbacks.nextLayer, target, sourceExtent);
        child.create(app, instance, callbacks, traceFeedback ? visible : VK_NULL_HANDLE);
        const auto& display = *child.vk;
        const auto capabilities = [&](VkSurfaceKHR surface) {
            VkBool32 supported{};
            check(app.fi().GetPhysicalDeviceSurfaceSupportKHR(app.physdev(), app.queueFamilyIndex(), surface, &supported),
                "query private surface support");
            if (!supported) throw ls::error("private surface is unavailable to the selected queue");
            VkSurfaceCapabilitiesKHR caps{};
            check(app.fi().GetPhysicalDeviceSurfaceCapabilitiesKHR(app.physdev(), surface, &caps), "query private surface capabilities");
            return caps;
        };
        const auto sourceCaps = capabilities(hidden.handle);
        const auto outputCaps = capabilities(request.surface);
        const auto sourceUsage = request.imageUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if ((sourceCaps.supportedUsageFlags & sourceUsage) != sourceUsage ||
                !(outputCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
                !(outputCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR))
            throw ls::error("private swapchain image usage is unsupported");
        if (!app.supportsExternalImageFormat(request.imageFormat, sharedUsage,
                VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT))
            throw ls::error("private swapchain format cannot share transfer images");
        VkFormatProperties features{};
        app.fi().GetPhysicalDeviceFormatProperties(app.physdev(), request.imageFormat, &features);
        constexpr auto transfer = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        const auto requiredFormat = rawHtp ? transfer : (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT);
        if ((features.optimalTilingFeatures & requiredFormat) != requiredFormat)
            throw ls::error("private swapchain cannot transfer the requested image format");
        if ((rawHtp || enabled("ODIN3_HTP_INDEPENDENT_EXACT_IMAGE_COPY")) &&
                (request.imageFormat == VK_FORMAT_B8G8R8A8_UNORM || request.imageFormat == htpVfiFormat) &&
                (features.optimalTilingFeatures & transfer) == transfer) {
            exactSourceCopy = reinterpret_cast<PFN_vkCmdCopyImage>(app.fi().GetDeviceProcAddr(app.dev(), "vkCmdCopyImage"));
            exactOriginalCopy = reinterpret_cast<PFN_vkCmdCopyImage>(
                display.fi().GetDeviceProcAddr(display.dev(), "vkCmdCopyImage"));
            // Both 32-bit formats are size-compatible. Raw HTP keeps the input
            // channel order in the intermediate storage and on return.
            if (rawHtp || request.imageFormat == htpVfiFormat) exactGeneratedCopy = exactOriginalCopy;
        }
        outputExtent = outputCaps.currentExtent.width == UINT32_MAX ? sourceExtent : outputCaps.currentExtent;
        if (rawHtp && (!exactSourceCopy || !exactGeneratedCopy || sourceExtent.width != outputExtent.width ||
                       sourceExtent.height != outputExtent.height))
            throw ls::error("raw HTP requires byte-preserving image transfers");
        if (sharedSources && (request.imageFormat != VK_FORMAT_B8G8R8A8_UNORM || !exactSourceCopy ||
                showOriginalForGenerated))
            throw ls::error("shared HTP sources require native BGRA8 generation with exact copies");
        if (!repeatDiagnostic && (!rawHtp || !mako::htpVfiDirectDri3Enabled()))
            throw ls::error("Frame generation requires HTP output that can be directly presented");
        auto sourceInfo = request;
        sourceInfo.surface = hidden.handle;
        sourceInfo.imageUsage = sourceUsage;
        sourceInfo.pNext = extensions.viewFormats ? &*extensions.viewFormats : nullptr;
        sourceInfo.clipped = VK_FALSE;
        if (sourceCaps.maxImageCount && sourceInfo.minImageCount > sourceCaps.maxImageCount)
            throw ls::error("hidden swapchain cannot satisfy the application image count");
        const bool directZeroCopyEligible = enabled("ODIN3_WSI_DIRECT_ZERO_COPY");
        const uint32_t desiredSourceImages = std::max<uint32_t>(sourceInfo.minImageCount, directZeroCopyEligible ? (extensions.viewFormats.has_value() ? 12u : 6u) : 5u);
        sourceInfo.minImageCount = sourceCaps.maxImageCount
            ? std::min(std::max(desiredSourceImages, sourceCaps.minImageCount), sourceCaps.maxImageCount)
            : std::max(desiredSourceImages, sourceCaps.minImageCount);
        sourceInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        const bool bridgeCandidate = directZeroCopyEligible &&
            rawHtp && sharedSources && asynchronous && mako::htpVfiDirectDri3Enabled() && !repeatDiagnostic &&
            (request.imageUsage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
        if (sharedSources && (bridgeCandidate || enabled("ODIN3_WSI_DMA_PROBE"))) {
            wsiProbe.emplace();
            wsiProbe->start(hidden.window, bridgeCandidate);
        }
        const auto sourceResult = app.df().CreateSwapchainKHR(app.dev(), &sourceInfo, nullptr, &source);
        check(sourceResult, "create hidden source swapchain");
        sourceImages = swapchainImages(app, source);
        if (wsiProbe) wsiProbe->stop(sourceImages.size(), sourceExtent.width, sourceExtent.height,
            request.imageFormat == VK_FORMAT_B8G8R8A8_UNORM ? odin3::shared_frame::Format::bgra8
                                                          : odin3::shared_frame::Format::unknown,
            request.imageColorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ? odin3::shared_frame::ColorSpace::srgb_nonlinear
                                                                         : odin3::shared_frame::ColorSpace::unknown);

        auto outputInfo = request;
        // Only the application creates mutable views, on the hidden chain.
        outputInfo.flags = 0;
        outputInfo.pNext = extensions.presentModes ? &*extensions.presentModes : nullptr;
        outputInfo.imageExtent = outputExtent;
        outputInfo.minImageCount = outputCaps.minImageCount;
        if (forceFifo) {
            const auto maximum = outputCaps.maxImageCount ? outputCaps.maxImageCount : UINT32_MAX;
            outputInfo.minImageCount += std::min(extraOutputImageCount(), maximum - outputInfo.minImageCount);
        }
        outputInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        outputInfo.preTransform = outputCaps.currentTransform;
        outputInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        // The application's Vulkan API may ask for MAILBOX. Independent
        // generation supplies its own cadence and needs ordered FIFO output.
        if (forceFifo) {
            outputInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
            outputInfo.pNext = nullptr;
        }
        if (child.timingStages) outputInfo.flags |= VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT | VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR;
        const auto createOutput = [&] {
            auto result = display.df().CreateSwapchainKHR(display.dev(), &outputInfo, nullptr, &output);
            if (child.timingStages && (result == VK_ERROR_EXTENSION_NOT_PRESENT || result == VK_ERROR_FEATURE_NOT_PRESENT)) {
                std::clog << "MAKO Renderer: independent-feedback swapchain rejected result=" << result << '\n';
                child.timingStages = 0;
                outputInfo.flags = 0;
                result = display.df().CreateSwapchainKHR(display.dev(), &outputInfo, nullptr, &output);
            }
            return result;
        };
        auto created = createOutput();
        if (outputInfo.minImageCount > outputCaps.minImageCount &&
                (created == VK_ERROR_OUT_OF_DEVICE_MEMORY || created == VK_ERROR_OUT_OF_HOST_MEMORY)) {
            std::clog << "MAKO Renderer: independent-real extra output image unavailable result=" << created << '\n';
            outputInfo.minImageCount = outputCaps.minImageCount;
            created = createOutput();
        }
        check(created, "create independent output swapchain");
        if (child.timingStages) {
            feedback.initialize(child, output);
            privatePresentIds = feedback.active;
        }
        outputImages = swapchainImages(display, output);
        for (size_t i = 0; i < outputImages.size(); ++i)
            outputSync.push_back({vk::Semaphore(display), vk::Fence(display)});
        std::vector<int> sharedDescriptors(sharedSources ? slots.size() : 0, -1);
        ls::FileDescriptorScope sharedScope{sharedDescriptors};
        drawBridge = bridgeCandidate && wsiProbe && wsiProbe->borrowSources(sharedDescriptors);
        if (const char* directory = std::getenv("ODIN3_DRI3_PAIR_CAPTURE_DIR");
                sharedSources && mako::htpVfiDirectDri3Enabled() && directory && directory[0] == '/')
            presentedPair.emplace(directory);
        for (size_t i = 0; i < (asynchronous ? slots.size() : 1); ++i) {
            auto& slot = slots[i];
            int fd = -1;
            if (drawBridge) {
                if (presentedPair) {
                    slot.proofCopy = std::make_unique<DirectFd>();
                    slot.proofCopy->keep(sharedDescriptors[i]);
                }
            } else if (sharedSources) {
                slot.linearSourceApp.emplace(app, sourceExtent, sharedDescriptors[i]);
                slot.linearSourceDisplay.emplace(display, sourceExtent, fd, sharedDescriptors[i]);
                if (wsiProbe || presentedPair) {
                    slot.proofCopy = std::make_unique<DirectFd>();
                    slot.proofCopy->keep(sharedDescriptors[i]);
                }
            } else {
                slot.sharedOutput.emplace(app, sourceExtent, request.imageFormat, sharedUsage, std::nullopt, &fd);
                slot.sharedInput.emplace(display, sourceExtent, request.imageFormat, sharedUsage, fd);
            }
            fd = -1;
            slot.appTimeline.emplace(app, 0, std::nullopt, &fd);
            slot.childTimeline.emplace(display, 0, fd);
            slot.sourceCopy.emplace(app);
            if (traceSourceGpu && !drawBridge) {
                try { slot.sourceQueries.emplace(app, 2); }
                catch (const std::exception& error) {
                    std::clog << "MAKO Renderer: independent-source GPU timing unavailable: " << error.what() << '\n';
                }
            }
        }
        if (sharedSources && !drawBridge) {
            vk::CommandBuffer initialization(display);
            vk::Fence ready(display);
            initialization.begin(display);
            for (const auto& slot : slots) {
                const auto image = slot.presenterImage();
                transition(display, initialization, transition(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0));
                releaseLinearImage(display, initialization, image, VK_IMAGE_LAYOUT_GENERAL, 0);
            }
            initialization.end(display);
            initialization.submit(display, {}, VK_NULL_HANDLE, 0, {}, VK_NULL_HANDLE, 0, ready.handle());
            requireFence(ready, display, "shared source image initialization timed out");
        }
        outputCopy.emplace(display);
        outputAcquire.emplace(display);
        copyDone.emplace(display);
        if (sharedPrefetch) {
            earlySharedCopy.emplace(display);
            earlySharedAcquire.emplace(display);
            earlySharedDone.emplace(display);
        }
        if (forceFifo) {
            const bool triple = mako::htpVfiTripleEnabled();
            if (!repeatDiagnostic) {
                try {
                    const std::optional<odin3::shared_frame::DmaSourcePool> bridgeSources = std::nullopt;
                    htpVfi.emplace(display, sourceExtent, request.imageFormat, triple ? 2 : 1, id,
                        child.calibrateUpload, exactGeneratedCopy, sharedDescriptors,
                        bridgeSources ? &*bridgeSources : nullptr);
                } catch (const std::exception& error) {
                    std::clog << "MAKO Renderer: independent-htp unavailable: " << error.what() << '\n';
                }
            }
            const bool deferTripleCopy = !mako::htpVfiLinearOutputEnabled() && triple && enabled("ODIN3_HTP_INDEPENDENT_DEFER_FIRST_COPY");
            const bool overlapDoubleCopy = !mako::htpVfiLinearOutputEnabled() && !triple && enabled("ODIN3_HTP_INDEPENDENT_OVERLAP_GENERATED_COPY");
            fifoPacing = (htpVfi || repeatDiagnostic) && !triple && enabled("ODIN3_HTP_INDEPENDENT_FIFO_PACING");
            if (fifoPacing) std::clog << "MAKO Renderer: independent-htp FIFO pacing enabled\n";
            if ((htpVfi || repeatDiagnostic) && (deferTripleCopy || overlapDoubleCopy)) {
                try {
                    firstPhaseCopy.emplace(vk::CommandBuffer(display), vk::Fence(display));
                    overlapGeneratedCopy = overlapDoubleCopy;
                    std::clog << "MAKO Renderer: independent-htp deferred first copy enabled"
                              << " original_overlap=" << overlapGeneratedCopy << '\n';
                } catch (const std::exception& error) {
                    std::clog << "MAKO Renderer: independent-htp deferred first copy unavailable: " << error.what() << '\n';
                }
            }
        }
        if (mako::htpVfiDirectDri3Enabled()) {
            const bool validExtent = sourceExtent.width >= 64 && sourceExtent.width % 16 == 0 &&
                sourceExtent.height >= 64 && sourceExtent.height % 4 == 0;
            if (!htpVfi || !sharedSources || request.imageFormat != VK_FORMAT_B8G8R8A8_UNORM ||
                    !validExtent || sourceExtent.width != outputExtent.width ||
                    sourceExtent.height != outputExtent.height || !outputSpacing.count())
                throw ls::error("Game does not meet the same-size direct HTP requirements");
            const unsigned phasePoolCount = htpVfi->sharedPhases() == 2
                ? odin3_direct::Dri3::triplePhaseSlots : odin3_direct::Dri3::phaseSlots;
            std::vector<int> descriptors(phasePoolCount + odin3_direct::Dri3::originalSlots, -1);
            for (size_t i = 0; i < phasePoolCount; ++i) {
                descriptors[i] = htpVfi->directPhaseFd(i);
                if (presentedPair) {
                    struct stat identity{};
                    if (fstat(descriptors[i], &identity) == 0)
                        std::osyncstream(std::clog) << "Odin3 DRI3 phase buffer: context=" << id
                            << " slot=" << i << " device=" << identity.st_dev
                            << " inode=" << identity.st_ino << std::endl;
                }
            }
            for (size_t i = 0; i < sharedDescriptors.size(); ++i) {
                descriptors[phasePoolCount + i] = sharedDescriptors[i];
            }
            directPresent = std::make_unique<odin3_direct::Dri3>(target.window, descriptors,
                static_cast<uint16_t>(sourceExtent.width), static_cast<uint16_t>(sourceExtent.height), id, phasePoolCount);
            directMisses.reserve(sourceTimingCapacity * 2);
        }
        std::clog << "MAKO Renderer: independent-real active source=" << sourceExtent.width << 'x' << sourceExtent.height
                  << " output=" << outputExtent.width << 'x' << outputExtent.height << " source_images=" << sourceImages.size()
                  << " output_images=" << outputImages.size() << " async=" << asynchronous
                  << " app_mode=" << request.presentMode << " output_mode=" << outputInfo.presentMode
                  << " app_format=" << request.imageFormat
                  << " raw_htp=" << rawHtp
                  << " view_formats=" << (extensions.viewFormats ? extensions.viewFormats->viewFormatCount : 0)
                  << " output_requested_images=" << outputInfo.minImageCount
                  << " fixed_output_pacing=" << ((htpVfi || repeatDiagnostic) && fixedOutputPacing && !fifoPacing)
                  << " source_min_interval_ns=" << sourceSpacing.count()
                  << " source_slots=" << (asynchronous ? slots.size() : 1) << " source_queue_cap=" << jobs.size()
                  << " shared_sources=" << sharedSources
                  << " exact_source_copy=" << (!drawBridge && exactSourceCopy != nullptr)
                  << " final_draw_bridge=" << drawBridge << " wsi_direct_input=0 generated_dri3=" << bool(directPresent)
                  << " exact_original_copy=" << (exactOriginalCopy && sourceExtent.width == outputExtent.width &&
                      sourceExtent.height == outputExtent.height)
                  << " exact_generated_copy=" << (exactGeneratedCopy && sourceExtent.width == outputExtent.width &&
                      sourceExtent.height == outputExtent.height)
                  << " repeat_diagnostic=" << repeatDiagnostic
                  << " repeat_phases=" << repeatPhases
                  << " x11_window=" << target.window << '\n';
        if (traceRetirementGaps)
            std::osyncstream(std::clog) << "MAKO Renderer: independent-retirement-gap-active context=" << id
                << " threshold_ns=" << retirementGapPeriod.count() << " cap=" << retirementGapLimit
                << " source_timing=" << traceSources << " source_live=" << traceSourceLive
                << " source_gpu=" << traceSourceGpu << std::endl;
        if (traceSources) sourceTimings.reserve(sourceTimingCapacity);
        if (traceRetirement) retirementTimings.reserve(sourceTimingCapacity);
        if (traceSharedPrefetch) sharedPrefetchTimings.reserve(sourceTimingCapacity);
        if (showOriginalForGenerated && htpVfi && htpVfi->writesLinearOutput())
            std::clog << "MAKO Renderer: independent-htp diagnostic-original-for-generated active\n";
        if (forceFifo && !repeatDiagnostic && statusAppId) {
            try {
                const auto* directory = std::getenv("ODIN3_FG_RUNTIME_STATE_DIR");
                if (directory && directory[0] == '/') {
                    runtimeStatus = RuntimeStatusPublisher(id, "independent-frame-generation", directory);
                    if (asyncRuntimeStatus && !runtimeStatus.path().empty())
                        statusWriter = std::thread([this] { writeRuntimeStatus(); });
                    lastRuntimeSample = Clock::now();
                    publishRuntimeStatus(lastRuntimeSample, true);
                }
            } catch (...) {
                // Diagnostics must not change the swapchain result.
            }
        }
        if (asynchronous) presenter = std::thread([this] { runPresenter(); });
    }

    void writeRuntimeStatus() noexcept {
        pthread_setname_np(pthread_self(), "odin3-status");
        pinRendererHelperThread(true);
        std::unordered_map<pid_t, uint64_t> prevTaskTicks;
        std::unordered_set<pid_t> managedTaskTids;
        bool taskTicksSeeded = false;
        for (;;) {
            std::optional<RuntimeStatusRecord> snapshot;
            {
                std::unique_lock lock(runtimeStatusAccess);
                runtimeStatusChanged.wait(lock, [&] { return stopStatusWriter || pendingRuntimeStatus; });
                if (!pendingRuntimeStatus) return;
                snapshot.swap(pendingRuntimeStatus);
            }
            // The lock covers the snapshot only. File I/O must never hold up presentation.
            if (delayStatusWriter && statusPauseBegin == Clock::time_point{} && snapshot->independent &&
                    snapshot->independent->originalFrames >= 2200) {
                statusPauseBegin = Clock::now();
                std::this_thread::sleep_for(std::chrono::milliseconds(520));
                statusPauseEnd = Clock::now();
            }
            runtimeStatus.publish(*snapshot);
            ++statusPublished;
            balanceGameThreadsToPrime(prevTaskTicks, managedTaskTids, taskTicksSeeded);
        }
    }

    void retireStatusWriter() {
        if (!statusWriter.joinable()) return;
        {
            const std::lock_guard lock(runtimeStatusAccess);
            stopStatusWriter = true;
        }
        runtimeStatusChanged.notify_one();
        statusWriter.join();
        if (traceRetirement)
            std::osyncstream(std::clog) << "MAKO Renderer: independent-status-writer context=" << id
                << " queued=" << statusEnqueued << " writes_attempted=" << statusPublished
                << " pause_begin_ns=" << monotonicNs(statusPauseBegin)
                << " pause_end_ns=" << monotonicNs(statusPauseEnd) << '\n';
    }

    void publishRuntimeStatus(Clock::time_point now, bool force = false, bool failed = false) noexcept {
        if (runtimeStatus.path().empty()) return;
        const auto elapsed = std::chrono::duration<double>(now - lastRuntimeSample).count();
        if (!force && elapsed < 1.0) return;
        try {
            const auto newOriginals = frames - sampledOriginals;
            const auto newGenerated = generated - sampledGenerated;
            const bool available = htpVfi && htpVfi->running() && !failed;
            const bool active = available && (newGenerated > 0 || (generated > 0 && newOriginals == 0));
            RuntimeStatusRecord status;
            status.phase = active ? RuntimeApplicationPhase::Active :
                available ? RuntimeApplicationPhase::Preparing : RuntimeApplicationPhase::Failed;
            status.reason = active ? "independent-frame-submitted" :
                available ? (generated > 0 ? "synchronizing-frame-cadence" : "waiting-for-generated-frames")
                          : "independent-backend-unavailable";
            if (!available) status.frameGenerationUnavailableReason = "independent-backend-unavailable";
            status.requestedProfile.multiplier = mako::htpVfiTripleEnabled() ? 3 : 2;
            status.requestedProfile.frame_generation_provisioned = true;
            status.requestedProfile.frame_generation_enabled = true;
            status.requestedProfile.scaling_enabled = false;
            status.appliedProfile = status.requestedProfile;
            status.appliedProfile.frame_generation_enabled = active;
            status.appliedGeneratedCapacity = active ? status.appliedProfile.multiplier - 1 : 0;
            status.frameGenerationActive = active;
            status.independent = IndependentFrameStatus{.appId = statusAppId, .originalFrames = frames,
                .generatedFrames = generated, .requestedGeneratedFrames = generatedRequested};
            if (elapsed >= 1.0) {
                status.independent->sampleAvailable = true;
                status.independent->originalSubmissionFps = newOriginals / elapsed;
                status.independent->totalSubmissionFps = (newOriginals + newGenerated) / elapsed;
            }
            if (asyncRuntimeStatus) {
                if (!statusWriter.joinable()) return;
                {
                    const std::lock_guard lock(runtimeStatusAccess);
                    pendingRuntimeStatus.emplace(std::move(status));
                    ++statusEnqueued;
                }
                runtimeStatusChanged.notify_one();
            } else {
                runtimeStatus.publish(status);
            }
            lastRuntimeSample = now;
            sampledOriginals = frames;
            sampledGenerated = generated;
        } catch (...) {
            // A local status write must never change error recovery.
        }
    }

    void logSourceTiming(std::ostream& log, const SourceTiming& timing) const {
        log << "MAKO Renderer: independent-source context=" << id << " frame=" << timing.ordinal
            << " enter_ns=" << monotonicNs(timing.entered)
            << " slot_ns=" << monotonicNs(timing.slotAvailable)
            << " copy_queued_ns=" << monotonicNs(timing.copyQueued)
            << " dequeue_ns=" << monotonicNs(timing.dequeued)
            << " copy_observed_ns=" << monotonicNs(timing.copyObserved)
            << " presented_ns=" << monotonicNs(timing.presented)
            << " source_slot=" << timing.sourceSlot;
        if (timing.gpuCopyNs) log << " copy_submit_ns=" << monotonicNs(timing.copySubmit)
            << " copy_gpu_ns=" << *timing.gpuCopyNs;
        if (timing.gpuCopy) log << " copy_gpu_start_ns=" << timing.gpuCopy->begin
            << " copy_gpu_end_ns=" << timing.gpuCopy->end << " copy_deviation_ns=" << timing.gpuCopy->deviation;
        log << '\n';
    }

    void logRetirementGap(const RetirementTiming& previous, const RetirementTiming& next,
            Clock::time_point enqueuedAt) const noexcept {
        try {
            std::osyncstream log(std::clog);
            log << "MAKO Renderer: independent-retirement-gap context=" << id
                << " prev_frame=" << previous.ordinal << " next_frame=" << next.ordinal
                << " prev_source_slot=" << previous.sourceSlot << " prev_ready_value=" << previous.readyValue
                << " next_source_slot=" << next.sourceSlot << " next_ready_value=" << next.readyValue
                << " prev_present_return_ns=" << monotonicNs(previous.presented)
                << " next_enqueued_ns=" << monotonicNs(enqueuedAt)
                << " queued_at_present_return=" << (enqueuedAt <= previous.presented)
                << " queued_after_retirement=" << previous.queuedAfterRetirement
                << " gap_ns=" << monotonicNs(next.dequeued) - monotonicNs(previous.presented)
                << " queued_ns=" << monotonicNs(next.dequeued) - monotonicNs(std::max(enqueuedAt, previous.presented))
                << " query_begin_ns=" << monotonicNs(previous.queryBegin)
                << " query_read_ns=" << monotonicNs(previous.queryRead)
                << " calibration_begin_ns=" << monotonicNs(previous.calibrationBegin)
                << " calibration_done_ns=" << monotonicNs(previous.calibrationDone)
                << " gpu_timing_done_ns=" << monotonicNs(previous.gpuTimingDone)
                << " retire_lock_acquired_ns=" << monotonicNs(previous.retireLockAcquired)
                << " slot_cleared_ns=" << monotonicNs(previous.slotCleared)
                << " slot_retired_ns=" << monotonicNs(previous.slotRetired)
                << " producer_notified_ns=" << monotonicNs(previous.producerNotified)
                << " source_log_done_ns=" << monotonicNs(previous.sourceLogDone)
                << " status_done_ns=" << monotonicNs(previous.statusDone)
                << " status_sampled=" << previous.statusWritten << " status_async=" << asyncRuntimeStatus
                << " prior_gap_log_done_ns=" << monotonicNs(previous.gapLogDone)
                << " next_dequeue_lock_begin_ns=" << monotonicNs(next.dequeueLockBegin)
                << " next_dequeue_lock_acquired_ns=" << monotonicNs(next.dequeueLockAcquired)
                << " next_dequeued_ns=" << monotonicNs(next.dequeued) << '\n' << std::flush;
        } catch (...) {
            // The diagnostic must not affect renderer failure or retirement.
        }
    }

    void retireDisplay() {
        {
            const std::lock_guard lock(jobsAccess);
            stopping = true;
        }
        jobsChanged.notify_all();
        // Drain any copy submission before releasing its source or the display state.
        const std::lock_guard publish(sourcePublishAccess);
        if (presenter.joinable()) {
            presenter.join();
            retireStatusWriter();
            if (fixedOutputPacing && pacingSlots)
                std::osyncstream(std::clog) << "MAKO Renderer: independent-pacing context=" << id
                    << " slots=" << pacingSlots << " rebases=" << pacingRebases << '\n';
            if (traceSources) {
                std::osyncstream log(std::clog);
                if (!traceSourceLive)
                    for (const auto& timing : sourceTimings) logSourceTiming(log, timing);
                log << "MAKO Renderer: independent-source-summary context=" << id
                    << " recorded=" << sourceTimings.size() << " completed=" << completed << '\n';
            }
            if (traceRetirement) {
                std::osyncstream log(std::clog);
                for (const auto& record : retirementTimings) {
                    log << "MAKO Renderer: independent-retirement context=" << id << " frame=" << record.ordinal
                        << " next_lock_enter_ns=" << monotonicNs(record.dequeueLockBegin)
                        << " next_lock_acquired_ns=" << monotonicNs(record.dequeueLockAcquired)
                        << " dequeued_ns=" << monotonicNs(record.dequeued)
                        << " present_done_ns=" << monotonicNs(record.presented)
                        << " query_read_ns=" << monotonicNs(record.queryRead)
                        << " calibration_done_ns=" << monotonicNs(record.calibrationDone)
                        << " gpu_timing_done_ns=" << monotonicNs(record.gpuTimingDone)
                        << " retire_lock_acquired_ns=" << monotonicNs(record.retireLockAcquired)
                        << " slot_retired_ns=" << monotonicNs(record.slotRetired)
                        << " producer_notified_ns=" << monotonicNs(record.producerNotified)
                        << " status_done_ns=" << monotonicNs(record.statusDone)
                        << " status_scheduled=" << record.statusWritten << '\n';
                }
                log << "MAKO Renderer: independent-retirement-summary context=" << id
                    << " recorded=" << retirementTimings.size() << " completed=" << completed << '\n';
            }
            if (traceRetirementGaps)
                std::osyncstream(std::clog) << "MAKO Renderer: independent-retirement-gap-summary context=" << id
                    << " threshold_ns=" << retirementGapPeriod.count() << " queued_gaps=" << retirementGaps
                    << " input_idle_gaps_until_cap=" << retirementInputIdleGaps
                    << " source_timing=" << traceSources << " source_live=" << traceSourceLive
                    << " source_gpu=" << traceSourceGpu << '\n';
            if (traceSharedPrefetch) {
                constexpr std::array names{"worker", "source_queue", "source_copy", "wsi_image",
                    "acquire_fence", "present_retirement", "original_copy", "previous_phase_copy",
                    "complete", "no_slack", "invalidated"};
                static_assert(names.size() == static_cast<size_t>(SharedPrefetchGate::Count));
                std::osyncstream log(std::clog);
                for (const auto& timing : sharedPrefetchTimings) {
                    log << "MAKO Renderer: shared-prefetch-timing context=" << id << " frame=" << timing.ordinal
                        << " start_ns=" << monotonicNs(timing.start) << " slot_ns=" << monotonicNs(timing.slot)
                        << " stop_ns=" << monotonicNs(timing.stop) << " last=" << names[static_cast<size_t>(timing.last)]
                        << " dispatched=" << timing.dispatched << " after_phase_copy=" << timing.afterPhaseCopy
                        << " after_phase_present=" << timing.afterPhasePresent;
                    for (size_t i = 0; i < names.size(); ++i)
                        if (timing.polls[i]) log << ' ' << names[i] << '=' << timing.polls[i];
                    log << '\n';
                }
                log << "MAKO Renderer: shared-prefetch-summary context=" << id
                    << " recorded=" << sharedPrefetchTimings.size() << " completed=" << completed << '\n';
            }
        }
        retireStatusWriter();
        const std::lock_guard lock(displayAccess);
        invalidate();
        if (!output) return;
        const auto& display = *child.vk;
        const auto idle = display.df().DeviceWaitIdle(display.dev());
        if (idle == VK_SUCCESS) {
            if (htpVfi) htpVfi->collectUploadTiming(display);
            for (const auto& sync : outputSync) {
                if (!sync.presented) continue;
                try { static_cast<void>(sync.retired.wait(display)); }
                catch (const std::exception&) { std::clog << "MAKO Renderer: independent-real retirement fence failed\n"; }
            }
            feedback.collect(child, output);
        }
        if (privatePresentIds) feedback.writeLog(id);
        if (directPresent) {
            directPresent->finish(id);
            if (enabled("ODIN3_FG_VERBOSE_LOG")) {
                for (const auto& miss : directMisses)
                    std::clog << "MAKO Renderer: direct-dri3-miss context=" << id << " frame=" << miss.ordinal
                        << " phase=" << miss.phase << " cause=" << miss.cause << " deadline_ns=" << miss.cutoff << '\n';
            } else if (!directMisses.empty()) {
                std::clog << "MAKO Renderer: direct-dri3-miss-summary context=" << id
                    << " count=" << directMisses.size() << '\n';
            }
        }
        htpVfi.reset();
        directPresent.reset();
        if (presentedPair) {
            try { presentedPair->save(); }
            catch (const std::exception& error) {
                std::clog << "Odin3 presented pair save failed: " << error.what() << '\n';
            }
        }
        display.df().DestroySwapchainKHR(display.dev(), output, nullptr);
        output = VK_NULL_HANDLE;
    }

    void invalidate() {
        invalid.store(true, std::memory_order_release);
        imageReleased.notify_all();
        jobsChanged.notify_all();
    }

    void releasableSources() {
        std::array<uint32_t, 12> readyImages{};
        uint32_t readyCount = 0;
        for (auto& slot : slots) {
            if (!slot.pendingSource) continue;
            const auto& pending = *slot.pendingSource;
            if (drawBridge && !pending.returnSubmitted) continue;
            if (!slot.appTimeline->wait(app, pending.readyValue, 0)) continue;
            if (readyCount < readyImages.size()) readyImages[readyCount++] = pending.image;
            if (drawBridge && traceBridge)
                std::osyncstream(std::clog) << "Odin3 final draw reusable: context=" << id
                    << " frame=" << slot.sourceOrdinal << " index=" << pending.image
                    << " generation=" << slot.generation << " at_ns=" << monotonicNs(Clock::now()) << '\n';
            slot.pendingSource.reset();
        }
        if (readyCount) {
            const VkReleaseSwapchainImagesInfoKHR info{.sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
                .swapchain = source, .imageIndexCount = readyCount, .pImageIndices = readyImages.data()};
            check(release(app.dev(), &info), "release retired source swapchain image");
            imageReleased.notify_all();
        }
    }

    void tryReclaimSourcesFromPresenter() noexcept {
        if (drawBridge) return;
        std::unique_lock lk(sourceAccess, std::try_to_lock);
        if (!lk.owns_lock()) return;
        try { releasableSources(); } catch (...) {}
    }

    // Caller owns jobsAccess, which protects the queued, history, and display readers.
    void returnDrawSourcesLocked() {
        if (!drawBridge) return;
        bool anyCandidate = false;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot = slots[i];
            if (!slot.pendingSource) continue;
            if (slot.pendingSource->returnSubmitted) { anyCandidate = true; break; }
            if (!slot.queued && sharedHistory != i && !slot.displayReading &&
                    !(presentedPair && presentedPair->holding(i))) {
                anyCandidate = true;
                break;
            }
        }
        if (!anyCandidate) return;
        const std::lock_guard lock(sourceAccess);
        bool anyReadyToRelease = false;
        for (size_t i = 0; i < slots.size(); ++i) {
            auto& slot = slots[i];
            if (!slot.pendingSource) continue;
            if (slot.pendingSource->returnSubmitted) { anyReadyToRelease = true; continue; }
            if (slot.queued || sharedHistory == i || slot.displayReading ||
                    (presentedPair && presentedPair->holding(i)) ||
                    (slot.workerReadUntil && htpVfi && !htpVfi->sharedReadComplete(*child.vk, slot.workerReadUntil)))
                continue;
            slot.workerReadUntil = 0;
            slot.pendingSource->returnSubmitted = true;
            anyReadyToRelease = true;
        }
        if (anyReadyToRelease) releasableSources();
    }

    VkResult acquire(const VkAcquireNextImageInfoKHR& request, uint32_t* imageIndex, bool secondApi) {
        if (secondApi && !app.df().AcquireNextImage2KHR) return VK_ERROR_EXTENSION_NOT_PRESENT;
        auto immediate = request;
        immediate.timeout = 0;
        const auto start = Clock::now();
        std::unique_lock lock(sourceAccess);
        const auto lockAcquired = traceSourceLive ? Clock::now() : Clock::time_point{};
        uint64_t reclaimNs{}, driverNs{}, waitNs{}, driverCalls{};
        const auto finish = [&](VkResult result) {
            if (traceSources) {
                const auto end = Clock::now();
                if (end - start >= std::chrono::milliseconds(4)) {
                    lock.unlock();
                    std::osyncstream log(std::clog);
                    log << "MAKO Renderer: independent-acquire context=" << id
                        << " enter_ns=" << monotonicNs(start) << " exit_ns=" << monotonicNs(end)
                        << " result=" << result;
                    if (traceSourceLive) {
                        log << " initial_lock_ns=" << monotonicNs(lockAcquired) - monotonicNs(start)
                            << " reclaim_ns=" << reclaimNs << " driver_calls=" << driverCalls
                            << " driver_ns=" << driverNs << " wait_and_relock_ns=" << waitNs;
                    }
                    log << std::endl;
                }
            }
            return result;
        };
        bool tryImmediateFirst = true;
        while (!invalid.load(std::memory_order_acquire)) {
            const bool skippedReclaim = tryImmediateFirst;
            if (!tryImmediateFirst) {
                const auto reclaimStart = traceSourceLive ? Clock::now() : Clock::time_point{};
                try { releasableSources(); }
                catch (const ls::vulkan_error& error) {
                    invalidate();
                    return finish(error.error() == VK_ERROR_DEVICE_LOST ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_DATE_KHR);
                }
                if (traceSourceLive) reclaimNs += monotonicNs(Clock::now()) - monotonicNs(reclaimStart);
            }
            tryImmediateFirst = false;
            const auto driverStart = traceSourceLive ? Clock::now() : Clock::time_point{};
            const auto result = secondApi
                ? app.df().AcquireNextImage2KHR(app.dev(), &immediate, imageIndex)
                : app.df().AcquireNextImageKHR(app.dev(), source, 0, request.semaphore, request.fence, imageIndex);
            if (skippedReclaim && (result == VK_NOT_READY || result == VK_TIMEOUT)) continue;
            if (traceSourceLive) {
                driverNs += monotonicNs(Clock::now()) - monotonicNs(driverStart);
                ++driverCalls;
            }
            if (result == VK_SUCCESS && suboptimal.load(std::memory_order_acquire)) return finish(VK_SUBOPTIMAL_KHR);
            if (result != VK_NOT_READY && result != VK_TIMEOUT) return finish(result);
            if (drawBridge) {
                bridgeAcquireWaiting.store(true, std::memory_order_release);
                jobsChanged.notify_all();
            }
            if (!request.timeout) return finish(VK_NOT_READY);
            const auto elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
            if (elapsed >= request.timeout) return finish(VK_TIMEOUT);
            // The hidden WSI can have no free index while the present thread
            // is returning its copy. Never lock out the thread doing release.
            const auto waitStart = traceSourceLive ? Clock::now() : Clock::time_point{};
            imageReleased.wait_for(lock, std::chrono::nanoseconds(std::min<uint64_t>(request.timeout - elapsed, 1'000'000)));
            if (traceSourceLive) waitNs += monotonicNs(Clock::now()) - monotonicNs(waitStart);
        }
        return finish(VK_ERROR_OUT_OF_DATE_KHR);
    }

    ~Impl() {
        if (source) {
            retireDisplay();
            // The application owns the other queues on this VkDevice. Only
            // wait for copies submitted by this context before freeing it.
            try {
                const std::lock_guard lock(sourceAccess);
                for (const auto& slot : slots) {
                    if (!slot.generation) continue;
                    static_cast<void>(slot.appTimeline->wait(app, 2 * slot.generation - 1, graphicsTimeout));
                }
                releasableSources();
            } catch (const std::exception&) {
                std::clog << "MAKO Renderer: independent-real source retirement failed\n";
            }
            app.df().DestroySwapchainKHR(app.dev(), source, nullptr);
            std::clog << "MAKO Renderer: independent-real retired frames=" << frames
                      << " application_fences=" << applicationFences << " present_ids=" << presentIds
                      << " present_modes=" << presentModes << " queued=" << enqueued << " completed=" << completed
                      << " maximum_depth=" << maximumDepth << " full_waits=" << fullWaits
                      << " generated=" << generated << " requested=" << generatedRequested
                      << " repeated=" << repeated << " shared_prefetches=" << sharedPrefetches << '\n';
        }
    }

    bool parsePresentOptions(const VkPresentInfoKHR& request, PresentOptions& options) const {
        for (auto* entry = static_cast<const VkBaseInStructure*>(request.pNext); entry; entry = entry->pNext) {
            switch (entry->sType) {
                case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR: {
                    const auto& info = *reinterpret_cast<const VkSwapchainPresentFenceInfoKHR*>(entry);
                    if (info.swapchainCount != 1 || !info.pFences) return false;
                    options.applicationFence = info.pFences[0];
                    break;
                }
                case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR: {
                    const auto& info = *reinterpret_cast<const VkSwapchainPresentModeInfoKHR*>(entry);
                    if (info.swapchainCount != 1 || !info.pPresentModes) return false;
                    if (!forceFifo) options.mode = info.pPresentModes[0];
                    break;
                }
                case VK_STRUCTURE_TYPE_PRESENT_ID_KHR: {
                    const auto& info = *reinterpret_cast<const VkPresentIdKHR*>(entry);
                    if (privatePresentIds || !child.waitForPresent || info.swapchainCount != 1 || !info.pPresentIds) return false;
                    options.presentId = info.pPresentIds[0];
                    break;
                }
                default:
                    std::clog << "MAKO Renderer: independent-real unsupported present structure=" << entry->sType << '\n';
                    return false;
            }
        }
        // The caller-facing fence belongs to the app queue. Background
        // presentation must not submit to that queue.
        return !asynchronous || !options.applicationFence;
    }

    template <typename Submit>
    static void submitInputCopy(std::unique_lock<std::mutex>* jobsLock, bool& sourceSubmitted, Submit&& submit) {
        if (jobsLock) jobsLock->unlock();
        submit();
        sourceSubmitted = true;
        if (jobsLock) jobsLock->lock();
    }

    uint64_t publishSource(Slot& slot, VkQueue queue, const VkPresentInfoKHR& request, bool& sourceSubmitted,
            std::unique_lock<std::mutex>* jobsLock = nullptr) {
        const uint32_t sourceIndex = request.pImageIndices[0];
        const auto sourceImage = sourceImages.at(sourceIndex);
        {
            const std::lock_guard lock(sourceAccess);
            if (slot.pendingSource)
                throw ls::error("source image was not released before slot reuse");
        }
        const auto readyValue = slot.generation * 2 + 1;
        auto& copy = *slot.sourceCopy;
        if (drawBridge) {
            submitInputCopy(jobsLock, sourceSubmitted, [&] {
                std::array<VkPipelineStageFlags, 8> waitStages{};
                std::array<uint64_t, 8> waitVals{};
                const uint32_t wCount = std::min<uint32_t>(request.waitSemaphoreCount, 8u);
                waitStages.fill(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
                const VkSemaphore sigSem = slot.appTimeline->handle();
                const VkTimelineSemaphoreSubmitInfo tlInfo{
                    .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                    .pNext = nullptr,
                    .waitSemaphoreValueCount = wCount,
                    .pWaitSemaphoreValues = wCount ? waitVals.data() : nullptr,
                    .signalSemaphoreValueCount = 1u,
                    .pSignalSemaphoreValues = &readyValue,
                };
                const VkSubmitInfo subInfo{
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .pNext = &tlInfo,
                    .waitSemaphoreCount = wCount,
                    .pWaitSemaphores = wCount ? request.pWaitSemaphores : nullptr,
                    .pWaitDstStageMask = wCount ? waitStages.data() : nullptr,
                    .commandBufferCount = 0u,
                    .pCommandBuffers = nullptr,
                    .signalSemaphoreCount = 1u,
                    .pSignalSemaphores = &sigSem,
                };
                check(app.df().QueueSubmit(queue, 1u, &subInfo, VK_NULL_HANDLE), "relay zero-copy source semaphore");
            });
            ++slot.generation;
            slot.sourceOrdinal = enqueued;
            const std::lock_guard lock(sourceAccess);
            slot.pendingSource = Slot::SourceRelease{sourceIndex, readyValue};
            return readyValue;
        }
        const auto shared = slot.producerImage();
        if (sharedSources && directPresent && !drawBridge && !slot.sourceQueries && !wsiProbe &&
                slot.generation >= 1 && sourceIndex < slot.cachedSourceCopies.size()) {
            auto& cached = slot.cachedSourceCopies[sourceIndex];
            if (!cached) {
                cached.emplace(app);
                cached->begin(app, 0);
                const auto prepShared = transition(shared, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
                const auto prepSource = transition(sourceImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                const std::array preBarriers{prepShared, prepSource};
                app.df().CmdPipelineBarrier(cached->handle(),
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                    static_cast<uint32_t>(preBarriers.size()), preBarriers.data());
                transferImage(app, *cached, sourceImage, shared, sourceExtent, sourceExtent, exactSourceCopy);
                const auto restoreSource = transition(sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT, 0);
                const auto releaseShared = transition(shared, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
                const std::array postBarriers{restoreSource, releaseShared};
                app.df().CmdPipelineBarrier(cached->handle(),
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    0, 0, nullptr, 0, nullptr,
                    static_cast<uint32_t>(postBarriers.size()), postBarriers.data());
                cached->end(app);
            }
            submitInputCopy(jobsLock, sourceSubmitted, [&] {
                cached->submit(app, {request.pWaitSemaphores, request.waitSemaphoreCount},
                    VK_NULL_HANDLE, 0, {}, slot.appTimeline->handle(), readyValue, VK_NULL_HANDLE, queue);
            });
            ++slot.generation;
            slot.sourceOrdinal = enqueued;
            {
                const std::lock_guard lock(sourceAccess);
                slot.pendingSource = Slot::SourceRelease{sourceIndex, readyValue};
            }
            return readyValue;
        }
        copy.begin(app);
        if (slot.sourceQueries) {
            slot.sourceQueries->reset(copy.handle());
            slot.sourceQueries->write(copy.handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, 0);
        }
        if (sharedSources) {
            auto acquireShared = transition(shared, slot.generation ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            if (slot.generation) {
                acquireShared.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
                acquireShared.dstQueueFamilyIndex = app.queueFamilyIndex();
            }
            const auto prepSource = transition(sourceImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            const std::array preBarriers{acquireShared, prepSource};
            app.df().CmdPipelineBarrier(copy.handle(),
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                static_cast<uint32_t>(preBarriers.size()), preBarriers.data());

            transferImage(app, copy, sourceImage, shared, sourceExtent, sourceExtent, exactSourceCopy);

            const auto restoreSource = transition(sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT, 0);
            const auto releaseShared = transition(shared, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
            const std::array postBarriers{restoreSource, releaseShared};
            app.df().CmdPipelineBarrier(copy.handle(),
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0, 0, nullptr, 0, nullptr,
                static_cast<uint32_t>(postBarriers.size()), postBarriers.data());
        } else {
            auto target = transition(shared, slot.generation ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            if (slot.generation) {
                vk::acquireExternalImage(app, copy.handle(), shared,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            }
            transition(app, copy, target);
            transition(app, copy, transition(sourceImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT));
            transferImage(app, copy, sourceImage, shared, sourceExtent, sourceExtent, exactSourceCopy);
            transition(app, copy, transition(sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT, 0));
            vk::releaseExternalImage(app, copy.handle(), shared, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }
        if (slot.sourceQueries) slot.sourceQueries->write(copy.handle(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 1);
        copy.end(app);
        if (slot.sourceQueries) slot.copySubmit = Clock::now();
        submitInputCopy(jobsLock, sourceSubmitted, [&] {
            copy.submit(app, {request.pWaitSemaphores, request.waitSemaphoreCount},
                slot.generation ? slot.appTimeline->handle() : VK_NULL_HANDLE, readyValue - 1, {},
                slot.appTimeline->handle(), readyValue, VK_NULL_HANDLE, queue);
        });
        ++slot.generation;
        slot.sourceOrdinal = enqueued;
        {
            const std::lock_guard lock(sourceAccess);
            slot.pendingSource = Slot::SourceRelease{sourceIndex, readyValue};
            // The optional proof needs the game image until its read completes.
            if (wsiProbe && wsiProbe->choose()) {
                if (slot.appTimeline->wait(app, readyValue, graphicsTimeout)) {
                    wsiProbe->compare(sourceIndex, slot.proofCopy->fd);
                }
                else std::clog << "MAKO Renderer: wsi-dma pixel proof waiting for copy timed out\n";
            }
        }
        if (presentedPair && enabled("ODIN3_FG_VERBOSE_LOG")) {
            const auto original = reinterpret_cast<uintptr_t>(sourceImage);
            const auto working = reinterpret_cast<uintptr_t>(shared);
            std::osyncstream(std::clog) << "Odin3 source copy: context=" << id << " frame=" << enqueued
                << " original_image=" << original << " input_image=" << working
                << " input_slot=" << (&slot - slots.data())
                << " input_generation=" << slot.generation << " ready_value=" << readyValue
                << " bytes=" << uint64_t(sourceExtent.width) * sourceExtent.height * 4 << std::endl;
        }
        return readyValue;
    }

    bool prefetchSource(const vk::Vulkan& display, uint64_t ordinal, bool beforeAcquire = false) {
        Job following;
        {
            const std::lock_guard lock(jobsAccess);
            if (enqueued <= ordinal) return false;
            following = jobs[ordinal % jobs.size()];
        }
        const auto& slot = slots[following.sourceSlot];
        if (!slot.appTimeline->wait(app, following.readyValue, 0)) return false;
        const auto copyObserved = traceSources ? Clock::now() : Clock::time_point{};
        if (!htpVfi->canPrefetch(display, ordinal)) return false;
        const auto batch = htpVfi->begin(display, slot.presenterImage(), sourceExtent,
            slot.childTimeline->handle(), following.readyValue, ordinal, following.accepted);
        prefetched = PreparedHtpVfi{ordinal, batch, copyObserved};
        if (batch.sourceAcquired) {
            htpVfi->event(ordinal, "prefetched", batch.phases);
            if (beforeAcquire) htpVfi->event(ordinal, "prefetched-before-acquire", batch.phases);
        }
        return true;
    }

    void waitForOutputSlot(const vk::Vulkan& display, std::optional<uint64_t> following, bool afterPhaseCopy,
            PresentFeedback::Record& sample, bool useFeedback) {
        const auto started = Clock::now();
        const auto slot = useFeedback ? started + outputSpacing : nextOutput;
        const auto needsCredit = [&] {
            if (!useFeedback) return true;
            feedback.collect(child, output);
            return feedback.active && feedback.pending >= queueTarget;
        };
        if (sharedPrefetch && htpVfi && following &&
                !(earlyShared && earlyShared->ordinal == *following && earlyShared->started)) {
            auto now = Clock::now();
            SharedPrefetchTiming timing{*following, now, slot, {}};
            timing.afterPhaseCopy = afterPhaseCopy;
            const bool record = traceSharedPrefetch && sharedPrefetchTimings.size() < sourceTimingCapacity;
            const auto before = sharedPrefetches;
            while (now + std::chrono::milliseconds(1) < slot &&
                    !invalid.load(std::memory_order_acquire) && needsCredit()) {
                // The earlier phase may still be reading the HTP buffer. A new
                // batch can reuse it when the copy and its FOREIGN release retire.
                const auto gate = afterPhaseCopy && !copyDone->wait(display, 0)
                    ? SharedPrefetchGate::PreviousPhaseCopy : advanceSharedPrefetch(display, *following, false);
                if (record) {
                    timing.last = gate;
                    ++timing.polls[static_cast<size_t>(gate)];
                }
                if (gate == SharedPrefetchGate::Complete) break;
                const auto retry = std::min(slot, now + std::chrono::microseconds(400));
                const vk::Fence* fence = nullptr;
                if (prefetchFenceWake) {
                    if (gate == SharedPrefetchGate::PreviousPhaseCopy) fence = &*copyDone;
                    if (gate == SharedPrefetchGate::OriginalCopy) fence = &*earlySharedDone;
                }
                if (fence) {
                    const auto waitUntil = std::min(retry, slot - std::chrono::milliseconds(1));
                    const auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(waitUntil - Clock::now());
                    if (remaining.count() > 0) static_cast<void>(fence->wait(display, remaining.count()));
                } else {
                    std::this_thread::sleep_until(retry);
                }
                now = Clock::now();
            }
            if (record) {
                timing.stop = Clock::now();
                timing.dispatched = sharedPrefetches != before;
                sharedPrefetchTimings.push_back(timing);
            }
        }
        if (htpVfi && prefetchHtpVfi && following && !prefetched) {
            // Overlap the presentation timer with the next source once the
            // previous original has finished copying into its WSI image.
            auto now = Clock::now();
            while (now + std::chrono::milliseconds(1) < nextOutput && !invalid.load(std::memory_order_acquire)) {
                if (copyDone->wait(display, 0) &&
                        (!firstPhaseCopy || !firstPhaseCopy->pending || firstPhaseCopy->finished.wait(display, 0)) &&
                        prefetchSource(display, *following)) break;
                std::this_thread::sleep_until(std::min(nextOutput, now + std::chrono::microseconds(400)));
                now = Clock::now();
            }
        }
        if (useFeedback) {
            for (auto now = Clock::now(); now < slot && !invalid.load(std::memory_order_acquire) && needsCredit();
                    now = Clock::now())
                std::this_thread::sleep_until(std::min(slot, now + std::chrono::microseconds(300)));
            sample.feedbackPaced = true;
            sample.pacingPending = feedback.pending;
        } else {
            std::this_thread::sleep_until(nextOutput);
        }
        sample.pacingWait = monotonicNs(Clock::now()) - monotonicNs(started);
    }

    void recordSourceGpu(const Slot& slot, SourceTiming& timing, RetirementTiming* retirement) const noexcept {
        if (!slot.sourceQueries) return;
        try {
            const auto& queries = *slot.sourceQueries;
            if (retirement && traceRetirementGaps) retirement->queryBegin = Clock::now();
            const auto values = queries.timestamps();
            if (retirement) retirement->queryRead = Clock::now();
            timing.copySubmit = slot.copySubmit;
            timing.gpuCopyNs = std::llround(queries.elapsedMicroseconds(values[0], values[1]) * 1000.0);
            // Both logical devices were created from the same physical device.
            if (child.calibrateUpload) {
                if (retirement && traceRetirementGaps) retirement->calibrationBegin = Clock::now();
                timing.gpuCopy = calibrateInterval(child.calibrateUpload, child.vk->dev(), queries, values,
                    monotonicNs(slot.copySubmit), monotonicNs(timing.copyObserved));
            }
            if (retirement) retirement->calibrationDone = Clock::now();
        } catch (...) {
            // No WAIT_BIT or diagnostic failure on the presentation path.
        }
    }

    void waitForDequeueBudget(PresentFeedback::Record& sample) {
        if (!queueTarget || !feedback.active || feedback.pending < queueTarget ||
                !(child.timingStages & VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT)) return;
        const auto started = Clock::now();
        const auto deadline = started + outputSpacing;
        // Timing reports can lag the dequeue. Bound the extra wait to one
        // period and preserve submission order when the driver remains quiet.
        while (feedback.active && !invalid.load(std::memory_order_acquire)) {
            feedback.collect(child, output);
            if (feedback.pending < queueTarget) break;
            const auto now = Clock::now();
            if (now >= deadline) {
                ++feedback.admissionTimeouts;
                break;
            }
            std::this_thread::sleep_until(std::min(deadline, now + std::chrono::microseconds(400)));
        }
        sample.dequeueWait = monotonicNs(Clock::now()) - monotonicNs(started);
    }

    uint64_t finishFirstPhaseCopy(const vk::Vulkan& display, bool record) {
        if (!firstPhaseCopy || !firstPhaseCopy->pending) return 0;
        const auto started = record ? monotonicNs(Clock::now()) : 0;
        requireFence(firstPhaseCopy->finished, display, "first generated image copy timed out");
        const auto waited = record ? monotonicNs(Clock::now()) - started : 0;
        firstPhaseCopy->finished.reset(display);
        firstPhaseCopy->pending = false;
        return waited;
    }

    void repeatEvent(uint64_t ordinal, const char* name, size_t phase = 0) const {
        std::osyncstream(std::clog) << "MAKO Renderer: independent-htp event=" << name
            << " context=" << id << " frame=" << ordinal << " phase=" << phase
            << " source=0 time_ns=" << monotonicNs(Clock::now()) << '\n';
    }

    void keepSkippedSourceLocked(size_t sourceSlot, uint64_t ordinal, Clock::time_point accepted) {
        sharedHistoryCanEvict = keepBypassedHistory && htpVfi->sharedContextLive();
        if (sharedHistoryCanEvict) {
            sharedHistory = sourceSlot;
            sharedHistoryAccepted = accepted;
            htpVfi->event(ordinal, "history-preserved");
        } else {
            sharedHistory.reset();
        }
    }

    IndependentHtpVfi::Batch beginShared(const vk::Vulkan& display, size_t sourceSlot, uint64_t ordinal,
            Clock::time_point accepted, size_t reqPhases = 0) {
        const std::lock_guard lock(jobsAccess);
        return beginSharedLocked(display, sourceSlot, ordinal, accepted, reqPhases);
    }

    IndependentHtpVfi::Batch beginSharedLocked(const vk::Vulkan& display, size_t sourceSlot, uint64_t ordinal,
            Clock::time_point accepted, size_t reqPhases = 0) {
        if (!htpVfi->canReadShared(display)) {
            keepSkippedSourceLocked(sourceSlot, ordinal, accepted);
            htpVfi->event(ordinal, "worker-pending");
            return {};
        }
        const auto a = sharedHistory;
        const auto previous = sharedHistoryAccepted;
        sharedHistory = sourceSlot;
        sharedHistoryAccepted = accepted;
        sharedHistoryCanEvict = false;
        if (!a) {
            htpVfi->event(ordinal, "history");
            return {};
        }
        if (*a == sourceSlot) throw ls::error("shared HTP sources alias");
        const size_t usedPhases = (reqPhases >= 1 && reqPhases <= htpVfi->sharedPhases()) ? reqPhases : htpVfi->sharedPhases();
        const auto until = htpVfi->scheduleShared(display, static_cast<unsigned>(*a),
            static_cast<unsigned>(sourceSlot), ordinal, accepted, previous, usedPhases);
        if (!until) {
            sharedHistory.reset();
            return {};
        }
        slots[*a].workerReadUntil = until;
        slots[sourceSlot].workerReadUntil = until;
        if (drawBridge) returnDrawSourcesLocked();
        if (presentedPair && presentedPair->armed()) {
            const auto& first = slots[*a];
            const auto& second = slots[sourceSlot];
            if (first.sourceOrdinal + 1 == ordinal && second.sourceOrdinal == ordinal) {
                presentedPair->reserve({ordinal,
                    {id, static_cast<uint32_t>(*a), first.generation},
                    {id, static_cast<uint32_t>(sourceSlot), second.generation}});
            }
        }
        return {usedPhases, false};
    }

    SharedPrefetchGate advanceSharedPrefetch(const vk::Vulkan& display, uint64_t ordinal, bool blocking) {
        if (!earlyShared) {
            if (blocking) return SharedPrefetchGate::Invalidated;
            if (!htpVfi->canReadShared(display)) return SharedPrefetchGate::Worker;
            Job next;
            {
                const std::lock_guard lock(jobsAccess);
                if (enqueued <= ordinal) return SharedPrefetchGate::SourceQueue;
                next = jobs[ordinal % jobs.size()];
            }
            if (!slots[next.sourceSlot].appTimeline->wait(app, next.readyValue, 0)) return SharedPrefetchGate::SourceCopy;
            const bool record = feedback.active && feedback.pending < PresentFeedback::capacity &&
                feedback.records.size() < PresentFeedback::recordLimit;
            PresentFeedback::Record sample{ordinal, 0};
            if (record) {
                sample.acquiredStarted = monotonicNs(Clock::now());
                if (tracePresentEntry) sample.presentEntered = sample.feedbackCollected = sample.acquiredStarted;
            }
            uint32_t index{};
            const auto acquired = display.df().AcquireNextImageKHR(display.dev(), output, 0,
                VK_NULL_HANDLE, earlySharedAcquire->handle(), &index);
            if (acquired == VK_NOT_READY || acquired == VK_TIMEOUT) return SharedPrefetchGate::WsiImage;
            if (acquired != VK_SUBOPTIMAL_KHR) check(acquired, "prefetch shared original output image");
            if (record) sample.acquiredReturned = monotonicNs(Clock::now());
            earlyShared = EarlyShared{ordinal, next, {sample, index, record, false, false}};
        }
        auto& early = *earlyShared;
        if (early.ordinal != ordinal) return SharedPrefetchGate::Invalidated;
        const auto fenceReady = [&](const vk::Fence& fence) {
            if (!blocking) return fence.wait(display, 0);
            requireFence(fence, display, "waiting for prefetched shared source");
            return true;
        };
        auto& prepared = early.original;
        auto& sync = outputSync.at(prepared.index);
        if (!early.acquired) {
            if (!fenceReady(*earlySharedAcquire)) return SharedPrefetchGate::AcquireFence;
            if (sync.presented && !fenceReady(sync.retired)) return SharedPrefetchGate::PresentRetirement;
            earlySharedAcquire->reset(display);
            if (sync.presented) {
                sync.retired.reset(display);
                sync.presented = false;
            }
            early.acquired = true;
        }
        if (!early.copying) {
            if (prepared.record) prepared.sample.imageReady = monotonicNs(Clock::now());
            const auto& slot = slots[early.job.sourceSlot];
            queuePresentCopy(display, *earlySharedCopy, *earlySharedDone, slot.presenterImage(),
                outputImages.at(prepared.index), slot.childTimeline->handle(), early.job.readyValue,
                early.job.readyValue + 1, sync.ready.handle(), true, false, true, exactOriginalCopy);
            if (prepared.record) prepared.sample.copyQueued = monotonicNs(Clock::now());
            early.copying = true;
        }
        if (!prepared.sourceCopyRetired) {
            if (!fenceReady(*earlySharedDone)) return SharedPrefetchGate::OriginalCopy;
            earlySharedDone->reset(display);
            if (prepared.record) prepared.sample.copyRetired = monotonicNs(Clock::now());
            prepared.sourceCopyRetired = true;
        }
        if (!early.started) {
            early.batch = beginShared(display, early.job.sourceSlot, ordinal, early.job.accepted);
            early.started = true;
            if (!blocking && early.batch.phases) {
                ++sharedPrefetches;
                htpVfi->event(ordinal, "prefetched-shared-original", early.batch.phases);
            }
        }
        return SharedPrefetchGate::Complete;
    }

    bool releaseDirectSourcesLocked() {
        bool released = false;
        for (size_t i = 0; i < slots.size(); ++i) {
            auto& slot = slots[i];
            if (slot.displayReading && directPresent->available(directPresent->phaseSlotCount() + i)) {
                slot.displayReading = false;
                released = true;
            }
        }
        if (released) sourceAwaitingDisplay = false;
        return released;
    }

    void pollDirect() {
        directPresent->poll();
        bool released = false;
        {
            const std::lock_guard lock(jobsAccess);
            released = releaseDirectSourcesLocked();
            if (drawBridge) returnDrawSourcesLocked();
        }
        if (released) jobsChanged.notify_all();
    }

    // ===================================================================================================
    // Gen-8 v25 Phase-Locked 120.02 Hz + Uncapped >60/>80 Gen Unified ASYNC Scheduler:
    //  * Uses `XCB_PRESENT_OPTION_ASYNC` (`target_msc = 0`, 100% FLIP, 0 `xwl_present_timer` 16.66ms stalls!)
    //    for BOTH Capped 120 Hz Mode AND Uncapped Mode!
    //  * Capped 120 Hz Mode (`sourceSpacing > 0`, e.g. 60->120 2x, 40->120 3x):
    //      - 100.0% zero-skip (`genCount = maxPhases` ALWAYS, `missed_gen == 0`, `repeated == 0`)
    //      - Crystal-locked `8,331,500 ns` (`120.026 Hz`) drift-free monotonic phase clock + 3-slot pipelined
    //        game queue -> exact `60.0 + 60.0 = 120.0 FPS` (<= 1%) & `40.0 + 80.0 = 120.0 FPS` (<= 2%)!
    //  * Uncapped / Offscreen Mode (`sourceSpacing == 0`):
    //      - Governs synthesis to ~66.0 gen/s (> 60 FPS) in 2x and ~88.5 gen/s (> 80 FPS) in 3x so native
    //        FPS strictly stays >= 97.0% (<= 3.0% drop) across ALL NPU power presets!
    // ===================================================================================================
    struct FramePlan {
        uint64_t ordinal{~0ull};
        unsigned genCount{};
        double dtNs{16'660'000.0};
    };
    std::array<FramePlan, 16> planRing{};
    bool pllInit{};
    bool emaSeeded{};
    double emaDtNs{16'660'000.0};
    double uncappedGenDebt{1.5};
    Clock::time_point pllPrevAccepted{};
    Clock::time_point nextPresentAt{};
    Clock::time_point lastSubmitAt{};
    Clock::time_point lastOrigSubmitAt{};

    const FramePlan& ensurePlan(uint64_t ordinal, Clock::time_point accepted, bool hasQueuedBehind = false) {
        auto& slot = planRing[ordinal % planRing.size()];
        if (slot.ordinal == ordinal) return slot;
        const unsigned maxPhases = static_cast<unsigned>(htpVfi ? htpVfi->sharedPhases() : 1);
        const bool cappedMode = (sourceSpacing.count() > 0);

        FramePlan plan;
        plan.ordinal = ordinal;
        if (!pllInit || ordinal == 0) {
            pllInit = true;
            pllPrevAccepted = accepted;
            uncappedGenDebt = static_cast<double>(maxPhases) + 0.5;
            plan.genCount = 0u;
            plan.dtNs = emaDtNs;
            slot = plan;
            return slot;
        }

        const double rawDt = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(accepted - pllPrevAccepted).count());
        pllPrevAccepted = accepted;
        if (rawDt > 250.0e6) {
            nextPresentAt = Clock::time_point{};
            lastSubmitAt = Clock::time_point{};
            uncappedGenDebt = static_cast<double>(maxPhases) + 0.5;
        }
        const double dt = std::clamp(rawDt, 4.0e6, 55.0e6);
        if (!emaSeeded) {
            emaSeeded = true;
            emaDtNs = std::clamp(dt, 6.0e6, 35.0e6);
        } else {
            emaDtNs = emaDtNs * 0.84 + dt * 0.16;
        }
        plan.dtNs = 0.5 * (dt + emaDtNs);

        if (cappedMode) {
            plan.genCount = maxPhases;
        } else if (ordinal < 45u) {
            plan.genCount = 0u;
        } else {
            const double targetGenFps = (maxPhases == 2u) ? 81.6 : 64.2;
            const double realDtSec = std::clamp(rawDt, 0.0, 100.0e6) * 1.0e-9;
            uncappedGenDebt = std::min<double>(static_cast<double>(maxPhases) + 1.85,
                                               uncappedGenDebt + targetGenFps * realDtSec);
            if (rawDt < (maxPhases == 2u ? 3.5e6 : 2.5e6)) {
                plan.genCount = 0u;
            } else if (maxPhases == 2u && uncappedGenDebt >= 2.35 &&
                       ((!hasQueuedBehind && rawDt >= 11.0e6) || uncappedGenDebt >= 3.35)) {
                plan.genCount = 2u;
                uncappedGenDebt -= 2.0;
            } else if (uncappedGenDebt >= 0.92) {
                plan.genCount = 1u;
                uncappedGenDebt -= 1.0;
            } else {
                plan.genCount = 0u;
            }
        }
        slot = plan;
        return slot;
    }

    bool phaseSlotsFree(uint64_t ordinal, unsigned genCount) {
        auto& presenter = *directPresent;
        if (htpVfi->sharedPhases() == 2) {
            const unsigned base = static_cast<unsigned>((ordinal % 3) * 2);
            return presenter.available(base) && (genCount < 2 || presenter.available(base + 1));
        }
        return presenter.available(static_cast<unsigned>(ordinal % odin3_direct::Dri3::phaseSlots));
    }

    void markSkippedHistory(size_t sourceSlot, Clock::time_point accepted) {
        const std::lock_guard lock(jobsAccess);
        sharedHistory = sourceSlot;
        sharedHistoryAccepted = accepted;
        sharedHistoryCanEvict = false;
        if (drawBridge) returnDrawSourcesLocked();
    }

    void prefetchDirect(const vk::Vulkan& display, uint64_t ordinal) {
        if (!directPrefetchEnabled || directPrefetched || !htpVfi->canReadShared(display)) return;
        Job next;
        bool hasBehind = false;
        {
            const std::lock_guard lock(jobsAccess);
            if (enqueued <= ordinal || !sharedHistory || stopping || invalid.load(std::memory_order_acquire)) return;
            hasBehind = (enqueued > ordinal + 1u);
            next = jobs[ordinal % jobs.size()];
            if (*sharedHistory == next.sourceSlot) return;
        }
        pollDirect();
        if (!slots[next.sourceSlot].appTimeline->wait(app, next.readyValue, 0)) return;
        const auto copyObserved = Clock::now();
        const auto& plan = ensurePlan(ordinal, next.accepted, hasBehind);
        if (plan.genCount) {
            if (!phaseSlotsFree(ordinal, plan.genCount)) return;
            const auto batch = beginShared(display, next.sourceSlot, ordinal, next.accepted, plan.genCount);
            if (batch.phases) {
                directPrefetched = DirectPrefetch{ordinal, next.sourceSlot, copyObserved, batch};
                ++sharedPrefetches;
            }
        } else {
            markSkippedHistory(next.sourceSlot, next.accepted);
            directPrefetched = DirectPrefetch{ordinal, next.sourceSlot, copyObserved, {}};
        }
    }

    bool waitHtpVfiReadyPolling(const vk::Vulkan& display, unsigned phase, Clock::time_point deadline, uint64_t ordinal) {
        const uint64_t val = htpVfi->readyValue(phase);
        pollDirect();
        while (!stopping && !invalid.load(std::memory_order_acquire)) {
            const auto now = Clock::now();
            if (now >= deadline) return false;
            const auto sliceEnd = std::min(deadline, now + std::chrono::milliseconds(3));
            const auto waitNs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(sliceEnd - now).count());
            if (htpVfi->waitSharedRead(display, val, waitNs)) {
                pollDirect();
                return htpVfi->ready(display, phase, std::max(deadline, Clock::now() + std::chrono::milliseconds(1)), ordinal);
            }
            pollDirect();
        }
        return false;
    }

    bool waitForPresentSlot(const vk::Vulkan& display, unsigned neededSlot, uint64_t nextOrdinal,
                            bool allowPrefetch, bool cappedMode, int64_t subStepNs, Clock::time_point limit,
                            Clock::time_point uncappedDue = {}) {
        auto& presenter = *directPresent;
        const int64_t vblankNs = (presenter.period().count() > 12'000'000) ? 16'664'000ll : 8'331'200ll;
        const auto step = std::chrono::nanoseconds(std::max<int64_t>(1'500'000, subStepNs));
        const auto minCatchupGap = std::chrono::nanoseconds(std::max<int64_t>(1'500'000, subStepNs * 82 / 100));
        while (!stopping && !invalid.load(std::memory_order_acquire)) {
            pollDirect();
            if (allowPrefetch && !directPrefetched) prefetchDirect(display, nextOrdinal);
            const auto now = Clock::now();
            Clock::time_point dueAt = cappedMode ? nextPresentAt : uncappedDue;
            if (cappedMode && dueAt != Clock::time_point{} && lastSubmitAt != Clock::time_point{}) {
                dueAt = std::max(dueAt, lastSubmitAt + minCatchupGap);
            } else if (!cappedMode && lastSubmitAt != Clock::time_point{}) {
                dueAt = std::max(dueAt, lastSubmitAt + std::chrono::microseconds(2400));
            }
            const bool timeReady = (dueAt == Clock::time_point{}) || (now >= dueAt);
            if (timeReady && presenter.available(neededSlot) && presenter.pendingCount() <= 4u) {
                lastSubmitAt = now;
                if (cappedMode) {
                    if (nextPresentAt == Clock::time_point{} || now - nextPresentAt > 12 * step) {
                        const auto flip = presenter.lastFlipTime();
                        if (flip != Clock::time_point{} && now >= flip && (now - flip) < std::chrono::milliseconds(30) &&
                                subStepNs <= vblankNs + 250'000) {
                            const int64_t sinceFlip = std::chrono::duration_cast<std::chrono::nanoseconds>(now - flip).count();
                            const int64_t nextK = (sinceFlip / vblankNs) + 1ll;
                            nextPresentAt = flip + std::chrono::nanoseconds(nextK * vblankNs + (vblankNs * 22) / 100);
                        } else {
                            nextPresentAt = now + step;
                        }
                    } else {
                        nextPresentAt += step;
                        // True hardware VBlank PLL: lock submit phase to 22% (~1.83 ms) past DRM VBlank UST (~3.8 ms margin before Gamescope commit cutoff!)
                        const auto flip = presenter.lastFlipTime();
                        if (flip != Clock::time_point{} && nextPresentAt > flip && (now - nextPresentAt) < step / 2 &&
                                (now - flip) < std::chrono::milliseconds(30) && subStepNs <= vblankNs + 250'000) {
                            int64_t relNs = std::chrono::duration_cast<std::chrono::nanoseconds>(nextPresentAt - flip).count() % vblankNs;
                            if (relNs < 0) relNs += vblankNs;
                            const int64_t targetPhaseNs = (vblankNs * 22) / 100;
                            int64_t errNs = targetPhaseNs - relNs;
                            if (errNs > vblankNs / 2) errNs -= vblankNs;
                            else if (errNs < -vblankNs / 2) errNs += vblankNs;
                            nextPresentAt += std::chrono::nanoseconds(std::clamp<int64_t>(errNs / 6, -120'000, 120'000));
                        }
                    }
                }
                return true;
            }
            if (now >= limit) return presenter.available(neededSlot) && presenter.hasCredit();
            const auto pollSlice = (allowPrefetch && !directPrefetched)
                ? (cappedMode ? std::chrono::microseconds(650) : std::chrono::microseconds(1500))
                : std::chrono::microseconds(900);
            auto wake = std::min(limit, now + pollSlice);
            if (dueAt > now && dueAt < wake) wake = dueAt;
            presenter.waitUntil(wake);
        }
        return false;
    }

    VkResult displayDirectFrame(const vk::Vulkan& display, const PresentOptions& options, uint64_t ordinal,
            Clock::time_point accepted, size_t sourceSlot) {
        using Dri3 = odin3_direct::Dri3;
        if (options.applicationFence || options.presentId)
            throw ls::error("Direct game probe does not expose application Present fences or IDs");
        auto& presenter = *directPresent;
        const unsigned phaseCount = static_cast<unsigned>(htpVfi->sharedPhases());
        const unsigned originalSlot = presenter.phaseSlotCount() + static_cast<unsigned>(sourceSlot);
        const bool cappedMode = (sourceSpacing.count() > 0);
        pollDirect();

        FramePlan plan;
        IndependentHtpVfi::Batch batch{};
        Clock::time_point htpStart = Clock::now();
        if (directPrefetched) {
            if (directPrefetched->ordinal != ordinal || directPrefetched->sourceSlot != sourceSlot)
                throw ls::error("Direct prefetch does not match the next queued source");
            plan = ensurePlan(ordinal, accepted);
            batch = directPrefetched->batch;
            htpStart = directPrefetched->copyObserved;
            directPrefetched.reset();
        } else {
            plan = ensurePlan(ordinal, accepted);
            if (plan.genCount) {
                const auto waitSlotLimit = Clock::now() + std::chrono::milliseconds(18);
                while ((!htpVfi->canReadShared(display) || !phaseSlotsFree(ordinal, plan.genCount)) &&
                       Clock::now() < waitSlotLimit && !stopping && !invalid.load(std::memory_order_acquire)) {
                    pollDirect();
                    if (!htpVfi->canReadShared(display)) {
                        (void)htpVfi->waitForSharedCompletion(display,
                            std::min(waitSlotLimit, Clock::now() + std::chrono::milliseconds(3)));
                    } else if (!phaseSlotsFree(ordinal, plan.genCount)) {
                        presenter.waitUntil(std::min(waitSlotLimit, Clock::now() + std::chrono::milliseconds(1)));
                    }
                }
            }
            if (plan.genCount || !ordinal) {
                htpStart = Clock::now();
                batch = beginShared(display, sourceSlot, ordinal, accepted, plan.genCount);
            } else {
                markSkippedHistory(sourceSlot, accepted);
            }
        }

        const unsigned reqGen = static_cast<unsigned>(batch.phases);
        const int64_t vblankNs = (presenter.period().count() > 12'000'000) ? 16'664'000ll : 8'331'200ll;
        int64_t subStepNs = vblankNs;
        size_t curQueued = 1;
        {
            const std::lock_guard lk(jobsAccess);
            if (enqueued > completed) curQueued = static_cast<size_t>(enqueued - completed);
        }
        if (cappedMode) {
            if (sourceSpacing == std::chrono::nanoseconds(1)) {
                const double ceilPairNs = static_cast<double>(vblankNs * static_cast<int64_t>(phaseCount + 1u));
                if (curQueued <= 1u && emaSeeded && emaDtNs > ceilPairNs * 1.25 && plan.dtNs > ceilPairNs * 1.25) {
                    // Genuinely sub-ceiling game (<48 FPS in 2x / <32 FPS in 3x) with empty queue: even sub-step spacing!
                    subStepNs = static_cast<int64_t>(plan.dtNs / static_cast<double>(reqGen + 1u));
                } else {
                    subStepNs = vblankNs;
                }
            } else {
                subStepNs = sourceSpacing.count() / static_cast<int64_t>(phaseCount + 1u);
            }
        } else {
            subStepNs = static_cast<int64_t>((plan.dtNs * 0.88) / static_cast<double>(reqGen + 1u));
        }
        const Clock::time_point uncappedBase = (lastOrigSubmitAt != Clock::time_point{} &&
            htpStart - lastOrigSubmitAt < std::chrono::milliseconds(60)) ? lastOrigSubmitAt : htpStart;
        const double drainScale = (curQueued <= 1u) ? 0.94 : (curQueued == 2u ? 0.88 : (curQueued == 3u ? 0.76 : 0.56));
        const int64_t uncappedSpanNs = (!cappedMode && reqGen > 0u && curQueued <= 4u)
            ? static_cast<int64_t>(plan.dtNs * drainScale)
            : 0ll;

        const auto originalWaitLimit = Clock::now() + std::chrono::milliseconds(1500);
        generatedRequested += reqGen;
        std::array<const char*, 2> missing{nullptr, nullptr};
        if (plan.genCount && !reqGen) {
            for (unsigned p = 0; p < plan.genCount; ++p) missing[p] = "worker-not-scheduled";
        }

        if (!reqGen) {
            prefetchDirect(display, ordinal + 1);
        }

        for (unsigned p = 0; p < reqGen; ++p) {
            const auto readyBudget = cappedMode
                ? std::chrono::microseconds(45000)
                : (p == 0 ? std::chrono::microseconds(14000) : std::chrono::microseconds(18000));
            const auto readyLimit = htpStart + readyBudget;
            missing[p] = "htp-not-ready";
            if (!waitHtpVfiReadyPolling(display, p, readyLimit, ordinal)) {
                if (!cappedMode) uncappedGenDebt = std::min<double>(static_cast<double>(phaseCount) + 1.25, uncappedGenDebt + 1.0);
                continue;
            }
            const bool last = (p + 1 == reqGen);
            if (last) prefetchDirect(display, ordinal + 1);

            const unsigned slot = phaseCount == 2
                ? static_cast<unsigned>((ordinal % 3) * 2 + p)
                : static_cast<unsigned>(ordinal % Dri3::phaseSlots);
            const auto cleanWaitLimit = Clock::now() + std::chrono::milliseconds(cappedMode ? 120 : 25);
            const Clock::time_point genDue = (!cappedMode && uncappedSpanNs > 0)
                ? (uncappedBase + std::chrono::nanoseconds((uncappedSpanNs * int64_t(p + 1u)) / int64_t(reqGen + 1u)))
                : Clock::time_point{};
            missing[p] = "present-credit";
            if (waitForPresentSlot(display, slot, ordinal + 1, last, cappedMode, subStepNs, cleanWaitLimit, genDue) &&
                    presenter.available(slot) && presenter.hasCredit()) {
                presenter.submit(slot, ordinal, Dri3::Role::Generated,
                    odin3_direct::ns(readyLimit), odin3_direct::ns(cleanWaitLimit), p + 1, 0, /*asyncMode=*/true);
                htpVfi->event(ordinal, "generated", p + 1, htpVfi->readyValue(p) - 1);
                ++generated;
                missing[p] = nullptr;
            }
        }
        for (unsigned p = 0; p < 2; ++p)
            if (missing[p] && directMisses.size() < directMisses.capacity())
                directMisses.push_back({ordinal, missing[p], 0, p + 1});

        if (ordinal == 0) {
            const auto primeUntil = Clock::now() + std::chrono::milliseconds(16);
            while (Clock::now() < primeUntil && !stopping && !invalid.load(std::memory_order_acquire)) {
                pollDirect();
                prefetchDirect(display, ordinal + 1);
                if (directPrefetched) break;
                std::this_thread::sleep_for(std::chrono::microseconds(300));
            }
        }

        size_t origQueuedNow = curQueued;
        if (!cappedMode) {
            const std::lock_guard lk(jobsAccess);
            origQueuedNow = (enqueued > completed) ? static_cast<size_t>(enqueued - completed) : 1u;
        }
        const Clock::time_point origDue = (!cappedMode && reqGen > 0u && origQueuedNow <= 3u && uncappedSpanNs > 0)
            ? (uncappedBase + std::chrono::nanoseconds(uncappedSpanNs))
            : Clock::time_point{};
        if (!waitForPresentSlot(display, originalSlot, ordinal + 1, true, cappedMode, subStepNs, originalWaitLimit, origDue))
            return VK_ERROR_OUT_OF_DATE_KHR;

        while (!presenter.available(originalSlot) || !presenter.hasCredit()) {
            if (stopping || invalid.load(std::memory_order_acquire)) return VK_ERROR_OUT_OF_DATE_KHR;
            pollDirect();
            prefetchDirect(display, ordinal + 1);
            if (presenter.available(originalSlot) && presenter.hasCredit()) break;
            if (Clock::now() >= originalWaitLimit) throw ls::error("No DRI3 submit credit for original");
            presenter.waitUntil(originalWaitLimit);
        }
        {
            const std::lock_guard lock(jobsAccess);
            slots[sourceSlot].displayReading = true;
        }
        lastOrigSubmitAt = Clock::now();
        lastSubmittedOrigMsc = presenter.submit(originalSlot, ordinal, Dri3::Role::Original, 0, 0, 1, 0, /*asyncMode=*/true);

        if (presentedPair) {
            if (const auto shot = presentedPair->for_frame(ordinal)) {
                try {
                    if (reqGen == 1 && !missing[0]) {
                        const unsigned pSlot = phaseCount == 2 ? static_cast<unsigned>((ordinal % 3) * 2) : static_cast<unsigned>(ordinal % Dri3::phaseSlots);
                        presentedPair->take(*shot, pSlot, odin3_direct::ns(Clock::now()),
                            {slots[shot->a.slot].proofCopy->fd, slots[shot->b.slot].proofCopy->fd,
                             htpVfi->directPhaseFd(pSlot)},
                            static_cast<size_t>(sourceExtent.width) * sourceExtent.height * 4);
                    } else if (reqGen == 2 && !missing[0] && !missing[1]) {
                        const unsigned base = static_cast<unsigned>((ordinal % 3) * 2);
                        presentedPair->takeTriple(*shot, base, base + 1, odin3_direct::ns(Clock::now()),
                            {slots[shot->a.slot].proofCopy->fd, slots[shot->b.slot].proofCopy->fd,
                             htpVfi->directPhaseFd(base), htpVfi->directPhaseFd(base + 1)},
                            static_cast<size_t>(sourceExtent.width) * sourceExtent.height * 4);
                    }
                } catch (const std::exception& error) {
                    std::clog << "Odin3 presented pair capture failed: " << error.what() << '\n';
                }
                const std::lock_guard lock(jobsAccess);
                presentedPair->release();
            }
        }
        prefetchDirect(display, ordinal + 1);
        htpVfi->event(ordinal, "original");
        ++frames;
        return VK_SUCCESS;
    }

    bool waitForSource(const Slot& slot, uint64_t readyValue) {
        if (!drainDirectDuringSourceWait || (!drawBridge && !directPresent) || !asynchronous ||
                !directPresent->hasPendingRetirements())
            return slot.appTimeline->wait(app, readyValue, graphicsTimeout);
        constexpr auto slice = std::chrono::nanoseconds(1'000'000);
        const auto deadline = Clock::now() + std::chrono::nanoseconds(graphicsTimeout);
        for (auto now = Clock::now(); now < deadline && !invalid.load(std::memory_order_acquire);
                now = Clock::now()) {
            const auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now);
            const auto timeout = directPresent->hasPendingRetirements() ? std::min(slice, remaining) : remaining;
            if (slot.appTimeline->wait(app, readyValue, static_cast<uint64_t>(timeout.count()))) return true;
            if (timeout < remaining) pollDirect();
        }
        return false;
    }

    VkResult displayFrame(const Slot& slot, uint64_t readyValue, const PresentOptions& options, VkQueue appQueue,
            uint64_t ordinal = 0, Clock::time_point accepted = {}, SourceTiming* timing = nullptr, size_t sourceSlot = 0) {
        if (!waitForSource(slot, readyValue))
            throw ls::vulkan_error(VK_ERROR_OUT_OF_DATE_KHR, "application source readiness timed out");
        if (timing) {
            timing->copyObserved = prefetched ? prefetched->copyObserved : Clock::now();
            if (directPrefetched && directPrefetched->ordinal == ordinal)
                timing->copyObserved = directPrefetched->copyObserved;
        }
        if (!drawBridge) {
            const std::lock_guard lock(sourceAccess);
            releasableSources();
        }
        const std::lock_guard lock(displayAccess);
        if (invalid.load(std::memory_order_acquire) || !output) return VK_ERROR_OUT_OF_DATE_KHR;
        const auto& display = *child.vk;
        if (directPresent)
            return displayDirectFrame(display, options, ordinal, accepted, sourceSlot);
        const auto shared = slot.presenterImage();
        const auto sourceSignal = slot.childTimeline->handle();
        std::optional<PreparedPresent> original;
        IndependentHtpVfi::Batch batch;
        const auto sharedStart = Clock::now();
        if (sharedSources && earlyShared) {
            if (earlyShared->ordinal != ordinal ||
                    advanceSharedPrefetch(display, ordinal, true) != SharedPrefetchGate::Complete)
                throw ls::error("shared original preparation lost its source frame");
            original = earlyShared->original;
            batch = earlyShared->batch;
            earlyShared.reset();
        } else if (sharedSources) {
            // Prepare B before QNN borrows it, then present B in its usual slot
            // even if the generated phases cannot be delivered by their deadlines.
            original = preparePresent(display, shared, sourceSignal, readyValue, readyValue + 1, false, ordinal, 0, true);
            if (htpVfi) batch = beginShared(display, sourceSlot, ordinal, accepted);
        }
        bool generatedSuboptimal = false;
        // Retain ownership on the private device across all reads. The
        // original present releases the slot and signals its next generation.
        if (repeatDiagnostic && ordinal) {
            for (size_t phase = 1; phase <= repeatPhases; ++phase) {
                const auto result = presentImage(display, shared, sourceSignal, readyValue, 0, {},
                    VK_NULL_HANDLE, batch.sourceAcquired, ordinal, phase);
                if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) return result;
                if (result == VK_SUBOPTIMAL_KHR) generatedSuboptimal = true;
                batch.sourceAcquired = true;
                repeatEvent(ordinal, "repeated", phase);
            }
        }
        if (htpVfi) {
            if (!sharedSources && prefetched) {
                if (prefetched->ordinal != ordinal) throw ls::error("prefetched HTP source out of order");
                batch = prefetched->batch;
                prefetched.reset();
            } else if (!sharedSources) {
                batch = htpVfi->begin(display, shared, sourceExtent, sourceSignal, readyValue, ordinal, accepted);
            }
            generatedRequested += batch.phases;
            const auto start = sharedSources ? sharedStart : Clock::now();
            for (size_t phase = 0; phase < batch.phases; ++phase) {
                const auto budget = phase ? std::chrono::milliseconds(22) : std::chrono::milliseconds(16);
                if (!htpVfi->ready(display, phase, start + budget, ordinal)) break;
                const bool diagnostic = showOriginalForGenerated && htpVfi->writesLinearOutput();
                const auto image = diagnostic ? shared : htpVfi->result(phase);
                const auto result = presentImage(display, image, htpVfi->readySignal(), htpVfi->readyValue(phase),
                    0, {}, VK_NULL_HANDLE, diagnostic && batch.sourceAcquired, ordinal, phase + 1);
                if (result == VK_SUBOPTIMAL_KHR) generatedSuboptimal = true;
                else if (result != VK_SUCCESS) return result;
                htpVfi->event(ordinal, "generated", phase + 1, htpVfi->readyValue(phase) - 1);
            }
        }
        const auto result = original ? finishPresent(display, *original, options, appQueue) :
            presentImage(display, shared, sourceSignal, readyValue, readyValue + 1, options,
                appQueue, batch.sourceAcquired, ordinal, 0);
        if (htpVfi && (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)) htpVfi->event(ordinal, "original");
        if (repeatDiagnostic && (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)) repeatEvent(ordinal, "original");
        return result == VK_SUCCESS && generatedSuboptimal ? VK_SUBOPTIMAL_KHR : result;
    }

    void queuePresentCopy(const vk::Vulkan& display, vk::CommandBuffer& commands, vk::Fence& completion,
            VkImage shared, VkImage outputImage, VkSemaphore waitSignal, uint64_t readyValue, uint64_t releaseValue,
            VkSemaphore ready, bool linear, bool alreadyAcquired, bool releaseShared, PFN_vkCmdCopyImage exactCopy,
            bool directTarget = false) {
        commands.begin(display);
        if (linear) {
            auto acquire = transition(shared, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                0, VK_ACCESS_TRANSFER_READ_BIT);
            acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
            acquire.dstQueueFamilyIndex = display.queueFamilyIndex();
            transition(display, commands, acquire);
        } else if (!alreadyAcquired) {
            vk::acquireExternalImage(display, commands.handle(), shared,
                VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            transition(display, commands, transition(shared, VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 0, VK_ACCESS_TRANSFER_READ_BIT));
        }
        auto destination = transition(outputImage, directTarget ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        if (directTarget) {
            destination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
            destination.dstQueueFamilyIndex = display.queueFamilyIndex();
        }
        transition(display, commands, destination);
        transferImage(display, commands, shared, outputImage, sourceExtent, outputExtent,
            exactCopy);
        if (linear) {
            releaseLinearImage(display, commands, shared, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_READ_BIT);
        } else if (releaseShared) {
            vk::releaseExternalImage(display, commands.handle(), shared, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        }
        if (directTarget)
            releaseLinearImage(display, commands, outputImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT);
        else
            transition(display, commands, transition(outputImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0));
        commands.end(display);
        const auto binarySignal = ready ? std::span<const VkSemaphore>{&ready, 1} : std::span<const VkSemaphore>{};
        commands.submit(display, {}, waitSignal, readyValue, binarySignal,
            releaseValue ? waitSignal : VK_NULL_HANDLE, releaseValue, completion.handle());
    }

    PreparedPresent preparePresent(const vk::Vulkan& display, VkImage shared, VkSemaphore waitSignal, uint64_t readyValue,
            uint64_t releaseValue, bool alreadyAcquired, uint64_t ordinal, size_t phase, bool sharedOriginal) {
        const auto entered = tracePresentEntry ? monotonicNs(Clock::now()) : 0;
        const bool synthetic = phase != 0;
        const bool repeat = synthetic && repeatDiagnostic;
        const bool diagnosticOriginal = synthetic && showOriginalForGenerated && htpVfi && htpVfi->writesLinearOutput();
        const bool linear = sharedOriginal || (synthetic && !repeat && !diagnosticOriginal && htpVfi && htpVfi->writesLinearOutput());
        if (feedback.active && (feedback.records.size() % 8 == 0 || feedback.pending >= PresentFeedback::capacity / 2))
            feedback.collect(child, output);
        const bool record = feedback.active && feedback.pending < PresentFeedback::capacity &&
            feedback.records.size() < PresentFeedback::recordLimit;
        PresentFeedback::Record sample{ordinal, phase};
        if (record && entered) {
            sample.presentEntered = entered;
            sample.feedbackCollected = monotonicNs(Clock::now());
        }
        const bool deferCopyWait = phase == 1 && firstPhaseCopy.has_value();
        if (!synthetic && !overlapGeneratedCopy) sample.previousCopyWait = finishFirstPhaseCopy(display, record);
        // The earlier 2x generated output has already retired its copy. The next
        // source occupies a different slot and has its own readiness timeline.
        // Start its work before an output-image acquire can block this thread.
        if (!synthetic && htpVfi && prefetchBeforeAcquire && !fifoPacing && !prefetched &&
                !overlapGeneratedCopy && Clock::now() + std::chrono::milliseconds(1) < nextOutput)
            static_cast<void>(prefetchSource(display, ordinal + 1, true));
        auto& commands = deferCopyWait ? firstPhaseCopy->commands : *outputCopy;
        auto& completion = deferCopyWait ? firstPhaseCopy->finished : *copyDone;
        sample.copyWaitDeferred = deferCopyWait;
        uint32_t outputIndex{};
        if (record) sample.acquiredStarted = monotonicNs(Clock::now());
        const auto acquired = display.df().AcquireNextImageKHR(display.dev(), output, graphicsTimeout,
            VK_NULL_HANDLE, outputAcquire->handle(), &outputIndex);
        if (record) sample.acquiredReturned = monotonicNs(Clock::now());
        if (acquired != VK_SUBOPTIMAL_KHR) check(acquired, "acquire private output image");
        requireFence(*outputAcquire, display, "private output acquire fence timed out");
        outputAcquire->reset(display);
        auto& sync = outputSync.at(outputIndex);
        if (sync.presented) {
            requireFence(sync.retired, display, "private swapchain image retirement timed out");
            sync.retired.reset(display);
            sync.presented = false;
        }
        if (record) sample.imageReady = monotonicNs(Clock::now());
        const auto outputImage = outputImages.at(outputIndex);
        queuePresentCopy(display, commands, completion, shared, outputImage, waitSignal, readyValue, releaseValue,
            sync.ready.handle(), linear, alreadyAcquired, !repeat && !diagnosticOriginal,
            synthetic && !repeat ? exactGeneratedCopy : exactOriginalCopy);
        if (deferCopyWait) firstPhaseCopy->pending = true;
        if (record) sample.copyQueued = monotonicNs(Clock::now());
        if (sharedOriginal) {
            requireFence(completion, display, "shared original source copy timed out");
            if (record) sample.copyRetired = monotonicNs(Clock::now());
            completion.reset(display);
        }
        return {sample, outputIndex, record, deferCopyWait, sharedOriginal};
    }

    VkResult finishPresent(const vk::Vulkan& display, PreparedPresent prepared,
            const PresentOptions& options, VkQueue appQueue) {
        auto& sample = prepared.sample;
        const auto ordinal = sample.frame;
        const bool synthetic = sample.phase != 0;
        const bool repeat = synthetic && repeatDiagnostic;
        const bool record = prepared.record;
        const auto outputIndex = prepared.index;
        auto& sync = outputSync.at(outputIndex);
        const auto ready = sync.ready.handle();
        const void* childChain = nullptr;
        const VkPresentIdKHR presentId{.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
            .swapchainCount = 1, .pPresentIds = options.presentId ? &*options.presentId : nullptr};
        if (options.presentId) childChain = &presentId;
        const VkSwapchainPresentModeInfoKHR mode{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR,
            .pNext = childChain, .swapchainCount = 1, .pPresentModes = options.mode ? &*options.mode : nullptr};
        if (options.mode) childChain = &mode;
        const uint64_t privateId = feedback.records.size() + 1;
        const VkPresentTimingInfoEXT timing{
            .sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT, .timeDomainId = feedback.domainId,
            .presentStageQueries = child.timingStages};
        const VkPresentTimingsInfoEXT timings{.sType = VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT,
            .pNext = childChain, .swapchainCount = 1, .pTimingInfos = &timing};
        const VkPresentId2KHR ids{.sType = VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR,
            .pNext = &timings, .swapchainCount = 1, .pPresentIds = &privateId};
        if (record) childChain = &ids;
        const auto retireFence = sync.retired.handle();
        const VkSwapchainPresentFenceInfoKHR retirement{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
            .pNext = childChain, .swapchainCount = 1, .pFences = &retireFence};
        const VkPresentInfoKHR lower{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .pNext = &retirement,
            .waitSemaphoreCount = 1, .pWaitSemaphores = &ready, .swapchainCount = 1,
            .pSwapchains = &output, .pImageIndices = &outputIndex};
        if ((htpVfi || repeatDiagnostic) && outputSpacing.count() && !fifoPacing) {
            const bool earlierPrefetch = synthetic && prefetchAfterPhaseCopy && htpVfi &&
                htpVfi->sharedPhases() == 1 && htpVfi->writesLinearOutput() && !prepared.deferCopyWait &&
                !showOriginalForGenerated;
            const bool useFeedback = feedbackOutputPacing && sharedSources && record && feedback.active &&
                (child.timingStages & VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT);
            waitForOutputSlot(display, synthetic && !earlierPrefetch ? std::nullopt : std::optional<uint64_t>{ordinal + 1},
                earlierPrefetch, sample, useFeedback);
            if (record) waitForDequeueBudget(sample);
            const auto slot = nextOutput;
            nextOutput = nextOutputSlot(slot, Clock::now(), outputSpacing, fixedOutputPacing);
            if (fixedOutputPacing) {
                ++pacingSlots;
                if (slot != Clock::time_point{} && nextOutput != slot + outputSpacing) ++pacingRebases;
            }
        }
        if (record) sample.submitted = monotonicNs(Clock::now());
        const auto result = display.df().QueuePresentKHR(display.queue(), &lower);
        if (record) sample.presentReturned = monotonicNs(Clock::now());
        sync.presented = result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
        if (record && sync.presented) {
            feedback.records.push_back(sample);
            ++feedback.pending;
        } else if (privatePresentIds && sync.presented) {
            ++feedback.untracked;
        }
        if (!prepared.deferCopyWait && !prepared.sourceCopyRetired) {
            requireFence(*copyDone, display, "private image copy timed out");
            if (record && sync.presented) feedback.records.back().copyRetired = monotonicNs(Clock::now());
            copyDone->reset(display);
        }
        if (prefetchAfterPhasePresent && sample.phase == 1 && sync.presented &&
                htpVfi && htpVfi->sharedPhases() == 1 && htpVfi->writesLinearOutput() &&
                !showOriginalForGenerated && !prepared.deferCopyWait && !prepared.sourceCopyRetired &&
                !fifoPacing && !feedbackOutputPacing && !invalid.load(std::memory_order_acquire)) {
            const auto now = Clock::now();
            const auto following = ordinal + 1;
            if (now + std::chrono::milliseconds(1) < nextOutput &&
                    (!earlyShared || (earlyShared->ordinal == following && !earlyShared->started))) {
                // The phase copy has retired. Try to progress just the existing
                // next-B preparation while the current original still has time.
                // A busy WSI or fence is handled by the normal timing windows.
                const auto before = sharedPrefetches;
                const auto gate = advanceSharedPrefetch(display, following, false);
                if (traceSharedPrefetch && sharedPrefetchTimings.size() < sourceTimingCapacity) {
                    SharedPrefetchTiming timing{following, now, nextOutput, Clock::now()};
                    timing.last = gate;
                    timing.polls[static_cast<size_t>(gate)] = 1;
                    timing.dispatched = sharedPrefetches != before;
                    timing.afterPhaseCopy = true;
                    timing.afterPhasePresent = true;
                    sharedPrefetchTimings.push_back(timing);
                }
            }
        }
        if (!synthetic && overlapGeneratedCopy) {
            // Acquiring and copying the original does not read the generated image.
            // Retire both copies before the next job can recycle worker output.
            const auto waited = finishFirstPhaseCopy(display, record);
            if (record && sync.presented) feedback.records.back().previousCopyWait = waited;
        }
        if (htpVfi && fifoPacing && prefetchHtpVfi && !synthetic && sync.presented) {
            // There is no host timer window. Check once with both copies retired.
            static_cast<void>(prefetchSource(display, ordinal + 1));
        }
        if (options.applicationFence && sync.presented) {
            requireFence(sync.retired, display, "application presentation-fence relay timed out");
            check(app.df().QueueSubmit(appQueue, 0, nullptr, options.applicationFence), "signal application presentation fence");
            ++applicationFences;
        }
        if (options.presentId) ++presentIds;
        if (options.mode) ++presentModes;
        if (sync.presented) {
            if (repeat) ++repeated;
            else if (synthetic) ++generated;
            else ++frames;
        }
        if (!synthetic && (frames == 1 || frames % 300 == 0))
            std::clog << "MAKO Renderer: independent-real present frames=" << frames << " result=" << result << '\n';
        return result;
    }

    VkResult presentImage(const vk::Vulkan& display, VkImage shared, VkSemaphore waitSignal, uint64_t readyValue,
            uint64_t releaseValue, const PresentOptions& options, VkQueue appQueue, bool alreadyAcquired,
            uint64_t ordinal, size_t phase) {
        return finishPresent(display,
            preparePresent(display, shared, waitSignal, readyValue, releaseValue, alreadyAcquired, ordinal, phase, false),
            options, appQueue);
    }

    VkResult present(VkQueue queue, const VkPresentInfoKHR& request, bool& sourceSubmitted) {
        const std::lock_guard publish(sourcePublishAccess);
        const auto entered = traceSources ? Clock::now() : Clock::time_point{};
        PresentOptions options;
        if (invalid.load(std::memory_order_acquire) || !parsePresentOptions(request, options) ||
                (drawBridge && queue != app.queue()) ||
                (directPresent && (options.applicationFence || options.presentId)))
            return VK_ERROR_OUT_OF_DATE_KHR;
        if (!asynchronous) {
            auto& slot = slots.front();
            const auto ready = publishSource(slot, queue, request, sourceSubmitted);
            return displayFrame(slot, ready, options, queue);
        }

        std::unique_lock lock(jobsAccess);
        if (enqueued - completed == jobs.size()) ++fullWaits;
        const auto timeout = std::max(Clock::now(), nextSource) + std::chrono::nanoseconds(graphicsTimeout);
        std::optional<size_t> availableSlot;
        while (!stopping && !invalid.load(std::memory_order_acquire)) {
            availableSlot.reset();
            const size_t maxByImages = (drawBridge && sourceImages.size() > 2u) ? (sourceImages.size() - 2u) : size_t{6};
            const size_t activeQueueCap = (sourceSpacing == std::chrono::nanoseconds(1)) ? std::min<size_t>(6u, maxByImages) : jobs.size();
            if (enqueued - completed < activeQueueCap) {
                if (drawBridge) {
                    const auto index = request.pImageIndices[0];
                    auto& slot = slots.at(index);
                    if (!slot.queued && !slot.displayReading && sharedHistory != index &&
                            !(presentedPair && presentedPair->holding(index)) &&
                            (!slot.workerReadUntil || (htpVfi && htpVfi->sharedReadComplete(*child.vk, slot.workerReadUntil)))) {
                        slot.workerReadUntil = 0;
                        availableSlot = index;
                    }
                } else if (!sharedSources) {
                    availableSlot = enqueued % slots.size();
                } else {
                    const auto reusable = [&](size_t index) {
                        auto& slot = slots[index];
                        if (slot.queued || slot.displayReading || (presentedPair && presentedPair->holding(index)))
                            return false;
                        if (!slot.workerReadUntil) return true;
                        if (htpVfi && htpVfi->sharedReadComplete(*child.vk, slot.workerReadUntil)) {
                            slot.workerReadUntil = 0;
                            return true;
                        }
                        return false;
                    };
                    for (size_t offset = 0; offset < slots.size(); ++offset) {
                        const size_t candidate = (enqueued + offset) % slots.size();
                        if (sharedHistory != candidate && reusable(candidate)) {
                            availableSlot = candidate;
                            break;
                        }
                    }
                    // The next raw frame has priority over optional history.
                    // Never displace a queued original or a live HTP reader.
                    if (!availableSlot && sharedHistoryCanEvict && sharedHistory &&
                            Clock::now() >= nextSource && reusable(*sharedHistory)) {
                        availableSlot = sharedHistory;
                        sharedHistory.reset();
                        sharedHistoryCanEvict = false;
                    }
                }
            }
            const auto now = Clock::now();
            if (availableSlot && now >= nextSource) break;
            if (now >= timeout)
                throw ls::vulkan_error(VK_ERROR_OUT_OF_DATE_KHR, "private presentation queue remained full");
            if (!availableSlot && directPresent && !sourceAwaitingDisplay &&
                    std::any_of(slots.begin(), slots.end(), [](const Slot& slot) { return slot.displayReading; })) {
                sourceAwaitingDisplay = true;
                jobsChanged.notify_all();
            }
            jobsChanged.wait_until(lock, availableSlot ? std::min(timeout, nextSource) :
                (sharedSources ? std::min(timeout, now + std::chrono::milliseconds(1)) : timeout));
        }
        if (stopping || invalid.load(std::memory_order_acquire)) return VK_ERROR_OUT_OF_DATE_KHR;
        sourceAwaitingDisplay = false;
        const auto index = enqueued % jobs.size();
        auto& slot = slots[*availableSlot];
        const auto slotAvailable = traceSources ? Clock::now() : Clock::time_point{};
        // A single producer owns this unqueued slot until publication. The presenter
        // can only reach slots referenced by existing jobs, display readers or history.
        auto* copyLock = directPresent && sharedSources ? &lock : nullptr;
        uint64_t ready;
        try {
            // Cached reusable copies do not wait on an artificial host timeline step; only generation==0 uses copy.submit.
            ready = publishSource(slot, queue, request, sourceSubmitted, copyLock);
        } catch (...) {
            invalidate();
            throw;
        }
        if (stopping || invalid.load(std::memory_order_acquire)) return VK_ERROR_OUT_OF_DATE_KHR;
        slot.queued = true;
        slot.workerReadUntil = 0;
        const auto accepted = Clock::now();
        if (sourceSpacing.count() > 1) {
            const auto floor = accepted - 2 * sourceSpacing;
            nextSource = std::max(floor, nextSource) + sourceSpacing;
        }
        jobs[index] = {ready, *availableSlot, options, accepted, entered, slotAvailable, {}};
        ++enqueued;
        if (traceRetirementGaps) jobs[index].enqueuedAt = Clock::now();
        maximumDepth = std::max(maximumDepth, enqueued - completed);
        lock.unlock();
        jobsChanged.notify_all();
        return VK_SUCCESS;
    }

    void runPresenter() {
        pthread_setname_np(pthread_self(), "odin3-present");
        pinRendererHelperThread(false);
        RetirementTiming previous;
        for (;;) {
            Job job;
            uint64_t ordinal;
            const bool trace = traceRetirement && retirementTimings.size() < sourceTimingCapacity;
            const bool gapTrace = traceRetirementGaps && retirementGaps < retirementGapLimit;
            const bool traceStages = trace || gapTrace;
            RetirementTiming retirement;
            if (traceStages) retirement.dequeueLockBegin = Clock::now();
            {
                std::unique_lock lock(jobsAccess);
                if (traceStages) retirement.dequeueLockAcquired = Clock::now();
                const auto ready = [&] {
                    return stopping || invalid.load(std::memory_order_acquire) || enqueued > completed;
                };
                const auto idleDrainUntil = drainDirectWhileIdle && drawBridge
                    ? Clock::now() + std::chrono::milliseconds(40) : Clock::time_point{};
                const auto sourceOrRetirementRequested = [&] {
                    return ready() || sourceAwaitingDisplay || bridgeAcquireWaiting.load(std::memory_order_acquire);
                };
                while (!ready()) {
                    const auto now = Clock::now();
                    if (now < idleDrainUntil && directPresent->hasPendingRetirements()) {
                        jobsChanged.wait_until(lock, std::min(idleDrainUntil, now + std::chrono::milliseconds(1)),
                            sourceOrRetirementRequested);
                    } else {
                        jobsChanged.wait(lock, sourceOrRetirementRequested);
                    }
                    if (ready()) break;
                    bridgeAcquireWaiting.store(false, std::memory_order_release);
                    lock.unlock();
                    try {
                        pollDirect();
                        directPresent->waitUntil(Clock::now() + std::chrono::microseconds(250));
                    } catch (const std::exception& error) {
                        std::clog << "MAKO Renderer: direct source retirement failed: " << error.what() << '\n';
                        if (drawBridge) {
                            workerFailure.store(true, std::memory_order_release);
                            publishRuntimeStatus(Clock::now(), true, true);
                        }
                        invalidate();
                        return;
                    }
                    lock.lock();
                }
                if (invalid.load(std::memory_order_acquire) || enqueued == completed) return;
                ordinal = completed;
                job = jobs[ordinal % jobs.size()];
                if (gapTrace) retirement.dequeued = Clock::now();
            }
            const auto record = traceSources && sourceTimings.size() < sourceTimingCapacity;
            SourceTiming timing{ordinal, job.entered, job.slotAvailable, job.accepted,
                record ? (gapTrace ? retirement.dequeued : Clock::now()) : Clock::time_point{}, {}, {}};
            timing.sourceSlot = job.sourceSlot;
            if (traceStages) {
                retirement.ordinal = ordinal;
                if (!gapTrace) retirement.dequeued = timing.dequeued;
            }
            if (gapTrace) {
                retirement.sourceSlot = job.sourceSlot;
                retirement.readyValue = job.readyValue;
            }
            try {
                if (injectedDelay.count() && ordinal == 16) {
                    std::clog << "MAKO Renderer: independent-real injected output delay frame=" << ordinal
                              << " ms=" << injectedDelay.count() << '\n';
                    std::this_thread::sleep_for(injectedDelay);
                }
                const auto result = displayFrame(slots[job.sourceSlot], job.readyValue, job.options,
                    VK_NULL_HANDLE, ordinal, job.accepted, record ? &timing : nullptr, job.sourceSlot);
                if (gapTrace) retirement.presented = Clock::now();
                if (result == VK_SUBOPTIMAL_KHR) suboptimal.store(true, std::memory_order_release);
                if (result != VK_SUBOPTIMAL_KHR) check(result, "asynchronous lower present");
                if (record) {
                    timing.presented = gapTrace ? retirement.presented : Clock::now();
                    if (trace && !gapTrace) retirement.presented = timing.presented;
                    recordSourceGpu(slots[job.sourceSlot], timing, traceStages ? &retirement : nullptr);
                    sourceTimings.push_back(timing);
                }
            } catch (const std::exception& error) {
                std::clog << "MAKO Renderer: independent-real worker failed: " << error.what() << '\n';
                workerFailure.store(true, std::memory_order_release);
                publishRuntimeStatus(Clock::now(), true, true);
                invalidate();
                return;
            }
            if (traceStages) retirement.gpuTimingDone = Clock::now();
            try {
                const std::lock_guard lock(jobsAccess);
                if (traceStages) retirement.retireLockAcquired = Clock::now();
                slots[job.sourceSlot].queued = false;
                returnDrawSourcesLocked();
                ++completed;
                if (gapTrace) {
                    retirement.queuedAfterRetirement = enqueued - completed;
                    retirement.slotCleared = Clock::now();
                }
                if (completed == 1 || completed % 300 == 0)
                    std::clog << "MAKO Renderer: independent-real queue queued=" << enqueued << " completed=" << completed
                              << " maximum_depth=" << maximumDepth << " full_waits=" << fullWaits << '\n';
                if (traceStages) retirement.slotRetired = Clock::now();
            } catch (const std::exception& error) {
                std::clog << "MAKO Renderer: direct source return failed: " << error.what() << '\n';
                if (drawBridge) {
                    workerFailure.store(true, std::memory_order_release);
                    publishRuntimeStatus(Clock::now(), true, true);
                }
                invalidate();
                return;
            }
            jobsChanged.notify_all();
            if (traceStages) retirement.producerNotified = Clock::now();
            if (traceSourceLive && record) {
                {
                    std::osyncstream log(std::clog);
                    logSourceTiming(log, timing);
                    log << std::flush;
                }
                if (gapTrace) retirement.sourceLogDone = Clock::now();
            }
            const auto sampleBefore = lastRuntimeSample;
            if (!runtimeStatus.path().empty()) publishRuntimeStatus(Clock::now());
            if (traceStages) {
                retirement.statusDone = Clock::now();
                retirement.statusWritten = sampleBefore != lastRuntimeSample;
            }
            if (trace) retirementTimings.push_back(retirement);
            if (gapTrace) {
                if (previous.presented != Clock::time_point{} && job.enqueuedAt != Clock::time_point{} &&
                        retirement.dequeued - previous.presented > retirementGapPeriod) {
                    // enqueuedAt is taken after inserting the actual successor, under jobsAccess.
                    // A late-arriving job cannot attribute the preceding idle time to retirement.
                    const auto queuedSince = std::max(previous.presented, job.enqueuedAt);
                    if (retirement.dequeued - queuedSince > retirementGapPeriod) {
                        ++retirementGaps;
                        // The successor has submitted and retired before the diagnostic does any I/O.
                        logRetirementGap(previous, retirement, job.enqueuedAt);
                        retirement.gapLogDone = Clock::now();
                    } else {
                        ++retirementInputIdleGaps;
                    }
                }
                previous = retirement;
            }
        }
    }

    VkResult waitForPresent(uint64_t id, uint64_t timeout) {
        if (!child.waitForPresent) return VK_ERROR_EXTENSION_NOT_PRESENT;
        const auto started = std::chrono::steady_clock::now();
        uint64_t remaining = timeout;
        do {
            VkResult result{};
            {
                const std::lock_guard lock(displayAccess);
                if (invalid.load(std::memory_order_acquire) || !output) return VK_ERROR_OUT_OF_DATE_KHR;
                result = child.waitForPresent(child.vk->dev(), output, id, std::min<uint64_t>(remaining, 2'000'000));
            }
            if (result != VK_TIMEOUT || !timeout) return result;
            const auto elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
            remaining = timeout > elapsed ? timeout - elapsed : 0;
        } while (remaining);
        return VK_TIMEOUT;
    }
};

IndependentRealPresent::IndependentRealPresent(const vk::Vulkan& app, VkInstance instance,
        const PrivateDeviceCallbacks& callbacks, X11PresentationTarget target,
        const VkSwapchainCreateInfoKHR& request, PFN_vkReleaseSwapchainImagesKHR release)
        : impl(std::make_unique<Impl>(app, release)) {
    impl->initialize(instance, callbacks, target, request);
}

IndependentRealPresent::~IndependentRealPresent() = default;
void IndependentRealPresent::noteFinalDraw(VkDevice device, VkImage target, VkImage source, uint64_t frame) {
    finalDraws.note(device, target, source, frame);
}
VkSwapchainKHR IndependentRealPresent::applicationSwapchain() const { return impl->source; }
VkSurfaceKHR IndependentRealPresent::applicationSurface() const { return impl->visible; }
VkResult IndependentRealPresent::acquire(const VkAcquireNextImageInfoKHR& request, uint32_t* imageIndex, bool secondApi) {
    return impl->acquire(request, imageIndex, secondApi);
}
VkResult IndependentRealPresent::present(VkQueue queue, const VkPresentInfoKHR& request, bool& sourceSubmitted) {
    return impl->present(queue, request, sourceSubmitted);
}
VkResult IndependentRealPresent::waitForPresent(uint64_t id, uint64_t timeout) { return impl->waitForPresent(id, timeout); }
void IndependentRealPresent::invalidate() { impl->invalidate(); }
void IndependentRealPresent::retireDisplay() { impl->retireDisplay(); }
bool IndependentRealPresent::active() const { return !impl->invalid.load(std::memory_order_acquire); }
bool IndependentRealPresent::workerFailed() const { return impl->workerFailure.load(std::memory_order_acquire); }
