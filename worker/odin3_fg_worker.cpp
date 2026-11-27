// SPDX-License-Identifier: GPL-3.0-or-later
// Odin 3 out-of-process Hexagon HTP v79 + Oryon NEON hybrid frame-generation worker.
#include "mako_fg_protocol.hpp"
#include "odin3_shared_dma.hpp"
#include "mako-common/configuration/htp_deadline.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <vulkan/vulkan.h>
#include "odin3_tier_engine.hpp"
#include "odin3_htp_direct_pair.hpp"

using namespace mako::backend::fg_host;

namespace {
enum class Mode { Stall, Exit, NoReply, Ready, Late, HtpVfi };

uint64_t monotonicNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool htpVfiDirectHtpEnabled() {
    const char* value = std::getenv("ODIN3_HTP_DIRECT_HTP");
    return value && std::string_view(value) == "1";
}

bool htpVfiRawHtpEnabled() {
    const char* value = std::getenv("ODIN3_HTP_RAW_HTP");
    return value && std::string_view(value) == "1";
}

bool htpVfiHostSignalRequested() {
    const char* value = std::getenv("ODIN3_HTP_RAW_SIGNAL_ON_HOST");
    return value && std::string_view(value) == "1";
}

bool copiesFrames(Mode mode) {
    return mode == Mode::Ready || mode == Mode::Late || mode == Mode::HtpVfi;
}

struct WorkerTimingLog {
    const char* action;
    uint64_t source;
    uint64_t nanoseconds;
    double graphUs = -1;
};

void writeTimingLog(const WorkerTimingLog& row) {
    if (row.graphUs >= 0) {
        dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=%s source=%ju graph_us=%.3f\n",
            getpid(), row.action, uintmax_t(row.source), row.graphUs);
    } else {
        dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=%s source=%ju monotonic_us=%ju\n",
            getpid(), row.action, uintmax_t(row.source), uintmax_t(row.nanoseconds / 1000));
    }
}

class BufferedWorkerTiming;
thread_local BufferedWorkerTiming* timingBuffer = nullptr;

class BufferedWorkerTiming {
    static constexpr size_t capacity = 131072;
    std::vector<WorkerTimingLog> entries;
    uint64_t dropped{};

public:
    explicit BufferedWorkerTiming(bool active) {
        if (!active) return;
        entries.reserve(capacity);
        timingBuffer = this;
    }

    ~BufferedWorkerTiming() {
        if (timingBuffer != this) return;
        timingBuffer = nullptr;
        for (const auto& entry : entries) writeTimingLog(entry);
        dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=htp-buffered-timing-summary records=%zu untracked=%ju\n",
            getpid(), entries.size(), uintmax_t(dropped));
    }

    void add(WorkerTimingLog row) {
        if (entries.size() == capacity) {
            ++dropped;
            return;
        }
        entries.push_back(row);
    }
};

bool verboseWorkerLogging() {
    static const bool enabled = [] {
        const char* value = std::getenv("ODIN3_FG_VERBOSE_LOG");
        return value && std::string_view(value) == "1";
    }();
    return enabled;
}

void timingLog(WorkerTimingLog row) {
    // Keep startup and fatal error diagnostics (row.source == 0) immediate; silence per-frame journald spam in production!
    if (timingBuffer && row.source) timingBuffer->add(row);
    else if (!row.source || verboseWorkerLogging()) writeTimingLog(row);
}

void event(const char* action, uint64_t source = 0) {
    if (source && !timingBuffer && !verboseWorkerLogging()) return;
    timingLog({action, source, monotonicNs()});
}

void waitTimingEvent(const char* action, uint64_t source) {
    static const bool enabled = [] {
        const char* value = std::getenv("ODIN3_HTP_WORKER_WAIT_TIMING");
        return value && std::string_view(value) == "1";
    }();
    if (enabled) event(action, source);
}

void check(bool condition, const char* message) {
    if (!condition) {
        event(message);
        _exit(1);
    }
}

void prepareDirectHtpLoader(char** argv) {
    const char* fastrpc = std::getenv("ODIN3_FASTRPC_LIB_DIR");
    const char* qairt = std::getenv("ODIN3_QAIRT_LIB_DIR");
    const auto absoluteDirectory = [](const char* path) {
        return path && path[0] == '/' && !std::strchr(path, ':');
    };
    check(absoluteDirectory(fastrpc) && absoluteDirectory(qairt), "htp-direct-htp-loader-path-invalid");
    const std::string prefix = std::string(fastrpc) + ':' + qairt;
    const char* inherited = std::getenv("LD_LIBRARY_PATH");
    const std::string_view previous = inherited ? inherited : "";
    if (previous == prefix ||
        (previous.starts_with(prefix) && previous[prefix.size()] == ':'))
        return;

    // Proton can replace the search path. Do this before the worker handles a
    // request so glibc uses the private paths without changing the game's loader.
    const std::string libraryPath = previous.empty() ? prefix : prefix + ':' + std::string(previous);
    check(setenv("LD_LIBRARY_PATH", libraryPath.c_str(), 1) == 0, "htp-direct-htp-loader-env-failed");
    event("htp-direct-htp-loader-reexec");
    execv("/proc/self/exe", argv);
    check(false, "htp-direct-htp-loader-reexec-failed");
}

void reply(int socket, uint32_t first = 0, uint32_t second = 0,
           const void* data = nullptr, size_t size = 0) {
    Reply message{};
    message.first = first;
    message.second = second;
    if (size) std::memcpy(message.text, data, size);
    const size_t bytes = offsetof(Reply, text) + size;
    check(send(socket, &message, bytes, MSG_NOSIGNAL) == ssize_t(bytes), "reply-failed");
}

struct VulkanDevice {
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family{};
    uint32_t timestampBits{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceLimits limits{};
    PFN_vkGetCalibratedTimestampsKHR calibratedTimestamps{};
    mutable std::mutex queueLock;
    GpuInfo identity{};

    explicit VulkanDevice(Mode mode) {
        const VkApplicationInfo app{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
        const VkInstanceCreateInfo basic{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
        check(vkCreateInstance(&basic, nullptr, &instance) == VK_SUCCESS, "instance-failed");
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance, &count, nullptr) == VK_SUCCESS, "enumerate-failed");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance, &count, devices.data()) == VK_SUCCESS, "enumerate-failed");
        VkPhysicalDevice selected{};
        for (const auto candidate : devices) {
            VkPhysicalDeviceIDProperties id{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
            vkGetPhysicalDeviceProperties2(candidate, &properties);
            if (properties.properties.vendorID != 0x5143)
                continue;
            selected = candidate;
            identity.vendor = properties.properties.vendorID;
            identity.id = properties.properties.deviceID;
            limits = properties.properties.limits;
            std::memcpy(identity.name, properties.properties.deviceName, sizeof(identity.name));
            std::memcpy(identity.driverUuid, id.driverUUID, sizeof(identity.driverUuid));
            std::memcpy(identity.deviceUuid, id.deviceUUID, sizeof(identity.deviceUuid));
            break;
        }
        check(selected != VK_NULL_HANDLE, "adreno-missing");
        physical = selected;
        vkGetPhysicalDeviceQueueFamilyProperties(selected, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(selected, &count, queues.data());
        while (family < count && !(queues[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++family;
        check(family < count, "graphics-queue-missing");
        timestampBits = queues[family].timestampValidBits;
        const float priority = 1.0F;
        VkDeviceQueueCreateInfo queue{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family,
            .queueCount = 1, .pQueuePriorities = &priority};
        const bool motion = mode == Mode::HtpVfi && mako::htpVfiMotionEnabled();
        const VkPhysicalDeviceVulkan12Features features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .scalarBlockLayout = motion ? VK_TRUE : VK_FALSE,
            .timelineSemaphore = VK_TRUE};
        const VkPhysicalDeviceFeatures core{
            .shaderStorageImageWriteWithoutFormat = motion ? VK_TRUE : VK_FALSE};
        std::vector<const char*> extensions{VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
                                            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
        const auto* calibration = std::getenv("ODIN3_HTP_CLOCK_CALIBRATION");
        const bool calibrationRequested = mode == Mode::HtpVfi && calibration && std::string_view(calibration) == "1";
        const auto* globalPriority = std::getenv("ODIN3_HTP_QUEUE_PRIORITY");
        const bool priorityRequested = mode == Mode::HtpVfi && globalPriority && std::string_view(globalPriority) == "high";
        const auto available = calibrationRequested || priorityRequested ?
            deviceExtensions() : std::vector<VkExtensionProperties>{};
        const auto supports = [&](std::string_view name) {
            return std::any_of(available.begin(), available.end(), [&](const auto& extension) {
                return name == extension.extensionName;
            });
        };
        if (mode == Mode::HtpVfi) {
            extensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
            extensions.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
        }
        const bool enableCalibration = calibrationRequested && timestampBits &&
            supports(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) && supportsMonotonicCalibration();
        if (enableCalibration) extensions.push_back(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
        const bool enablePriority = priorityRequested && supports(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME) &&
            supportsHighPriority();
        const VkDeviceQueueGlobalPriorityCreateInfoKHR highPriority{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
            .globalPriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR};
        if (enablePriority) {
            extensions.push_back(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
            queue.pNext = &highPriority;
        }
        VkDeviceCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features,
            .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(), .pEnabledFeatures = &core};
        auto created = vkCreateDevice(selected, &ci, nullptr, &device);
        if (enablePriority && (created == VK_ERROR_NOT_PERMITTED_KHR || created == VK_ERROR_INITIALIZATION_FAILED)) {
            dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=queue-priority-fallback vk_result=%d\n",
                getpid(), static_cast<int>(created));
            queue.pNext = nullptr;
            --ci.enabledExtensionCount;
            device = VK_NULL_HANDLE;
            created = vkCreateDevice(selected, &ci, nullptr, &device);
        }
        check(created == VK_SUCCESS, "device-failed");
        if (priorityRequested)
            event(queue.pNext ? "queue-priority-high" : "queue-priority-default");
        if (enableCalibration)
            calibratedTimestamps = reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(
                vkGetDeviceProcAddr(device, "vkGetCalibratedTimestampsKHR"));
        if (calibrationRequested)
            event(calibratedTimestamps ? "clock-calibration-ready" : "clock-calibration-unavailable");
        vkGetDeviceQueue(device, family, 0, &this->queue);
        vkGetPhysicalDeviceMemoryProperties(selected, &memory);
    }

    ~VulkanDevice() {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

    VkSemaphore importTimeline(int fd) const {
        VkSemaphore semaphore{};
        const VkSemaphoreTypeCreateInfo type{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
        const VkSemaphoreCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type};
        check(vkCreateSemaphore(device, &ci, nullptr, &semaphore) == VK_SUCCESS, "semaphore-failed");
        const auto import = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(
            vkGetDeviceProcAddr(device, "vkImportSemaphoreFdKHR"));
        check(import != nullptr, "semaphore-import-unavailable");
        const VkImportSemaphoreFdInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR, .semaphore = semaphore,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, .fd = fd};
        check(import(device, &info) == VK_SUCCESS, "semaphore-import-failed");
        return semaphore;
    }

    uint32_t memoryIndex(uint32_t allowed, VkMemoryPropertyFlags required) const {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((allowed & (1u << i)) && (memory.memoryTypes[i].propertyFlags & required) == required)
                return i;
        check(false, "memory-type-unavailable");
        return 0;
    }

    bool wait(VkSemaphore timeline, uint64_t value, uint64_t timeout) const {
        const VkSemaphoreWaitInfo info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
            .pSemaphores = &timeline, .pValues = &value};
        const auto result = vkWaitSemaphores(device, &info, timeout);
        check(result == VK_TIMEOUT || result == VK_SUCCESS, "timeline-wait-failed");
        return result == VK_SUCCESS;
    }

private:
    std::vector<VkExtensionProperties> deviceExtensions() const {
        uint32_t count{};
        if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr) != VK_SUCCESS || !count)
            return {};
        std::vector<VkExtensionProperties> extensions(count);
        if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()) != VK_SUCCESS)
            return {};
        extensions.resize(count);
        return extensions;
    }

    bool supportsHighPriority() const {
        uint32_t count{};
        vkGetPhysicalDeviceQueueFamilyProperties2(physical, &count, nullptr);
        if (family >= count) return false;
        std::vector<VkQueueFamilyProperties2> families(count,
            VkQueueFamilyProperties2{.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2});
        VkQueueFamilyGlobalPriorityPropertiesKHR priorities{
            .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR};
        families[family].pNext = &priorities;
        vkGetPhysicalDeviceQueueFamilyProperties2(physical, &count, families.data());
        const auto end = priorities.priorities + priorities.priorityCount;
        return std::find(priorities.priorities, end, VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR) != end;
    }

    bool supportsMonotonicCalibration() const {
        uint32_t count{};
        const auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR"));
        if (!query || query(physical, &count, nullptr) != VK_SUCCESS || !count)
            return false;
        std::vector<VkTimeDomainKHR> domains(count);
        if (query(physical, &count, domains.data()) != VK_SUCCESS)
            return false;
        const auto contains = [&](VkTimeDomainKHR domain) {
            return std::find(domains.begin(), domains.begin() + count, domain) != domains.begin() + count;
        };
        return contains(VK_TIME_DOMAIN_DEVICE_KHR) && contains(VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR);
    }
};

struct Image {
    VkImage handle{};
    VkDeviceMemory memory{};
};

struct GpuWorkTiming {
    uint64_t source, submitNs, submitEndNs;
};

void logCalibratedGpuWork(const VulkanDevice& gpu, const char* action, const GpuWorkTiming& work,
                          const std::array<uint64_t, 2>& timestamps) {
    if (!gpu.calibratedTimestamps) return;
    constexpr std::array domains{
        VkCalibratedTimestampInfoKHR{.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
            .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR},
        VkCalibratedTimestampInfoKHR{.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
            .timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR}};
    std::array<uint64_t, 2> reference{};
    uint64_t deviation{};
    const auto before = monotonicNs();
    const auto result = gpu.calibratedTimestamps(gpu.device, domains.size(), domains.data(), reference.data(), &deviation);
    const auto after = monotonicNs();
    if (result != VK_SUCCESS || deviation > 100'000 || reference[1] + deviation < before ||
            reference[1] > after + deviation || after - work.submitNs > 250'000'000)
        return;
    const uint64_t mask = UINT64_MAX >> (64 - gpu.timestampBits);
    const uint64_t half = uint64_t{1} << (gpu.timestampBits - 1);
    const auto toMonotonic = [&](uint64_t timestamp) {
        const auto forward = (timestamp - reference[0]) & mask;
        const double ticks = forward < half ? double(forward) : -double((reference[0] - timestamp) & mask);
        return static_cast<int64_t>(reference[1]) +
            static_cast<int64_t>(std::llround(ticks * gpu.limits.timestampPeriod));
    };
    const auto start = toMonotonic(timestamps[0]);
    const auto end = toMonotonic(timestamps[1]);
    if (start < static_cast<int64_t>(work.submitNs) - static_cast<int64_t>(deviation) ||
            end < start || end > static_cast<int64_t>(after + deviation))
        return;
    dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=%s source=%llu "
        "submit_ns=%llu gpu_start_ns=%lld gpu_end_ns=%lld deviation_ns=%llu age_ns=%llu "
        "tid=%d submit_end_ns=%llu\n",
        getpid(), action, static_cast<unsigned long long>(work.source),
        static_cast<unsigned long long>(work.submitNs), static_cast<long long>(start),
        static_cast<long long>(end), static_cast<unsigned long long>(deviation),
        static_cast<unsigned long long>(after - work.submitNs), gettid(),
        static_cast<unsigned long long>(work.submitEndNs));
}

#include "odin3_dma_buffer.hpp"
#include "live_htp_profile.hpp"
#include "odin3_htp_vfi_model.hpp"

class CopyQueue {
    static constexpr VkExtent3D mark{32, 24, 1};
    static constexpr VkImageSubresourceLayers color{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    const VulkanDevice& gpu;
    const bool sharedInput;
    const bool directOutput;
    const bool directInput;
    const bool hostSignal;
    const VkExtent3D extent;
    const VkFormat format;
    std::array<Image, 2> source;
    std::array<Image, 2> outputs{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkBuffer stamp{};
    VkDeviceMemory stampMemory{};
    std::array<bool, 2> wroteOutput{};
    const bool useRaw = htpVfiRawHtpEnabled();
    const bool serialRawPhases = [] {
        const char* value = std::getenv("ODIN3_HTP_RAW_SERIAL_PHASES");
        return value && std::string_view(value) == "1";
    }();
    std::optional<RawHtpVfiModel> rawModel;
    VkQueryPool outputQueries{};
    std::optional<GpuWorkTiming> outputTiming;

    void begin() const {
        check(vkResetCommandBuffer(command, 0) == VK_SUCCESS, "worker-command-reset");
        const VkCommandBufferBeginInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        check(vkBeginCommandBuffer(command, &info) == VK_SUCCESS, "worker-command-begin");
    }

    void acquireOutput(size_t output, VkAccessFlags write) const {
        const bool external = wroteOutput.at(output);
        transfer(outputs.at(output).handle, external ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED,
                 external ? gpu.family : VK_QUEUE_FAMILY_IGNORED,
                 external ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, 0, write);
    }

    bool renderRaw(uint64_t frame, VkSemaphore timeline, uint64_t readyValue, unsigned output,
            unsigned batchPhases, RawHtpVfiModel::PhaseTrace* trace) {
        const auto phaseValue = readyValue + output;
        const VkBufferImageCopy region{.imageSubresource = color, .imageExtent = extent};
        event("htp-pack-start", phaseValue);
        try {
            if (output == 0 && directInput) {
                // The schedule handler has observed the Renderer upload's
                // timeline after it released the linear input to FOREIGN.
                if (!sharedInput) rawModel->bindLinearInputs(frame);
                if (trace) trace->inputReady = monotonicNs();
                rawModel->estimate(phaseValue, trace);
            } else if (output == 0) {
                const size_t index = frame % source.size();
                const auto plan = rawModel->planInputs(frame);
                const std::array endpoints{source[1 - index].handle, source[index].handle};
                for (unsigned i = plan.reuseA ? 1u : 0u; i < endpoints.size(); ++i) {
                    const auto image = endpoints[i];
                    const auto buffer = rawModel->input(i ? plan.b : plan.a);
                    transfer(image, VK_QUEUE_FAMILY_EXTERNAL, gpu.family, VK_IMAGE_LAYOUT_GENERAL,
                             0, VK_ACCESS_TRANSFER_READ_BIT);
                    htpVfiDmaOwnership(gpu, command, buffer, false, VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT);
                    vkCmdCopyImageToBuffer(command, image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &region);
                    htpVfiDmaOwnership(gpu, command, buffer, true, VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT);
                    transfer(image, gpu.family, VK_QUEUE_FAMILY_EXTERNAL, VK_IMAGE_LAYOUT_GENERAL,
                             VK_ACCESS_TRANSFER_READ_BIT, 0);
                }
                if (trace) {
                    trace->inputCopies = plan.reuseA ? 1u : 2u;
                    trace->inputCopyBytes = uint64_t(extent.width) * extent.height * 4 * trace->inputCopies;
                    trace->inputSubmit = monotonicNs();
                }
                rawModel->submitInputs(command, timeline, readyValue);
                if (trace) trace->inputReady = monotonicNs();
                rawModel->markInputsReady(frame);
                waitTimingEvent("htp-pack-ready", phaseValue);
                rawModel->estimate(phaseValue, trace);
                if (trace) trace->commandResetBegin = monotonicNs();
                begin();
                if (trace) trace->commandResetEnd = monotonicNs();
            }
            rawModel->render(output, batchPhases, phaseValue, trace);
            if (trace) trace->readable = monotonicNs();
        } catch (const std::exception& error) {
            rawModel->discardLumaHistory();
            dprintf(STDERR_FILENO, "Odin3 HTP generation failed: %s\n", error.what());
            return false;
        }
        event("htp-result-ready", phaseValue);
        return true;
    }

    void recordRawOutput(unsigned output) {
        const VkBufferImageCopy region{.imageSubresource = color, .imageExtent = extent};
        if (outputQueries) {
            vkCmdResetQueryPool(command, outputQueries, 0, 2);
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, outputQueries, 0);
        }
        const auto buffer = rawModel->output(output);
        htpVfiDmaOwnership(gpu, command, buffer, false, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        acquireOutput(output, VK_ACCESS_TRANSFER_WRITE_BIT);
        const auto image = outputs.at(output).handle;
        vkCmdCopyBufferToImage(command, buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        htpVfiDmaOwnership(gpu, command, buffer, true, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        transfer(image, gpu.family, VK_QUEUE_FAMILY_EXTERNAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, 0);
        if (outputQueries) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, outputQueries, 1);
    }

    Image importImage(int fd, VkImageUsageFlags usage) const {
        Image image;
        const VkExternalMemoryImageCreateInfo external{
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        const VkImageCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &external,
            .imageType = VK_IMAGE_TYPE_2D, .format = format, .extent = extent,
            .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        check(vkCreateImage(gpu.device, &ci, nullptr, &image.handle) == VK_SUCCESS, "image-create-failed");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(gpu.device, image.handle, &requirements);
        const VkMemoryDedicatedAllocateInfo dedicated{
            .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = image.handle};
        const VkImportMemoryFdInfoKHR import{
            .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, .pNext = &dedicated,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, .fd = fd};
        const VkMemoryAllocateInfo allocate{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &import,
            .allocationSize = requirements.size,
            .memoryTypeIndex = gpu.memoryIndex(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        check(vkAllocateMemory(gpu.device, &allocate, nullptr, &image.memory) == VK_SUCCESS, "image-import-failed");
        check(vkBindImageMemory(gpu.device, image.handle, image.memory, 0) == VK_SUCCESS, "image-bind-failed");
        return image;
    }

    void transfer(VkImage image, uint32_t from, uint32_t to, VkImageLayout previous,
                  VkAccessFlags read, VkAccessFlags write) const {
        const VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = read, .dstAccessMask = write,
            .oldLayout = previous, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = from, .dstQueueFamilyIndex = to,
            .image = image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    void createStamp() {
        constexpr std::array<uint8_t, 4> magenta8{255, 0, 255, 255};
        constexpr std::array<uint8_t, 4> cyan8{0, 255, 255, 255};
        constexpr std::array<uint16_t, 4> magenta16{0x3c00, 0, 0x3c00, 0x3c00};
        constexpr std::array<uint16_t, 4> cyan16{0, 0x3c00, 0x3c00, 0x3c00};
        const bool wide = format == VK_FORMAT_R16G16B16A16_SFLOAT;
        const size_t pixelSize = wide ? sizeof(magenta16) : sizeof(magenta8);
        const VkDeviceSize size = mark.width * mark.height * pixelSize;
        const VkBufferCreateInfo ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
        check(vkCreateBuffer(gpu.device, &ci, nullptr, &stamp) == VK_SUCCESS, "stamp-buffer-failed");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(gpu.device, stamp, &requirements);
        const VkMemoryAllocateInfo allocation{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = gpu.memoryIndex(requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        check(vkAllocateMemory(gpu.device, &allocation, nullptr, &stampMemory) == VK_SUCCESS, "stamp-memory-failed");
        check(vkBindBufferMemory(gpu.device, stamp, stampMemory, 0) == VK_SUCCESS, "stamp-bind-failed");
        void* mapped{};
        check(vkMapMemory(gpu.device, stampMemory, 0, size, 0, &mapped) == VK_SUCCESS, "stamp-map-failed");
        auto* bytes = static_cast<unsigned char*>(mapped);
        for (size_t i = 0; i < mark.width * mark.height; ++i) {
            const bool left = (i % mark.width) < mark.width / 2;
            const void* pixel = wide ? static_cast<const void*>((left ? magenta16 : cyan16).data())
                                     : static_cast<const void*>((left ? magenta8 : cyan8).data());
            std::memcpy(bytes + i * pixelSize, pixel, pixelSize);
        }
        vkUnmapMemory(gpu.device, stampMemory);
    }

public:
    CopyQueue(const VulkanDevice& device, const Request& open, const std::array<int, 4>& images, Mode mode,
              std::array<int, 10> spareInputFds, std::array<int, 4> extraOutputFds,
              const odin3::shared_frame::DmaSourcePool* describedSources = nullptr)
        : gpu(device), sharedInput(open.encoding == rawSharedSourceEncoding),
          directOutput(open.encoding == rawLinearOutputEncoding || open.encoding == rawLinearIoEncoding || sharedInput),
          directInput(open.encoding == rawLinearInputEncoding || open.encoding == rawLinearIoEncoding || sharedInput),
          hostSignal(sharedInput && htpVfiHostSignalRequested()),
          extent{open.width, open.height, 1},
          format(open.encoding == 1 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM) {
        check(!sharedInput || (mode == Mode::HtpVfi && mako::htpVfiSharedSourcesEnabled() && spareInputFds[0] >= 0),
              "htp-shared-input-protocol-mismatch");
        check(!directOutput || (mode == Mode::HtpVfi && mako::htpVfiLinearOutputEnabled()), "htp-linear-protocol-mismatch");
        check(!directInput || (mode == Mode::HtpVfi && mako::htpVfiLinearInputEnabled()), "htp-linear-input-protocol-mismatch");
        check(!hostSignal || (directOutput && htpVfiRawHtpEnabled()), "host-signal-requires-shared-raw-htp");
        check((open.encoding == 0 || open.encoding == 1 || directOutput || directInput) && extent.width >= mark.width &&
              extent.height >= mark.height, "copy-format-or-extent-unsupported");
        if (!directInput) {
            source[0] = importImage(images[0], mako::htpVfiSourceImageUsage);
            source[1] = importImage(images[1], mako::htpVfiSourceImageUsage);
        }
        const size_t outputCount = images[3] < 0 ? 1 : 2;
        if (!directOutput)
            for (size_t i = 0; i < outputCount; ++i)
                outputs[i] = importImage(images[2 + i], mako::htpVfiOutputImageUsage);
        const VkCommandPoolCreateInfo ci{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = gpu.family};
        check(vkCreateCommandPool(gpu.device, &ci, nullptr, &pool) == VK_SUCCESS, "copy-pool-failed");
        const VkCommandBufferAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        check(vkAllocateCommandBuffers(gpu.device, &allocate, &command) == VK_SUCCESS, "copy-command-failed");
        if (mode == Mode::HtpVfi) {
            check(useRaw, "htp-raw-mode-required");
                const bool validExtent = extent.width >= 64 && extent.width % 16 == 0 &&
                    extent.height >= 64 && extent.height % 4 == 0;
                const char* joinedContext = std::getenv("ODIN3_HTP_RAW_JOINED_CONTEXT");
                const char* firstContext = std::getenv("ODIN3_HTP_RAW_FIRST_CONTEXT");
                const char* motionContext = std::getenv("ODIN3_HTP_RAW_MOTION_CONTEXT");
                const char* bundleDir = std::getenv("ODIN3_FG_MODEL_BUNDLE_DIR");
                const bool hasExplicitCtx = (joinedContext && joinedContext[0] == '/') || (firstContext && firstContext[0] == '/');
                const char* bundleRes = (!hasExplicitCtx && bundleDir && bundleDir[0] == '/')
                    ? mako::odin3ModelBundleSubdir(extent.width, extent.height) : nullptr;
                const bool hasMotionCtx = (motionContext && motionContext[0] == '/') ||
                    (bundleRes && std::filesystem::is_regular_file(std::filesystem::path(bundleDir) / bundleRes / "vfi-motion3x.bin"));
                const bool joined2x = validExtent && outputCount == 1 &&
                    ((joinedContext && joinedContext[0] == '/') || bundleRes != nullptr) &&
                    directInput && directOutput && sharedInput &&
                    spareInputFds[1] >= 0 && extraOutputFds[0] >= 0 && extraOutputFds[1] >= 0;
                const bool joined3x = validExtent && outputCount == 2 && !hasMotionCtx &&
                    directInput && directOutput && sharedInput &&
                    spareInputFds[1] >= 0 && extraOutputFds[2] >= 0 && extraOutputFds[3] >= 0;
                const bool joined = joined2x || joined3x;
                const bool bundleExtentUnsupported = !hasExplicitCtx && bundleDir && bundleDir[0] == '/' && !bundleRes;
                if (!validExtent || bundleExtentUnsupported || format != VK_FORMAT_R8G8B8A8_UNORM) {
                    event("htp-raw-format-unsupported");
                } else {
                    try {
                        const auto outputFds = directOutput ? std::array{images[2], images[3]} : std::array{-1, -1};
                        const auto inputFds = directInput
                            ? std::array{images[0], images[1], spareInputFds[0], spareInputFds[1],
                                         spareInputFds[2], spareInputFds[3], spareInputFds[4], spareInputFds[5],
                                         spareInputFds[6], spareInputFds[7], spareInputFds[8], spareInputFds[9]}
                            : std::array{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
                        rawModel.emplace(gpu, outputCount, outputFds, inputFds, extraOutputFds,
                                         odin3_direct_pair::FrameSize{extent.width, extent.height}, joined, describedSources);
                    } catch (const std::exception& error) {
                        dprintf(STDERR_FILENO, "Odin3 HTP setup failed: %s\n", error.what());
                    }
                }
                if (directOutput) {
                    for (size_t i = 0; i < outputCount; ++i) close(images[2 + i]);
                    for (const int fd : extraOutputFds)
                        if (fd >= 0) close(fd);
                    check(rawModel.has_value(), "htp-linear-htp-setup-failed");
                    event("htp-raw-linear-output", outputCount);
                    if (extraOutputFds[0] >= 0) event("htp-direct-dri3-output-pool", directDri3PoolSize(outputCount == 2));
                }
                if (directInput) {
                    close(images[0]);
                    close(images[1]);
                    for (const int fd : spareInputFds)
                        if (fd >= 0) close(fd);
                    check(rawModel.has_value(), "htp-linear-input-htp-setup-failed");
                    event("htp-raw-linear-input", sharedInput ? sourceCount(mako::htpVfiDirectDri3Enabled()) : 2);
                }
            const auto* timing = std::getenv("ODIN3_HTP_OUTPUT_TIMING");
            if (!directOutput && gpu.timestampBits && (gpu.calibratedTimestamps || (timing && std::string_view(timing) == "1"))) {
                const VkQueryPoolCreateInfo queries{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
                if (vkCreateQueryPool(gpu.device, &queries, nullptr, &outputQueries) != VK_SUCCESS)
                    event("output-timestamps-unavailable");
            }
        } else {
            createStamp();
        }
    }

    ~CopyQueue() {
        collectOutputTiming();
        if (outputQueries) vkDestroyQueryPool(gpu.device, outputQueries, nullptr);
        rawModel.reset();
        vkDestroyCommandPool(gpu.device, pool, nullptr);
        for (const auto image : {source[0], source[1], outputs[0], outputs[1]}) {
            vkDestroyImage(gpu.device, image.handle, nullptr);
            vkFreeMemory(gpu.device, image.memory, nullptr);
        }
        vkDestroyBuffer(gpu.device, stamp, nullptr);
        vkFreeMemory(gpu.device, stampMemory, nullptr);
    }

    void bindSharedInputs(unsigned a, unsigned b, uint64_t ordinal) {
        check(sharedInput && rawModel.has_value(), "missing-htp-shared-source-context");
        rawModel->bindSharedInputs(a, b, ordinal);

    }

    bool signalsOnHost() const { return hostSignal; }

    bool submit(VkSemaphore timeline, uint64_t readyValue, uint64_t frame,
                std::span<const float> phases, std::span<uint64_t> status) {
        const uint64_t finalValue = readyValue + phases.size();
        if (useRaw) {
            const auto near = [](float a, float b) { return std::abs(a - b) < 0.0001f; };
            const bool supported = rawModel && !phases.empty() && phases.size() <= rawModel->phaseCount() &&
                ((phases.size() == 1 && near(phases[0], 0.5f)) ||
                 (phases.size() == 2 && near(phases[0], 1.0f / 3.0f) && near(phases[1], 2.0f / 3.0f)));
            if (!supported) {
                event(rawModel ? "htp-raw-phase-unsupported" : "htp-raw-unavailable", readyValue);
                check(!phases.empty() && !status.empty(), "htp-raw-no-phase-status");
                std::atomic_ref(status[motionBatchWord]).store(finalValue << 1, std::memory_order_release);
                submitOutput(timeline, readyValue, finalValue, false);
                return false;
            }
        }
        const size_t index = frame % source.size();
        for (size_t output = 0; output < phases.size(); ++output) {
            const uint64_t waitValue = readyValue + output;
            bool accepted = true;
            auto* trace = rawModel ? rawModel->beginPhaseTrace(waitValue) : nullptr;
            const bool overlapRawPhase = rawModel && !serialRawPhases && output == 1;
            if (output) {
                const auto delaySecondPhase = [frame, waitValue] {
                    static const bool testDelay = [] {
                        const char* value = std::getenv("ODIN3_HTP_TEST_DELAY_SECOND_PHASE");
                        return value && std::string_view(value) == "1";
                    }();
                    if (testDelay && frame == 16) {
                        event("delay-second-phase", waitValue);
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                };
                if (overlapRawPhase) {
                    delaySecondPhase();
                    // HTP writes a different buffer while Vulkan reads the first output.
                    // Retire the Vulkan work before reusing its command buffer or queries.
                    accepted = renderRaw(frame, timeline, readyValue, output, static_cast<unsigned>(phases.size()), trace);
                }
                waitTimingEvent("previous-phase-wait-begin", waitValue);
                check(gpu.wait(timeline, waitValue, 2000000000ULL), "previous-phase-not-retired");
                waitTimingEvent("previous-phase-wait-end", waitValue);
                if (!overlapRawPhase)
                    delaySecondPhase();
            }
            if (trace) trace->queryBegin = monotonicNs();
            collectOutputTiming();
            if (trace) trace->queryEnd = monotonicNs();
            if (!hostSignal) begin();
            if (trace && !hostSignal) trace->recordingBegan = monotonicNs();
            if (rawModel) {
                if (!overlapRawPhase)
                    accepted = renderRaw(frame, timeline, readyValue, output, static_cast<unsigned>(phases.size()), trace);
                if (accepted && !directOutput)
                    recordRawOutput(output);
            } else {
                recordCopy(index);
            }
            // Retire a rejected pair on this same queue, including all reserved values.
            const uint64_t doneValue = accepted ? waitValue + 1 : finalValue;
            if (!accepted)
                check(!status.empty(), "htp-motion-status-missing");
            if (accepted && !hostSignal)
                check(vkEndCommandBuffer(command) == VK_SUCCESS, "worker-command-end");
            if (status.size() == motionStatusWords && phases.size() == 2 && output == 0 && accepted)
                std::atomic_ref(status[motionFirstPhaseWord]).store((doneValue << 1) | 1u, std::memory_order_release);
            if (!status.empty() && doneValue == finalValue)
                std::atomic_ref(status[motionBatchWord]).store((doneValue << 1) | accepted, std::memory_order_release);
            if (trace) trace->outputSubmit = monotonicNs();
            submitOutput(timeline, waitValue, doneValue, accepted);
            if (trace) {
                trace->outputReturned = monotonicNs();
                if (accepted) rawModel->collectLiveProfile(*trace);
            }
            if (!accepted)
                return false;
            wroteOutput.at(output) = true;
        }
        return true;
    }

private:
    void collectOutputTiming() {
        if (!outputTiming)
            return;
        std::array<uint64_t, 2> timestamps{};
        if (vkGetQueryPoolResults(gpu.device, outputQueries, 0, timestamps.size(), sizeof(timestamps),
                timestamps.data(), sizeof(timestamps[0]), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const uint64_t mask = UINT64_MAX >> (64 - gpu.timestampBits);
            const auto ticks = (timestamps[1] - timestamps[0]) & mask;
            dprintf(STDERR_FILENO, "Odin3 FG worker: pid=%d action=output-gpu-timing source=%llu gpu_ns=%.0f\n",
                getpid(), static_cast<unsigned long long>(outputTiming->source),
                double(ticks) * gpu.limits.timestampPeriod);
            logCalibratedGpuWork(gpu, "output-calibrated", *outputTiming, timestamps);
        }
        outputTiming.reset();
    }

    void submitOutput(VkSemaphore timeline, uint64_t waitValue, uint64_t doneValue, bool writes) {
        if (hostSignal) {
            // Shared raw input was acquired before evaluation. No Vulkan command writes this output.
            // The renderer still acquires the FOREIGN image after waiting on this same payload.
            check(gpu.wait(timeline, waitValue, 0), "host-output-predecessor-incomplete");
            const VkSemaphoreSignalInfo signal{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = timeline, .value = doneValue};
            check(vkSignalSemaphore(gpu.device, &signal) == VK_SUCCESS, "linear-output-host-signal-failed");
            return;
        }
        const VkTimelineSemaphoreSubmitInfo values{.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &waitValue,
            .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &doneValue};
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &values,
            .waitSemaphoreCount = 1, .pWaitSemaphores = &timeline, .pWaitDstStageMask = &waitStage,
            .commandBufferCount = writes ? 1u : 0u, .pCommandBuffers = writes ? &command : nullptr,
            .signalSemaphoreCount = 1, .pSignalSemaphores = &timeline};
        const std::lock_guard lock(gpu.queueLock);
        if (writes && outputQueries) outputTiming = GpuWorkTiming{waitValue, monotonicNs(), 0};
        check(vkQueueSubmit(gpu.queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS, "worker-output-submit");
        if (writes && outputQueries) outputTiming->submitEndNs = monotonicNs();
    }

    void recordCopy(size_t index) const {
        const auto in = source.at(index).handle;
        const auto output = outputs.front();
        transfer(in, VK_QUEUE_FAMILY_EXTERNAL, gpu.family, VK_IMAGE_LAYOUT_GENERAL,
                 0, VK_ACCESS_TRANSFER_READ_BIT);
        acquireOutput(0, VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkImageCopy frame{.srcSubresource = color, .dstSubresource = color, .extent = extent};
        vkCmdCopyImage(command, in, VK_IMAGE_LAYOUT_GENERAL, output.handle, VK_IMAGE_LAYOUT_GENERAL, 1, &frame);
        const VkMemoryBarrier writes{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &writes, 0, nullptr, 0, nullptr);
        const VkBufferImageCopy marker{.imageSubresource = color,
            .imageOffset = {int32_t((extent.width - mark.width) / 2),
                            int32_t((extent.height - mark.height) / 2), 0},
            .imageExtent = mark};
        vkCmdCopyBufferToImage(command, stamp, output.handle, VK_IMAGE_LAYOUT_GENERAL, 1, &marker);
        transfer(in, gpu.family, VK_QUEUE_FAMILY_EXTERNAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_TRANSFER_READ_BIT, 0);
        transfer(output.handle, gpu.family, VK_QUEUE_FAMILY_EXTERNAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    }
};


void context(int socket, const VulkanDevice& gpu, VkSemaphore timeline, Mode mode,
             Request open, std::array<int, 4> images, int statusFd, std::array<int, 10> spareInputFds,
             std::array<int, 4> extraOutputFds, std::optional<odin3::shared_frame::DmaSourcePool> describedSources) {
    std::span<uint64_t> status;
    if (statusFd >= 0) {
        struct stat info {};
        check(fstat(statusFd, &info) == 0, "htp-motion-status-stat-failed");
        check(info.st_size == off_t(sizeof(uint64_t)) || info.st_size == off_t(motionStatusWords * sizeof(uint64_t)),
            "htp-motion-status-size-invalid");
        const auto size = static_cast<size_t>(info.st_size);
        void* mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, statusFd, 0);
        close(statusFd);
        check(mapping != MAP_FAILED, "htp-motion-status-map-failed");
        status = {static_cast<uint64_t*>(mapping), size / sizeof(uint64_t)};
    }
    std::optional<CopyQueue> copy;
    const bool shared = open.encoding == rawSharedSourceEncoding;
    const bool direct = shared && mako::htpVfiDirectDri3Enabled();
    if (copiesFrames(mode)) copy.emplace(gpu, open, images, mode, spareInputFds, extraOutputFds,
        describedSources ? &*describedSources : nullptr);
    reply(socket, open.width, open.height);
    event("context-open");
    const auto* bufferTiming = std::getenv("ODIN3_HTP_BUFFER_RAW_TIMING");
    const BufferedWorkerTiming buffered(htpVfiRawHtpEnabled() && bufferTiming && std::string_view(bufferTiming) == "1");
    uint64_t source = 1;
    uint64_t lastOutput = 0;
    uint64_t frame = 0;
    bool pending = false;
    Request request{};
    while (recv(socket, &request, sizeof request, 0) == ssize_t(sizeof request)) {
        if (request.op == Ready) {
            reply(socket, !pending && (!lastOutput || gpu.wait(timeline, lastOutput, 0)));
            continue;
        }
        const bool async = mako::htpVfiAsyncScheduleEnabled();
        const uint32_t limit = mako::htpVfiTripleEnabled() && mode == Mode::HtpVfi && images[3] >= 0 ? 2 : 1;
        check(request.op == (async ? ScheduleAsync : Schedule) &&
              request.count <= limit && !pending, "unexpected-request");
        if (shared) {
            const auto slots = request.timeout & sharedSourceSlotMask;
            const auto count = sourceCount(direct);
            check(copy.has_value() && request.count && (direct || (slots && slots <= count * count)),
                "invalid-shared-source-schedule");
            const auto encoded = static_cast<unsigned>(slots - (direct ? 0 : 1));
            const auto a = encoded / count, b = encoded % count;
            check(a != b, "shared-source-endpoints-alias");
            copy->bindSharedInputs(a, b, request.timeout >> sharedSourceOrdinalShift);
        }
        if (request.count == 0) {
            if (!async) reply(socket);
            event("history", source++);
            ++frame;
            continue;
        }
        pending = true;
        if (mode == Mode::NoReply) {
            event("schedule-reply-withheld", source);
            continue;
        }
        if (!async) reply(socket);
        waitTimingEvent("source-wait-begin", source);
        check(gpu.wait(timeline, source, 2000000000ULL), "source-wait-failed");
        event("source-ready", source);
        if (copy) {
            if (mode == Mode::Late && lastOutput == 0) {
                event("delay-first-output", source);
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            const auto phases = std::span{request.times}.first(request.count);
            float priorPhase = 0;
            for (const float phase : phases) {
                check(std::isfinite(phase) && phase > priorPhase && phase < 1, "invalid-output-phase");
                priorPhase = phase;
            }
            const bool accepted = copy->submit(timeline, source, frame++, phases, status);
            lastOutput = source + request.count;
            const char* completed = copy->signalsOnHost() ? "host-output-signaled" : "gpu-output-submitted";
            event(accepted ? completed : "motion-rejected", source);
            source = lastOutput + 1;
            pending = false;
            continue;
        }
        if (mode == Mode::Exit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
            event("exit-before-output", source);
            _exit(42);
        }
        event("output-withheld", source);
    }
    if (lastOutput)
        check(gpu.wait(timeline, lastOutput, 2000000000ULL), "copy-retirement-failed");
    copy.reset();
    if (const auto* trace = std::getenv("ODIN3_WSI_DIRECT_TRACE");
            trace && std::string_view(trace) == "1" && describedSources)
        dprintf(STDERR_FILENO, "Odin3 shared DMA retired: pid=%d epoch=%ju last_output=%ju at_ns=%ju\n",
            getpid(), uintmax_t(describedSources->epoch), uintmax_t(lastOutput), uintmax_t(monotonicNs()));
    if (!status.empty())
        munmap(status.data(), status.size_bytes());
    vkDestroySemaphore(gpu.device, timeline, nullptr);
    close(socket);
}
}

int main(int, [[maybe_unused]] char** argv) {
    const char* value = std::getenv("ODIN3_FG_WORKER_MODE");
    const std::string_view mode = (value && *value) ? value : "htp";
    check(mode == "stall" || mode == "exit" || mode == "no-reply" ||
          mode == "ready" || mode == "late" || mode == "htp", "mode-required");
    const Mode test = mode == "ready" ? Mode::Ready : mode == "late" ? Mode::Late :
        mode == "stall" ? Mode::Stall : mode == "exit" ? Mode::Exit :
        mode == "htp" ? Mode::HtpVfi : Mode::NoReply;
    if (test == Mode::HtpVfi && (htpVfiDirectHtpEnabled() || htpVfiRawHtpEnabled()))
        prepareDirectHtpLoader(argv);
    check(!mako::htpVfiAsyncScheduleEnabled() || test != Mode::NoReply, "async-mode-expects-no-reply");
    const bool motion = mako::htpVfiMotionEnabled();
    check(!motion || test == Mode::HtpVfi, "motion-mode-requires-htp-model");
    if (htpVfiRawHtpEnabled()) {
        check(test == Mode::HtpVfi && motion && !htpVfiDirectHtpEnabled(), "htp-raw-htp-worker-contract");

    }
    if (htpVfiDirectHtpEnabled()) {
        check(test == Mode::HtpVfi, "htp-direct-htp-requires-htp-mode");
        check(motion, "htp-direct-htp-requires-motion");

    }
    const pid_t child = fork();
    check(child >= 0, "fork-failed");
    if (child > 0) _exit(0);
    setenv("DISABLE_MAKO", "1", 1);
    odin3_tier::pinToEfficiencyCluster();
    VulkanDevice gpu(test);
    std::vector<std::thread> contexts;
    while (true) {
        Request request{};
        odin3::shared_frame::DmaSourcePool sourcePool{};
        std::array<iovec, 2> pieces{{{&request, sizeof request}, {&sourcePool, sizeof sourcePool}}};
        alignas(cmsghdr) std::array<char, CMSG_SPACE(maxOpenDescriptors * sizeof(int))> storage{};
        msghdr message{};
        message.msg_iov = pieces.data();
        message.msg_iovlen = pieces.size();
        message.msg_control = storage.data();
        message.msg_controllen = storage.size();
        const ssize_t bytes = recvmsg(hostDescriptor, &message, MSG_CMSG_CLOEXEC);
        if (bytes < 0 && errno == EINTR) continue;
        if (bytes <= 0) break;
        check(bytes == ssize_t(sizeof request + (request.op == OpenDescribed ? sizeof sourcePool : 0)) && !(message.msg_flags & (MSG_CTRUNC | MSG_TRUNC)),
              "control-message-truncated");
        if (request.op == Hello) {
            constexpr char name[] = "Odin3 HTP FG worker";
            const auto protocol = mako::htpVfiDirectDri3Enabled() ? (request.timeout == describedSourceProtocol ? describedSourceProtocol : htpVfiDirectDri3Protocol) :
                mako::htpVfiSharedSourcesEnabled() ? htpVfiSharedSourceProtocol :
                mako::htpVfiLinearInputEnabled() ? htpVfiLinearIoProtocol :
                mako::htpVfiLinearOutputEnabled() ? htpVfiLinearOutputProtocol :
                mako::htpVfiTripleEnabled() ? htpVfiTripleProtocol : motion ? htpVfiMotionProtocol :
                mako::htpVfiAsyncScheduleEnabled() ? htpVfiAsyncProtocol : hostProtocol;
            reply(hostDescriptor, protocol, sizeof(Request), name, sizeof name);
        } else if (request.op == Devices) {
            reply(hostDescriptor, 1, 0, &gpu.identity, sizeof gpu.identity);
        } else if (request.op == Supports) {
            reply(hostDescriptor, 0);
        } else {
            const auto* item = CMSG_FIRSTHDR(&message);
            const bool shared = request.encoding == rawSharedSourceEncoding;
            const bool pooled = shared && mako::htpVfiDirectDri3Enabled();
            const size_t extraSources = shared ? (sourceCount(pooled) - 2u) : 0u;
            check((request.op == Open || request.op == OpenDescribed) && request.count > 0 && request.count <= maxOutputs && item &&
                  (!shared || (motion && mako::htpVfiSharedSourcesEnabled() && test == Mode::HtpVfi &&
                      (pooled ? request.count == directDri3PoolSize(mako::htpVfiTripleEnabled()) : request.count <= 2))) &&
                  item->cmsg_level == SOL_SOCKET && item->cmsg_type == SCM_RIGHTS &&
                  item->cmsg_len == CMSG_LEN((request.count + 4 + motion + extraSources) * sizeof(int)),
                  "control-open-invalid");
            std::array<int, maxOpenDescriptors> descriptors;
            const size_t imagesEnd = request.count + 4;
            const size_t descriptorCount = imagesEnd + motion + extraSources;
            std::memcpy(descriptors.data(), CMSG_DATA(item), descriptorCount * sizeof(int));
            std::optional<odin3::shared_frame::DmaSourcePool> describedSources;
            if (request.op == OpenDescribed) {
                try {
                    if (!pooled) throw std::invalid_argument("Described sources require direct mode");
                    const std::array inputFds{descriptors[1], descriptors[2], descriptors[imagesEnd + 1], descriptors[imagesEnd + 2]};
                    odin3::shared_frame::validate_dma_source_pool(sourcePool, inputFds, request.width, request.height);
                    describedSources = sourcePool;
                } catch (const std::exception& error) {
                    Reply failure{};
                    failure.failed = 1;
                    std::snprintf(failure.text, sizeof failure.text, "Shared source registration rejected: %s", error.what());
                    (void)send(descriptors[0], &failure, sizeof failure, MSG_NOSIGNAL);
                    for (size_t i = 0; i < descriptorCount; ++i) close(descriptors[i]);
                    continue;
                }
            }
            const bool second = mako::htpVfiTripleEnabled() && test == Mode::HtpVfi && request.count >= 2;
            const std::array images{descriptors[1], descriptors[2], descriptors[4], second ? descriptors[5] : -1};
            const std::array extraOutputFds = pooled
                ? (second ? std::array{descriptors[6], descriptors[7], descriptors[8], descriptors[9]}
                          : std::array{descriptors[5], descriptors[6], -1, -1})
                : std::array{-1, -1, -1, -1};
            for (size_t i = 1; i < imagesEnd; ++i) {
                const bool usedForCopy = copiesFrames(test) &&
                    (i == 1 || i == 2 || i == 4 || (i == 5 && second) || (pooled && i >= 4));
                if (i != 3 && !usedForCopy) close(descriptors[i]);
            }
            const auto timeline = gpu.importTimeline(descriptors[3]);
            std::array<int, 10> spareInputFds{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
            for (size_t k = 0; k < std::min(extraSources, spareInputFds.size()); ++k)
                spareInputFds[k] = descriptors[imagesEnd + motion + k];
            contexts.emplace_back(context, descriptors[0], std::cref(gpu), timeline, test, request, images,
                                  motion ? descriptors[imagesEnd] : -1, spareInputFds, extraOutputFds, describedSources);
        }
    }
    for (auto& worker : contexts) worker.join();
    return 0;
}
