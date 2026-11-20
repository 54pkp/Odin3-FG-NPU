// A MAKO protocol peer for presentation testing. It either injects a fault or
// submits a real Vulkan image copy carrying a small capture signature.
#include "mako_fg_protocol.hpp"
#include "mako-common/configuration/anvil_deadline.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <sys/socket.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <vulkan/vulkan.h>
#ifdef MAKO_ANVIL_MODEL
#include "fsr4npu.h"
#include "mako_anvil_spv.hpp"
#include "mako_kfg_motion.h"
#endif

using namespace mako::backend::fg_host;

namespace {
enum class Mode { Stall, Exit, NoReply, Ready, Late, Anvil };

bool copiesFrames(Mode mode) {
    return mode == Mode::Ready || mode == Mode::Late || mode == Mode::Anvil;
}

void event(const char* action, uint64_t source = 0) {
    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    dprintf(STDERR_FILENO, "ANVIL fault worker: pid=%d action=%s source=%llu monotonic_us=%lld\n",
        getpid(), action, static_cast<unsigned long long>(source),
        static_cast<long long>(stamp));
}

void check(bool condition, const char* message) {
    if (!condition) {
        event(message);
        _exit(1);
    }
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
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceLimits limits{};
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
        const float priority = 1.0F;
        const VkDeviceQueueCreateInfo queue{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family,
            .queueCount = 1, .pQueuePriorities = &priority};
        const bool motion = mode == Mode::Anvil && mako::anvilMotionEnabled();
        const VkPhysicalDeviceVulkan12Features features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .scalarBlockLayout = motion ? VK_TRUE : VK_FALSE,
            .timelineSemaphore = VK_TRUE};
        const VkPhysicalDeviceFeatures core{
            .shaderStorageImageWriteWithoutFormat = motion ? VK_TRUE : VK_FALSE};
        const char* extensions[] = {VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
                                    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                                    VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME};
        const VkDeviceCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features,
            .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue,
            .enabledExtensionCount = mode == Mode::Anvil ? 4u : 2u,
            .ppEnabledExtensionNames = extensions, .pEnabledFeatures = &core};
        check(vkCreateDevice(selected, &ci, nullptr, &device) == VK_SUCCESS, "device-failed");
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
};

struct Image {
    VkImage handle{};
    VkDeviceMemory memory{};
};

#ifdef MAKO_ANVIL_MODEL
#include "mako_anvil_model.inl"
#endif

class CopyQueue {
    static constexpr VkExtent3D mark{32, 24, 1};
    static constexpr VkImageSubresourceLayers color{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    const VulkanDevice& gpu;
    const VkExtent3D extent;
    const VkFormat format;
    std::array<Image, 2> source;
    Image output;
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkBuffer stamp{};
    VkDeviceMemory stampMemory{};
    bool wroteOutput{false};
#ifdef MAKO_ANVIL_MODEL
    std::optional<AnvilModel> model;
    bool tensorsUsed = false;
#endif

    void begin() const {
        check(vkResetCommandBuffer(command, 0) == VK_SUCCESS, "worker-command-reset");
        const VkCommandBufferBeginInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        check(vkBeginCommandBuffer(command, &info) == VK_SUCCESS, "worker-command-begin");
    }

    void acquireOutput(VkAccessFlags write) const {
        transfer(output.handle, wroteOutput ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED,
                 wroteOutput ? gpu.family : VK_QUEUE_FAMILY_IGNORED,
                 wroteOutput ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, 0, write);
    }

#ifdef MAKO_ANVIL_MODEL
    bool generate(size_t index, VkSemaphore timeline, uint64_t sourceValue) {
        event("anvil-pack-start", sourceValue);
        for (const auto& image : source)
            transfer(image.handle, VK_QUEUE_FAMILY_EXTERNAL, gpu.family, VK_IMAGE_LAYOUT_GENERAL,
                     0, VK_ACCESS_SHADER_READ_BIT);
        model->recordMotion(command, index);
        if (tensorsUsed)
            model->tensorOwnership(command, false, VK_ACCESS_SHADER_WRITE_BIT);
        model->dispatch(command, index, true);
        for (const auto& image : source)
            transfer(image.handle, gpu.family, VK_QUEUE_FAMILY_EXTERNAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_ACCESS_SHADER_READ_BIT, 0);
        model->tensorOwnership(command, true, VK_ACCESS_SHADER_WRITE_BIT);
        check(vkEndCommandBuffer(command) == VK_SUCCESS, "anvil-pack-command-end");
        tensorsUsed = true;
        if (!model->submitPack(command, timeline, sourceValue))
            return false;
        const auto stats = model->infer();
        dprintf(STDERR_FILENO,
            "ANVIL HTP worker: pid=%d source=%llu exec_us=%llu rtt_us=%llu\n", getpid(),
            static_cast<unsigned long long>(sourceValue),
            static_cast<unsigned long long>(stats.exec_ns / 1000),
            static_cast<unsigned long long>(stats.rtt_ns / 1000));
        event("anvil-result-ready", sourceValue);
        begin();
        model->tensorOwnership(command, false, VK_ACCESS_SHADER_READ_BIT);
        acquireOutput(VK_ACCESS_SHADER_WRITE_BIT);
        model->dispatch(command, index, false, sourceValue);
        model->tensorOwnership(command, true, VK_ACCESS_SHADER_READ_BIT);
        transfer(output.handle, gpu.family, VK_QUEUE_FAMILY_EXTERNAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_SHADER_WRITE_BIT, 0);
        return true;
    }
#endif

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
    CopyQueue(const VulkanDevice& device, const Request& open, const std::array<int, 3>& images, Mode mode)
        : gpu(device), extent{open.width, open.height, 1},
          format(open.encoding == 0 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R16G16B16A16_SFLOAT) {
        check((open.encoding == 0 || open.encoding == 1) && extent.width >= mark.width &&
              extent.height >= mark.height, "copy-format-or-extent-unsupported");
        source[0] = importImage(images[0], mako::anvilSourceImageUsage);
        source[1] = importImage(images[1], mako::anvilSourceImageUsage);
        output = importImage(images[2], mako::anvilOutputImageUsage);
        const VkCommandPoolCreateInfo ci{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = gpu.family};
        check(vkCreateCommandPool(gpu.device, &ci, nullptr, &pool) == VK_SUCCESS, "copy-pool-failed");
        const VkCommandBufferAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        check(vkAllocateCommandBuffers(gpu.device, &allocate, &command) == VK_SUCCESS, "copy-command-failed");
        if (mode == Mode::Anvil) {
#ifdef MAKO_ANVIL_MODEL
            check(extent.width == 1280 && extent.height == 720, "anvil-requires-720p");
            model.emplace(gpu, source, output, format);
#else
            check(false, "anvil-model-build-required");
#endif
        } else {
            createStamp();
        }
    }

    ~CopyQueue() {
#ifdef MAKO_ANVIL_MODEL
        model.reset();
#endif
        vkDestroyCommandPool(gpu.device, pool, nullptr);
        for (const auto image : {source[0], source[1], output}) {
            vkDestroyImage(gpu.device, image.handle, nullptr);
            vkFreeMemory(gpu.device, image.memory, nullptr);
        }
        vkDestroyBuffer(gpu.device, stamp, nullptr);
        vkFreeMemory(gpu.device, stampMemory, nullptr);
    }

    bool submit(VkSemaphore timeline, uint64_t readyValue, size_t index, uint64_t* status) {
        begin();
        const uint64_t doneValue = readyValue + 1;
#ifdef MAKO_ANVIL_MODEL
        if (model) {
            const bool accepted = generate(index, timeline, readyValue);
            if (status)
                std::atomic_ref(*status).store((doneValue << 1) | accepted, std::memory_order_release);
            if (!accepted) {
                check(status != nullptr, "anvil-motion-status-missing");
                // Retire rejected work on the same device queue as its image
                // releases so reusing the source has a Vulkan memory dependency.
                const VkTimelineSemaphoreSubmitInfo values{
                    .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                    .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &doneValue};
                const VkSubmitInfo retire{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .pNext = &values, .signalSemaphoreCount = 1, .pSignalSemaphores = &timeline};
                const std::lock_guard lock(gpu.queueLock);
                check(vkQueueSubmit(gpu.queue, 1, &retire, VK_NULL_HANDLE) == VK_SUCCESS,
                      "anvil-motion-retire-failed");
                return false;
            }
        } else
#else
        static_cast<void>(status);
#endif
            recordCopy(index);
        check(vkEndCommandBuffer(command) == VK_SUCCESS, "worker-command-end");

        const VkTimelineSemaphoreSubmitInfo values{.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &readyValue,
            .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &doneValue};
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &values,
            .waitSemaphoreCount = 1, .pWaitSemaphores = &timeline, .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1, .pCommandBuffers = &command,
            .signalSemaphoreCount = 1, .pSignalSemaphores = &timeline};
        const std::lock_guard lock(gpu.queueLock);
        check(vkQueueSubmit(gpu.queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS, "worker-output-submit");
        wroteOutput = true;
        return true;
    }

private:
    void recordCopy(size_t index) const {
        const auto in = source.at(index).handle;
        transfer(in, VK_QUEUE_FAMILY_EXTERNAL, gpu.family, VK_IMAGE_LAYOUT_GENERAL,
                 0, VK_ACCESS_TRANSFER_READ_BIT);
        acquireOutput(VK_ACCESS_TRANSFER_WRITE_BIT);
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
             Request open, std::array<int, 3> images, int statusFd) {
    uint64_t* status = nullptr;
    if (statusFd >= 0) {
        void* mapping = mmap(nullptr, sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_SHARED, statusFd, 0);
        close(statusFd);
        check(mapping != MAP_FAILED, "anvil-motion-status-map-failed");
        status = static_cast<uint64_t*>(mapping);
    }
    std::optional<CopyQueue> copy;
    if (copiesFrames(mode)) copy.emplace(gpu, open, images, mode);
    reply(socket, open.width, open.height);
    event("context-open");
    uint64_t source = 1;
    uint64_t lastOutput = 0;
    size_t slot = 0;
    bool pending = false;
    Request request{};
    while (recv(socket, &request, sizeof request, 0) == ssize_t(sizeof request)) {
        if (request.op == Ready) {
            reply(socket, !pending && (!lastOutput || gpu.wait(timeline, lastOutput, 0)));
            continue;
        }
        const bool async = mako::anvilAsyncScheduleEnabled();
        check(request.op == (async ? ScheduleAsync : Schedule) &&
              request.count <= 1 && !pending, "unexpected-request");
        if (request.count == 0) {
            if (!async) reply(socket);
            event("history", source++);
            ++slot;
            continue;
        }
        pending = true;
        if (mode == Mode::NoReply) {
            event("schedule-reply-withheld", source);
            continue;
        }
        if (!async) reply(socket);
        check(gpu.wait(timeline, source, 2000000000ULL), "source-wait-failed");
        event("source-ready", source);
        if (copy) {
            if (mode == Mode::Late && lastOutput == 0) {
                event("delay-first-output", source);
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            const bool accepted = copy->submit(timeline, source, slot++ % 2, status);
            lastOutput = source + 1;
            event(accepted ? "gpu-output-submitted" : "motion-rejected", source);
            source += 2;
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
    if (status)
        munmap(status, sizeof(*status));
    vkDestroySemaphore(gpu.device, timeline, nullptr);
    close(socket);
}
}

int main() {
    const char* value = std::getenv("MAKO_FAULT_WORKER_MODE");
    const std::string_view mode = value ? value : "";
    check(mode == "stall" || mode == "exit" || mode == "no-reply" ||
          mode == "ready" || mode == "late" || mode == "anvil", "mode-required");
    const Mode test = mode == "ready" ? Mode::Ready : mode == "late" ? Mode::Late :
        mode == "stall" ? Mode::Stall : mode == "exit" ? Mode::Exit :
        mode == "anvil" ? Mode::Anvil : Mode::NoReply;
    check(!mako::anvilAsyncScheduleEnabled() || test != Mode::NoReply, "async-mode-expects-no-reply");
    const bool motion = mako::anvilMotionEnabled();
    check(!motion || test == Mode::Anvil, "motion-mode-requires-anvil-model");
    const pid_t child = fork();
    check(child >= 0, "fork-failed");
    if (child > 0) _exit(0);
    setenv("DISABLE_MAKO", "1", 1);
    VulkanDevice gpu(test);
    std::vector<std::thread> contexts;
    while (true) {
        Request request{};
        iovec piece{&request, sizeof request};
        alignas(cmsghdr) std::array<char, CMSG_SPACE((maxOutputs + 5) * sizeof(int))> storage{};
        msghdr message{};
        message.msg_iov = &piece;
        message.msg_iovlen = 1;
        message.msg_control = storage.data();
        message.msg_controllen = storage.size();
        const ssize_t bytes = recvmsg(hostDescriptor, &message, MSG_CMSG_CLOEXEC);
        if (bytes < 0 && errno == EINTR) continue;
        if (bytes <= 0) break;
        check(bytes == ssize_t(sizeof request) && !(message.msg_flags & (MSG_CTRUNC | MSG_TRUNC)),
              "control-message-truncated");
        if (request.op == Hello) {
            constexpr char name[] = "ANVIL deadline fault fixture";
            const auto protocol = motion ? anvilMotionProtocol :
                mako::anvilAsyncScheduleEnabled() ? anvilAsyncProtocol : hostProtocol;
            reply(hostDescriptor, protocol, sizeof(Request), name, sizeof name);
        } else if (request.op == Devices) {
            reply(hostDescriptor, 1, 0, &gpu.identity, sizeof gpu.identity);
        } else if (request.op == Supports) {
            reply(hostDescriptor, 0);
        } else {
            const auto* item = CMSG_FIRSTHDR(&message);
            check(request.op == Open && request.count > 0 && request.count <= maxOutputs && item &&
                  item->cmsg_level == SOL_SOCKET && item->cmsg_type == SCM_RIGHTS &&
                  item->cmsg_len == CMSG_LEN((request.count + 4 + motion) * sizeof(int)), "control-open-invalid");
            std::array<int, maxOutputs + 5> descriptors;
            const size_t imagesEnd = request.count + 4;
            const size_t descriptorCount = imagesEnd + motion;
            std::memcpy(descriptors.data(), CMSG_DATA(item), descriptorCount * sizeof(int));
            const std::array images{descriptors[1], descriptors[2], descriptors[4]};
            for (size_t i = 1; i < imagesEnd; ++i) {
                const bool usedForCopy = copiesFrames(test) && (i == 1 || i == 2 || i == 4);
                if (i != 3 && !usedForCopy) close(descriptors[i]);
            }
            const auto timeline = gpu.importTimeline(descriptors[3]);
            contexts.emplace_back(context, descriptors[0], std::cref(gpu), timeline, test, request, images,
                                  motion ? descriptors[imagesEnd] : -1);
        }
    }
    for (auto& worker : contexts) worker.join();
    return 0;
}
