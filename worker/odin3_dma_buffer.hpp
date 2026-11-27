// SPDX-License-Identifier: GPL-3.0-or-later
// External buffers shared by the Vulkan worker and HTP.
void importHtpVfiDmaBuffer(const VulkanDevice& gpu, int fd, VkDeviceSize bytes,
                          VkBufferUsageFlags usage, bool verifyImport,
                          VkBuffer& buffer, VkDeviceMemory& memory) {
    constexpr auto handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    bool dedicatedOnly = false;
    if (verifyImport) {
        const VkPhysicalDeviceExternalBufferInfo info{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            .usage = usage, .handleType = handleType};
        VkExternalBufferProperties support{.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        vkGetPhysicalDeviceExternalBufferProperties(gpu.physical, &info, &support);
        const auto features = support.externalMemoryProperties.externalMemoryFeatures;
        check(features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT, "htp-dma-not-importable");
        dedicatedOnly = features & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT;
    }
    const VkExternalMemoryBufferCreateInfo external{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = handleType};
    const VkBufferCreateInfo create{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &external,
        .size = bytes, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    check(vkCreateBuffer(gpu.device, &create, nullptr, &buffer) == VK_SUCCESS, "htp-tensor-buffer-failed");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(gpu.device, buffer, &req);
    const auto properties = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
        vkGetDeviceProcAddr(gpu.device, "vkGetMemoryFdPropertiesKHR"));
    VkMemoryFdPropertiesKHR compatible{.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    check(properties && properties(gpu.device, handleType, fd, &compatible) == VK_SUCCESS && req.size <= bytes,
          "htp-tensor-dma-properties");
    const VkMemoryDedicatedAllocateInfo dedicated{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = buffer};
    const VkImportMemoryFdInfoKHR import{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .pNext = dedicatedOnly ? &dedicated : nullptr, .handleType = handleType, .fd = fd};
    const VkMemoryAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &import,
        .allocationSize = bytes, .memoryTypeIndex = gpu.memoryIndex(req.memoryTypeBits & compatible.memoryTypeBits, 0)};
    const auto allocated = vkAllocateMemory(gpu.device, &allocate, nullptr, &memory);
    if (allocated != VK_SUCCESS)
        close(fd);
    check(allocated == VK_SUCCESS && vkBindBufferMemory(gpu.device, buffer, memory, 0) == VK_SUCCESS,
          "htp-tensor-import-failed");
}

void htpVfiDmaOwnership(const VulkanDevice& gpu, VkCommandBuffer command, VkBuffer buffer,
                       bool toHtp, VkAccessFlags access, VkPipelineStageFlags stage) {
    const VkBufferMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = toHtp ? access : 0, .dstAccessMask = toHtp ? 0 : access,
        .srcQueueFamilyIndex = toHtp ? gpu.family : VK_QUEUE_FAMILY_FOREIGN_EXT,
        .dstQueueFamilyIndex = toHtp ? VK_QUEUE_FAMILY_FOREIGN_EXT : gpu.family,
        .buffer = buffer, .size = VK_WHOLE_SIZE};
    const auto top = VkPipelineStageFlags(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    const auto bottom = VkPipelineStageFlags(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vkCmdPipelineBarrier(command, toHtp ? stage : top, toHtp ? bottom : stage,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
}
