// Included by the private MAKO worker after the Vulkan device and image helpers.
class AnvilModel
{
    static constexpr VkDeviceSize inputBytes = 1280 * 720 * 6;
    static constexpr VkDeviceSize outputBytes = 1280 * 720 * 3;
    const VulkanDevice& gpu;
    fsr4npu* client{};
    VkDeviceMemory memory{};
    VkBuffer tensors{};
    VkSampler sampler{};
    std::array<VkImageView, 3> views{};
    VkDescriptorSetLayout bindings{};
    VkDescriptorPool descriptors{};
    std::array<VkDescriptorSet, 2> sets{};
    VkPipelineLayout layout{};
    VkPipeline pack{}, resolve{};
    VkFence packDone{};
    std::array<uint32_t, 3> params{};
    bool traceMotion{false};
    std::unique_ptr<AnvilMotion, decltype(&anvil_motion_destroy)> motion{nullptr, anvil_motion_destroy};

    void importTensors(uint64_t (&offsets)[FSR4NPU_MAXT]) {
        fsr4npu_info model{};
        check(fsr4npu_connect("fsr4npud_anvil_s_playtest", 100, &client, &model) == S_OK &&
              model.backend == B_QNN && model.ntensors == 2 && model.bytes[0] == inputBytes &&
              model.bytes[1] == outputBytes && std::strcmp(model.graph, "anvil_s_float") == 0,
              "anvil-htp-graph-unavailable");
        int fd = -1;
        uint64_t bytes = 0;
        check(fsr4npu_alloc(client, 0, &fd, &bytes, offsets) == S_OK && fd >= 0, "anvil-htp-alloc-failed");
        const VkDeviceSize alignment = gpu.limits.minStorageBufferOffsetAlignment;
        check(offsets[0] <= bytes && inputBytes <= bytes - offsets[0] &&
              offsets[1] <= bytes && outputBytes <= bytes - offsets[1] &&
              offsets[0] + inputBytes <= offsets[1] &&
              offsets[0] % alignment == 0 && offsets[1] % alignment == 0 &&
              inputBytes <= gpu.limits.maxStorageBufferRange, "anvil-htp-tensor-layout");
        const VkExternalMemoryBufferCreateInfo external{
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        };
        const VkBufferCreateInfo create{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &external,
            .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        check(vkCreateBuffer(gpu.device, &create, nullptr, &tensors) == VK_SUCCESS, "anvil-tensor-buffer-failed");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(gpu.device, tensors, &req);
        const auto properties = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
            vkGetDeviceProcAddr(gpu.device, "vkGetMemoryFdPropertiesKHR"));
        VkMemoryFdPropertiesKHR compatible{.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
        check(properties && properties(gpu.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                      fd, &compatible) == VK_SUCCESS && req.size <= bytes,
              "anvil-tensor-dma-properties");
        const VkImportMemoryFdInfoKHR import{
            .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = fd};
        const VkMemoryAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &import,
            .allocationSize = bytes, .memoryTypeIndex = gpu.memoryIndex(
                req.memoryTypeBits & compatible.memoryTypeBits, 0)};
        check(vkAllocateMemory(gpu.device, &allocate, nullptr, &memory) == VK_SUCCESS &&
              vkBindBufferMemory(gpu.device, tensors, memory, 0) == VK_SUCCESS, "anvil-tensor-import-failed");
    }

    VkPipeline pipeline(const uint32_t* code, size_t size) const {
        const VkShaderModuleCreateInfo shaderInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size, .pCode = code};
        VkShaderModule shader{};
        check(vkCreateShaderModule(gpu.device, &shaderInfo, nullptr, &shader) == VK_SUCCESS, "anvil-shader-failed");
        const VkComputePipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
            .layout = layout};
        VkPipeline result{};
        check(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &info, nullptr, &result) == VK_SUCCESS,
              "anvil-pipeline-failed");
        vkDestroyShaderModule(gpu.device, shader, nullptr);
        return result;
    }

    void createDescriptors(const std::array<Image, 2>& source, Image output, VkFormat format,
                           const uint64_t (&offsets)[FSR4NPU_MAXT]) {
        const VkSamplerCreateInfo sample{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
        check(vkCreateSampler(gpu.device, &sample, nullptr, &sampler) == VK_SUCCESS, "anvil-sampler-failed");
        const std::array images{source[0].handle, source[1].handle, output.handle};
        for (size_t i = 0; i < images.size(); ++i) {
            const VkImageViewCreateInfo view{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = images[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            check(vkCreateImageView(gpu.device, &view, nullptr, &views[i]) == VK_SUCCESS, "anvil-view-failed");
        }
        if (mako::anvilMotionEnabled()) {
            const char* spv = std::getenv("MAKO_ANVIL_SPV_DIR");
            check(spv && *spv, "anvil-motion-shaders-required");
            const std::lock_guard lock(gpu.queueLock);
            motion.reset(anvil_motion_create(gpu.physical, gpu.device, gpu.queue, gpu.family, views.data(), spv));
            check(motion != nullptr, "anvil-motion-init-failed");
        }
        constexpr std::array types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        std::array<VkDescriptorSetLayoutBinding, types.size()> entries{};
        for (uint32_t i = 0; i < entries.size(); ++i)
            entries[i] = {i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        const VkDescriptorSetLayoutCreateInfo descriptorLayout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = entries.size(), .pBindings = entries.data()};
        check(vkCreateDescriptorSetLayout(gpu.device, &descriptorLayout, nullptr, &bindings) == VK_SUCCESS,
              "anvil-bindings-failed");
        const std::array<VkDescriptorPoolSize, 3> sizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}}};
        const VkDescriptorPoolCreateInfo pool{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = static_cast<uint32_t>(sets.size()), .poolSizeCount = sizes.size(), .pPoolSizes = sizes.data()};
        check(vkCreateDescriptorPool(gpu.device, &pool, nullptr, &descriptors) == VK_SUCCESS, "anvil-pool-failed");
        const std::array layouts{bindings, bindings};
        const VkDescriptorSetAllocateInfo alloc{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptors, .descriptorSetCount = layouts.size(), .pSetLayouts = layouts.data()};
        check(vkAllocateDescriptorSets(gpu.device, &alloc, sets.data()) == VK_SUCCESS, "anvil-set-failed");
        const VkDescriptorBufferInfo input{tensors, offsets[0], inputBytes};
        const VkDescriptorBufferInfo residual{tensors, offsets[1], outputBytes};
        for (size_t current = 0; current < sets.size(); ++current) {
            const VkDescriptorImageInfo previousImage{sampler, views[1 - current], VK_IMAGE_LAYOUT_GENERAL};
            const VkDescriptorImageInfo currentImage{sampler, views[current], VK_IMAGE_LAYOUT_GENERAL};
            const VkDescriptorImageInfo generated{VK_NULL_HANDLE, views[2], VK_IMAGE_LAYOUT_GENERAL};
            const VkDescriptorImageInfo aligned{sampler, motion ? anvil_motion_image(motion.get()) : views[0],
                VK_IMAGE_LAYOUT_GENERAL};
            std::array<VkWriteDescriptorSet, types.size()> writes{};
            for (uint32_t i = 0; i < writes.size(); ++i)
                writes[i] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[current],
                    .dstBinding = i, .descriptorCount = 1, .descriptorType = types[i]};
            writes[0].pImageInfo = &previousImage;
            writes[1].pImageInfo = &currentImage;
            writes[2].pBufferInfo = &input;
            writes[3].pBufferInfo = &residual;
            writes[4].pImageInfo = &generated;
            writes[5].pImageInfo = &aligned;
            vkUpdateDescriptorSets(gpu.device, writes.size(), writes.data(), 0, nullptr);
        }
    }

public:
    AnvilModel(const VulkanDevice& device, const std::array<Image, 2>& source, Image output, VkFormat format)
        : gpu(device) {
        uint64_t offsets[FSR4NPU_MAXT]{};
        importTensors(offsets);
        createDescriptors(source, output, format, offsets);
        const char* tag = std::getenv("MAKO_ANVIL_CAPTURE_TAG");
        params = {tag && std::string_view(tag) == "1", motion != nullptr};
        const char* motionTrace = std::getenv("MAKO_ANVIL_MOTION_TRACE");
        traceMotion = motionTrace && std::string_view(motionTrace) == "1";
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params)};
        const VkPipelineLayoutCreateInfo pipeLayout{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &bindings,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
        check(vkCreatePipelineLayout(gpu.device, &pipeLayout, nullptr, &layout) == VK_SUCCESS,
              "anvil-pipeline-layout-failed");
        pack = pipeline(mako_anvil_pack, sizeof mako_anvil_pack);
        resolve = format == VK_FORMAT_R16G16B16A16_SFLOAT ?
            pipeline(mako_anvil_resolve_wide, sizeof mako_anvil_resolve_wide) :
            pipeline(mako_anvil_resolve, sizeof mako_anvil_resolve);
        const VkFenceCreateInfo fence{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(gpu.device, &fence, nullptr, &packDone) == VK_SUCCESS, "anvil-fence-failed");
    }

    ~AnvilModel() {
        vkDestroyFence(gpu.device, packDone, nullptr);
        vkDestroyPipeline(gpu.device, pack, nullptr);
        vkDestroyPipeline(gpu.device, resolve, nullptr);
        vkDestroyPipelineLayout(gpu.device, layout, nullptr);
        vkDestroyDescriptorPool(gpu.device, descriptors, nullptr);
        vkDestroyDescriptorSetLayout(gpu.device, bindings, nullptr);
        motion.reset();
        for (const auto view : views)
            vkDestroyImageView(gpu.device, view, nullptr);
        vkDestroySampler(gpu.device, sampler, nullptr);
        vkDestroyBuffer(gpu.device, tensors, nullptr);
        vkFreeMemory(gpu.device, memory, nullptr);
        fsr4npu_close(client);
    }

    void tensorOwnership(VkCommandBuffer command, bool toHtp, VkAccessFlags use) const {
        const VkBufferMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = toHtp ? use : 0, .dstAccessMask = toHtp ? 0 : use,
            .srcQueueFamilyIndex = toHtp ? gpu.family : VK_QUEUE_FAMILY_FOREIGN_EXT,
            .dstQueueFamilyIndex = toHtp ? VK_QUEUE_FAMILY_FOREIGN_EXT : gpu.family,
            .buffer = tensors, .size = VK_WHOLE_SIZE};
        vkCmdPipelineBarrier(command,
            toHtp ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            toHtp ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    void dispatch(VkCommandBuffer command, size_t current, bool prepare, uint64_t sourceValue = 0) const {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, prepare ? pack : resolve);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[current], 0, nullptr);
        auto constants = params;
        constants[2] = (static_cast<uint32_t>(getpid()) << 16) |
            static_cast<uint32_t>(sourceValue & 0xffffu);
        vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(constants), constants.data());
        vkCmdDispatch(command, prepare ? 40 : 160, 90, 1);
    }

    void recordMotion(VkCommandBuffer command, size_t current) const {
        if (motion) {
            const std::lock_guard lock(gpu.queueLock);
            check(anvil_motion_record(motion.get(), command, static_cast<uint32_t>(current)) == 0,
                  "anvil-motion-record-failed");
        }
    }

    bool submitPack(VkCommandBuffer command, VkSemaphore timeline, uint64_t sourceValue) const {
        check(vkResetFences(gpu.device, 1, &packDone) == VK_SUCCESS, "anvil-pack-fence-reset");
        // The host readiness check cannot make source writes visible to this
        // device. Wait on the imported payload before acquiring the images.
        const VkTimelineSemaphoreSubmitInfo values{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &sourceValue};
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &values, .waitSemaphoreCount = 1, .pWaitSemaphores = &timeline,
            .pWaitDstStageMask = &stage,
            .commandBufferCount = 1, .pCommandBuffers = &command};
        {
            const std::lock_guard lock(gpu.queueLock);
            check(vkQueueSubmit(gpu.queue, 1, &submit, packDone) == VK_SUCCESS, "anvil-pack-submit");
        }
        check(vkWaitForFences(gpu.device, 1, &packDone, VK_TRUE, 100000000ULL) == VK_SUCCESS, "anvil-pack-timeout");
        if (motion && traceMotion) {
            uint16_t stats[4];
            anvil_motion_stats(motion.get(), stats);
            dprintf(STDERR_FILENO,
                "ANVIL fault worker: pid=%d action=motion-stats source=%llu fp16=%04x,%04x,%04x,%04x\n",
                getpid(), static_cast<unsigned long long>(sourceValue),
                unsigned(stats[0]), unsigned(stats[1]), unsigned(stats[2]), unsigned(stats[3]));
        }
        return !motion || !anvil_motion_cut(motion.get());
    }

    fsr4npu_stats infer() const {
        fsr4npu_stats stats{};
        check(fsr4npu_run(client, -1, 100, &stats) == S_OK && stats.exec_ns != 0, "anvil-htp-frame-failed");
        return stats;
    }
};
