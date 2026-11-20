/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "mako_kfg_motion.h"
#include "kfg_vk.h"
#include <stdlib.h>
#include <string.h>

int kfg_vk_borrow(kfg_vk*, VkPhysicalDevice, VkDevice, VkQueue, uint32_t, const char*);

enum { width = 1280, height = 720, input_pair = 2048, stats_bytes = 8 };

struct AnvilMotion {
	kfg_vk vk;
	kfg_vk_image color[2];
	kfg_vk_image luma[2][KFG_LEVELS + 1];
	kfg_vk_image flow[KFG_LEVELS + 1];
	kfg_vk_image stats, scratch[2], selection, aligned;
	VkBuffer cut_buffer;
	VkDeviceMemory cut_memory;
	uint16_t* cut;
};

static const kfg_vk_image* image_for(void* user, kfg_image_ref ref)
{
	AnvilMotion* m = user;
	switch (ref.role) {
	case KFG_ROLE_COLOR: return &m->color[ref.ring];
	case KFG_ROLE_LUMA: return &m->luma[ref.ring][ref.level];
	case KFG_ROLE_FLOW: return &m->flow[ref.level];
	case KFG_ROLE_STATS: return &m->stats;
	case KFG_ROLE_SELECT: return &m->scratch[ref.ring & 1];
	case KFG_ROLE_PAIR_SELECT: return &m->selection;
	case KFG_ROLE_OUTPUT: return &m->aligned;
	default: return NULL;
	}
}

static int create_readback(AnvilMotion* m)
{
	const VkBufferCreateInfo ci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = stats_bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	};
	if (vkCreateBuffer(m->vk.device, &ci, NULL, &m->cut_buffer) != VK_SUCCESS)
		return -1;
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(m->vk.device, m->cut_buffer, &req);
	VkPhysicalDeviceMemoryProperties memory;
	vkGetPhysicalDeviceMemoryProperties(m->vk.phys, &memory);
	const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
		if (!(req.memoryTypeBits & (1u << i)) || (memory.memoryTypes[i].propertyFlags & wanted) != wanted)
			continue;
		const VkMemoryAllocateInfo ai = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = req.size, .memoryTypeIndex = i,
		};
		if (vkAllocateMemory(m->vk.device, &ai, NULL, &m->cut_memory) != VK_SUCCESS ||
		    vkBindBufferMemory(m->vk.device, m->cut_buffer, m->cut_memory, 0) != VK_SUCCESS ||
		    vkMapMemory(m->vk.device, m->cut_memory, 0, stats_bytes, 0, (void**)&m->cut) != VK_SUCCESS)
			return -1;
		return 0;
	}
	return -1;
}

AnvilMotion* anvil_motion_create(VkPhysicalDevice physical, VkDevice device, VkQueue queue, uint32_t family,
                                 const VkImageView sources[2], const char* spv_dir)
{
	AnvilMotion* m = calloc(1, sizeof(*m));
	if (!m)
		return NULL;
	if (kfg_vk_borrow(&m->vk, physical, device, queue, family, spv_dir))
		goto fail;
	kfg_analysis_pixels = 320 * 180;
	uint32_t analysis[2];
	kfg_analysis_size(width, height, analysis);
	for (uint32_t i = 0; i < 2; ++i) {
		m->color[i].view = sources[i];
		for (uint32_t level = 1; level <= KFG_LEVELS; ++level) {
			uint32_t dims[2];
			kfg_level_size(analysis[0], analysis[1], level, dims);
			if (kfg_vk_image_create(&m->vk, &m->luma[i][level], VK_FORMAT_R8_UNORM, dims[0], dims[1]))
				goto fail;
			if (i == 0 && level >= 2 &&
			    kfg_vk_image_create(&m->vk, &m->flow[level], VK_FORMAT_R16G16B16A16_SFLOAT, dims[0], dims[1]))
				goto fail;
		}
	}
	uint32_t dims[2];
	kfg_select_size(analysis[0], analysis[1], dims);
	for (uint32_t i = 0; i < 2; ++i)
		if (kfg_vk_image_create(&m->vk, &m->scratch[i], VK_FORMAT_R16G16B16A16_SFLOAT, dims[0], dims[1]))
			goto fail;
	if (kfg_vk_image_create(&m->vk, &m->selection, VK_FORMAT_R16G16B16A16_SFLOAT, dims[0], dims[1]) ||
	    kfg_vk_image_create(&m->vk, &m->stats, VK_FORMAT_R16G16B16A16_SFLOAT, 7, 1) ||
	    kfg_vk_image_create(&m->vk, &m->aligned, VK_FORMAT_R8G8B8A8_UNORM, 2 * width, height) ||
	    create_readback(m))
		goto fail;
	return m;
fail:
	anvil_motion_destroy(m);
	return NULL;
}

void anvil_motion_destroy(AnvilMotion* m)
{
	if (!m)
		return;
	if (m->cut)
		vkUnmapMemory(m->vk.device, m->cut_memory);
	vkDestroyBuffer(m->vk.device, m->cut_buffer, NULL);
	vkFreeMemory(m->vk.device, m->cut_memory, NULL);
	for (uint32_t level = 1; level <= KFG_LEVELS; ++level) {
		kfg_vk_image_destroy(&m->vk, &m->luma[0][level]);
		kfg_vk_image_destroy(&m->vk, &m->luma[1][level]);
		kfg_vk_image_destroy(&m->vk, &m->flow[level]);
	}
	kfg_vk_image_destroy(&m->vk, &m->scratch[0]);
	kfg_vk_image_destroy(&m->vk, &m->scratch[1]);
	kfg_vk_image_destroy(&m->vk, &m->selection);
	kfg_vk_image_destroy(&m->vk, &m->stats);
	kfg_vk_image_destroy(&m->vk, &m->aligned);
	kfg_vk_destroy(&m->vk);
	free(m);
}

VkImageView anvil_motion_image(const AnvilMotion* m) { return m->aligned.view; }

int anvil_motion_record(AnvilMotion* m, VkCommandBuffer command, uint32_t current)
{
	kfg_vk* vk = &m->vk;
	if (vkResetDescriptorPool(vk->device, vk->dpool, 0) != VK_SUCCESS)
		return -1;
	vk->ubo_used = 0;
	const VkCommandBuffer internal = vk->cmd;
	vk->cmd = command;
	const uint32_t previous = 1 - current;
	const kfg_tuning tune = kfg_default_tuning();
	kfg_plan plan;
	kfg_plan_analyze(&plan, width, height, current, previous, previous, 0, -1, &tune);
	int error = kfg_vk_record(vk, &plan, image_for, m);
	if (!error) {
		kfg_plan_analyze(&plan, width, height, previous, current, current, 1, -1, &tune);
		error = kfg_vk_record(vk, &plan, image_for, m);
	}
	if (!error) {
		kfg_plan_generate(&plan, width, height, previous, current, current, 0.5f, input_pair);
		error = kfg_vk_record(vk, &plan, image_for, m);
	}
	vk->cmd = internal;
	if (error)
		return -1;

	const VkBufferImageCopy copy = {
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageExtent = {1, 1, 1},
	};
	vkCmdCopyImageToBuffer(command, m->stats.image, VK_IMAGE_LAYOUT_GENERAL, m->cut_buffer, 1, &copy);
	const VkMemoryBarrier host = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
	};
	vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
	                     0, 1, &host, 0, NULL, 0, NULL);
	return 0;
}

int anvil_motion_cut(const AnvilMotion* m)
{
	// The R16 stats field is FP16. Zero is ordinary motion and 2.0 is
	// screen-space composition. Unknown values fail closed like a hard cut.
	return m->cut[0] != 0 && m->cut[0] != 0x4000;
}

void anvil_motion_stats(const AnvilMotion* m, uint16_t stats[4])
{
	memcpy(stats, m->cut, stats_bytes);
}
