/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct AnvilMotion AnvilMotion;
AnvilMotion* anvil_motion_create(VkPhysicalDevice physical, VkDevice device, VkQueue queue, uint32_t family,
                                 const VkImageView sources[2], const char* spv_dir);
void anvil_motion_destroy(AnvilMotion* motion);
VkImageView anvil_motion_image(const AnvilMotion* motion);
int anvil_motion_record(AnvilMotion* motion, VkCommandBuffer command, uint32_t current);
/* Call after the fence for anvil_motion_record completes. */
int anvil_motion_cut(const AnvilMotion* motion);
void anvil_motion_stats(const AnvilMotion* motion, uint16_t stats[4]);
#ifdef __cplusplus
}
#endif
