/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "x11_presentation_target.hpp"

#include <cstdint>
#include <memory>
#include <vulkan/vk_layer.h>

namespace vk { class Vulkan; }

namespace mako::layer {
    struct PrivateDeviceCallbacks {
        PFN_vkLayerCreateDevice create{};
        PFN_vkLayerDestroyDevice destroy{};
        PFN_vkGetInstanceProcAddr currentLayer{};
        PFN_vkGetInstanceProcAddr nextLayer{};
        bool synchronizedGraphicsQueue{};
    };

    class IndependentRealPresent {
    public:
        IndependentRealPresent(const vk::Vulkan& app, VkInstance instance,
            const PrivateDeviceCallbacks& callbacks, X11PresentationTarget target,
            const VkSwapchainCreateInfoKHR& request, PFN_vkReleaseSwapchainImagesKHR release);
        ~IndependentRealPresent();
        IndependentRealPresent(const IndependentRealPresent&) = delete;
        IndependentRealPresent& operator=(const IndependentRealPresent&) = delete;

        static void noteFinalDraw(VkDevice device, VkImage target, VkImage source, uint64_t frame);
        VkSwapchainKHR applicationSwapchain() const;
        VkSurfaceKHR applicationSurface() const;
        VkResult acquire(const VkAcquireNextImageInfoKHR& request, uint32_t* imageIndex, bool secondApi);
        VkResult present(VkQueue queue, const VkPresentInfoKHR& request, bool& sourceSubmitted);
        VkResult waitForPresent(uint64_t id, uint64_t timeout);
        void invalidate();
        void retireDisplay();
        bool active() const;
        bool workerFailed() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
}
