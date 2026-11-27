/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <chrono>
#include <cstdlib>
#include <string_view>
#include <vulkan/vulkan_core.h>

namespace mako {
    inline constexpr auto htpVfiPresentBudget = std::chrono::milliseconds(7);
    inline constexpr auto htpVfiReadyWindowCeiling = 2 * htpVfiPresentBudget;
    inline constexpr auto htpVfiScheduleReplyBudget = std::chrono::milliseconds(2);
    inline constexpr VkImageUsageFlags htpVfiSourceImageUsage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;
    inline constexpr VkImageUsageFlags htpVfiOutputImageUsage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

    inline bool htpVfiDeadlineEnabled() noexcept {
        static const bool enabled = [] {
            const char* value = std::getenv("ODIN3_HTP_DEADLINE");
            return value && std::string_view(value) == "1";
        }();
        return enabled;
    }

    inline bool htpVfiRenderBoundaryEnabled() noexcept {
        static const bool enabled = [] {
            const char* value = std::getenv("ODIN3_HTP_RENDER_BOUNDARY");
            return htpVfiDeadlineEnabled() && value && std::string_view(value) == "1";
        }();
        return enabled;
    }

    inline bool htpVfiReadyWindowEnabled() noexcept {
        static const bool enabled = [] {
            const char* value = std::getenv("ODIN3_HTP_READY_WINDOW");
            return htpVfiRenderBoundaryEnabled() && value && std::string_view(value) == "1";
        }();
        return enabled;
    }

    inline bool htpVfiAsyncScheduleEnabled() noexcept {
        static const bool enabled = [] {
            const char* value = std::getenv("ODIN3_HTP_ASYNC_SCHEDULE");
            return htpVfiDeadlineEnabled() && value && std::string_view(value) == "1";
        }();
        return enabled;
    }

    inline bool htpVfiMotionEnabled() noexcept {
        static const bool enabled = [] {
            const char* value = std::getenv("ODIN3_HTP_MOTION");
            return htpVfiAsyncScheduleEnabled() && value && std::string_view(value) == "1";
        }();
        return enabled;
    }

    inline bool htpVfiTripleEnabled() noexcept {
        static const bool enabled = [] {
            const char* value = std::getenv("ODIN3_HTP_3X");
            return htpVfiMotionEnabled() && value && std::string_view(value) == "1";
        }();
        return enabled;
    }

    inline bool htpVfiLinearOutputEnabled() noexcept {
        static const bool enabled = [] {
            const char* raw = std::getenv("ODIN3_HTP_RAW_HTP");
            const char* linear = std::getenv("ODIN3_HTP_RAW_LINEAR_OUTPUT");
            return htpVfiMotionEnabled() && raw && std::string_view(raw) == "1" &&
                linear && std::string_view(linear) == "1";
        }();
        return enabled;
    }

    inline bool htpVfiLinearInputEnabled() noexcept {
        static const bool enabled = [] {
            const char* raw = std::getenv("ODIN3_HTP_RAW_HTP");
            const char* linear = std::getenv("ODIN3_HTP_RAW_LINEAR_INPUT");
            return htpVfiMotionEnabled() && raw && std::string_view(raw) == "1" &&
                linear && std::string_view(linear) == "1";
        }();
        return enabled;
    }

    inline bool htpVfiSharedSourcesEnabled() noexcept {
        static const bool enabled = [] {
            const char* value = std::getenv("ODIN3_HTP_SHARED_SOURCES");
            return htpVfiLinearInputEnabled() && htpVfiLinearOutputEnabled() && value && std::string_view(value) == "1";
        }();
        return enabled;
    }

    inline bool htpVfiDirectDri3Enabled() noexcept {
        static const bool enabled = [] {
            const char* value = std::getenv("ODIN3_HTP_DIRECT_DRI3");
            return htpVfiSharedSourcesEnabled() && value && std::string_view(value) == "1";
        }();
        return enabled;
    }

    inline constexpr const char* odin3ModelBundleSubdir(uint32_t width, uint32_t height) noexcept {
        if (width == 640 && height == 360) return "360p";
        if (width == 960 && height == 540) return "540p";
        if (width == 1280 && height == 720) return "720p";
        if (width == 1600 && height == 900) return "900p";
        if (width == 1920 && height == 1080) return "1080p";
        return nullptr;
    }

    inline bool odin3ModelBundleHelperUnused() noexcept {
        static const bool enabled = [] {
            return true;
        }();
        return enabled;
    }
}
