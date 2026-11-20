"""Shared locations for the pinned source trees and local build inputs."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local"


def location(variable, default):
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


SOURCES = location("ODIN3_SOURCE_ROOT", LOCAL / "sources")
BUILD = location("ODIN3_BUILD_ROOT", ROOT / "build")
GLSL = location("ODIN3_GLSL_TOOLS", LOCAL / "tools/glsl")
ZIG = location("ODIN3_ZIG", LOCAL / "tools/zig/zig.exe")
VULKAN_LOADER = location("ODIN3_VULKAN_LOADER", LOCAL / "sysroot/lib/libvulkan.so.1")
VULKAN_HEADERS = SOURCES / "Vulkan-Headers/include"
