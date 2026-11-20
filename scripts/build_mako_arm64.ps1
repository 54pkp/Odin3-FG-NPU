param(
    [string]$BuildDirectory,
    [ValidateRange(1, 32)][int]$Jobs = 6,
    [switch]$ConfigureOnly,
    [switch]$BuildFaultWorker
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
function Get-LocalPath([string]$Variable, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Variable)
    if (!$value) { $value = $Default }
    return [IO.Path]::GetFullPath($value)
}
$sources = Get-LocalPath ODIN3_SOURCE_ROOT "$root/local/sources"
$buildRoot = Get-LocalPath ODIN3_BUILD_ROOT "$root/build"
$cmake = Get-LocalPath ODIN3_CMAKE "$root/local/tools/cmake/bin/cmake.exe"
$ninja = Get-LocalPath ODIN3_NINJA "$root/local/tools/ninja/ninja.exe"
$zig = Get-LocalPath ODIN3_ZIG "$root/local/tools/zig/zig.exe"
$llvm = Get-LocalPath ODIN3_LLVM_TOOLS "$root/local/tools/llvm/bin"
$headers = Get-LocalPath ODIN3_SYSROOT_INCLUDE "$root/local/sysroot/include"
$vulkan = "$sources/Vulkan-Headers/include"
$source = "$sources/MAKO/engine"
$build = if ($BuildDirectory) { [IO.Path]::GetFullPath($BuildDirectory) } else { "$buildRoot/mako" }
$dsoSource = Join-Path $root 'native/zig_dso_lifetime.c'
$target = 'aarch64-linux-gnu.2.39'

foreach ($path in @($cmake, $ninja, $zig, "$llvm/llvm-ar.exe", "$llvm/llvm-ranlib.exe",
                    "$headers/X11/Xlib.h", "$headers/xcb/xcb.h", "$vulkan/vulkan/vulkan_core.h",
                    "$source/CMakeLists.txt", $dsoSource)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required build input is missing: $path"
    }
}

New-Item -ItemType Directory -Path $build -Force | Out-Null
$dsoRevision = (Get-FileHash -Algorithm SHA256 -LiteralPath $dsoSource).Hash.Substring(0, 12)
$dsoObject = (Join-Path $build "zig-dso-$dsoRevision.o").Replace('\', '/')
if (-not (Test-Path -LiteralPath $dsoObject -PathType Leaf)) {
    & $zig cc -target $target -fPIC -Wall -Wextra -Werror -c $dsoSource -o $dsoObject
    if ($LASTEXITCODE -ne 0) { throw 'Zig DSO startup object build failed' }
}

& $cmake -S $source -B $build -G Ninja `
    '-DCMAKE_SYSTEM_NAME=Linux' '-DCMAKE_SYSTEM_PROCESSOR=aarch64' `
    "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_CXX_COMPILER=$zig" `
    "-DCMAKE_CXX_COMPILER_ARG1=c++ -target $target" `
    "-DCMAKE_AR=$llvm/llvm-ar.exe" "-DCMAKE_RANLIB=$llvm/llvm-ranlib.exe" `
    "-DCMAKE_SHARED_LINKER_FLAGS=`"$dsoObject`"" `
    "-DCMAKE_CXX_FLAGS=-isystem $headers -D_LIBCPP_ENABLE_EXPERIMENTAL" '-DCMAKE_BUILD_TYPE=Release' `
    "-DMAKO_VULKAN_HEADERS_INCLUDE_DIR=$vulkan" '-DMAKO_REQUIRE_NATIVE_PACKAGE_HEADERS=ON' `
    '-DMAKO_BUILD_VK_LAYER=ON' '-DMAKO_BUILD_CLI=OFF' '-DMAKO_BUILD_UI=OFF' `
    '-DMAKO_FG=ON' '-DMAKO_REQUIRE_FG=OFF' '-DBUILD_TESTING=OFF'
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }

if (-not $ConfigureOnly) {
    & $cmake --build $build --target mako-render --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'MAKO Renderer build failed' }
}

if ($BuildFaultWorker) {
    $vulkanLoader = Get-LocalPath ODIN3_VULKAN_LOADER "$root/local/sysroot/lib/libvulkan.so.1"
    if (-not (Test-Path -LiteralPath $vulkanLoader -PathType Leaf)) {
        throw "The AArch64 Vulkan loader used for link-time resolution is missing: $vulkanLoader"
    }
    & $zig c++ -target $target -std=c++20 -O2 -Wall -Wextra -Werror `
        -Wno-missing-field-initializers -I "$source/mako-backend/src" -I "$source/mako-common/include" -I $vulkan `
        "$root/native/mako_fault_worker.cpp" $vulkanLoader -o "$build/mako-fault-worker"
    if ($LASTEXITCODE -ne 0) { throw 'MAKO fault worker build failed' }
}
