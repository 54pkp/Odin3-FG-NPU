# 构建与运行

Windows 主机使用 Git、CPython 3.12、Zig 0.14.1、CMake、Ninja、LLVM 的 `llvm-ar.exe` 和 `llvm-ranlib.exe`，以及 `glslangValidator.exe`、`spirv-val.exe`。目标是 `aarch64-linux-gnu.2.39`。Linux sysroot 需要 X11、XCB、Xau 头文件和 ARM64 的 `libvulkan.so.1`。QAIRT 2.49.0.260730 需要自行准备 Windows 转换工具、QNN 头文件、ARM64 Linux 动态库和 v79 unsigned DSP 库。

## Windows 构建

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
& ./.venv/Scripts/python.exe scripts/prepare_sources.py
& ./scripts/build_mako_arm64.ps1
& ./.venv/Scripts/python.exe scripts/build_mako_anvil_worker.py
& ./.venv/Scripts/python.exe scripts/build_kfg_anvil_probe.py
& ./.venv/Scripts/python.exe scripts/build_daemon.py
& ./.venv/Scripts/python.exe scripts/stage_runtime.py
```

`prepare_sources.py` 从固定提交获取依赖，并在 MAKO 工作树应用 `patches/mako-odin3-anvil.patch`。可重复运行。默认路径和可覆盖的环境变量如下。

| 输入或输出 | 默认位置 | 环境变量 |
| --- | --- | --- |
| 上游源码树 | `local/sources` | `ODIN3_SOURCE_ROOT` |
| 产物 | `build` | `ODIN3_BUILD_ROOT` |
| Zig | `local/tools/zig/zig.exe` | `ODIN3_ZIG` |
| CMake | `local/tools/cmake/bin/cmake.exe` | `ODIN3_CMAKE` |
| Ninja | `local/tools/ninja/ninja.exe` | `ODIN3_NINJA` |
| LLVM 归档工具目录 | `local/tools/llvm/bin` | `ODIN3_LLVM_TOOLS` |
| glslang/SPIR-V 工具目录 | `local/tools/glsl` | `ODIN3_GLSL_TOOLS` |
| Linux 头文件 | `local/sysroot/include` | `ODIN3_SYSROOT_INCLUDE` |
| ARM64 Vulkan loader | `local/sysroot/lib/libvulkan.so.1` | `ODIN3_VULKAN_LOADER` |
| QAIRT SDK 根目录 | `local/qairt-sdk` | `ODIN3_QAIRT_SDK` |
| 单独解压的 QNN 头文件目录 | `<SDK>/include/QNN` | `ODIN3_QNN_INCLUDE` |

## ANVIL-S W8A8 上下文

1. 从发布方取得 ANVIL-S 的 1080p ONNX。`specialize_720p.py` 核对 SHA-256 `65e871aed25b331e3a3747cf9019942121a94af71997476f5804a24b768cd0d8`，再修改 720p 相关的形状常量。 [发布权重](https://huggingface.co/NihilDigit/anvil/tree/fa97e30f6aabe897bd4f6f82d971ad607f5d61fa/onnx)
2. 用 `scripts/video_baseline.py --upstream local/sources/anvil --video <校准视频> --output local/calibration --resolution 720p --pairs 12` 准备校准帧。实机记录使用上游发布的 *tractor*。 [发布样例](https://github.com/NihilDigit/mpv-android-anvil/tree/1efc82739f6f7034cd63e7ee8b3ec5431df27467/app/src/main/assets/anvil/demo)
3. 用 `.venv/Scripts/python.exe scripts/specialize_720p.py <发布的ONNX> local/anvil-s-720p.onnx` 生成 720p 模型。再用 QAIRT 对应的 Python 环境运行 `scripts/compile_w8a8.py --sdk <SDK根目录> --model local/anvil-s-720p.onnx --calibration local/calibration --output local/quantized --width 1280 --height 720 --io-layout NHWC`。
4. 在奥丁 3 的临时目录放入 `scripts/run_htp.sh`、`anvil-s.w8a8.dlc`、`qairt-linux/`，以及一个 `calibration-nhwc/` 下的浮点验证样本（命名为 `pair.raw`）。执行 `bash run_htp.sh result 30`，生成 `result/anvil-s-v79.bin` 并运行 HTP 验证。`scripts/stage_qairt.py --output <目录>` 可从固定版本 SDK 归档提取这条路径所需的 Linux 文件。

运行阶段的模型 ABI 为 `anvil_s_float`，输入 U8 NHWC `1×720×1280×6`，输出 U8 NHWC `1×720×1280×3`。输入编码 scale 为 `1/255`、offset 为 `0`，输出 scale 为 `0.003843763610`、offset 为 `−126`。重新校准后应核对编码，编码变化时同步更新 `native/mako_anvil.comp` 中的残差反量化常量。

## 奥丁 3 手动验证

把 `build/runtime` 复制到掌机用户目录。把对应的 `qairt-linux` 和 `anvil-s-v79.bin` 放在运行目录，或分别用 `ODIN3_QAIRT_LINUX` 和 `ODIN3_ANVIL_CONTEXT` 指定。赋予脚本、`anvil-npud` 与 `mako-anvil-worker` 执行权限后，在终端前台运行 `./start-htp.sh`。等待私有 socket 报告 `backend=qnn graph=anvil_s_float`。

为测试游戏使用单独的临时启动项，让 `odin3-fg.sh` 承接该游戏原始启动命令。游戏设为 1280×720 SDR。TUNIC 的 DXVK 对照使用 `DXVK_FRAME_RATE=60` 和 120 Hz 显示模式。可用例如 `DXVK_FRAME_RATE=60 /absolute/path/odin3-fg.sh -- %command%`；如原启动项中含系统包装器，把它保留在 `--` 之后。验证结束后恢复原启动项并结束前台 daemon。

这一包装器沿用已验证的 MAKO、Vela 和 LSFG 隔离参数。仓库打包脚本与启动配置应优先用于私有测试，游戏及 Proton 前缀的备份和复制由测试者管理。
