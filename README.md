# Odin3-FG-NPU

在 AYN Odin 3（Snapdragon 8 Elite / SM8750）上运行的实验性帧生成。MAKO Renderer 获取游戏原帧，Adreno 830 执行运动估计与重建，Hexagon HTP v79 执行 ANVIL-S W8A8 残差网络。当前实机路径为 1280×720、SDR、固定 2×，通过 Gamescope 提呈。

当前版本修正了 Vulkan 跨设备图像交接，并在全屏菜单突然变化时使用屏幕空间重建，保留一个清晰的生成帧。《TUNIC》已验证从启动开启补帧、进入存档，以及六次打开和六次关闭暂停菜单。稳态 120 FPS 零漏帧、更多游戏和 Vela 同时启用仍待验证。实机数据和适用范围见 [状态](docs/STATUS.md)。

- [Windows 构建、模型准备和掌机运行](docs/BUILD.md)
- [依赖固定版本](sources.json)
- [许可证和来源](docs/LICENSES.md)

源码与补丁在 `native/`、`scripts/` 和 `patches/`。QAIRT、网络权重、游戏文件、编译产物、存档和完整实机采集保存在本地。
