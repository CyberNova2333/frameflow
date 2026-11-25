# 接入与运行排错 FAQ

本页汇总浏览器接入、设备能力探测、颜色/坐标系对齐与 Release 模型加载中的常见阻断与排查要点。

## 1. `probeCapabilities()` 返回 `{ supported: false }`

- **表现**：在极老旧手机 WebView 或软件渲染环境下，既无可用 `WebGPU` 适配器，也不支持 `WebGL2` 半浮点/浮点颜色渲染缓冲。
- **处理**：这是预期内的主动拒绝门禁——FrameFlow 刻意不做 CPU/WASM 软推理兜底以免拖垮游戏核心；宿主界面应依据 `caps.supported === false` 直接隐藏或禁用插帧与 AI 超分选项。

## 2. 接入 WebGL2 / WebGPU 后画面色偏（红蓝反色）或上下颠倒

- **红蓝反色**：模型内部输入约定 `[0, 1]` 范围的 **BGR** 三通道；确认输入接入路径经 `tex_in` 着色器提取或自备缓冲区时严格按 B-G-R 顺序排布。
- **上下颠倒**：WebGL FBO 纹理默认左下角为原点，而 PyTorch 与 WebGPU 约定左上角为 `y=0`。复用外部 WebGL 纹理时须在 `createSession` 传入 `flipSrcY: true`。

## 3. 多标签/独立 Canvas 调用 `copyExternalImageToTexture` 报规范错误或高耗时

- **成因**：对已调用 `transferControlToOffscreen()` 的画布从另一设备或主线程跨 `GPUDevice` 拷贝会受 Chromium 规范限制，在 Dawn 后端也会引入 14–24 ms 跨设备拷贝延迟。
- **处理**：参考[接入指南](/docs/integration)，通过 `createFrameFlow({ backend: 'webgpu', device })` 共享模拟器自身的逻辑 `GPUDevice` 并使用 `session.setExternalInputTexture()`。

## 4. `parseModel` 抛出 `format_version` 或 `missing weight` 异常

- **排查**：确认托管目录下的 `graph.json`、`model.json` 与 `model.bin` 来自同一次发版导出（切勿混用不同预设 `std`/`lite` 的单文件）；同时检查静态服务未对 `.bin` 做截断，保证字节数为偶数。
