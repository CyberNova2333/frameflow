---
layout: home

hero:
  name: "FrameFlow"
  text: "跨浏览器 AI 插帧与超分工作流套件"
  tagline: "集成 GameFI 光流插帧（Std/Lite）与 GameSR 2× 超分模型，面向网页端游戏与模拟器（PVSSPW / PPSSPW）的零依赖 WebGPU / WebGL2 运行时。"
  actions:
    - theme: brand
      text: "起步与模拟器集成"
      link: /docs/
    - theme: alt
      text: "API 与模型格式"
      link: /api-reference/
    - theme: alt
      text: "架构与源码指南"
      link: /docs/architecture

features:
  - title: "WebGPU → WebGL2 自动降级"
    details: "零 npm 依赖与零 WASM 纯 ES 模块；优先启用 shader-f16 计算着色器，无 WebGPU 时自动切换至 RGBA16F + MRT 片元着色器。"
  - title: "GameFI + GameSR 双家族"
    details: "提供 GameFI-Std（775K）与 GameFI-Lite（222K）光流插帧以及 GameSR-Std（11.5K）与 GameSR-Lite（6.9K）2× 实时超分。"
  - title: "同设备零拷贝与开源合训"
    details: "支持共享外部 GPUDevice 或 WebGL2RenderingContext 接入模拟器，配合开放许可素材、静态 HUD 抑制及双端自动化发版流水线。"
---

::: info 能力边界与核心指标
- 浏览器端 `runtime/` 总计零外部依赖，在无头 Edge 157（Intel Xe-LPG）下与 PyTorch 参考导出保持 **69–72 dB PSNR**（`GameFI`）与 **62–63 dB PSNR**（`GameSR-2x`）数值对齐。
- 原生支持 `2×`（`t=0.5`）、`3×`（`t=1/3, 2/3`）及任意相位 `t ∈ (0, 1)`，并与 `GameSR` 共享统一的 `format_version = 1` 三文件产物协议。
:::

::: warning 明确不做什么
- **不在 Git 仓库内保存模型权重或训练视频**：`*.bin` / `*.pt` 与原始素材保持忽略，已训权重通过 GitHub Release 与 CNB OCI 制品库独立分发。
- **不提供 CPU / WASM 软推理兜底**：当设备既不支持 WebGPU 也不支持半浮点/浮点帧缓冲时，`probeCapabilities()` 明确返回不支持。
- **热路径禁止阻塞回读与非 2× 超分**：`FrameFlowPacer` 不做 `mapAsync` 回读等待，`GameSRNet` 固定为 `2×` 残差像素重排放大。
:::

## 浏览器快速接入示例

```js
import { createFrameFlow, loadModelFromUrl, FrameFlowPacer } from 'frameflow/runtime';

const engine = await createFrameFlow({ backend: 'auto' }); // WebGPU -> WebGL2
const fiModel = await loadModelFromUrl(engine, '/assets/models/gamefi-lite/');
const session = engine.createSession(fiModel, {
  width: 960,
  height: 544,
  internal: [480, 288],
  k: 1,
  stillGuard: true,
});

session.pushFrame(0, frameACanvas);
session.pushFrame(1, frameBCanvas);
session.pair(0, 1);
session.mid(0.5, 0);
```
