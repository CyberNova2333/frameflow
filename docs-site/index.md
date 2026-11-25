---
layout: home

hero:
  name: "FrameFlow"
  text: "跨浏览器 AI 插帧与超分工作流套件"
  tagline: "集成 GameFI 光流插帧（Std/Lite）与 GameSR 2× 超分模型，以零依赖 WebGPU Compute / WebGL2 RGBA16F 运行时服务 PVSSPW、PPSSPW 与通用 Canvas。"
  actions:
    - theme: brand
      text: "起步与模拟器接入"
      link: /docs/
    - theme: alt
      text: "API 与组件参考"
      link: /api-reference/
    - theme: alt
      text: "契约与模型格式"
      link: /docs/contracts
    - theme: alt
      text: "架构全景与图谱"
      link: /docs/architecture

features:
  - title: "WebGPU → WebGL2 自动降级"
    details: "零 npm 依赖与零 WASM 纯 ES 模块；优先启用 shader-f16 计算着色器，无 WebGPU 时自动切换至 RGBA16F + MRT 片元着色器。"
  - title: "GameFI + GameSR 双模型家族"
    details: "提供 GameFI-Std（775,811 参数）/ Lite（222,179 参数）光流插帧与 GameSR-Std / Lite（~11.5K / ~6.9K 参数）2× 实时超分。"
  - title: "同设备零拷贝与无阻塞节拍"
    details: "支持复用外部 GPUDevice（PVSSPW）或 WebGL2RenderingContext（PPSSPW），配合 FrameFlowPacer 与静态 HUD 零光流保护。"
  - title: "纯开源素材训练与合规发版"
    details: "严格基于 Blender CC-BY 3.0、Xiph、SYNLA MIT 与合成文字样本训练，随包绑定五份许可文书、DATA_PROVENANCE.json 与 SHA256SUMS。"
---

## 能力边界与明确非目标

::: info 能做到什么程度（已实现能力）
- 浏览器端零 npm/WASM 依赖运行 **WebGPU Compute**（`f16` 自动降 `f32`）与 **WebGL2**（`RGBA16F` + MRT），无头 Edge 157（Intel Xe-LPG）下与 PyTorch 参考导出数值对齐达 **69–72 dB PSNR**（`GameFI`）与 **62–63 dB PSNR**（`GameSR-2x`）。
- 支持共享外部 `GPUDevice` / `WebGL2RenderingContext`、`2×`/`3×` 与任意相位 `t ∈ (0, 1)`、`FrameFlowPacer` 节拍队列、以及可串联的 `GameFI -> GameSR-2x` 后处理流程；完整接口与参数见 [API 与组件参考](/api-reference/)。
:::

::: warning 明确不做什么（边界与硬限制）
- **绝不在 Git 仓库提交二进制权重或非商业素材**：`*.bin` / `*.pt` 与训练视频保持 gitignore 忽略，已训模型由 GitHub Releases 与 CNB OCI 配合 `SHA256SUMS.txt` 与 `DATA_PROVENANCE.json` 单独分发。
- **不做 CPU / WASM 软推理兜底**：设备不支持 WebGPU 且缺失 WebGL2 半浮点/浮点 FBO 时，`probeCapabilities()` 明确报告不支持，绝不抢占模拟器主线程算力。
- **固定 `2×` 超分且热路径禁止同步回读**：`SrModelConfig` 强制 `scale = 2`；节拍器与呈现热循环禁止调用阻塞式 `mapAsync` 或 `readRgba()`。
:::

## 架构全景

<DiagramFrame
  src="/diagrams/frameflow-architecture.html"
  title="FrameFlow 训练导出、双后端运行时与宿主零拷贝架构图（支持点击节点查看模块边界 / 平移 / 缩放）"
  aspect="940 / 330"
  new-tab-label="在新窗口全屏打开 ↗"
/>

## 快速上手演示

```js
import { probeCapabilities, createFrameFlow, loadModelFromUrl } from 'frameflow/runtime';

const caps = await probeCapabilities();
if (caps.supported) {
  const engine = await createFrameFlow({ backend: 'auto' }); // WebGPU -> WebGL2
  const fiModel = await loadModelFromUrl(engine, '/assets/models/gamefi-lite/');
  const session = engine.createSession(fiModel, {
    width: 960, height: 544, internal: [480, 288], k: 1, stillGuard: true,
  });
  session.pushFrame(0, frameACanvas);
  session.pushFrame(1, frameBCanvas);
  session.pair(0, 1);
  session.mid(0.5, 0);
}
```

## 文档分层导航

::: tip 按照读者需求快速跳转
- **接入网页端与模拟器宿主**：阅读 [浏览器与模拟器接入](/docs/integration) 与 [接入与运行排错 FAQ](/docs/troubleshooting)，对接 PVSSPW `GPUDevice` 与 PPSSPW `WebGL2RenderingContext`；
- **核对模型格式、训练与性能参数**：查阅 [GameFI / GameSR 契约与格式](/docs/contracts)、[双后端计算与节拍器调优](/docs/performance) 与 [训练、蒸馏与双端发版](/docs/training)；
- **源码追踪与架构审计**：浏览 [架构全景与交互图谱](/docs/architecture)、[一次真实运行走查](/docs/walkthrough)、[代码地图](/docs/code-map) 与 [工程决策与合规边界](/docs/decisions)。
:::
