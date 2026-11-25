---
layout: page
sidebar: false
title: API 与格式参考
description: FrameFlow 运行时 probeCapabilities、createFrameFlow、FrameFlowSession、FrameFlowPacer、format_version=1 产物与打包发版 CLI 参考。
---

<script setup lang="ts">
import { apiSpecZh } from '../.vitepress/data/api-spec'
</script>

<ApiExplorer :spec="apiSpecZh">

### 能力探测与双后端创建 (`createFrameFlow`) {#create-frameflow}

优先尝试 WebGPU Compute（支持 `shader-f16` 自动启用，否则降回 `f32`）并无缝回退到 WebGL2 `RGBA16F` + MRT；支持共享外部 `{ device }`（PVSSPW）或 `{ gl }`（PPSSPW），均不支持时显式拒绝，不做 CPU/WASM 软推理。

### 零拷贝纹理会话 (`FrameFlowSession`) {#frameflow-session}

静态展开计算图为 `pairSteps` 与 `midSteps` 避免运行期重复分配；支持 `setExternalInputTexture` 零拷贝对接，`GameSR` 严格固定 2× 输出，`readRgba()` 严禁放进热循环。

### 无阻塞帧节拍器 (`FrameFlowPacer`) {#frameflow-pacer}

按 4–200 ms 源帧滑动平均值调度 `2×`（`t=0.5`）与 `3×`（`t=1/3, 2/3`）生成任务，超出 `factor + 2` 主动淘汰陈旧项。

### 统一三文件产物约定 (`format_version = 1`) {#model-format-v1}

由 `graph.json`、`model.json` 与小端序 IEEE-754 `binary16` `model.bin` 组成；全部隐层通道数限制为 4 的倍数。

### 数据制备与发行包打包 CLI (`package_release.py`) {#package-release-cli}

随权重归档附带五份许可文书、`DATA_PROVENANCE.json`、`metrics.jsonl` 与 `SHA256SUMS.txt` 发布至 GitHub Release 与 CNB OCI。

</ApiExplorer>
