# 浏览器与模拟器接入指南（PVSSPW / PPSSPW / 通用 Canvas）

[`runtime/`](https://github.com/CyberNova2333/frameflow/tree/main/runtime) 是零 npm 依赖、零 WASM 的纯 ES Module 推理引擎，提供 **WebGPU Compute（优先）→ WebGL2 Fragment Shader（兜底全浏览器）** 的统一接口。

## 一、能力探测与后端选择策略

```js
import { probeCapabilities, createFrameFlow, loadModelFromUrl } from 'frameflow/runtime';

const caps = await probeCapabilities();
if (!caps.supported) {
  // 极老设备（连 WebGL2 半浮点 FBO 都不支持）：直接隐藏插帧开关，不走 CPU 软推理
  console.info('FrameFlow unsupported on this GPU:', caps);
} else {
  const engine = await createFrameFlow({ backend: 'auto' }); // WebGPU -> WebGL2
  console.info('Active backend:', engine.info());
}
```

| 后端 | 适用浏览器 | 推荐模型档位 | 推荐内部分辨率 |
| --- | --- | --- | --- |
| **WebGPU**（`f16` 自动降 `f32`） | Chrome / Edge、Safari 26+（macOS/iOS 26+）、Win/Mac Firefox | 独显选 `GameFI-Std`，核显/手机选 `GameFI-Lite`（或 `Std` + `480×288`） | `960×544` 输出用 `[480, 288]`；独显可用 `'frame'` |
| **WebGL2**（`RGBA16F` + MRT） | Firefox Linux/Android、iOS 15–18 Safari、旧手机 WebView、**PPSSPW 同上下文** | **`GameFI-Lite`**（~434 KB） | PSP `1x (480×272)` 用 `'frame'`；`960×544` 用 `[480, 288]` |

## 二、从 GitHub Release 加载权重

从 [GitHub Releases](https://github.com/CyberNova2333/frameflow/releases) 下载对应发行包后，可将解压出来的 `graph.json`、`model.json`、`model.bin` 托管至同源静态目录或 CDN（推荐开启 `gzip`/`brotli` 与 `Cache-Control: immutable`）：

```js
const model = await loadModelFromUrl(engine, '/assets/models/gamefi-lite/');
```

如直接加载散文件命名（Release 中的 `gamefi-lite-graph.json` / `gamefi-lite-model.json` / `gamefi-lite-model.bin`），也可自行 `fetch` 后传入 `engine.loadModel(graphText, manifestText, binBytes)`。

## 三、接入 PVSSPW（WebGPU 同设备零拷贝）

PVSSPW（Vitaslop）自身运行在 WebGPU 上，接入时应直接复用 Vitaslop 的 `GPUDevice` 与原生显示纹理（**插帧放在超分 / FSR 1 之前**）：

```js
const engine = await createFrameFlow({ backend: 'webgpu', device: vitaslopDevice });
const model = await loadModelFromUrl(engine, '/assets/models/gamefi-lite/');
const session = engine.createSession(model, {
  width: 960,
  height: 544,
  internal: [480, 288],
  k: 1,
  stillGuard: true,
});

// 零拷贝绑定双缓冲 native present-stage GPUTexture（跳过跨设备 copyExternalImageToTexture）
session.setExternalInputTexture(0, stageTexA);
session.setExternalInputTexture(1, stageTexB);
session.pair(0, 1);
session.mid(0.5, 0);

// 取出插帧结果 GPUTexture，交给后续 present-scale / GameSR / FSR 1 渲染到屏幕
const midTexture = session.outputTexture(0);
```

> [!WARNING]
> **工程约束**：
> - 请勿对已调用 `transferControlToOffscreen()` 的 `<canvas>` 从主线程跨设备执行 `copyExternalImageToTexture`；务必共享同一 `GPUDevice`。
> - 热路径禁止调用阻塞式 `await mapAsync` 回读像素，以免阻塞呈现循环。

## 四、接入 PPSSPW（WebGL2 同上下文零拷贝）

PPSSPW 基于 Emscripten `ppsspp-wasm` 使用 WebGL2 渲染。使用 FrameFlow 的 **WebGL2 后端**可以直接复用 PPSSPW 的 `WebGL2RenderingContext`，避免 WebGL2 → WebGPU 跨 API 拷贝：

```js
const engine = await createFrameFlow({ backend: 'webgl2', gl: ppssppGl });
const model = await loadModelFromUrl(engine, '/assets/models/gamefi-lite/');
const session = engine.createSession(model, {
  width: 480,   // PSP 1x；2x 时为 960x544
  height: 272,
  internal: 'frame',
  k: 1,
  stillGuard: true,
  flipSrcY: true, // WebGL FBO 纹理为左下原点时开启 Y 翻转，与顶行对齐
});

session.setExternalInputTexture(0, pspFrameTexA);
session.setExternalInputTexture(1, pspFrameTexB);
session.pair(0, 1);
session.mid(0.5, 0);
session.present({ output: 0 });
```

## 五、实时帧节拍器（`2×` / `3×` 无阻塞调度）

```js
import { FrameFlowPacer } from 'frameflow/runtime';

const pacer = new FrameFlowPacer({ factor: 2 }); // 20 FPS 游戏可设为 factor: 3
let nextSlot = 0;

function onNewGameFrame(source, nowMs) {
  const slot = nextSlot;
  nextSlot = (nextSlot + 1) % 3;
  session.pushFrame(slot, source);

  const tasks = pacer.onSourceFrame(slot, nowMs);
  if (tasks.length && pacer.prevSlot >= 0) {
    session.pair(pacer.prevSlot, slot);
    for (const task of tasks) {
      session.mid(task.t, task.outSlot);
      task.ready = true;
    }
  }
}

function onRaf(nowMs) {
  requestAnimationFrame(onRaf);
  const frame = pacer.tick(nowMs);
  if (frame) {
    // WebGL2: session.present(frame)
    // WebGPU: session.presentToContext(canvasCtx, canvasFormat, frame)
  }
}
```
