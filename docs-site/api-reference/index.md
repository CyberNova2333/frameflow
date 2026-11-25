# API 与模型格式参考

本页依据 `runtime/index.js`、`runtime/loader.js`、`runtime/pacer.js` 与 `runtime/types.d.ts` 整理运行时接口与 `format_version = 1` 模型产物规范。

## 运行时顶层函数 (`runtime/index.js`)

| 函数名 | 签名与参数 | 返回值与说明 |
| --- | --- | --- |
| `probeCapabilities(options?)` | `{ powerPreference?: 'high-performance' \| 'low-power', gl?, canvas? }` | 返回 `{ supported, recommended: 'webgpu' \| 'webgl2' \| null, webgpu, webgl2 }`，不分配会话显存 |
| `createFrameFlow(options?)` | `{ backend?: 'auto' \| 'webgpu' \| 'webgl2', device?, gl?, canvas?, precision?: 'auto' \| 'f16' \| 'f32', powerPreference? }` | 传入 `device` 默认走 `webgpu`、传入 `gl` 默认走 `webgl2`；`auto` 依次尝试并均不支持时抛错 |
| `loadModelFromUrl(engine, baseUrl, stem?)` | `baseUrl` 为目录 URL，`stem` 默认 `'model'` | 并行 `fetch` `graph.json`、`{stem}.json` 与 `{stem}.bin` 后交给 `engine.loadModel` |

## 引擎与会话方法 (`FrameFlowEngine` / `FrameFlowSession`)

| 接口 / 方法 | 关键参数说明 | 行为边界 |
| --- | --- | --- |
| `engine.createSession(model, options)` | `{ width, height, internal?: 'frame' \| 'auto' \| [w, h], k?, stillGuard?, sampling?, useFilm?, useRefine?, flipSrcY? }` | 插帧内部宽高通过 `align32()` 向上对齐到 32 倍数；超分会话直接基于输入尺寸执行 2× 输出 |
| `session.setExternalInputTexture(slot, tex)` | `slot` 整数槽位，`tex` 为同设备 `GPUTexture` 或 `WebGLTexture` | 直接绑定外部渲染纹理，零拷贝跳过 JS 像素上传 |
| `session.pushFrame(slot, source)` | 支持 `Uint8Array` / `HTMLCanvasElement` / `OffscreenCanvas` / `ImageBitmap` / `HTMLVideoElement` | 当不使用外部共享纹理时从宿主上传帧画面 |
| `session.pair(slotA, slotB)` & `session.mid(t?, outSlot?)` | `t ∈ (0, 1)`，默认 `0.5` | 先由 `pair` 完成一次编码与粗估计，再按需多次调用 `mid(t)` 生成任意相位帧 |
| `session.upscale(srcSlot?, outSlot?)` | `srcSlot=0, outSlot=0` | `GameSR` 专用：将输入提升为固定的 `(2*width, 2*height)` 分辨率 |
| `session.outputTexture(slot?)` | 输出槽位编号 | 获取生成的 GPU 纹理，供模拟器交换链或后续超分/渲染直接采样 |
| `session.readRgba(outSlot?)` | `Promise<Uint8Array>` | 显式回读 RGBA 像素；**仅用于测试或截图，禁止放入实时游玩热循环** |

## 无阻塞帧节拍器 (`FrameFlowPacer`)

- `new FrameFlowPacer({ factor?: 2 | 3, maxBufferMs?: number })`：默认 `maxBufferMs = 120`。
- `onSourceFrame(slot, nowMs)`：基于 `4–200 ms` 区间动态估算 `emaIntervalMs`，返回需排队的 `{ t, dueMs, outSlot }` 任务；若队列超过 `factor + 2` 自动丢弃最旧项并递增 `stats.dropped`。
- `tick(nowMs)`：返回下一待展示项或状态，始终免去阻塞式 GPU 回读。

## 紧凑模型产物约定 (`format_version = 1`)

- `graph.json`：含 `meta` 与 `ops` 计算步骤数组；插帧含 `interpolate`、`conv`、`sad_volume`、`film_mlp` 与 `render_pix` 等算子，超分以 `render_sr`（`scale: 2`）收尾。
- `model.json`：`{ meta: { model, task, format_version: 1, n_params, cfg }, tensors: { name: { shape, offset } } }`。
- `model.bin`：按 `offset` 紧邻存放的小端 IEEE-754 `binary16` 字节流（总字节数必为偶数）。
