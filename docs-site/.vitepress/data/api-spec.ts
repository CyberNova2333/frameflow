import type { ApiCatalogSpec } from './api-types';

export const apiSpecZh: ApiCatalogSpec = {
  project: 'FrameFlow · 双后端推理 API、ModelFormat v1 与训练打包 CLI 参考',
  subtitle:
    '覆盖 runtime/index.js 统一入口、WebGPU/WebGL2 零拷贝外部纹理会话、FrameFlowPacer 无阻塞调度，以及 format_version=1 与发布打包契约。',
  environments: [
    { label: '本地静态资产挂载目录', value: '/assets/models' },
    { label: 'GitHub Releases 发行包基准', value: 'https://github.com/CyberNova2333/frameflow/releases/download' },
  ],
  searchPlaceholder: '搜索 createFrameFlow、setExternalInputTexture、FrameFlowPacer、GameSR、format_version…',
  emptyText: '没有找到匹配的 FrameFlow 运行时接口、模型结构或训练脚手架。',
  labels: {
    allGroups: '全部',
    canDo: '✅ 能做到',
    cannotDo: '⛔ 明确限制 / 不做什么',
    params: '参数、选项与边界约束',
    paramName: '字段 / 参数 / 选项',
    paramLocation: '作用域',
    paramType: '类型 / 默认值',
    paramRequired: '要求',
    paramDesc: '行为说明与校验边界',
    responses: '返回值与报错拦截',
    examples: '标准调用 / CLI 示例',
    responseSample: '返回结构 / 清单示例',
    copy: '复制',
    copied: '已复制',
    noParams: '无额外参数。',
    required: '必填',
    optional: '可选',
    deepDocs: '查看对应架构文档',
    totalItems: '组接口/组件',
    envLabel: '权重托管路径',
    mobilePicker: '切换接口',
    defaultLabel: '默认',
    plainIndexToggle: '纯文本检索与打印索引（Ctrl+K 站内搜索定位）',
  },
  groups: [
    { id: 'runtime-js', title: '浏览器 ES Module 运行时与节拍调度' },
    { id: 'models-format', title: 'GameFI / GameSR 架构与 v1 产物格式' },
    { id: 'training-cli', title: '开放数据准备、训练与合规发版' },
  ],
  items: [
    {
      id: 'create-frameflow',
      group: 'runtime-js',
      kind: 'SDK',
      title: '能力探测与后端引擎初始化 (probeCapabilities / createFrameFlow)',
      signature: 'import { probeCapabilities, createFrameFlow, loadModelFromUrl } from "frameflow/runtime"',
      auth: '零 npm 依赖 · 零 WASM 纯 ES Module',
      stability: '核心运行时入口',
      summary:
        '先探测或直接构造引擎：默认 auto 优先启用 WebGPU Compute（可选开启 shader-f16 并自动降级 f32），不具备时无缝切到 WebGL2 RGBA16F + MRT；传入 { device } 或 { gl } 可与模拟器共享同上下文。',
      relatedLink: '/docs/integration',
      relatedLabel: '查看 PVSSPW / PPSSPW 接入指南',
      capabilities: [
        'probeCapabilities() 在不分配显存纹理的情况下探测 WebGPU 与 WebGL2 浮点渲染目标可用性',
        '支持 { device: GPUDevice }（PVSSPW）与 { gl: WebGL2RenderingContext }（PPSSPW）同设备/同上下文共享',
      ],
      limitations: [
        '不做 CPU / WASM 软推理兜底：若 WebGPU 与 WebGL2 半浮点/浮点 FBO 均不可用，createFrameFlow 直接抛错拒绝',
        '不可将已调用 transferControlToOffscreen() 的 Canvas 从主线程跨 GPUDevice 做 copyExternalImageToTexture',
      ],
      params: [
        { name: 'backend', location: 'config', type: "'auto' | 'webgpu' | 'webgl2'", defaultValue: "'auto'", description: '传入 device 时自动倾向 webgpu，传入 gl 时倾向 webgl2' },
        { name: 'device', location: 'config', type: 'GPUDevice', description: '复用宿主既有 WebGPU 逻辑设备以实现零拷贝绑定' },
        { name: 'gl', location: 'config', type: 'WebGL2RenderingContext', description: '复用 PPSSPW 等现有 WebGL2 上下文' },
        { name: 'precision', location: 'config', type: "'auto' | 'f16' | 'f32'", defaultValue: "'auto'", description: '在支持 shader-f16 时启用半精度存储与计算，否则自动落回 f32' },
        { name: 'powerPreference', location: 'config', type: "'high-performance' | 'low-power'", defaultValue: "'high-performance'", description: '传递给 WebGPU adapter 请求的功耗偏好' },
      ],
      responses: [
        { status: 'OK · EngineReady', tone: 'ok', summary: '返回 FrameFlowEngine 实例，可通过 engine.info() 读取后端类型与浮点精度' },
        { status: 'Error · UnsupportedGPU', tone: 'err', summary: 'WebGPU 失败且 WebGL2 缺失半浮点/浮点帧缓冲扩展时抛出显式错误，不占用模拟线程' },
      ],
      examples: [
        {
          label: '浏览器自动探测并共享设备',
          lang: 'js',
          code: "import { probeCapabilities, createFrameFlow, loadModelFromUrl } from 'frameflow/runtime';\n\nconst caps = await probeCapabilities();\nif (!caps.supported) throw new Error('GPU unsupported');\n\nconst engine = await createFrameFlow({ backend: 'auto', precision: 'auto' });\nconst model = await loadModelFromUrl(engine, '{{BASE_URL}}/gamefi-lite/');",
        },
      ],
    },
    {
      id: 'frameflow-session',
      group: 'runtime-js',
      kind: 'CLASS',
      title: '零拷贝纹理会话与推理管线 (FrameFlowSession)',
      signature: 'engine.createSession(model, { width, height, internal, k, stillGuard, flipSrcY })',
      auth: '支持 setExternalInputTexture 零拷贝输入',
      stability: '核心会话 API',
      summary:
        '在创建期经 compilePlan 静态展开算子图并一次性分配显存缓冲区；插帧分为 pair(A, B) 编码与一次或多次 mid(t, outSlot) 合成，超分调用 upscale(srcSlot, outSlot) 输出固定 2× 结果。',
      relatedLink: '/docs/performance',
      relatedLabel: '查看双后端与调优细节',
      capabilities: [
        'pair() 与 mid() 分离：3x 模式下 pair 仅运行一次即可连续合成 t=1/3 与 t=2/3 两帧中间画面',
        '支持 setExternalInputTexture() 直接接入原生游戏渲染纹理，免去像素往返拷贝',
      ],
      limitations: [
        'GameSR 会话严格固定 2× 输出（outW = 2*width, outH = 2*height），不支持其他任意放大倍率',
        'readRgba() 会引发 GPU->CPU 显存同步等待，严格限用于单元测试或手动截图，严禁放进渲染热路径',
      ],
      params: [
        { name: 'width / height', location: 'config', type: 'number', required: true, description: '输入帧逻辑宽高；插帧内部工作平面会经 align32() 向上对齐至 32 倍数' },
        { name: 'internal', location: 'config', type: "'frame' | 'auto' | [number, number]", defaultValue: "'auto'", description: '核显/移动端运行 960x544 推荐使用 [480, 288] 降低代价体压力' },
        { name: 'k', location: 'config', type: 'number', defaultValue: '1', description: 'FineNet 精细光流迭代次数' },
        { name: 'flipSrcY', location: 'config', type: 'boolean', defaultValue: 'false', description: '接入左下原点 WebGL FBO 纹理时设 true 以对齐 BGR 顶行坐标系' },
      ],
      responses: [
        { status: 'OK · SingleSubmit', tone: 'ok', summary: 'pair() 与 mid()/upscale() 将当次算子链一次性编码提交至 GPUQueue 或 WebGL FBO' },
        { status: 'Error · InvalidGraph', tone: 'err', summary: 'model.bin 字节数为奇数、format_version !== 1 或张量越界时在加载期抛出异常' },
      ],
      examples: [
        {
          label: 'GameFI + GameSR 串联零拷贝示例',
          lang: 'js',
          code: "const fi = engine.createSession(fiModel, { width: 960, height: 544, internal: [480, 288], k: 1 });\nfi.setExternalInputTexture(0, texA);\nfi.setExternalInputTexture(1, texB);\nfi.pair(0, 1);\nfi.mid(0.5, 0);\nconst midTex = fi.outputTexture(0);",
        },
      ],
    },
    {
      id: 'frameflow-pacer',
      group: 'runtime-js',
      kind: 'CLASS',
      title: '无阻塞 2× / 3× 实时帧节拍器 (FrameFlowPacer)',
      signature: 'new FrameFlowPacer({ factor: 2 | 3, maxBufferMs: 120 })',
      auth: '纯 JS 定时节奏协调（zero readback）',
      stability: '实时呈现调度',
      summary:
        '独立跟踪源帧 4–200 ms 有效到达间隔的指数滑动平均 emaIntervalMs，按 factor 自动算出 t=0.5 或 t=1/3, 2/3 中间帧排期并在队列超出 factor + 2 时丢弃过期积压帧。',
      relatedLink: '/docs/walkthrough',
      relatedLabel: '查看完整主流程 Step 5',
      capabilities: [
        '完全不等待 GPU mapAsync 栅栏即完成队列排期与到期呈现判定',
        '记录 sources / presentedSource / presentedMid / dropped 统计信息辅助调优',
      ],
      limitations: [
        '仅支持 factor 为 2 或 3；仅应在模拟器真正产出全新画面帧时调用 onSourceFrame 避免重复静止帧触发无效插值',
      ],
      params: [
        { name: 'factor', location: 'config', type: '2 | 3', defaultValue: '2', description: '倍帧系数：2x（30->60 FPS）或 3x（20->60 FPS）' },
        { name: 'maxBufferMs', location: 'config', type: 'number', defaultValue: '120', description: '平滑间隔钳位上限（毫秒）' },
      ],
      responses: [
        { status: 'OK · TickReady', tone: 'ok', summary: 'onSourceFrame() 返回需即时生成的 { t, outSlot, dueMs } 列表，tick() 返回下一呈递帧描述符' },
      ],
      examples: [
        {
          label: '3x 模式调度循环',
          lang: 'js',
          code: "import { FrameFlowPacer } from 'frameflow/runtime';\nconst pacer = new FrameFlowPacer({ factor: 3, maxBufferMs: 120 });\nconst tasks = pacer.onSourceFrame(slot, performance.now()); // [{ t: 1/3 }, { t: 2/3 }]",
        },
      ],
    },
    {
      id: 'model-format-v1',
      group: 'models-format',
      kind: 'SCHEMA',
      title: '统一紧凑模型产物规范 (format_version = 1)',
      signature: 'graph.json + model.json + model.bin (little-endian IEEE-754 binary16)',
      auth: 'AGPL-3.0-only + v1.6.7-r4 附加条款（独立打包发布）',
      stability: '冻结互操作格式',
      summary:
        'PyTorch Exporter 与浏览器 loader.js 共同遵守的三文件静态图格式。所有隐层通道强制为 4 的倍数以兼顾 WGSL vec4 计算与 WebGL2 RGBA16F 四通道纹理打包。',
      relatedLink: '/docs/contracts',
      relatedLabel: '查看模型结构与预设表',
      capabilities: [
        '同时表达 GameFI（Std 775,811 / Lite 222,179 参数）与 GameSR（Std ~11.5K / Lite ~6.9K 参数）计算图',
        '保留 bgr_to_luma（0.114*B + 0.587*G + 0.299*R）与 sad_volume 逐像素等价约束',
      ],
      limitations: [
        '严禁把 *.bin / *.pt 二进制权重提交进 Git 仓库主树，须随 SHA256SUMS.txt 从 Release 或 OCI 获取',
        'GameSRNet 初始化强制 scale = 2，不接受非二倍超分参数',
      ],
      params: [
        { name: 'format_version', location: 'meta', type: 'number (const 1)', required: true, description: 'parseModel 校验版本必须严格等于 1' },
        { name: 'task', location: 'meta', type: "'interp' | 'sr'", required: true, description: '区分时间域插帧图与 2x 超分图' },
        { name: 'tensors[k]', location: 'field', type: '{ shape: number[], offset: number }', required: true, description: '按 binary16 元素计数偏移检索 model.bin' },
      ],
      responses: [
        { status: 'OK · Parity >= 62 dB', tone: 'ok', summary: '无头 Edge 157 (Intel Xe-LPG) 与 PyTorch 黄金输出对照：GameFI 69–72 dB，GameSR 62–63 dB' },
        { status: 'Error · ShapeOrOffsetError', tone: 'err', summary: '任一隐层通道非 4 的倍数会在 Python 配置初始化立即触发 ValueError' },
      ],
      examples: [
        {
          label: 'PyTorch 导出模型产物示例',
          lang: 'python',
          code: "from frameflow import GameFINet, export_model, GameSRNet, export_sr_model\nexport_model(GameFINet('lite'), 'dist/gamefi-lite', train_step=50000)\nexport_sr_model(GameSRNet('sr_lite'), 'dist/gamesr-lite', train_step=30000)",
        },
      ],
    },
    {
      id: 'package-release-cli',
      group: 'training-cli',
      kind: 'CLI',
      title: '合训数据准备、发版打包与校验 CLI (package_release.py)',
      signature: 'python3 scripts/package_release.py --tag <tag> --presets std lite --out dist',
      auth: '自动打包五份根许可文书与 DATA_PROVENANCE.json',
      stability: '发版流水线工具',
      summary:
        '把已训最优选取导出目录（export_best / export）整理为直下散文件与合规 .tar.gz 压缩包，并自动生成覆盖全部产物的 SHA256SUMS.txt 与 RELEASE_NOTES.md。',
      relatedLink: '/docs/training',
      relatedLabel: '查看 CNB 48GB GPU 与 Release 说明',
      capabilities: [
        '训练材料限定 Blender CC-BY 3.0 / Xiph / SYNLA MIT / 合成 HUD，按 SHA1 95/5 确定性划分防泄漏',
        '同一打包逻辑同时服务 CNB OCI 制品（.cnb.yml）与 GitHub Actions（train-and-release.yml）',
      ],
      limitations: [
        '严禁 引入 CC BY-NC（例如 FlyHigh 外部参考权重）或无许可动漫/商业游戏录屏混入公开发版训练',
      ],
      params: [
        { name: '--tag', location: 'flag', type: 'string', defaultValue: 'weights-v0.1.0', description: '发行版本标签，嵌入压缩包名与 RELEASE_NOTES' },
        { name: '--presets', location: 'flag', type: 'string[]', defaultValue: 'std lite', description: '待打包预设名（支持 std/lite 或 sr_std/sr_lite）' },
        { name: '--data-manifest', location: 'flag', type: 'string', defaultValue: 'data/manifest.json', description: '写入压缩包的 DATA_PROVENANCE.json 溯源清单源路径' },
      ],
      responses: [
        { status: 'Exit 0 · dist/ Ready', tone: 'ok', summary: '写出直拉散文件、*.tar.gz 合规包、SHA256SUMS.txt 与 RELEASE_NOTES.md' },
        { status: 'RuntimeError · NoExports', tone: 'err', summary: '若检查点目录缺少完整三文件导出则中止发版' },
      ],
      examples: [
        {
          label: 'Starter 训练与发版一键命令',
          lang: 'bash',
          code: "make env-check && make test\nmake data-starter\nmake train-std && make train-lite\nmake package-release RELEASE_TAG=weights-v0.1.0",
        },
      ],
    },
  ],
};

export const apiSpecEn: ApiCatalogSpec = {
  project: 'FrameFlow · Runtime SDK, ModelFormat v1 & Packaging CLI Reference',
  subtitle:
    'Covers runtime/index.js capability probing & engine init, zero-copy WebGPU/WebGL2 sessions, FrameFlowPacer scheduling, format_version=1 artifacts, and release packaging.',
  environments: [
    { label: 'Static Self-Hosted Models Directory', value: '/assets/models' },
    { label: 'GitHub Releases Asset Base', value: 'https://github.com/CyberNova2333/frameflow/releases/download' },
  ],
  searchPlaceholder: 'Search createFrameFlow, setExternalInputTexture, FrameFlowPacer, GameSR, format_version…',
  emptyText: 'No matching FrameFlow runtime API, model schema, or training CLI found.',
  labels: {
    allGroups: 'All',
    canDo: '✅ Supported Capabilities',
    cannotDo: '⛔ Explicit Non-Goals & Limits',
    params: 'Parameters, Options & Validation Guards',
    paramName: 'Field / Option / Flag',
    paramLocation: 'Scope',
    paramType: 'Type / Default',
    paramRequired: 'Required',
    paramDesc: 'Behavior & Rejection Criteria',
    responses: 'Outcomes & Errors',
    examples: 'Usage & CLI Samples',
    responseSample: 'Payload / Manifest Sample',
    copy: 'Copy',
    copied: 'Copied',
    noParams: 'No extra parameters.',
    required: 'Required',
    optional: 'Optional',
    deepDocs: 'Open Related Guide',
    totalItems: 'APIs / Specs',
    envLabel: 'Model Base Path',
    mobilePicker: 'Select API or Spec',
    defaultLabel: 'Default',
    plainIndexToggle: 'Plaintext Search & Print Index (Ctrl+K target)',
  },
  groups: [
    { id: 'runtime-js', title: 'Browser ES Module Runtime & Pacer' },
    { id: 'models-format', title: 'GameFI / GameSR & v1 Format' },
    { id: 'training-cli', title: 'Open-Media Prep & Release CLI' },
  ],
  items: [
    {
      id: 'create-frameflow',
      group: 'runtime-js',
      kind: 'SDK',
      title: 'Capability Probing & Engine Creation (probeCapabilities / createFrameFlow)',
      signature: 'import { probeCapabilities, createFrameFlow, loadModelFromUrl } from "frameflow/runtime"',
      auth: 'Zero npm deps · Zero WASM pure ES modules',
      stability: 'Core Public Entrypoint',
      summary:
        'Tries WebGPU Compute (auto-enabling shader-f16 with seamless f32 fallback) first and degrades automatically to WebGL2 RGBA16F + MRT; accepts { device } or { gl } for zero-copy emulator sharing.',
      relatedLink: '/en/docs/integration',
      relatedLabel: 'Open Browser & Emulator Integration',
      capabilities: [
        'probeCapabilities() checks WebGPU and WebGL2 float FBO support without allocating session textures',
        'Supports zero-copy shared { device: GPUDevice } (PVSSPW) and { gl: WebGL2RenderingContext } (PPSSPW)',
      ],
      limitations: [
        'No CPU / WASM inference fallback: explicitly rejects browsers lacking both WebGPU and float/half-float FBO targets',
        'Never call cross-device copyExternalImageToTexture on an OffscreenCanvas-transferred element',
      ],
      params: [
        { name: 'backend', location: 'config', type: "'auto' | 'webgpu' | 'webgl2'", defaultValue: "'auto'", description: 'Follows device -> webgpu or gl -> webgl2 hints when supplied' },
        { name: 'device', location: 'config', type: 'GPUDevice', description: 'Reuse host GPUDevice to avoid cross-device texture copies' },
        { name: 'gl', location: 'config', type: 'WebGL2RenderingContext', description: 'Reuse host WebGL2 context' },
        { name: 'precision', location: 'config', type: "'auto' | 'f16' | 'f32'", defaultValue: "'auto'", description: 'Selects f16 WGSL compute when shader-f16 is exposed' },
      ],
      responses: [
        { status: 'OK · EngineReady', tone: 'ok', summary: 'Returns an initialized FrameFlowEngine with inspectable engine.info()' },
        { status: 'Error · UnsupportedGPU', tone: 'err', summary: 'Throws explicit unsupported error rather than starving emulator CPUs' },
      ],
      examples: [
        {
          label: 'Auto Backend Probe & Model Load',
          lang: 'js',
          code: "import { probeCapabilities, createFrameFlow, loadModelFromUrl } from 'frameflow/runtime';\n\nconst caps = await probeCapabilities();\nif (caps.supported) {\n  const engine = await createFrameFlow({ backend: 'auto' });\n  const model = await loadModelFromUrl(engine, '{{BASE_URL}}/gamefi-lite/');\n}",
        },
      ],
    },
    {
      id: 'frameflow-session',
      group: 'runtime-js',
      kind: 'CLASS',
      title: 'Static-Plan Session & Zero-Copy Execution (FrameFlowSession)',
      signature: 'engine.createSession(model, { width, height, internal, k, stillGuard, flipSrcY })',
      auth: 'Supports setExternalInputTexture()',
      stability: 'Core Session API',
      summary:
        'Unrolls repeat blocks into static pairSteps and midSteps at initialization; runs pair() once per input pair before one or more mid(t) synthesis calls or fixed 2x upscale().',
      relatedLink: '/en/docs/performance',
      relatedLabel: 'Open Dual-Backend Mechanics',
      capabilities: [
        'Decouples pair(slotA, slotB) from mid(t, outSlot) so 3x interpolation reuses encoded features',
        'Accepts external GPU textures without CPU round-trips',
      ],
      limitations: [
        'GameSR sessions strictly scale by 2x (no non-2x factor support)',
        'readRgba() stalls on GPU->CPU sync and must only be used in offline tests or captures',
      ],
      params: [
        { name: 'width / height', location: 'config', type: 'number', required: true, description: 'Input frame size; interpolation work buffers pad via align32()' },
        { name: 'internal', location: 'config', type: "'frame' | 'auto' | [number, number]", defaultValue: "'auto'", description: '[480, 288] recommended on iGPUs/WebGL2 for 960x544 frames' },
        { name: 'flipSrcY', location: 'config', type: 'boolean', defaultValue: 'false', description: 'Set true when sampling bottom-up WebGL FBO textures' },
      ],
      responses: [
        { status: 'OK · Enqueued', tone: 'ok', summary: 'Encodes pair() / mid() / upscale() passes cleanly onto the shared command queue' },
      ],
      examples: [
        {
          label: 'Shared-GPUDevice Zero-Copy Session',
          lang: 'js',
          code: "const fi = engine.createSession(fiModel, { width: 960, height: 544, internal: [480, 288], k: 1 });\nfi.setExternalInputTexture(0, texA);\nfi.setExternalInputTexture(1, texB);\nfi.pair(0, 1);\nfi.mid(0.5, 0);\nconst midTex = fi.outputTexture(0);",
        },
      ],
    },
    {
      id: 'frameflow-pacer',
      group: 'runtime-js',
      kind: 'CLASS',
      title: 'Non-Blocking 2x / 3x Frame Pacer (FrameFlowPacer)',
      signature: 'new FrameFlowPacer({ factor: 2 | 3, maxBufferMs: 120 })',
      auth: 'Zero hot-path readback stalls',
      stability: 'Core Pacer',
      summary:
        'Maintains an EMA frame interval over 4–200 ms source deltas, schedules t=0.5 or t=1/3, 2/3 mid tasks, and sheds queued frames whenever depth exceeds factor + 2.',
      relatedLink: '/en/docs/walkthrough',
      relatedLabel: 'Open Walkthrough Step 5',
      capabilities: [
        'Zero synchronous GPU fence or mapAsync inside tick() / onSourceFrame()',
        'Tracks sources, presentedSource, presentedMid, and dropped counters',
      ],
      limitations: [
        'Supports factor 2 or 3 only; feed unique newly-presented frames only',
      ],
      params: [
        { name: 'factor', location: 'config', type: '2 | 3', defaultValue: '2', description: 'Output multiplier (2x or 3x)' },
        { name: 'maxBufferMs', location: 'config', type: 'number', defaultValue: '120', description: 'Upper clamp for EMA interval (ms)' },
      ],
      responses: [
        { status: 'OK · Scheduled', tone: 'ok', summary: 'Returns mid tasks on source arrivals and source/mid present slots on tick()' },
      ],
      examples: [
        {
          label: 'Schedule 3x Interpolation',
          lang: 'js',
          code: "import { FrameFlowPacer } from 'frameflow/runtime';\nconst pacer = new FrameFlowPacer({ factor: 3 });\nconst tasks = pacer.onSourceFrame(slot, performance.now());",
        },
      ],
    },
    {
      id: 'model-format-v1',
      group: 'models-format',
      kind: 'SCHEMA',
      title: 'Unified Artifact Schema (format_version = 1)',
      signature: 'graph.json + model.json + model.bin (little-endian binary16)',
      auth: 'AGPL-3.0-only + Additional Terms v1.6.7-r4',
      stability: 'Frozen Interchange Contract',
      summary:
        'Shared 3-file format bridging PyTorch exporters and browser loaders. Enforces multiples-of-4 hidden channels for WGSL vec4 vectorization and WebGL2 RGBA16F texture packing.',
      relatedLink: '/en/docs/contracts',
      relatedLabel: 'Open Model Contracts',
      capabilities: [
        'Covers both GameFI (775,811 Std / 222,179 Lite) and GameSR (~11.5K Std / ~6.9K Lite)',
        'Guarantees >= 62 dB PSNR (69–72 dB GameFI, 62–63 dB GameSR) vs PyTorch in headless Edge 157',
      ],
      limitations: [
        'Binary weights (.bin/.pt) never live in Git; fetched separately via Releases or OCI',
        'GameSRNet strictly locks scale = 2',
      ],
      params: [
        { name: 'format_version', location: 'meta', type: '1', required: true, description: 'Validated strictly equal to 1 by parseModel()' },
        { name: 'tensors[k]', location: 'field', type: '{ shape, offset }', required: true, description: 'Element offset inside little-endian binary16 model.bin' },
      ],
      responses: [
        { status: 'OK · Validated', tone: 'ok', summary: 'Decodes to Float32Array and retains Uint16Array slices for f16 buffers' },
      ],
      examples: [
        {
          label: 'Export GameFI & GameSR',
          lang: 'python',
          code: "from frameflow import GameFINet, export_model, GameSRNet, export_sr_model\nexport_model(GameFINet('lite'), 'dist/gamefi-lite', train_step=50000)\nexport_sr_model(GameSRNet('sr_lite'), 'dist/gamesr-lite', train_step=30000)",
        },
      ],
    },
    {
      id: 'package-release-cli',
      group: 'training-cli',
      kind: 'CLI',
      title: 'Open-Media Prep & Compliance Packaging CLI (package_release.py)',
      signature: 'python3 scripts/package_release.py --tag <tag> --presets std lite --out dist',
      auth: 'Bundles 5 license files + DATA_PROVENANCE.json',
      stability: 'CI & Local Release CLI',
      summary:
        'Packages export_best/ directories into direct-fetch files, compliance tarballs, SHA256SUMS.txt, and RELEASE_NOTES.md across CNB OCI and GitHub Releases.',
      relatedLink: '/en/docs/training',
      relatedLabel: 'Open Training & Release Guide',
      capabilities: [
        'Deterministic 95/5 SHA1 split over CC-BY 3.0 / Xiph / SYNLA MIT / synthetic CJK+ASCII UI cards',
        'Synthetic static-HUD overlay supervision prevents HUD smearing',
      ],
      limitations: [
        'Zero non-commercial (CC BY-NC) or copyrighted game capture assets permitted',
      ],
      params: [
        { name: '--tag', location: 'flag', type: 'string', defaultValue: 'weights-v0.1.0', description: 'Release tag stamped in archive names and notes' },
        { name: '--presets', location: 'flag', type: 'string[]', defaultValue: 'std lite', description: 'Target presets (std/lite or sr_std/sr_lite)' },
      ],
      responses: [
        { status: 'Exit 0 · Bundled', tone: 'ok', summary: 'Outputs tar.gz bundles, SHA256SUMS.txt, and RELEASE_NOTES.md into --out' },
      ],
      examples: [
        {
          label: 'Local Training & Release Package',
          lang: 'bash',
          code: "make env-check && make test\nmake data-starter\nmake train-std && make train-lite\nmake package-release RELEASE_TAG=weights-v0.1.0",
        },
      ],
    },
  ],
};
