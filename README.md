# FrameFlow

**FrameFlow** 是一款面向浏览器与网页模拟器（如 PVSSPW、PPSSPW）的跨浏览器光流 AI 插帧训练与推理引擎，采用 **统一紧凑权重格式（`graph.json` + `model.json` + `model.bin`）** 与 **零依赖、零 WASM 的 WebGPU / WebGL2 双后端自动降级运行时**。

## 模型规格与实测基准

所有隐藏层通道数均为 4 的倍数，同时适配 WebGPU 向量化计算与 WebGL2 `RGBA16F` 四通道纹理打包：

| 预设 | 模型名 | 参数量 | f16 权重体积 | 典型定位 |
|---|---|---:|---:|---|
| `std` | `FrameFlow-Std` | 775,811 | ~1.52 MB | 桌面核显/独显 WebGPU、旗舰移动 WebGPU |
| `lite` | `FrameFlow-Lite` | 222,179 | ~434 KB | WebGL2 全浏览器兜底、中端移动端、PSP `1x/2x` |

**Intel Xe-LPG 核显（无头 Edge 157，带真实 GPU 完成栅栏，离线电池态）实测对齐与耗时**：

| 后端（零 WASM 纯 JS/Shader） | 模型 | PyTorch 对齐 PSNR（`t=0.5` / `t=1/3`） | 最大像素差 | `960×544`（`int 480×288`） | PSP `480×272` |
|---|---|---:|---:|---:|---:|
| **WebGPU**（`f16` 自动降 `f32`） | `FrameFlow-Std` | **71.42 / 71.52 dB** | 1 / 255 | **19.11 ms**（12.13 + 6.71） | — |
| **WebGPU**（`f16` 自动降 `f32`） | `FrameFlow-Lite` | **72.11 / 72.55 dB** | 1 / 255 | **8.67 ms**（4.95 + 3.72） | **7.85 ms** |
| **WebGL2**（`RGBA16F` + MRT） | `FrameFlow-Std` | **69.04 / 68.88 dB** | 1–2 / 255 | **26.92 ms** | — |
| **WebGL2**（`RGBA16F` + MRT） | `FrameFlow-Lite` | **71.05 / 71.33 dB** | 1 / 255 | **16.59 ms** | **9.25 ms** |

核心特性：
- **全浏览器覆盖**：优先 WebGPU Compute（Chrome/Edge/Safari 26+/Win&Mac Firefox），自动回退 WebGL2 Fragment Shaders（Firefox Linux/Android、iOS 15–18 Safari、WebView）；不支持浮点/半浮点渲染目标时优雅关闭，不做抢占模拟器 CPU 的软推理。
- **零拷贝共享上下文**：WebGPU 后端支持传入外部 `GPUDevice` + `GPUTexture`（直接共享 PVSSPW / Vitaslop 设备）；WebGL2 后端支持传入外部 `WebGL2RenderingContext` + `WebGLTexture`（直接共享 PPSSPW 同上下文纹理）。
- **任意相位 `t ∈ (0, 1)`**：同一套权重同时支持 **2×（30→60 FPS）** 与 **3×（20→60 FPS）**；内置静态 HUD 保护（训练期合成 HUD 掩码监督 + 推理期 `stillGuard`）。

## 浏览器端快速接入

```js
import { createFrameFlow, loadModelFromUrl, FrameFlowPacer } from 'frameflow/runtime';

// 自动按 WebGPU -> WebGL2 降级；也可传入 { device }（PVSSPW）或 { gl }（PPSSPW）实现零拷贝共享
const engine = await createFrameFlow({ backend: 'auto' });
const model = await loadModelFromUrl(engine, '/models/frameflow-lite/');
const session = engine.createSession(model, {
  width: 960,
  height: 544,
  internal: [480, 288], // 核显 / WebGL2 推荐档位；PSP 1x (480x272) 可用 'frame'
  k: 1,
  stillGuard: true,
});

// 喂入游戏帧（支持 HTMLCanvasElement / OffscreenCanvas / ImageBitmap / Uint8Array 或外部纹理）
session.pushFrame(0, frameACanvas);
session.pushFrame(1, frameBCanvas);
session.pair(0, 1);
session.mid(0.5, 0); // 生成 t=0.5 中间帧至 output slot 0
```

## 文档导航

- [**训练与双端发版指南** (`docs/TRAINING.md`)](docs/TRAINING.md)：CNB 48GB 云端 GPU 一键训练、GitHub Actions 流水线、本地 RTX 训练与自动写入 GitHub Release
- [**浏览器与模拟器接入指南** (`docs/INTEGRATION.md`)](docs/INTEGRATION.md)：PVSSPW（WebGPU 共享设备零拷贝）、PPSSPW（WebGL2 共享上下文零拷贝）与通用 Canvas 集成

## 目录结构

```text
frameflow/
├── .cnb.yml                     # CNB 云原生构建：push/PR CI + 48GB GPU 手动训练与发版流水线
├── .cnb/web_trigger.yml         # CNB 分支详情页自定义按钮（一键选档位/步数/同步 GitHub Release）
├── .github/workflows/
│   ├── ci.yml                   # GitHub Actions CI：Python 模型 + Node 运行时单元测试
│   └── train-and-release.yml    # GitHub Actions 手动训练、产物归档与直写 GitHub Release
├── docs/
│   ├── TRAINING.md              # CNB GPU / GitHub Actions / 本地训练与权重发布说明
│   └── INTEGRATION.md           # PVSSPW / PPSSPW / 浏览器通用接入说明
├── runtime/                     # 零依赖、零 WASM 浏览器推理后端（WebGPU + WebGL2 + Pacer + TS 类型）
│   ├── index.js                 # 统一入口 createFrameFlow() 与能力探测 probeCapabilities()
│   ├── loader.js                # 共享模型解析、校验、静态图编译与 FiLM MLP
│   ├── webgpu-backend.js        # 纯 WGSL WebGPU Compute 后端（自动 f16/f32，支持共享 GPUDevice）
│   ├── webgl2-backend.js        # 纯 GLSL ES 3.00 WebGL2 后端（RGBA16F TEXTURE_2D_ARRAY + MRT，支持共享 GL）
│   ├── pacer.js                 # 无回读阻塞 2x/3x 实时帧节拍器
│   └── types.d.ts               # 完整 TypeScript 声明
├── frameflow/                   # PyTorch 模型、算子、损失、合成 HUD 增强、数据集与权重导出器
├── scripts/
│   ├── prepare_data.py          # 下载 CC-BY / 开源 50-60fps 视频并自动切镜/滤静止帧划分 train/val
│   ├── train.py                 # bf16 训练、验证、知识蒸馏与权重自动导出
│   ├── package_release.py       # 打包权重压缩包、单文件权重、SHA256SUMS.txt 与 RELEASE_NOTES.md
│   ├── publish_github_release.sh# 自动创建/更新 GitHub Release 并上传全部权重（支持 gh 或 GITHUB_TOKEN）
│   ├── eval_pairs.py            # 在真实连续游戏帧上评测并输出局部放大对比图
│   ├── verify_backends.py/.mjs  # 自研 WebGPU + WebGL2 后端与 PyTorch 数值对齐与基准
│   └── verify_parity.py/.mjs    # 与参考 FlyHigh WebGPU 引擎对齐校验
├── tests/                       # 快速 CPU Python 单元测试 + Node 运行时加载与节拍器测试
├── Makefile
├── package.json
├── pyproject.toml
├── meta.json
├── CONTRIBUTING.md
├── LICENSE                      # GNU AGPLv3 (AGPL-3.0-only)
├── LICENSE-ADDITIONAL_TERMS-zh-CN.md  # AGPLv3 第 7 条附加条款（v1.6.7-r4 简体中文正文）
├── LICENSE-ADDITIONAL_TERMS-en.md     # 附加条款英文参考译文
├── IMPLEMENTATION-GUIDE.zh-en.md      # 附加条款工程实施指南
└── ASSETS_LICENSES.md           # 人员署名、AI 工具、依赖与资产四分区清单
```

## 双端云训练与权重发布（GitHub Release）

仓库不提交二进制权重文件（`.bin` / `.pt`），正式训练权重统一发布至 **GitHub Releases**（以及 CNB Releases）：

1. **CNB 48GB GPU 云端一键训练（推荐，算力高于移动端独显）**：
   - 在 CNB 仓库 `main` 分支详情页点击自定义按钮 **「48GB GPU 训练并发布权重」**（由 [`.cnb/web_trigger.yml`](.cnb/web_trigger.yml) 与 [`.cnb.yml`](.cnb.yml) 驱动），自动在 `cnb:arch:amd64:gpu`（16 核 + 48GB 显存）上下载 CC-BY/Xiph 素材、训练 `FrameFlow-Std` 并蒸馏 `FrameFlow-Lite`，打包发布至 CNB Release；配置密钥后可直接同步推送至 GitHub Release。
2. **GitHub Actions 训练与直写 Release**：
   - 在 GitHub **Actions → Train & Publish Weights to GitHub Release** 手动触发 [`.github/workflows/train-and-release.yml`](.github/workflows/train-and-release.yml)（支持自托管 GPU Runner 或 `ubuntu-latest` 合成冒烟），训练完成后自动上传 Workflow Artifact 并写入 GitHub Release。
3. **本地 RTX 独显训练 + 一键发版**：

```bash
# 0. 检查环境、CPU 单元测试与双后端对齐
make env-check
make test
make verify-runtime

# 1. 下载并切分 100% 合规 CC-BY / 开源 50-60fps 训练素材
#    starter (~727 MB: Big Buck Bunny 60fps + Tears of Steel)
#    full    (额外加入 Sintel + Xiph 720p50/59.94fps 实拍序列)
make data-starter
# 可选：把自录 60fps 游戏/动漫视频放进 data/raw_local/ 后执行 make data-local

# 2. 训练 FrameFlow-Std（100K 步，RTX 5070 Ti bf16 约 1.2–1.6 小时，支持断点续训）
make train-std

# 3. 从 FrameFlow-Std 蒸馏 FrameFlow-Lite（50K 步，约 35–45 分钟）
make train-lite

# 4. 在真实游戏连续运动帧上生成对比图，并打包发布到 GitHub Release
make eval-game
make package-release RELEASE_TAG=weights-v0.1.0
make publish-github RELEASE_TAG=weights-v0.1.0
```

详细参数、密钥仓库配置与 Release 文件清单见 [**`docs/TRAINING.md`**](docs/TRAINING.md)。

## 许可证与附加条款

- 本项目第一方自研代码与自训权重依据 [GNU AGPLv3](LICENSE)（`AGPL-3.0-only`）条款发布。
- **附加条款与资产规则**：本项目依据 GNU AGPLv3 第 7 条适用 [`LICENSE-ADDITIONAL_TERMS-zh-CN.md`](LICENSE-ADDITIONAL_TERMS-zh-CN.md)（v1.6.7-r4；英文参考译文：[`LICENSE-ADDITIONAL_TERMS-en.md`](LICENSE-ADDITIONAL_TERMS-en.md)，以简体中文完整版正文为准），对来源陈述、修改标识、禁止冒充、扩展模块有条件附加许可及项目标识边界作出补充说明；工程落地参考见 [`IMPLEMENTATION-GUIDE.zh-en.md`](IMPLEMENTATION-GUIDE.zh-en.md)，人员署名、AI 辅助工具、外部依赖与公开训练数据来源及项目标识综合清单见 [`ASSETS_LICENSES.md`](ASSETS_LICENSES.md)。附加要求与第 9.5–9.6 款附加许可豁免均分别核实相应版权持有人有效授权，未取得该等授权的第三方材料仍受其原有有效许可约束。
