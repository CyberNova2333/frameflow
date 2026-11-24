# FrameFlow 训练与制品发布指南（CNB 48GB GPU / GitHub Actions / 本地 GPU）

本文档说明如何从零训练 100% 自主知识产权的 **GNU AGPLv3 + 附加条款（v1.6.7-r4）自训权重**（插帧模型 `GameFI-Std` 775K 参数与蒸馏版 `GameFI-Lite` 222K 参数；2× 超分模型见 `scripts/train_sr.py`），并**免配任何跨平台密钥**，直接将打包后的权重及五份许可合规文本发布为各平台原生制品（**CNB OCI 制品库** / **GitHub Actions Artifact**），需要公开发版时再按需发到 **GitHub Release**。

---

## 一、三种训练路径与制品去向对比

| 方式 | 推荐度 | 典型算力 | `Std 100K + Lite 50K` 预计耗时 | 默认制品去向（均免配额外密钥） |
|---|---|---|---|---|
| **1. CNB 云原生构建 GPU 节点** | **首选** | `cnb:arch:amd64:gpu`（16 核 + **48GB L40 级显存**） | **约 40–75 分钟**（`batch=32`, `bf16`） | **CNB OCI 制品库** `oci.cnb.cool/cybernova2333/frameflow/weights:<tag>`（+ 可选 CNB Release） |
| **2. 本地 NVIDIA 独显** | 备选（适合加入私录游戏素材） | 如 RTX 5070 Ti 12GB（`bf16`） | **约 1.8–2.3 小时**（`batch=16`） | 本地 `dist/` → 可按需传 **GitHub Release** |
| **3. GitHub Actions（Self-Hosted GPU / 冒烟）** | 适合已有自托管 GPU Runner | 自托管 GPU Runner（或 `ubuntu-latest` 仅冒烟） | 视 Runner GPU 而定 | **GitHub Actions Artifact**（页面直下，保留 30 天；填 `release_tag` 时顺手建 GitHub Release） |

> GitHub 免费公共 Runner（`ubuntu-latest`）无独立 GPU，仅适合在 `data_tier=synthetic`、步数设小（如 `100`）时做流水线冒烟；正式训练请使用 **CNB 48GB GPU**、**本地插电独显**或**挂载 GPU 的 Self-Hosted Runner**。

---

## 二、方式 A：CNB 云原生构建 48GB GPU 一键训练并直发 OCI 制品（推荐）

仓库已内置 [`.cnb.yml`](../.cnb.yml) 与 [`.cnb/web_trigger.yml`](../.cnb/web_trigger.yml)，**全程使用流水线内置临时 `CNB_TOKEN`，无需配置任何密钥仓库**：

1. 打开 CNB 仓库的 **代码 → 分支详情页（`main`）**。
2. 点击自定义按钮 **「48GB GPU 训练并发布权重制品」**（归属「模型训练」分组），选择参数：
   - **`DATA_TIER`**：
     - `starter`（~727 MB：*Big Buck Bunny 60fps* + *Tears of Steel 720p*，CC-BY 3.0，快速出第一版）
     - `full`（在 `starter` 基础上增加 *Sintel 1080p* + Xiph/SVT 4 组 50/59.94fps 实拍序列，运动覆盖更全）
     - `synthetic`（零下载快速冒烟验证）
   - **`TRAIN_PRESETS`**：默认 `both`（先训 `GameFI-Std`，再用冻结教师蒸馏 `GameFI-Lite`）。
   - **`STD_STEPS` / `LITE_STEPS` / `BATCH_SIZE`**：默认 `100000` / `50000` / `32`（48GB 显存可轻松跑 `batch=32`）。
   - **`ARTIFACT_TAG`**：例如 `weights-v0.1.0`；流水线会通过 `oras push` 直接推送至 CNB OCI 制品库，并同步更新 `:latest`。
   - **`ALSO_PUBLISH_RELEASE`**：默认 `false`（仅发 OCI 制品）；如需同时在 CNB Release 页面点鼠标下载附件，可切换为 `true`。
3. **拉取 CNB 权重制品（需要时可一键转存至 GitHub Release）**：
   ```bash
   # 在本地用 oras 拉取刚刚训完的制品到 dist/
   oras pull oci.cnb.cool/cybernova2333/frameflow/weights:weights-v0.1.0 -o dist/
   # 如需公开发布到 GitHub Release（使用本机已登录的 gh）：
   bash scripts/publish_github_release.sh weights-v0.1.0 dist CyberNova2333/frameflow
   ```

---

## 三、方式 B：GitHub Actions 训练并直发 Workflow Artifact

仓库已内置 [`.github/workflows/train-and-release.yml`](../.github/workflows/train-and-release.yml)，**默认直接发布为 Actions Artifact 制品，免配任何密钥**：

1. 打开 GitHub 仓库 → **Actions** → **Train & Publish Weights Artifact** → **Run workflow**。
2. 选择 `runner_label`：
   - 正式训练：选 `self-hosted`（需提前在有 NVIDIA GPU 的机器上注册 GitHub Actions Self-Hosted Runner）。
   - 流水线冒烟：选 `ubuntu-latest` + `data_tier: synthetic` + `std_steps: 40` + `lite_steps: 40`。
3. 填写 `artifact_tag`（默认 `weights-v0.1.0`），`release_tag` 默认留空：
   - 运行完成后直接在 Actions 运行详情页 **Artifacts** 区域下载 `frameflow-<tag>` 压缩包；
   - 如同时填写了 `release_tag`，流水线还会使用内置 `github.token` 顺手创建或更新对应的 **GitHub Release**。

---

## 四、方式 C：通用云端 GPU Notebook（如超算互联网 / 曙光智算 RTX 4090 / L20、AutoDL 等）

仓库提供了免 root、自动适配显存与 CPU 核数的一键脚本 [`scripts/run_cloud_train.sh`](../scripts/run_cloud_train.sh)（缺少系统 `ffmpeg` 时自动通过清华源安装纯 wheel `imageio-ffmpeg`，且自动复用容器内已有 PyTorch CUDA）：

- **选卡建议**：FrameFlow 训练仅占 2–4 GB 显存，优先选 **NVIDIA RTX 4090 24GB**（比 L20 48GB 单卡训练快约 35–45%，整套 `Std 100K + Lite 50K` 约 **40–55 分钟**）；其次选 **L20 48GB**。不建议在异构 DTK 加速卡上从头折腾算子兼容。

```bash
git clone https://github.com/CyberNova2333/frameflow.git
cd frameflow

# 1. 可选：先跑 1 分钟合成冒烟，确认 GPU / PyTorch / 打包全链路通畅
DATA_TIER=synthetic STD_STEPS=100 LITE_STEPS=100 bash scripts/run_cloud_train.sh

# 2. 正式训练（starter ~727MB，或改 DATA_TIER=full；自动断点续训，产出直接落在 dist/）
DATA_TIER=starter RELEASE_TAG=weights-v0.1.0 bash scripts/run_cloud_train.sh
```

> 若云端机房外网访问 `download.blender.org` 较慢，可在本机下好视频后通过 Notebook 文件管理器上传至 `data/raw/`（原文件名不变）或 `data/raw_local/`（配合 `DATA_TIER=local-only`），脚本会自动跳过下载直接切分训练。

---

## 五、方式 D：本地 RTX GPU 训练并一键发布到 GitHub Release

插上电源后，在仓库根目录执行：

```bash
# 1. 下载并切分合规开源素材（starter ~727 MB；完整版用 make data-full）
make data-starter

# （可选）将你自录的 60fps 开源游戏 / 动漫三渲二素材放进 data/raw_local/ 后运行：
# make data-local
# train-std / train-lite 会自动以 75% 开源通用 + 25% 本地素材的比例加权混训！

# 2. 训练 GameFI-Std（支持断点续训与 Ctrl+C 优雅存盘）
make train-std

# 3. 从 GameFI-Std 蒸馏 GameFI-Lite
make train-lite

# 4. 打包权重与 SHA-256 校验和，并直接发布到 GitHub Release（需已登录 gh）
.venv/bin/python scripts/package_release.py --tag weights-v0.1.0 --out dist
bash scripts/publish_github_release.sh weights-v0.1.0 dist CyberNova2333/frameflow
```

---

## 六、训练数据合规与切分机制

[`scripts/prepare_data.py`](../scripts/prepare_data.py) 仅收录可明确溯源的开放许可媒体（可用 `make data-list` / `make data-check` 核验）：

- **Blender Open Movies（CC-BY 3.0）**：*Big Buck Bunny 60fps*、*Tears of Steel 720p*、*Sintel 1080p*
- **Xiph.org / SVT Derf's Test Media**：`720p50_parkrun`、`720p50_mobcal`、`720p50_shields`、`720p5994_stockholm`
- **游戏专属增强（[`frameflow/dataset.py`](../frameflow/dataset.py)）**：
  - 切分阶段自动检测硬切镜（丢弃跨镜头三元组）并滤除片尾黑屏/静止帧，按 Clip 哈希做 95/5 无泄漏 train/val 划分；
  - 训练期按 35% 概率在 `(A, GT_t, B)` 上叠加 100% 静止的合成游戏 HUD（血条、技能文字框、圆角小地图、数字徽标），并施加显式零光流掩码监督；
  - 训练期随机采样 `t ∈ {0.25, 1/3, 0.5, 2/3, 0.75}`，使同一份权重原生支持 **2×（30→60）** 与 **3×（20→60）** 插帧。

---

## 七、制品与 Release 产物清单

每次发布的 GitHub / CNB Release 包含以下文件：

- `gamefi-std-<tag>.tar.gz` / `gamefi-lite-<tag>.tar.gz`：完整压缩包（含 `graph.json`、`model.json`、`model.bin`、五份许可合规文本 `LICENSE` / `LICENSE-ADDITIONAL_TERMS-{zh-CN,en}.md` / `IMPLEMENTATION-GUIDE.zh-en.md` / `ASSETS_LICENSES.md`、`DATA_PROVENANCE.json`、`metrics.jsonl`）
- `gamefi-std-{graph.json,model.json,model.bin}`、`gamefi-lite-{graph.json,model.json,model.bin}`：可直接供前端 / CDN 单文件下载的未压缩权重
- `SHA256SUMS.txt`：全部文件 SHA-256 校验和
