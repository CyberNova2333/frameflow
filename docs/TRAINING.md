# FrameFlow 训练与发版指南（CNB 48GB GPU / GitHub Actions / 本地 GPU）

本文档说明如何从零训练 100% 自主知识产权的 **GNU AGPLv3 + 附加条款（v1.6.7-r4）自训权重**（`FrameFlow-Std` 775K 参数与蒸馏版 `FrameFlow-Lite` 222K 参数），并将打包后的权重及五份许可合规文本自动发布到 **GitHub Release** 与 **CNB Release**。

---

## 一、三种训练路径对比

| 方式 | 推荐度 | 典型算力 | `Std 100K + Lite 50K` 预计耗时 | 产物去向 |
|---|---|---|---|---|
| **1. CNB 云原生构建 GPU 节点** | **首选** | `cnb:arch:amd64:gpu`（16 核 + **48GB L40 级显存**） | **约 40–75 分钟**（`batch=32`, `bf16`） | CNB Release + 可选自动同步 **GitHub Release** |
| **2. 本地 NVIDIA 独显** | 备选（适合加入私录游戏素材） | 如 RTX 5070 Ti 12GB（`bf16`） | **约 1.8–2.3 小时**（`batch=16`） | 本地 `dist/` → 一键脚本写入 **GitHub Release** |
| **3. GitHub Actions（Self-Hosted GPU / 云端）** | 适合已有自托管 GPU Runner | 自托管 GPU Runner（或 `ubuntu-latest` 仅冒烟） | 视 Runner GPU 而定 | Workflow Artifact + 自动写入 **GitHub Release** |

> GitHub 免费公共 Runner（`ubuntu-latest`）无独立 GPU，仅适合在 `data_tier=synthetic`、步数设小（如 `100`）时做流水线冒烟；正式训练请使用 **CNB 48GB GPU**、**本地插电独显**或**挂载 GPU 的 Self-Hosted Runner**。

---

## 二、方式 A：CNB 云原生构建 48GB GPU 一键训练（推荐）

仓库已内置 [`.cnb.yml`](../.cnb.yml) 与 [`.cnb/web_trigger.yml`](../.cnb/web_trigger.yml)：

1. 打开 CNB 仓库的 **代码 → 分支详情页（`main`）**。
2. 点击自定义按钮 **「48GB GPU 训练并发布权重」**（归属「模型训练」分组），在弹窗中选择参数：
   - **`DATA_TIER`**：
     - `starter`（~727 MB：*Big Buck Bunny 60fps* + *Tears of Steel 720p*，CC-BY 3.0，快速出第一版）
     - `full`（在 `starter` 基础上增加 *Sintel 1080p* + Xiph/SVT 4 组 50/59.94fps 实拍序列，运动覆盖更全）
     - `synthetic`（零下载快速冒烟验证）
   - **`TRAIN_PRESETS`**：默认 `both`（先训 `FrameFlow-Std`，再用冻结教师蒸馏 `FrameFlow-Lite`）。
   - **`STD_STEPS` / `LITE_STEPS` / `BATCH_SIZE`**：默认 `100000` / `50000` / `32`（48GB 显存可轻松跑 `batch=32`）。
   - **`RELEASE_TAG`**：例如 `weights-v0.1.0`；流水线会在训练完成后自动打包并发布到 CNB Release。
3. **（可选）让 CNB 训完后直接写入 GitHub Release**：
   - 在你的 CNB 私有密钥仓库中新建 `frameflow.yml`：
     ```yaml
     GITHUB_TOKEN: github_pat_你的细粒度Token（仅需该仓库 Contents: Read and write）
     GITHUB_REPO: CyberNova2333/frameflow
     ```
   - 在 [`.cnb.yml`](../.cnb.yml) 的 `web_trigger_train` 下取消 `imports:` 注释并填入该密钥文件 URL。
   - 触发按钮时将 `PUBLISH_GITHUB_RELEASE` 选为 **是**，CNB 会在打包完成后自动调用 [`scripts/publish_github_release.sh`](../scripts/publish_github_release.sh) 直接把权重、校验和与 `RELEASE_NOTES.md` 上传到 GitHub Release。

---

## 三、方式 B：GitHub Actions 训练与直写 Release

仓库已内置 [`.github/workflows/train-and-release.yml`](../.github/workflows/train-and-release.yml)：

1. 打开 GitHub 仓库 → **Actions** → **Train & Publish Weights to GitHub Release** → **Run workflow**。
2. 选择 `runner_label`：
   - 正式训练：选 `self-hosted`（需提前在有 NVIDIA GPU 的机器上注册 GitHub Actions Self-Hosted Runner）。
   - 流水线冒烟：选 `ubuntu-latest` + `data_tier: synthetic` + `std_steps: 40` + `lite_steps: 40`。
3. 填写 `release_tag`（如 `weights-v0.1.0`）运行，完成后会自动：
   - 上传 `dist/` 为 Workflow Artifact（保留 30 天）；
   - 创建或更新对应的 **GitHub Release** 并挂载所有模型权重、压缩包与 `SHA256SUMS.txt`。

---

## 四、方式 C：本地 RTX GPU 训练并一键发布到 GitHub Release

插上电源后，在仓库根目录执行：

```bash
# 1. 下载并切分合规开源素材（starter ~727 MB；完整版用 make data-full）
make data-starter

# （可选）将你自录的 60fps 开源游戏 / 动漫三渲二素材放进 data/raw_local/ 后运行：
# make data-local
# train-std / train-lite 会自动以 75% 开源通用 + 25% 本地素材的比例加权混训！

# 2. 训练 FrameFlow-Std（支持断点续训与 Ctrl+C 优雅存盘）
make train-std

# 3. 从 FrameFlow-Std 蒸馏 FrameFlow-Lite
make train-lite

# 4. 打包权重与 SHA-256 校验和，并直接发布到 GitHub Release（需已登录 gh）
.venv/bin/python scripts/package_release.py --tag weights-v0.1.0 --out dist
bash scripts/publish_github_release.sh weights-v0.1.0 dist CyberNova2333/frameflow
```

---

## 五、训练数据合规与切分机制

[`scripts/prepare_data.py`](../scripts/prepare_data.py) 仅收录可明确溯源的开放许可媒体（可用 `make data-list` / `make data-check` 核验）：

- **Blender Open Movies（CC-BY 3.0）**：*Big Buck Bunny 60fps*、*Tears of Steel 720p*、*Sintel 1080p*
- **Xiph.org / SVT Derf's Test Media**：`720p50_parkrun`、`720p50_mobcal`、`720p50_shields`、`720p5994_stockholm`
- **游戏专属增强（[`frameflow/dataset.py`](../frameflow/dataset.py)）**：
  - 切分阶段自动检测硬切镜（丢弃跨镜头三元组）并滤除片尾黑屏/静止帧，按 Clip 哈希做 95/5 无泄漏 train/val 划分；
  - 训练期按 35% 概率在 `(A, GT_t, B)` 上叠加 100% 静止的合成游戏 HUD（血条、技能文字框、圆角小地图、数字徽标），并施加显式零光流掩码监督；
  - 训练期随机采样 `t ∈ {0.25, 1/3, 0.5, 2/3, 0.75}`，使同一份权重原生支持 **2×（30→60）** 与 **3×（20→60）** 插帧。

---

## 六、Release 产物清单

每次发布的 GitHub / CNB Release 包含以下文件：

- `frameflow-std-<tag>.tar.gz` / `frameflow-lite-<tag>.tar.gz`：完整压缩包（含 `graph.json`、`model.json`、`model.bin`、五份许可合规文本 `LICENSE` / `LICENSE-ADDITIONAL_TERMS-{zh-CN,en}.md` / `IMPLEMENTATION-GUIDE.zh-en.md` / `ASSETS_LICENSES.md`、`DATA_PROVENANCE.json`、`metrics.jsonl`）
- `frameflow-std-{graph.json,model.json,model.bin}`、`frameflow-lite-{graph.json,model.json,model.bin}`：可直接供前端 / CDN 单文件下载的未压缩权重
- `SHA256SUMS.txt`：全部文件 SHA-256 校验和
