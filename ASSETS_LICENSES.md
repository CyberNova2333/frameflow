# FrameFlow 人员署名、AI 工具、代码与资产许可综合清单

**标准文件名：** `ASSETS_LICENSES.md`  
**配套模板版本：** v1.6.7-r4  
**项目名称：** `FrameFlow`  
**开发、维护或联合推出主体：** `CyberNova`  
**上游仓库地址：** `https://github.com/CyberNova2333/frameflow`  
**所对应的仓库版本及发行物：** `main` 分支初始完整版本（v0.1.0，涵盖 PyTorch 训练/导出套件、浏览器双后端运行时与双端 CI/训练流水线）  
**记录覆盖范围及尚未核实部分：** 已核实本仓库全部自研源码、文档与文档站静态资产（`frameflow/`、`runtime/`、`scripts/`、`tests/`、`docs/`、`docs-site/`、`repo-docs/`、`.github/`、`.cnb/`、`.cnb.yml`、`Makefile`、`package.json`、`pyproject.toml`），包括 `docs-site/.vitepress/theme/vendor/animejs/`（MIT v4.5.0）与自托管字体（SIL OFL 1.1）。尚未核实或明确排除在本仓库许可范围之外的内容：训练期由 `scripts/prepare_data.py` 按需从外部下载的第三方开放许可视频媒体、用户在本地 `data/raw_local/` 中自行提供的录屏素材、以及训练环境安装的外部 Python/FFmpeg 运行时依赖包。  
**清单最后更新日期：** `2026-11-25`  
**配套附加条款：** [中文完整文本](./LICENSE-ADDITIONAL_TERMS-zh-CN.md)；[英文参考译文](./LICENSE-ADDITIONAL_TERMS-en.md)；[工程实施指南](./IMPLEMENTATION-GUIDE.zh-en.md)

> **填写状态：本清单已依据本仓库实际文件、Git 提交记录与依赖清单核实填写。** 四个分区的排列与配套附加条款 v1.6.7-r4 第 3.2、7.1 款一致；本清单如实记录已核实范围与未覆盖范围，不扩大或缩减任何上游有效许可证授予的权利，也不以列名方式向第三方材料追加限制。

**法律效力与载体说明**

1. 本清单记录事实及授权依据，不由登记、标签、Git 字段或 `meta.json` 自行产生版权、转让、代表权限或同意；真实声明的效力与覆盖范围依附加条款第 1.2、1.7—1.9 款判断。
2. 下游可依第 3.7 款改用其他合规载体，无须沿用本文件格式或工作流。与实际采用材料有关、依法或依有效许可应保留的版权、许可、免责、`NOTICE` 和合理署名，仍须按该依据保留；原许可要求完整文本或特定位置、方式的，链接和摘要不能代替。纯工具记录、一般致谢和审计字段的边界见第 3.5—3.6 款。
3. 仓库根目录保留完整基础许可证 [`LICENSE`](./LICENSE)（GNU AGPLv3）及配套附加条款与指南文本；未来若引入任何第三方内嵌代码或独立资产，须同步保留其原始许可文本并在本清单对应分区登记。

工程实践参考见 [`IMPLEMENTATION-GUIDE.zh-en.md`](./IMPLEMENTATION-GUIDE.zh-en.md)（非规范性指南，不增设许可条件）。

**状态标签说明**

| 中文标签 | 英文等效标签 | 记录对象及效力边界 |
| --- | --- | --- |
| `[附加条款：已明确同意]` | `[Additional terms: Express agreement recorded]` | 记录可核实的实际权利人或其有效授权代表就特定材料、条款及版本作出的明确同意。须分别核实附加要求和第 9.5—9.6 款附加许可的覆盖范围；同意其中一项不自动证明同意另一项。 |
| `[附加条款：未发表意见]` | `[Additional terms: No recorded position]` | 本项目现有记录尚无该主体对相关材料适用相应条款的可核实明确意见。不表示拒绝、曾被单独征询、放弃权利或在其他场合从未表达意见；原有效许可、既有附加条款或例外仍依实际内容适用。 |
| `[附加条款：不适用（遵循原上游许可证）]` | `[Additional terms: Not applicable (original upstream license governs)]` | 用于第三分区的原上游及第三方材料，说明列名不使其纳入本项目新增附加要求，也不代表其权利人授予本项目第 9.5 款豁免。适用的原许可及组合义务仍须履行。 |

## 一、本项目人类作者与贡献者 (Project Human Authors & Contributors)

依据当前可核实的 Git 提交记录、仓库权利记录与根目录 `meta.json`，记录本项目维护者与原作者的真实角色。`Author`、`Committer`、`Co-authored-by` 与版权持有人并不当然相同。

### 1.1 主体、角色与贡献记录

| 记录编号 | 姓名、笔名或组织名称及真实角色 | Git 或平台账户关联 | 实际贡献及材料范围 | 提交、PR 或其他事实依据 | 附加条款意见状态及授权记录编号 |
| --- | --- | --- | --- | --- | --- |
| `P-01` | CyberNova（项目发起者、维护者、作者、版权持有人、许可人；Git 角色：Author, Committer） | `https://github.com/CyberNova2333` / `295488275+CyberNova2333@users.noreply.github.com` | 本仓库全部第一方原创源码、模型定义与训练/蒸馏/导出/发版脚本（`frameflow/`、`scripts/`）、零依赖浏览器 WebGPU/WebGL2 运行时与节拍器（`runtime/`）、测试套件（`tests/`）、CI/训练流水线配置及文档（`README.md`、`docs/`），以及由本仓库训练脚本在合规公开素材上训练并公开发布的第一方自训模型权重（`AGPL-3.0-only` + 附加条款 v1.6.7-r4） | 本仓库 Git 完整提交历史及根目录 `meta.json` | `[附加条款：已明确同意]`（见 `AUTH-01`） |

### 1.2 授权事实与范围索引

下表严格分别记录“附加要求的覆盖范围”与“第 9.5—9.6 款附加许可的覆盖范围”；同意其中一项不自动推定同意另一项，且任何声明均仅及于声明人享有合法权限的材料范围。

| 授权记录编号及关联主体 | 实际声明主体与代表权限依据 | 材料、提交及版本范围 | 附加要求的覆盖范围 | 第 9.5—9.6 款附加许可的覆盖范围 | 声明、有效许可或证明依据 |
| --- | --- | --- | --- | --- | --- |
| `AUTH-01`；`P-01` | CyberNova（本人，对其第一方原创受版权保护材料享有完整版权与许可权限） | 本仓库全部由 CyberNova 创作并提交的第一方源代码、文档、配置以及使用本项目合规数据流程发布的自训权重（v0.1.0 及后续由本人提交并标明适用的版本） | 已明确同意适用 `LICENSE-ADDITIONAL_TERMS-zh-CN.md` **v1.6.7-r4** 全部附加要求（AGPLv3 第 7(b)/(c)/(d) 条） | 已明确授权适用 `LICENSE-ADDITIONAL_TERMS-zh-CN.md` **v1.6.7-r4** 第 9.5—9.6 款有条件的附加许可豁免（仅限于本人享有权利的第一方材料，不代第三人授予） | 仓库根目录 `meta.json`（`additionalTermsConsent.CyberNova = true`）及本仓库公开发布记录 |

### 1.3 应保留的项目材料声明

| 对应材料或主体编号 | 应保留的具体版权、许可或合理署名内容及依据 | 实际承载文件或界面 | 适用范围与允许的调整方式 |
| --- | --- | --- | --- |
| `P-01` | GNU AGPLv3 完整原文、附加条款 v1.6.7-r4、合理署名与项目来源声明（依据 AGPLv3 第 4–7 条及附加条款第三条） | `LICENSE`、`LICENSE-ADDITIONAL_TERMS-zh-CN.md`、`LICENSE-ADDITIONAL_TERMS-en.md`、`IMPLEMENTATION-GUIDE.zh-en.md`、`ASSETS_LICENSES.md` 及第一方源文件 SPDX 头 | 适用于本项目第一方 AGPLv3 适用材料与打包发布的权重归档；下游可依附加条款第 3.7 款改用等效合规载体 |

## 二、AI Agent 与自动化辅助工具 (AI Agents & Automated Assistants)

本分区记录工程中实际使用的 AI 辅助工具。依据附加条款第 1.4、1.8 款，本分区不将工具、模型或云端环境视为独立的版权许可授权主体，也不对工具本身赋予附加条款同意状态。

| 工具记录编号及工具类别或名称 | 技术账户、机器人或环境标识 | 经核实的使用、控制或提交上下文 | 辅助范围及相关提交或材料 | 实际主体或组件记录关联 | 核实情况与必要说明 |
| --- | --- | --- | --- | --- | --- |
| `T-01`；Antigravity（Google DeepMind AI 编程辅助工具） | 未使用独立提交账户（全部提交由维护者 `P-01` CyberNova 身份签名） | 在人类维护者（`P-01` CyberNova）架构决策、需求指令、实机验证与审核下参与工程实现 | 模型/训练/导出代码草拟、WebGPU/WebGL2 双后端运行时实现、数值对齐测试脚本、双端流水线配置与文档起草（v0.1.0 初始版本） | `P-01`（工具本身非版权许可授权主体） | 工具本身不标注附加条款同意状态；具体权属、原创性与有效许可边界依附加条款第 1.4、1.8 款按实际人类权利人及适用法律判断 |

## 三、上游基线、依赖与第三方代码组件（含原始作者与贡献者） (Upstream Baselines, Dependencies & Third-Party Code Components)

本仓库**未内嵌、未分叉、未打包任何第三方源代码快照**；浏览器端运行时（`runtime/`）零 npm 依赖、零 WASM。下表如实登记训练环境外部依赖、按需下载的公开训练素材来源及可选的本地外部对照工具，便于完整溯源。

### 3.1 外部依赖、可选对照工具与公开训练素材来源清单

| 组件编号及名称 | 类型与实际引入方式 | 仓库位置、发行物位置或调用关系 | 上游来源及所采用版本、提交或归档标识 | 原始作者、版权持有人及相关贡献署名出处 | 实际核实范围与备注 |
| --- | --- | --- | --- | --- | --- |
| `C-01`；PyTorch & TorchVision | 外部 Python 训练/导出直接依赖（不随仓库分发，由用户环境或 CI 安装） | `pyproject.toml`、`frameflow/`、`scripts/` 仅通过标准 Python `import` 调用 | https://github.com/pytorch/pytorch（`torch>=2.6`、`torchvision>=0.21`） | Meta Platforms, Inc. & PyTorch Contributors | 仅用于模型训练、验证与导出，浏览器运行时不依赖 PyTorch |
| `C-02`；NumPy / Pillow / tqdm | 外部 Python 数值与图像处理依赖（不随仓库分发） | `pyproject.toml`、`frameflow/`、`scripts/` | PyPI 官方发行包（`numpy>=1.26`、`pillow>=10.0`、`tqdm>=4.66`） | NumPy Developers / Alex Clark & Contributors / tqdm Developers | 各自遵循 BSD-3-Clause / MIT-CMU / MPL-2.0 or MIT 原许可 |
| `C-03`；FFmpeg（命令行工具） | 外部 CLI 工具（不随仓库分发，仅在数据准备时通过子进程调用） | `scripts/prepare_data.py` 子进程调用 `ffmpeg` 解帧 | https://ffmpeg.org/ | FFmpeg Developers | 仅作为独立外部命令行进程运行，不与本仓库代码链接 |
| `C-04`；Blender Open Movies（*Big Buck Bunny*、*Tears of Steel*、*Sintel*） | 外部公开视频数据源（不随仓库分发，由 `scripts/prepare_data.py` 按需下载至被 `.gitignore` 忽略的 `data/`） | `scripts/prepare_data.py` URL 清单；训练打包时在 `DATA_PROVENANCE.json` 中记录溯源 | Blender Foundation（`download.blender.org` 官方公开分发链接） | Blender Foundation（CC BY 3.0） | 严格禁止将第三方原始视频文件提交入库；仅用于训练监督，发行物中提供来源清单 |
| `C-05`；Xiph.org / SVT Derf's Test Media（`720p50_*`、`720p5994_*`） | 外部公开测试视频序列（不随仓库分发，由 `scripts/prepare_data.py` 按需下载） | `scripts/prepare_data.py` URL 清单；训练打包时记录于 `DATA_PROVENANCE.json` | https://media.xiph.org/video/derf/ | Sveriges Television AB (SVT) / Xiph.org | 开放研究与分发测试序列，仅按需下载至本地忽略目录 |
| `C-06`；FlyHigh Engine WebGPU（可选外部对照参考） | 外部可选验证对照（**未随本仓库分发或内嵌任何代码或权重**） | `scripts/verify_parity.py` / `scripts/verify_parity.mjs` 通过命令行路径参数可选加载本地外部目录作数值对齐测试 | 外部独立项目（代码为 Apache-2.0；其自带权重为 CC BY-NC 4.0，明确禁止引入本仓库） | FlyHigh Engine 作者 | 本仓库 `runtime/` 与 `frameflow/` 均为独立从零编写，不含其源码、WASM 或 CC BY-NC 权重 |
| `C-07`；SYNLA-Dataset v1（`Dataset_Color`，不含 `SYNLA-Plus`） | 外部程序化动漫/线条合成数据集（不随仓库分发，由 `scripts/prepare_sr_data.py` 按需下载至被 `.gitignore` 忽略的 `data/`） | `scripts/prepare_sr_data.py`；训练打包时记录于 `DATA_PROVENANCE.json` | `https://github.com/bloc97/SYNLA-Dataset`（主分支 `Dataset_Color`，2,224 张纯粹程序化合成 PNG） | bloc97（MIT） | 已明确剔除含 DIV2K/网络抓取帧的 `SYNLA-Plus`，仅使用纯程序化合成的 MIT 子集 `Dataset_Color` 训练 `GameSR-2x` |
| `C-08`；Anime.js 4.5.0 | 第三方内嵌脚本（静态复制，仅文档站动效使用） | `docs-site/.vitepress/theme/vendor/animejs/` | https://github.com/juliangarnier/anime（v4.5.0） | Julian Garnier（`Copyright (c) 2025 Julian Garnier`） | 仅随独立 `docs-site/` 构建分发，完整 MIT 原文见 `docs-site/.vitepress/theme/vendor/animejs/LICENSE.md` |

### 3.2 原始许可、本地修改与组合边界

| 组件编号 | 原始部分的有效许可证及完整文本位置 | 本地新增或修改部分及其有效授权 | 构建、链接或交互方式与适用边界 | 原始材料的附加条款状态 |
| --- | --- | --- | --- | --- |
| `C-01` | BSD-3-Clause（上游 PyTorch 发行包内附 `LICENSE`） | 无本地修改 | 仅训练期 Python 环境依赖；与 `AGPL-3.0-only` 兼容 | `[附加条款：不适用（遵循原上游许可证）]` |
| `C-02` | BSD-3-Clause（NumPy）、MIT-CMU（Pillow）、MPL-2.0 / MIT（tqdm）（各自发行包内附） | 无本地修改 | 仅训练期 Python 环境依赖；与 `AGPL-3.0-only` 兼容 | `[附加条款：不适用（遵循原上游许可证）]` |
| `C-03` | LGPL-2.1-or-later / GPL-2.0-or-later（随用户系统安装的 FFmpeg 发行版） | 无本地修改 | 独立子进程 CLI 调用，不构成代码链接 | `[附加条款：不适用（遵循原上游许可证）]` |
| `C-04` | Creative Commons Attribution 3.0 Unported（CC BY 3.0，Blender Foundation） | 无视频二次分发，仅本地按需下载用于训练 | 外部数据集；若发布基于其训练的权重，随发行包 `DATA_PROVENANCE.json` 保留明确署名与许可来源 | `[附加条款：不适用（遵循原上游许可证）]` |
| `C-05` | Xiph.org / SVT 公开研究测试媒体条款（见 `media.xiph.org/video/derf/`） | 无视频二次分发，仅本地按需下载用于训练 | 外部数据集；随发行包 `DATA_PROVENANCE.json` 保留来源记录 | `[附加条款：不适用（遵循原上游许可证）]` |
| `C-06` | Apache-2.0（仅代码；本仓库未引入其代码或权重） | 未引入 | 仅作为开发者本机可选对齐脚本的外部参照路径 | `[附加条款：不适用（遵循原上游许可证）]` |
| `C-07` | MIT（bloc97/SYNLA-Dataset `Dataset_Color` 原仓库 `LICENSE`） | 无原始图片二次分发，仅本地按需下载用于训练 | 外部数据集；随发行包 `DATA_PROVENANCE.json` 保留 MIT 来源署名 | `[附加条款：不适用（遵循原上游许可证）]` |
| `C-08` | MIT（`docs-site/.vitepress/theme/vendor/animejs/LICENSE.md`） | 无本地源码修改 | 仅随独立文档站分发，与 AGPL-3.0-only 兼容 | `[附加条款：不适用（遵循原上游许可证）]` |

### 3.3 需保留的许可、NOTICE 与版权声明索引

| 组件编号 | 声明类别及原始出处 | 实际随附或提供位置 | 原许可要求保留的内容和适用范围 | 载体安排说明 |
| --- | --- | --- | --- | --- |
| `C-04` / `C-05` / `C-07` | 开放训练媒体与程序化数据集来源署名 | `scripts/prepare_data.py`、`scripts/prepare_sr_data.py` 及每次权重发版归档内的 `DATA_PROVENANCE.json` / `RELEASE_NOTES.md` | 注明 Blender Foundation（CC BY 3.0）、Xiph.org / SVT 与 bloc97 SYNLA v1（MIT）来源 | 由 `scripts/package_release.py` 自动随权重压缩包与 Release Notes 交付 |

## 四、独立美术、字体与项目标识资产 (Independent Artwork, Fonts & Project Identifier Assets)

本分区登记独立美术、字体与项目标识。当前初始版本（v0.1.0）未随仓库打包任何外部图片、图标或字体二进制文件。

### 4.1 资产、来源与许可清单

| 资产编号、名称及类别 | 实际文件、来源与所采用版本 | 作者、版权持有人及可核实依据 | 实际有效许可证或具体授权及完整文本位置 | 是否为项目标识、随包方式及替换情况 | 适用范围、修改与尚待核实事项 |
| --- | --- | --- | --- | --- | --- |
| `A-01`；FrameFlow 系列与模型档位标识（`FrameFlow`、`GameFI`、`GameFI-Std`、`GameFI-Lite`、`GameSR`、`GameSR-2x`、`GameSR-Std`、`GameSR-Lite`） | 见 `README.md`、`package.json`、`pyproject.toml`、`frameflow/config.py`、`frameflow/sr_config.py` | CyberNova（`P-01`） | 依据 `LICENSE-ADDITIONAL_TERMS-zh-CN.md` 第五条与第八条管理（不创设法律未赋予的注册商标排他权） | 是（项目文字标识）；下游二次分发或修改版本可依第八条更名并须如实区分来源 | 禁止冒充官方发布或虚构官方背书，具体规则见附加条款第五条与第八条 |
| `A-02`；本机合成 UI 训练图所用系统字体（`Noto Sans CJK`、`DejaVu Sans`） | 仅训练期由 `scripts/prepare_sr_data.py` 调用系统字体栅格化为像素训练样本；仓库及权重包内不含字体文件 | Google & Adobe（SIL OFL 1.1） / DejaVu Fonts Team（Bitstream Vera 宽松许可） | SIL OFL 1.1 / Bitstream Vera Font License（随操作系统字体包安装） | 未随仓库或权重打包 | 仅用于离线生成游戏 HUD/对话文字超分训练样本 |
| `A-03`；IBM Plex Sans（文档站自托管字体） | `docs-site/.vitepress/theme/fonts/ibm-plex-sans-latin-*.woff2`；来源：IBM Plex Sans | IBM Corp.（见同目录 `IBM-Plex-Sans-LICENSE.txt`） | SIL Open Font License 1.1（`docs-site/.vitepress/theme/fonts/IBM-Plex-Sans-LICENSE.txt`） | 否（非项目标识）；仅随 `docs-site` 构建静态分发；可自由替换 | 按 SIL OFL 1.1 随文档站浏览分发；未修改字形 |
| `A-04`；JetBrains Mono（文档站自托管字体） | `docs-site/.vitepress/theme/fonts/jetbrains-mono-latin-500-normal.woff2`；来源：JetBrains Mono | The JetBrains Mono Project Authors（见同目录 `LICENSE-jetbrains-mono.txt`） | SIL Open Font License 1.1（`docs-site/.vitepress/theme/fonts/LICENSE-jetbrains-mono.txt`） | 否（非项目标识）；仅随 `docs-site` 构建静态分发；可自由替换 | 按 SIL OFL 1.1 随文档站代码块渲染使用；未修改字形 |
