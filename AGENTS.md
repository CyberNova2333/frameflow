# AGENTS.md —— FrameFlow 协作与维护约定

## 常用验证命令

```sh
# 1. 快速 CPU 单元测试（PyTorch GameFI / GameSR + Node 运行时加载器与节拍器）
make test

# 2. Intel Xe-LPG 无头 Edge 157 双后端数值对齐校验（需可用硬件 GPU / Vulkan）
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json .venv/bin/python scripts/verify_backends.py
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json .venv/bin/python scripts/verify_sr_backends.py
```

## 模型与架构边界

- **命名分层**：`FrameFlow` 为仓库、工作流与浏览器运行时系列总称；`GameFI`（`GameFI-Std` / `GameFI-Lite`）指光流插帧模型，`GameSR`（`GameSR-Std` / `GameSR-Lite`）指 2× 超分模型。Python 侧同时保留 `FrameFlowNet` / `FrameFlowLoss` 向后兼容别名（`GameFINet = FrameFlowNet`、`GameFILoss = FrameFlowLoss`）。
- **权重隔离红线**：仓库严禁提交二进制模型权重（`*.bin` / `*.pt` / `*.pth`）、`dist/`、`dist_sr/`、`checkpoints/`、`artifacts/` 或原始训练媒体；权重统一通过 `scripts/package_release.py` 附带五份许可合规文本与 `DATA_PROVENANCE.json` / `SHA256SUMS.txt` 发往 GitHub Release 或 CNB OCI。
- **许可与数据源边界**：所有训练媒体与公开数据源必须在 `ASSETS_LICENSES.md` 与 `DATA_PROVENANCE.json` 明确溯源；严禁引入非商业许可（如 `CC BY-NC`）或来源不明的模型权重及素材，运行时不从参考仓复制非兼容代码。
- **后端兼容约束**：WebGPU（Compute）与 WebGL2（`RGBA16F` + MRT）须与 PyTorch 参考导出保持 `>= 50 dB` PSNR 对齐；热路径不得出现同步 `mapAsync` 回读阻塞，不支持半浮点/浮点渲染目标的设备优雅报告不支持，不做 CPU/WASM 软推理兜底。

## 提交与签名约定

- 提交标题遵循 Conventional Commits：`type(scope): 中文描述`（允许 `feat`、`fix`、`docs`、`style`、`refactor`、`perf`、`test`、`build`、`ci`、`chore`、`revert`、`data`，不超过 100 字符）。
- 人类贡献者为 Git `Author`，AI 编程助手仅作为 `Co-authored-by` 尾注参与者（只写通用工具名，不写具体模型档位或版本）。
- 非初始提交使用人类维护者已配置的 SSH 签名密钥签名（`gpg.format=ssh`）。
- 请勿包含会话链接、内部凭据或未核实数据。

## 项目简介与主页元数据 (`meta.json`)

- **项目名称**：FrameFlow · 跨浏览器 WebGPU/WebGL2 光流插帧与超分工作流套件
- **仓库地址**：`https://github.com/CyberNova2333/frameflow`
- **分类与主语言**：`media-engine` · `Python` / `JavaScript`
- **同步规则**：修改核心模型家族、预设规格或主要能力时，须同步检查并更新根目录 [`meta.json`](meta.json) 与中英文 README。
