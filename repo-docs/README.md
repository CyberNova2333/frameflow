# FrameFlow 源码阅读指南

当浏览器页面或网页模拟器需要把低帧率、低分辨率画面提升为高帧率或高清晰度输出时，模型权重、静态算子图、图形后端和节拍调度必须协同就绪。本指南沿一次模型导出与浏览器推理交接展开，帮助维护者区分模型结构、双后端算子等价性与运行时拒绝边界。

## 阅读路径

| 读者目标 | 从这里开始 | 读完后获得什么 |
| --- | --- | --- |
| 理解一次真实运行 | [主流程与拒绝边界](walkthroughs/one-real-run.md) | 能解释权重导出、后端选择、帧推理交接以及失败在哪里出现 |
| 定位修改入口 | [按职责找代码](code-map.md) | 在已声明范围内找到模型、运行时、脚本与测试的修改入口 |
| 深入双后端与帧节拍调度 | [WebGPU 与 WebGL2 双后端运行时](modules/dual-backend-runtime.md) | 理清半精度解码、四通道打包、外部纹理零拷贝与无阻塞节拍队列 |
| 核对插帧与超分模型结构 | [GameFI 与 GameSR 模型及静态图格式](modules/gamefi-and-gamesr-models.md) | 掌握预设参数、算子图字段、别名兼容以及插帧与超分输出约定 |
| 追踪合规数据与权重发版 | [开源许可数据、训练与双端发版流程](modules/training-and-release-pipeline.md) | 理清开源媒体切分、合成介面遮罩增强、校验打包与多平台发布 |
| 统一核心概念与符号含义 | [术语表](glossary.md) | 快速查对权重格式、通道约束、渲染采样与节拍器符号 |
| 审核解释依据 | [源码证据与两遍调查](references/source-evidence.md) | 区分静态源码确认、测试替身范围与实体硬件 GPU 验证 |
| 复核文档质量 | [读者模拟与剩余风险](references/quality-review.md) | 看懂本指南能回答什么，以及真机浏览器基准等未覆盖边界 |
| 查看同步边界 | [指南变更记录](change-log.md) | 找到源码锚点与本轮实际执行的验证记录 |

## 覆盖范围

本指南覆盖 Python 模型与导出层、纯前端双后端推理运行时与节拍器、数据准备与训练发版脚本，以及纯 CPU 单元测试。训练过程中按需从外部获取的视频媒体素材、云端 GPU 宿主机调度内部实现，以及各宿主模拟器自身的外壳页面，均不在本仓逐层展开。

本包采用 Standard 模式：主流程先建立端到端的导出与推理交接模型，专题模块页拆解双后端算子等价性、网络结构与合规发版约束，代码地图用于定位具体改动入口，证据台账与术语表提供可核对依据。

## 运行与验证入口

以下命令在仓库根运行；带硬件验证命令依赖本机具备 Vulkan 或可用 GPU 驱动：

```sh
# 1. 运行 Python 模型导出单元测试与 Node.js 运行时/节拍器单元测试（零 GPU 依赖）
make test

# 2. 单独检查 JS 语法与加载器/节拍器测试
make test-runtime

# 3. 检查 Python、CUDA PyTorch 与 ffmpeg 环境
make env-check

# 4. 运行双后端数值对齐校验（需要无头 Edge / GPU 环境）
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json .venv/bin/python scripts/verify_backends.py
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json .venv/bin/python scripts/verify_sr_backends.py
```

独立网页文档构建仅渲染说明页面与索引，不执行模型训练，也不打包任何 `.bin` / `.pt` 权重文件。网页版将指南之外的仓库文件链接保留为精确源码路径，避免生成失效跳转。

证据状态：除特别标注外，本页基于当前源码已确认。
