# FrameFlow 源码证据与调查台账

这是解释与证据边界台账：所有论述以仓库源码与实际执行命令为依据。当前对应源码锚点为 `f328c3d82bfff02b380689f1e1c2525bfdc07d07`，文档站点构建与纯代码逻辑校验另外记账，不互相替代。

## Evidence Traversal Log

| Pass | Purpose | Inspected evidence | What changed in the model |
| --- | --- | --- | --- |
| Pass 1 | 建立模型导出、能力探测与推理输出主路径 | `frameflow/export.py`、`frameflow/sr_export.py`、`runtime/index.js`、`runtime/loader.js`、`runtime/types.d.ts`、`AGENTS.md` | 确立统一三文件格式、`WebGPU -> WebGL2` 探测降级与外部共享纹理入口 |
| Pass 2 | 反查算子等价细节、节拍淘汰、拒绝条件与数据发版约束 | `frameflow/config.py`、`frameflow/model.py`、`frameflow/sr_config.py`、`frameflow/sr_model.py`、`frameflow/ops.py`、`frameflow/dataset.py`、`scripts/package_release.py`、`tests/test_frameflow.py`、`tests/test_runtime_node.mjs` | 补充四通道打包强制校验、静态 HUD 遮罩增强、零回读节拍边界及五份合规随包要求 |

## 证据表

| Claim | Evidence | Confidence | Caveat | Used by |
| --- | --- | --- | --- | --- |
| 插帧与超分均导出为统一 `format_version=1` 三文件结构 | [插帧导出](../../frameflow/export.py)、[超分导出](../../frameflow/sr_export.py)：`build_graph_json`、`build_sr_graph_json` | Confirmed | 源码静态核对及导出测试覆盖 | [主流程](../walkthroughs/one-real-run.md)、[模型模块](../modules/gamefi-and-gamesr-models.md) |
| 引擎优先选择 WebGPU 并自动降级 WebGL2，均不支持则拒绝启动 | [运行时入口](../../runtime/index.js)：`probeCapabilities`、`createFrameFlow` | Confirmed | 不存在 CPU 或 WASM 软推理后备路径 | [主流程](../walkthroughs/one-real-run.md)、[双后端模块](../modules/dual-backend-runtime.md) |
| 加载器手写 `binary16` 解码并在缺失张量时直接抛出异常 | [模型加载器](../../runtime/loader.js)：`f16ToF32`、`parseModel`、`compilePlan` | Confirmed | Node 单元测试验证典型边界值和真实临时导出图 | [主流程](../walkthroughs/one-real-run.md)、[双后端模块](../modules/dual-backend-runtime.md) |
| 支持共享外部 `GPUDevice` 或 `WebGL2RenderingContext` 进行零拷贝推理 | [计算后端](../../runtime/webgpu-backend.js)、[片元后端](../../runtime/webgl2-backend.js) | Confirmed | 具体宿主零拷贝效果还需对应模拟器 Worker 配合作业 | [代码地图](../code-map.md)、[双后端模块](../modules/dual-backend-runtime.md) |
| 节拍器以滑动间隔调度二倍与三倍插帧并在超时积压时裁切队列 | [帧节拍器](../../runtime/pacer.js)：`FrameFlowPacer.onSourceFrame`、`tick` | Confirmed | Node 单元测试覆盖 2x 与 3x 队列推进 | [主流程](../walkthroughs/one-real-run.md)、[双后端模块](../modules/dual-backend-runtime.md) |
| 隐藏通道强制为四的倍数且插帧参数分别为 775,811 与 222,179 | [插帧配置](../../frameflow/config.py)、[超分配置](../../frameflow/sr_config.py)、[模型单测](../../tests/test_frameflow.py) | Confirmed | `GameSR` 体积依据源码注释，实际精确值见导出清单 | [模型模块](../modules/gamefi-and-gamesr-models.md) |
| 训练前过滤切镜并叠加静态 HUD 遮罩施加零光流监督 | [数据制备](../../scripts/prepare_data.py)、[数据集](../../frameflow/dataset.py)、[插帧损失](../../frameflow/losses.py) | Confirmed | 开源媒体需自行按脚本指令联网或从本地目录读取 | [训练与发版模块](../modules/training-and-release-pipeline.md) |
| 打包脚手架随权重压缩包写入五份许可文书、数据清单与哈希汇总 | [打包脚本](../../scripts/package_release.py)、[发版指南](../../docs/TRAINING.md) | Confirmed | 本地目录已确认行为逻辑，线上 Release 产物另行验收 | [训练与发版模块](../modules/training-and-release-pipeline.md) |
| 无头 Edge 实测对齐维持 62–72 dB PSNR 且单帧耗时可控 | [对齐验证脚手架](../../scripts/verify_backends.py)、[根说明](../../README.md) | Inferred | 援引自既有测量记录与脚本阈值，本轮仅构建文档未重跑硬件 GPU 对齐 | [双后端模块](../modules/dual-backend-runtime.md) |

## 反证与后续问题

反证检查（Falsifying check）：如果发现浏览器端能够在 WebGPU 与 WebGL2 半浮点 FBO 均不可用时继续通过 JS 计算像素，或者导出脚本输出未校验 `format_version=1` 的旧格式而被成功载入，则说明本指南关于拒绝边界与格式约束的描述已经失效。

下一位读者可能问（Next reader question）：当只更换了新的自训权重包但没有更改模型拓扑时，需要更新代码里的什么？通常只需更换静态托管目录下的三文件并将 `loadModelFromUrl` 路径指向新版目录；只有新增了计算图节点种类或非二倍超分尺度，才需要改动双后端着色器与规划器。

## 范围与未验证事项

本台账完整审阅了 `frameflow/` 核心模块、`runtime/` 浏览器运行库、`tests/` 单元测试与核心 `scripts/` 脚本的边界定义；`verify_parity.mjs`、`eval_sr_visual.py` 的渲染绘图细节，以及云端计算集群底层宿主机镜像环境未在本轮做运行时实跑复核。GPU 真机测试、大数据集训练收敛质量与跨机型表现均需结合硬件运行日志独立判定。

证据状态：除特别标注外，本页基于当前源码已确认。
