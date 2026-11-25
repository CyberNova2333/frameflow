# FrameFlow 代码地图

先阅读[主流程与拒绝边界](walkthroughs/one-real-run.md)，再按职责定位改动位置。本页只覆盖已声明并经核对的核心目录，不把仓库内出现的所有辅助文件当作逐行验证模块。

| 路径 | 职责 | 关键代码 | 与主流程的关系 |
| --- | --- | --- | --- |
| `frameflow/` | 插帧与超分网络定义、算子、损失、数据集与导出器 | `model.py / sr_model.py / export.py` | 主流程起点：生成标准化三文件模型产物 |
| `runtime/` | 零依赖浏览器双后端引擎、静态图编译与帧节拍器 | `index.js / loader.js / pacer.js` | 主流程执行体：后端探测、图编译、推理与调度 |
| `scripts/` | 开源数据准备、训练蒸馏、对齐验证、打包与发版脚本 | `train.py / verify_backends.py / package_release.py` | 离线生产与验收链路，产出校验通过的模型包 |
| `tests/` | 纯 CPU 模型算子单测与 Node 加载器/节拍器测试 | `test_frameflow.py / test_runtime_node.mjs` | 快速回归防线，验证静态结构与格式约束 |

## `frameflow/`

保存插帧模型与二倍超分模型的 PyTorch 定义、损失函数、开源数据增强集以及统一格式导出器；所有隐藏层通道数受配置校验强制为四的倍数。

| 重要代码 | 功能 | 关键符号 | 调用方 / 使用方 |
| --- | --- | --- | --- |
| [插帧配置预设](../frameflow/config.py) | 定义 std 与 lite 档位通道、搜索半径及采样模式并校验四通道对齐 | `ModelConfig / PRESETS / get_preset` | 模型实例化、训练脚本与导出器 |
| [插帧网络本体](../frameflow/model.py) | 实现编码器、粗糙与精细光流头、时间调制与凸上采样，提供兼容别名 | `GameFINet / FrameFlowNet / FilmHead` | 训练主循环、权重导出器与评测脚本 |
| [插帧基础算子](../frameflow/ops.py) | 提供与着色器逐像素等价的亮度转换、代价体、反翘与上采样函数 | `bgr_to_luma / sad_volume / backward_warp` | 插帧网络前向传播与对齐参考 |
| [超分配置与网络](../frameflow/sr_model.py) | 定义轻量残差主干与双线性基准加像素重排二倍超分结构 | `GameSRNet / ResPair / SrModelConfig` | 超分训练、超分导出与视觉评测 |
| [插帧与超分导出](../frameflow/export.py) | 构造动态拓扑并写出小端十六位半精度权重与张量清单 | `build_graph_json / export_model / export_sr_model` | 训练收口保存、单元测试与权重发布 |
| [训练损失与数据集](../frameflow/losses.py) | 组合拉普拉斯、人口统计结构与合成静态界面零光流约束 | `FrameFlowLoss / GameSRLoss / overlay_synthetic_hud` | 插帧与超分训练脚本 |

相关验证：`tests/test_frameflow.py` 覆盖模型参数量、前向形状、梯度与导出读写。

## `runtime/`

向浏览器页面与网页模拟器暴露统一的纯 ES 模块接口，负责半精度权重解码、图形后端降级及无回读帧间隔调度。

| 重要代码 | 功能 | 关键符号 | 调用方 / 使用方 |
| --- | --- | --- | --- |
| [运行时总入口](../runtime/index.js) | 导出能力探测、自动后端创建与模型下载辅助函数 | `probeCapabilities / createFrameFlow / loadModelFromUrl` | 宿主页面、模拟器 Worker 与集成文档示例 |
| [模型加载与规划](../runtime/loader.js) | 解码十六位浮点数、校验张量偏移边界并展开静态图计划 | `parseModel / compilePlan / align32` | WebGPU 后端、WebGL2 后端与 Node 单测 |
| [计算着色器后端](../runtime/webgpu-backend.js) | 实现优先启用十六位扩展的纯 WGSL 推理引擎，支持设备共享 | `WebGPUEngine / probeWebGPU` | 支持 WebGPU 的浏览器与 PVSSPW 宿主 |
| [片元着色器后端](../runtime/webgl2-backend.js) | 实现半浮点纹理数组与多渲染目标全浏览器兜底路径 | `WebGL2Engine / probeWebGL2` | 无 WebGPU 浏览器与 PPSSPW 共享上下文 |
| [无阻塞帧节拍器](../runtime/pacer.js) | 维护源帧槽位间隔估算与二倍/三倍生成帧展示队列 | `FrameFlowPacer / onSourceFrame / tick` | 需要平滑倍帧调度的播放器循环 |
| [接口类型定义](../runtime/types.d.ts) | 声明引擎选项、模型元数据与会话生命周期接口 | `FrameFlowEngine / FrameFlowSession` | TypeScript 调用方与外部对接参考 |

相关验证：`tests/test_runtime_node.mjs` 离线验证解码、静态计划编译与节拍器队列逻辑。

## `scripts/`

负责开源许可媒体拉取、断点续训、无头浏览器数值对齐检验以及压缩包校验和生成。

| 重要代码 | 功能 | 关键符号 | 调用方 / 使用方 |
| --- | --- | --- | --- |
| [训练数据制备](../scripts/prepare_data.py) | 下载开源视频、过滤黑帧切镜并按哈希划定训练与验证集 | `resolve_ffmpeg / VideoSource` | 云端训练一键脚本与 Makefile 数据目标 |
| [超分数据制备](../scripts/prepare_sr_data.py) | 汇集开放影视帧、几何线稿与合成游戏界面文字生成验证集 | `is_val_key / extract_video_frames` | 超分训练前置步骤 |
| [训练与蒸馏入口](../scripts/train.py) | 执行混合精度训练、冻结教师蒸馏及最优权重自动导出 | `update_ema / run_validation` | 本地独显训练与云端工作流 |
| [双后端数值对齐校验](../scripts/verify_backends.py) | 驱动无头 Edge 运行真机着色器并对比 PyTorch 参考输出 | `init_nontrivial / psnr_u8` | 交付前数值对齐核查与 CI |
| [发行包合规打包](../scripts/package_release.py) | 收集模型三文件、五份许可文件、数据来源清单并生成校验摘要 | `pick_export_dir / sha256_file` | 本地或云端发版流水线 |
| [发布辅助与云入口](../scripts/publish_github_release.sh) | 检测设备环境并在云端触发训练、打包或上传至 Release | `run_cloud_train.sh / eval_pairs.py` | 维护者手动发版与 CNB/GitHub Actions |

相关验证：数值对齐脚本当场比对真实像素误差，不随纯 CPU 单测自动执行。

## `tests/`

不依赖 GPU 或网络下载即可快速运行的回归测试集合，用于守住导出规格与加载规划基础边界。

| 重要代码 | 功能 | 关键符号 | 调用方 / 使用方 |
| --- | --- | --- | --- |
| [PyTorch 单元测试](../tests/test_frameflow.py) | 校验模型参数数量、动态时间步前向形状、梯度回传与导出格式 | `TestFrameFlow / test_presets_and_param_counts` | Makefile `test` 目标与 CI |
| [Node 运行时单元测试](../tests/test_runtime_node.mjs) | 导出临时模型并用纯 JS 加载器验证十六位解码、计划展开与节拍队列 | `f16ToF32 / compilePlan / FrameFlowPacer` | Makefile `test-runtime` 与 CI |

相关验证：通过 `make test` 在五秒级本地 CPU 环境下完成执行。

## 覆盖范围

上述清单覆盖 FrameFlow 第一方 Python 模块、浏览器 JS 运行时、常用离线准备/验证/打包脚本以及 CPU 单测入口；实际训练产生的 `checkpoints/`、`dist/` 与 `data/` 目录已被 Git 忽略并在[指南变更记录](change-log.md)中另行区分产物状态。

证据状态：除特别标注外，本页基于当前源码已确认。
