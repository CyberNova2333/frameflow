# 概览与能力边界

**FrameFlow** 将光流 AI 插帧（**GameFI**）与 2× AI 超分（**GameSR**）收敛到一套轻量级训练、导出与浏览器端推理管线中。前端引擎完全以原生 JavaScript、WGSL 与 GLSL ES 3.00 写成，可直接独立接入 Canvas 渲染，或零拷贝挂接在基于 WebGPU / WebGL2 的网页模拟器之后。

## 套件构成一览

| 目录 / 层 | 职责范围 | 典型入口文件 |
| --- | --- | --- |
| 模型与训练核心 (`frameflow/`) | `GameFINet`、`GameSRNet` 网络结构、算子、损失、数据集增强与导出 | `frameflow/model.py`、`frameflow/sr_model.py` |
| 浏览器运行时 (`runtime/`) | 自动能力探测、WebGPU Compute / WebGL2 后端、半精度加载器与 `FrameFlowPacer` | `runtime/index.js`、`runtime/loader.js` |
| 数据、评测与发版 (`scripts/`) | 开放许可视频切分、混合精度训练、双后端 PSNR 对齐与合规压缩包生成 | `scripts/train.py`、`scripts/package_release.py` |
| 快速回归测试 (`tests/`) | 纯 CPU 模型测试与零 GPU Node 运行时解码及节拍器断言 | `tests/test_frameflow.py`、`tests/test_runtime_node.mjs` |

## 实测模型规格

| 模型名 | 预设 | 任务 | 参数量 | f16 体积 | 定位说明 |
| --- | --- | --- | ---: | ---: | --- |
| `GameFI-Std` | `std` | 插帧 (`interp`) | 775,811 | ~1.52 MB | 桌面核显/独显及旗舰移动 WebGPU |
| `GameFI-Lite` | `lite` | 插帧 (`interp`) | 222,179 | ~434 KB | WebGL2 全浏览器兜底与移动端首选 |
| `GameSR-Std` | `sr_std` | 2× 超分 (`sr`) | ~11,548 | ~22.6 KB | 较高轮廓还原度，可单开或接在插帧后 |
| `GameSR-Lite` | `sr_lite` | 2× 超分 (`sr`) | ~6,936 | ~13.5 KB | 超轻量 2× 去锯齿与文字锐化 |

> [!NOTE]
> `GameSR` 参数数和体积数据源自 `frameflow/sr_config.py` 源码注释，精准张量统计由导出清单 `model.json` 的 `n_params` 字段写入；权重获取请参阅 [GitHub Releases](https://github.com/CyberNova2333/frameflow/releases)。

继续阅读：[浏览器与模拟器接入](/docs/integration)、[训练、蒸馏与双端发版](/docs/training) 或查阅 [API 与模型格式参考](/api-reference/)。
