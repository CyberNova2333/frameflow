# 术语表

| 术语 | 项目里的意思 | 延伸阅读 |
| --- | --- | --- |
| `FrameFlow` | 整个项目的系列名、训练导出演算套件以及纯前端跨浏览器运行时总称。 | [源码阅读指南](README.md) |
| `GameFI` | 光流插帧模型家族名称，分为 `GameFI-Std` 与 `GameFI-Lite` 两档，导出标识任务类型为 `interp`。 | [GameFI 与 GameSR 模型及静态图格式](modules/gamefi-and-gamesr-models.md) |
| `GameFINet` / `FrameFlowNet` | Python 侧光流插帧主网络类；`GameFINet` 与 `FrameFlowNet` 为等价兼容别名。 | [GameFI 与 GameSR 模型及静态图格式](modules/gamefi-and-gamesr-models.md) |
| `GameSR` | 面向像素文字与图形边缘的固定二倍空间超分模型家族，分为 `GameSR-Std` 与 `GameSR-Lite`。 | [GameFI 与 GameSR 模型及静态图格式](modules/gamefi-and-gamesr-models.md) |
| `format_version = 1` | 紧凑模型产物的统一版本约束，由静态算子拓扑 `graph.json`、张量索引 `model.json` 与权重字节流 `model.bin` 组成。 | [一次真实主流程](walkthroughs/one-real-run.md) |
| `binary16` / `f16ToF32` | 小端序十六位 IEEE-754 半精度浮点数格式与纯 JS 标量解码函数，用于在无半精度原生支持时展平权重。 | [WebGPU 与 WebGL2 双后端运行时](modules/dual-backend-runtime.md) |
| `parseModel` / `compilePlan` | 共享加载器核心函数：前者解码并校验十六位权重和图引用完整，后者在创建会话时静态展开 `repeat` 循环与工作张量形状。 | [一次真实主流程](walkthroughs/one-real-run.md) |
| `pairSteps` / `midSteps` | `compilePlan` 划分的两段执行序列：成对特征与全局流估计只执行一次（`pairSteps`），随相位 `t` 变化的时间调制与细化按需执行（`midSteps`）。 | [WebGPU 与 WebGL2 双后端运行时](modules/dual-backend-runtime.md) |
| `GPUDevice` | WebGPU 逻辑设备接口；接入 PVSSPW 时由宿主与 `WebGPUEngine` 共享同一实例，实现原生游戏纹理零拷贝接入。 | [WebGPU 与 WebGL2 双后端运行时](modules/dual-backend-runtime.md) |
| `shader-f16` | WebGPU 可选硬件计算特性；适配器支持时自动请求开启以加速运算，驱动拒配或不支持时自动转回三十二位浮点。 | [WebGPU 与 WebGL2 双后端运行时](modules/dual-backend-runtime.md) |
| `RGBA16F TEXTURE_2D_ARRAY` / `MRT` | WebGL2 后端核心特征包封策略：把 4 个特征通道打包进单层二维纹理数组，并通过多渲染目标单次写入最高 16 个输出通道。 | [WebGPU 与 WebGL2 双后端运行时](modules/dual-backend-runtime.md) |
| `bgr_to_luma` | BGR 通道顺序到单通道亮度的固定转换权重（`0.114*B + 0.587*G + 0.299*R`），在 PyTorch 与两套着色器中逐字保持等价。 | [GameFI 与 GameSR 模型及静态图格式](modules/gamefi-and-gamesr-models.md) |
| `sad_volume` | 基于局部亮度差分并在三乘三窗内做零填充均值池化的相关性代价体算子，分别用于粗估计与精细光流对齐。 | [GameFI 与 GameSR 模型及静态图格式](modules/gamefi-and-gamesr-models.md) |
| `FilmHead` | 接收标量时间相位 `t` 生成每通道缩放因子与偏置的小型两层调制网络，使单模型支持二倍、三倍或任意相位插帧。 | [GameFI 与 GameSR 模型及静态图格式](modules/gamefi-and-gamesr-models.md) |
| `FrameFlowPacer` | 独立于 GPU 回读的实时展示节奏调度器，维护源帧间隔指数平滑值并按队列上限淘汰积压陈旧帧。 | [WebGPU 与 WebGL2 双后端运行时](modules/dual-backend-runtime.md) |
| `overlay_synthetic_hud` | 数据增强函数，在三连帧同一坐标叠加静止虚拟游戏控件并返回静态区域遮罩，配合零光流惩罚减轻界面文字重影。 | [开源许可数据、训练与双端发版流程](modules/training-and-release-pipeline.md) |
| `align32` | 把输入图像尺寸向上取整至 32 整数倍的工具函数，配合边缘像素复制避免下采样编码尺寸奇偶错位。 | [一次真实主流程](walkthroughs/one-real-run.md) |
| `DATA_PROVENANCE.json` | 随模型权重发行压缩包一同附带的开放许可素材溯源清单，保证每个发行版本来源清晰可核。 | [开源许可数据、训练与双端发版流程](modules/training-and-release-pipeline.md) |
