# WebGPU 与 WebGL2 双后端运行时

浏览器端画质增强必须面对两类约束：模拟器通常已经占用了主线程算力与独立的图形上下文，且移动端、Linux 核显或旧版 WebView 对现代图形接口的支持参差不齐。`runtime/` 目录在不引入任何第三方 npm 依赖或 WebAssembly 模块的前提下，围绕统一权重格式实现了先计算着色器、后片元着色器的自动降级方案。

## 双后端探测与零兜底拒绝机制

调用 `probeCapabilities()` 或 `createFrameFlow()` 时，运行时按预设顺序探测可用环境：
1. 若未显式指定 `webgl2`，优先探测 WebGPU 适配器并检查特性列表；只要可用，就在创建逻辑设备时按需开启 `shader-f16`，不满足条件则回退到同 API 下的三十二位浮点别名。
2. 当环境无 WebGPU 或创建设备抛出异常（例如 `auto` 模式下降级），立即转入 WebGL2 探测，要求上下文能够附着并渲染十六位半浮点或全浮点颜色缓冲。
3. 若两种浮点加速条件都不具备，引擎直接抛出显式错误并告知原因，拒绝退回主线程或 Worker 上的 CPU 软推理，确保不会反向抢占模拟器的核心算力。

对应源码：[运行时总入口](../../runtime/index.js) 与 [接口声明](../../runtime/types.d.ts)。

## 半精度解析与静态执行计划编译

不论采用哪种后端，加载模型都共享 `loader.js` 的同一套逻辑。模型二进制文件中全部参数严格采用小端序 IEEE-754 `binary16` 格式紧邻排列；`parseModel` 先校验 `model.bin` 字节长度为偶数、`format_version === 1`，并逐个检查张量维度乘积与偏移量之和是否越界，再由 `f16ToF32` 手写位运算展开为 `Float32Array`，同时保留原始 `Uint16Array` 视图以便直接上传给支持半精度存储的 GPU 缓冲。

紧接着 `compilePlan(modelData, workW, workH, k, opts)` 在创建会话时静态展开算子图中的所有 `repeat` 循环，并精确划分为两阶段：
- `pairSteps`：仅依赖参考帧 A 与 B 的图像特征编码、图像金字塔、全局 SAD 代价体与粗估流计算；同一对输入帧切换多个中间时刻 `t` 时无需重复跑 `pairSteps`。
- `midSteps`：依赖具体插值相位 `t` 的特征时间调制（`evalFilmMlp` 在 JS 侧先把一维 `t` 算成通道缩放偏置向量，免去 GPU 内重复执行小线性层）、精细光流残差修正以及最终合成输出。

当插帧输入的宽高不满足 32 像素对齐时，`align32` 会将工作尺度向上补齐，并在输入阶段使用边缘复制策略填充边界。

对应源码：[模型加载与规划](../../runtime/loader.js)。

## WebGPU 与 WebGL2 的数值等价实现细节

虽然前端计算分别采用 WGSL 计算着色器与 GLSL ES 3.00 片元着色器，二者在关键像素坐标与采样公式上保持与 PyTorch 参考模块严格等价，从而把无头 Edge 实测对齐维持在 62–72 dB PSNR 以上：

| 机制维度 | WebGPU Compute 后端 | WebGL2 Fragment Shader 后端 | 等价对齐依据 |
| --- | --- | --- | --- |
| 特征存储 | 平面一维存储缓冲 `array<f16>` / `array<f32>` | 四通道打包的 `RGBA16F` `TEXTURE_2D_ARRAY`（每层 4 通道） | 网络配置文件强制所有隐层通道数为 4 的倍数 |
| 卷积并行 | 工作组三维栅格直写计算结果 | 多渲染目标 `MRT drawBuffers` 每次输出 4 层（16 通道），提升 4 倍输入复用 | 计算图拓扑与卷积权重布局完全一致 |
| 图像垂直方向 | 像素左上角 `y=0`，对齐 PyTorch | 着色器显式通过 `uFlipSrcY` / 自上而下 `gl_FragCoord` 换算 | [算子实现](../../frameflow/ops.py) 与 `warp`/`render` 边界夹紧策略一致 |
| 外部纹理共享 | `createFrameFlow({ device })` 共享已有 `GPUDevice`，零拷贝读取外部纹理 | `createFrameFlow({ gl })` 共享已有 `WebGL2RenderingContext`，避免像素来回上传 | [接入说明](../../docs/INTEGRATION.md) 中的 PVSSPW 与 PPSSPW 对接方式 |

<details>
<summary>示例对比：同一组输入在不同后端与设备上的降级路径</summary>

1. **桌面或旗舰移动支持 WebGPU 共享设备（如 PVSSPW）**：引擎直接复用宿主的 `GPUDevice`，将游戏原生纹理通过 `setExternalInputTexture` 喂入，不经过主内存拷贝，单次 `pair()` 或 `mid()` 打包成单一命令缓冲区提交。
2. **Linux Firefox 或旧版移动 Safari 无 WebGPU（如 PPSSPW）**：`createFrameFlow({ backend: 'auto', gl })` 自动切换到 WebGL2，利用 `RGBA16F` 纹理阵列完成相同运算，并通过工作分辨率折衷（如四分之三/降尺度内部工作平面）保障帧时间可控。
3. **无半浮点纹理写回能力的极老设备**：`probeWebGL2` 检验帧缓冲完整性失败，`probeCapabilities().supported` 为 `false`，上层界面据此直接隐藏画质增强选项。

</details>

## 无回读帧节拍调度

在二倍或三倍插帧场景下，浏览器不能在渲染热路径上等待神经网络计算完成或同步回读张量，否则会将模拟器的模拟线程锁死在 GPU 栅栏上。`FrameFlowPacer` 独立管理游戏原始帧间隔估算与展示序列：

- 每当 `onSourceFrame(slot, nowMs)` 收到新源帧，若时间差落入合理范围（4 毫秒至 200 毫秒），更新指数滑动平均间隔 `emaIntervalMs`，据此按因子 2（`t = 0.5`）或因子 3（`t = 1/3, 2/3`）顺序排入中间帧与跟随其后的最新源帧展示计划。
- 展示时刻调用 `tick()` 推进队列；如果游戏或 GPU 产生积压使队列长度超出 `factor + 2` 上限，调度器立即从头部淘汰过时条目并累加 `stats.dropped` 计数，把展示延迟锁在可控窗口内。

对应源码：[无阻塞帧节拍器](../../runtime/pacer.js)。

证据状态：除特别标注外，本页基于当前源码已确认；跨浏览器对比帧时间援引自根目录基准说明与[数值对齐校验脚本](../../scripts/verify_backends.py)。
