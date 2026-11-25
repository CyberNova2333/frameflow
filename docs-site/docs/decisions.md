---
repoGuideSource: repo-docs/change-log.md
---

# 工程决策与合规边界（ADR 摘要）

本页汇总关键架构取舍、开源合规边界与既有实测/变更记录，便于维护者判断修改影响范围。

| 决策项 | 当前选择 | 放弃方案与原因 | 源码依据 |
| --- | --- | --- | --- |
| 运行时依赖边界 | 零 npm / 零 WASM 纯 ES 模块 + 原始着色器 | 放弃 ONNX Runtime Web / TF.js 降低包体、避免抢占模拟器内存并支持同设备纹理零拷贝 | `package.json`、`runtime/` |
| 不支持设备降级策略 | 拒绝启动、隐藏 UI，无 CPU/WASM 软推理 | 软件推理帧耗时远超原生游戏帧预算，会造成模拟器严重掉帧或卡死 | `runtime/index.js`、`docs/INTEGRATION.md` |
| 隐层通道倍数约束 | 强制为 `4` 的倍数（`__post_init__` 校验） | 保障 WebGL2 `RGBA16F` 每层正好打满 4 通道并配合 4-target MRT 单次输出 16 通道 | `frameflow/config.py`、`sr_config.py` |
| 权重仓库隔离策略 | Git 仅存代码与图谱生成逻辑，权重随 Release / OCI 单独发放 | 防止二进制权重污染 Git 历史，并强制捆绑 `DATA_PROVENANCE.json` 与五份许可文本 | `.gitignore`、`scripts/package_release.py` |
| 游戏 UI 保护手段 | 数据集叠加程序合成静态 HUD 并施加零光流监督 | 防止光流把动态背景拖曳到固定血条、小地图与对话文字边缘 | `frameflow/dataset.py`、`losses.py` |

---

<!--@include: ../../repo-docs/change-log.md-->
