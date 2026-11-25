---
repoGuideSource: repo-docs/modules/training-and-release-pipeline.md
---

<!--@include: ../../repo-docs/modules/training-and-release-pipeline.md-->

---

## 详细操作参数与命令

以下补充直接节选自仓储运行脚本与发布说明：

```bash
# 0. 检查 Python / CUDA PyTorch / ffmpeg 环境与 CPU 单元测试
make env-check
make test

# 1. 下载并切分开放许可媒体序列
make data-starter   # 或 make data-full

# 2. 训练教师并蒸馏轻量学生模型（GameFI）
make train-std
make train-lite

# 3. 训练 2x 超分模型（GameSR）
.venv/bin/python scripts/prepare_sr_data.py
.venv/bin/python scripts/train_sr.py --preset sr_std
.venv/bin/python scripts/train_sr.py --preset sr_lite --teacher checkpoints/sr_std/best.pt

# 4. 打包附带 5 份许可协议、DATA_PROVENANCE.json 与 SHA256SUMS.txt 的压缩包
make package-release RELEASE_TAG=weights-v0.1.0
```
