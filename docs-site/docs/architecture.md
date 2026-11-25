---
repoGuideSource: repo-docs/README.md
---

# 架构全景与交互图谱

FrameFlow 将 **PyTorch 训练导出层**、**零依赖双后端 JS 运行时** 与 **模拟器零拷贝共享接入** 收敛到一套 `format_version = 1` 契约之上，互不依赖臃肿运行时框架。

<DiagramFrame
  src="/diagrams/frameflow-architecture.html"
  title="FrameFlow 训练导出、双后端运行时与宿主零拷贝架构图（支持平移 / 缩放 / 节点探查）"
  aspect="940 / 330"
  new-tab-label="在新窗口全屏打开 ↗"
/>

## 源码深潜入口与活文档总览

- [一次真实运行走查（源码）](/docs/walkthrough)：顺着出口三文件 -> `probeCapabilities()` / `createFrameFlow()` -> `parseModel()` -> `pair()`/`mid()` -> `FrameFlowPacer` 跟踪全部阶段交接与失败边界；
- [代码地图（按职责定位）](/docs/code-map)：按 `frameflow/`、`runtime/`、`scripts/`、`tests/` 快速索引源码与验证命令；
- [工程决策与合规边界](/docs/decisions)：核对关键取舍及 `repo-docs/change-log.md` 验证同步台账。

---

<!--@include: ../../repo-docs/README.md-->

---

<!--@include: ../../repo-docs/glossary.md-->
