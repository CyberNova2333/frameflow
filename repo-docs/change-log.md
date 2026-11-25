# 指南变更记录

| Timestamp | Request | Actions | Verification | Result |
| --- | --- | --- | --- | --- |
| 2026-11-25 11:04 Asia/Shanghai | 补齐 FrameFlow `repo-docs/` 中文源码指南并接入 VitePress 双语文档站与 Pages 工作流 | 新建 Standard 模式源码指南（总览、五步主流程、代码地图、三篇专题模块、术语表、源码证据、读者复核），更新 `AGENTS.md`/`CLAUDE.md`、`meta.json`/`README*`，并构建 `docs-site/` 与 `.github/workflows/docs-pages.yml` | `validate_repo_docs.py`（0 errors, 2既有许可文本警告）；`pnpm --dir docs-site test`（1/1 通过）；`vitepress build --base /frameflow/`（2.04s 通过）；`make test`（Python 6/6 + Node 3/3 通过）；`meta.json` & UTF-8 校验通过 | Synced through f328c3d82bfff02b380689f1e1c2525bfdc07d07；上述校验已于本次提交前完整实跑通过 |

## 本轮验证与范围边界

- 本文档指南与文档站构建输出（VitePress 默认主题）相互独立，站点产物不包含任何 `.pt` 或 `.bin` 模型权重与训练数据素材。
- 纯代码与文档构建校验均已在提交前通过；涉及硬件驱动与浏览器的 GPU 数值对齐基准（`scripts/verify_backends.py`、`scripts/verify_sr_backends.py`）沿用既定基准记录边界，未在本轮纯文档变更中重跑。

Synced through f328c3d82bfff02b380689f1e1c2525bfdc07d07
