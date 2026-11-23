# 贡献指南 / Contributing

## 贡献许可与附加条款约定 / Contribution Licensing & Additional Terms

依据 [`LICENSE-ADDITIONAL_TERMS-zh-CN.md`](LICENSE-ADDITIONAL_TERMS-zh-CN.md)（v1.6.7-r4）第 1.7—1.9 款，向本项目提交拉取请求（PR）、补丁或其他贡献时，你须确认对所提交内容享有合法授权权利并同意按对应基础许可证（第一方原创代码与自训权重为 [GNU AGPLv3](LICENSE)）提供。如你希望将个人享有版权的贡献纳入 [`LICENSE-ADDITIONAL_TERMS-zh-CN.md`](LICENSE-ADDITIONAL_TERMS-zh-CN.md)（英文参考：[`LICENSE-ADDITIONAL_TERMS-en.md`](LICENSE-ADDITIONAL_TERMS-en.md)，实施指南：[`IMPLEMENTATION-GUIDE.zh-en.md`](IMPLEMENTATION-GUIDE.zh-en.md)）适用范围，请在 PR 说明或可核实记录中**分别明确声明**对附加要求以及对第 9.5–9.6 款有条件附加许可的授权范围；未作明确声明的贡献将如实记为 `[附加条款：未发表意见]` 并按原适用基础许可证处理。引入第三方组件、公开训练数据源或附带独立图标、美术、字体资产的，须按 [`ASSETS_LICENSES.md`](ASSETS_LICENSES.md) 如实登记来源、原始作者与有效许可原文；**严禁提交任何非商业（如 CC BY-NC）或来源不明的权重与素材文件**。

Pursuant to Sections 1.7–1.9 of [`LICENSE-ADDITIONAL_TERMS-zh-CN.md`](LICENSE-ADDITIONAL_TERMS-zh-CN.md) (v1.6.7-r4; English reference: [`LICENSE-ADDITIONAL_TERMS-en.md`](LICENSE-ADDITIONAL_TERMS-en.md); implementation guide: [`IMPLEMENTATION-GUIDE.zh-en.md`](IMPLEMENTATION-GUIDE.zh-en.md)), submitting a pull request, patch, or other contribution confirms that you hold the requisite rights to license your submission under the applicable base license ([GNU AGPLv3](LICENSE) for first-party code and self-trained weights). Express authorization for the Additional Requirements and for the conditional additional permission in Sections 9.5–9.6 is verified separately: please state your covered materials and consent scope explicitly in the PR or a verifiable record; submissions without an express verifiable statement are recorded as `[Additional terms: No recorded position]` and remain governed by their base license. Any third-party components, dataset sources, or standalone artwork, fonts, or icons must be truthfully documented in [`ASSETS_LICENSES.md`](ASSETS_LICENSES.md). Do not commit non-commercial (e.g., CC BY-NC) or unverified third-party weights or media assets.

## 提交与核验 / Commits and Verification

- 提交标题采用 Conventional Commits：`type(scope): 描述`（允许 `feat`、`fix`、`docs`、`style`、`refactor`、`perf`、`test`、`build`、`ci`、`chore`、`revert`、`data`），不超过 100 个字符。
- 提交前请至少运行轻量级测试：`make test`（Python 模型与导出单元测试 + Node 运行时加载与节拍器测试）；涉及 WebGPU / WebGL2 Shader 修改时，请额外运行 `make verify-runtime` 核验与 PyTorch 的 PSNR 数值对齐。
- 仓库不提交二进制权重（`*.bin` / `*.pt` / `*.pth`）与原始训练视频，正式权重通过 Release 流程附带五份许可合规文本与 `SHA256SUMS.txt` 发布。
