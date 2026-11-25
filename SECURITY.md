# 安全说明 / Security

<!-- repo-maintenance:begin -->
## 本地开发与制品边界 / Local development and release boundaries

1. **禁止入库私密数据与二进制权重**：不要将本地凭据、环境变量文件（`.env`）、私有密钥、训练数据集视频（`data/`）或二进制权重文件（`*.bin` / `*.pt` / `*.pth`）提交进版本控制。
2. **数据与许可合规核验**：`scripts/prepare_data.py` 与 `scripts/prepare_sr_data.py` 仅收录经过许可核验的开放媒体；发布包均须包含 `DATA_PROVENANCE.json`、`SHA256SUMS.txt` 与完整许可文本，防止来源不明或非商业授权权重流入下游分发。
3. **浏览器运行时安全边界**：`runtime/` 为纯前端计算/着色器 ES Module，不发起隐式遥测，不读写外部私密凭据；向共享上下文（`GPUDevice` / `WebGL2RenderingContext`）传递的外部纹理生命周期仍由宿主模拟器控制。

## 漏洞与合规问题报告 / Reporting vulnerabilities and licensing issues

如发现数值越界、浏览器崩溃路径、安全边界漏洞或任何第三方素材署名/许可遗漏，请通过 GitHub 仓库的私密安全通道或直接联系维护者（CyberNova），避免在公开 Issue 中粘贴敏感凭据或未核实的私钥内容。
<!-- repo-maintenance:end -->
