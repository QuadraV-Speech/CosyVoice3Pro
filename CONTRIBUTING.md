# Contributing to CosyVoice3Pro

感谢你帮助改进 CosyVoice3Pro。Bug 修复、文档、兼容性验证、性能数据和新
GPU 测试结果都很有价值。

## 提交 Issue 前

- 使用最新 `master` 复现问题。
- 查看已有 Issue，避免重复提交。
- Bug 请附上复现步骤、预期结果、实际结果和必要日志。
- 请删除日志中的公网地址、Token、音频隐私数据和其他敏感信息。

## 本地检查

提交 Pull Request 前至少执行：

```bash
python3 -m py_compile \
  gateway/app.py \
  gateway/legacy_tts.py \
  gateway/speaker_registration.py \
  scripts/client.py \
  scripts/benchmark.py

node --check gateway/web/app.js
bash -n manage.sh
git diff --check
```

服务可用时建议继续执行：

```bash
curl --fail-with-body http://127.0.0.1:18000/health

python3 scripts/benchmark.py \
  --speaker-id common_speaker_1 \
  --concurrency 1 \
  --requests 2
```

## API 边界

- 面向业务开发者的兼容接口位于 `/health`、`/register`、`/speakers` 和
  `/tts/`。
- Web 工作台应只使用上述 Public API。
- `/v2/*`、gRPC 和 Metrics 属于内部高级接口。
- 修改已有请求或响应字段时，请保留向后兼容，或在 PR 中明确标注破坏性
  变更及迁移方法。

## Pull Request

1. 每个 PR 只解决一个明确问题。
2. 说明为什么修改、如何修改以及如何验证。
3. UI 变化请附修改前后截图或动图。
4. 性能优化请提供复现命令、软硬件环境和修改前后数据。
5. 不要提交模型权重、Speaker 数据、生成音频、密钥或本机配置。

提交 PR 即表示你有权贡献相关代码和资源，并同意项目按仓库最终采用的
许可证分发这些贡献。
