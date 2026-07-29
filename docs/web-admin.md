# CosyVoice3Pro 同端口 Web 管理后台

## 1. 访问地址

网页后台：

```text
http://服务器地址:18000/
```

原 Triton HTTP API 地址保持不变：

```text
http://服务器地址:18000/v2/
```

其他端口保持不变：

| 端口 | 用途 |
| --- | --- |
| `18000` | Web 管理后台、声纹注册、统一 TTS 和 Triton HTTP Gateway |
| `18001` | Triton gRPC |
| `18002` | Triton Metrics |

## 2. 运行架构

```text
浏览器 / HTTP API 客户端
              |
          18000
              |
    CosyVoice3Pro Web Gateway
       |              |             |
       | /            | Public API  | Advanced API
       |              | /health     | /v2/*
       |              | /register   |
       |              | /speakers   |
       |              | /tts/       |
       v              v             v
  静态管理网页      业务 API     Triton HTTP :18100
                               |
                       CosyVoice3Pro Models
```

`18100` 只在容器内部使用，没有映射到宿主机。原有调用
`http://服务器:18000/v2/...` 会由 Gateway 原样代理，因此 curl 和现有
客户端不需要修改地址。网页本身只使用 Public API，不直接调用 `/v2/*`
或 `/admin/*`。

## 3. 页面功能

- Triton、Gateway 和模型健康状态
- 已注册 Speaker 列表和搜索
- 上传 WAV、MP3、M4A 等提示音频
- 使用公开 HTTP/HTTPS 音频 URL 注册声纹
- 服务端统一转换为 16kHz 单声道 FP32
- 注册或更新 Speaker
- 设置注册默认 `prompt` 画像
- 推理时使用默认画像或非空 `prompt` 单次覆盖
- 配置慢速、均衡、快速语速
- 配置较小、标准、较大音量
- 配置长文本分段字符数
- 选择 PCM、MP3、WAV、AAC、M4A、Opus、OGG、FLAC、WebM
- 在线试听并下载处理后的 16kHz 音频
- 删除 Speaker

浏览器上传的提示音频建议为 3～10 秒清晰单人声，服务端约束为
0.5～30 秒。

## 4. 启动和管理

```bash
bash manage.sh start
bash manage.sh restart
bash manage.sh status
bash manage.sh logs
```

健康检查：

```bash
curl --fail-with-body \
  http://127.0.0.1:18000/health
```

## 5. Speaker 数据备份

当前容器中的 Speaker 数据已经备份至：

```text
data/speakers
```

手动刷新备份：

```bash
bash manage.sh backup
```

新执行 `install` 创建容器时，该目录会挂载到：

```text
/workspace/cosyvoice_speaker_store
```

`install` 或 `remove` 在删除现有容器前也会自动执行 Speaker 数据备份。

## 6. 兼容与回滚

临时关闭 Web Gateway、让 Triton 重新直接监听 `18000`：

```bash
COSYVOICE_WEB_GATEWAY_ENABLED=false \
  bash manage.sh restart
```

恢复 Web Gateway：

```bash
bash manage.sh restart
```

关闭 Gateway 时：

- `/v2/*` Triton API 继续使用 `18000`
- `/` 不再提供管理网页
- `/health` 不再提供对外聚合健康检查
- `/register` 不再提供对外声纹注册
- `/speakers` 不再提供对外声纹查询和删除
- `/tts/` 不再提供音频流接口
- `18001` 和 `18002` 不变

## 7. 并发性能配置

默认 `COSYVOICE_PERFORMANCE_PROFILE=auto`。80GB GPU 自动启用双
token2wav、双 vocoder 的 `throughput` profile，其他 GPU 使用保守的
`balanced` profile。

手动切换并重启：

```bash
COSYVOICE_PERFORMANCE_PROFILE=throughput \
  bash manage.sh restart
```

```bash
COSYVOICE_PERFORMANCE_PROFILE=balanced \
  bash manage.sh restart
```

`/tts/` 会返回 `Server-Timing`、`X-CosyVoice-Inference-Ms` 和
`X-CosyVoice-Encode-Ms`，可用于区分模型排队和音频编码耗时。实例数、
显存参数、压力测试和调优方法见[性能基准文档](benchmark.md)。

## 8. 仓库结构

```text
gateway/
models/
scripts/client.py
manage.sh
docs/
data/speakers/
```

Web Gateway、Triton Models、命令行客户端和 Speaker 数据均属于
CosyVoice3Pro 自身，不依赖额外的 Python UI 服务。网页的文件上传和
音频 URL 注册都通过 `POST /register` 完成，由服务端统一下载、转码和
校验。列表、查询和删除分别通过 `GET /speakers`、
`GET /speakers/{speakerId}` 和 `DELETE /speakers/{speakerId}` 完成。

接口说明：

- [对外开发者 API](public-api.md)
- [内部 Triton 高级 API](advanced-api.md)

## 9. 安全说明

当前 `18000` Triton API、声纹注册、TTS 接口和管理网页没有应用层账号
认证，与升级前的 Triton API 权限模型一致。管理页面包含注册和删除操作，
应只向可信内网开放，或在外层反向代理、防火墙中增加认证和来源限制。
