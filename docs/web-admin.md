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
| `18000` | Web 管理后台、统一 TTS 和 Triton HTTP Gateway |
| `18001` | Triton gRPC |
| `18002` | Triton Metrics |

## 2. 运行架构

```text
浏览器 / HTTP API 客户端
              |
          18000
              |
    CosyVoice3Pro Web Gateway
       |          |             |
       | /        | /tts/       | /v2/*
       v          v             v
  静态管理网页  音频流接口   Triton HTTP :18100
                               |
                       CosyVoice3Pro Models
```

`18100` 只在容器内部使用，没有映射到宿主机。原有调用
`http://服务器:18000/v2/...` 会由 Gateway 原样代理，因此 curl 和现有
客户端不需要修改地址。

## 3. 页面功能

- Triton、Gateway 和模型健康状态
- 已注册 Speaker 列表和搜索
- 上传 WAV、MP3、M4A 等提示音频
- 浏览器端转换为 16kHz 单声道 FP32
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
curl -f http://127.0.0.1:18000/v2/health/ready
curl -f http://127.0.0.1:18000/v2/models/CosyVoice3Pro/ready
curl -sS http://127.0.0.1:18000/admin/api/info
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
- `/tts/` 不再提供音频流接口
- `18001` 和 `18002` 不变

## 7. 仓库结构

```text
gateway/
models/
scripts/client.py
manage.sh
docs/
data/speakers/
```

Web Gateway、Triton Models、命令行客户端和 Speaker 数据均属于
CosyVoice3Pro 自身，不依赖额外的 Python UI 服务。

## 8. 安全说明

当前 `18000` Triton API、TTS 接口和管理网页没有应用层账号认证，与
升级前的 Triton API 权限模型一致。管理页面包含注册和删除操作，应只向
可信内网开放，或在外层反向代理、防火墙中增加认证和来源限制。
