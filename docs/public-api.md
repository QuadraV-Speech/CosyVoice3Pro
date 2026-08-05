# CosyVoice3Pro 对外 API

本文档面向业务开发者。对外 API 使用普通 JSON、表单和音频流，不需要
了解 Triton Tensor 协议。

## 1. 基本信息

默认地址：

```text
http://127.0.0.1:18000
```

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| `GET` | `/health` | 服务与模型健康检查 |
| `POST` | `/register` | 上传音频或使用 URL 注册/更新声纹 |
| `GET` | `/speakers` | 查询全部声纹 |
| `GET` | `/speakers/{speakerId}` | 查询一个声纹 |
| `DELETE` | `/speakers/{speakerId}` | 删除一个声纹 |
| `POST` | `/tts/` | 生成处理后的音频 |
| `POST` | `/tts/stream` | SSE 在线流式合成与播报 |

Web 管理后台也只使用这些对外 API。

## 2. 通用规则

### 2.1 Speaker ID

`speakerId` 必须满足：

- 长度 1～128
- 只允许字母、数字、下划线、中划线和点
- 不允许包含 `..`

合法示例：

```text
narrator_01
user-1001.voice-a
```

### 2.2 Content-Type

- 上传文件：`multipart/form-data`
- URL 注册：`application/json` 或 `multipart/form-data`
- TTS：推荐 `multipart/form-data`
- 查询和删除：不需要请求体

### 2.3 错误响应

错误统一返回：

```json
{
  "detail": "具体错误信息"
}
```

常见状态码：

| 状态码 | 说明 |
| --- | --- |
| `400` | 必填字段为空、下载失败或音频无法解码 |
| `404` | Speaker 不存在 |
| `413` | 请求体或远程音频超过 32 MiB |
| `415` | Content-Type 不受支持 |
| `422` | 字段格式、枚举或参数组合非法 |
| `502` | 模型推理或 Speaker Registry 返回异常 |
| `503` | Triton、GPU 或模型暂时不可用 |

## 3. 健康检查

```bash
curl --fail-with-body \
  "http://127.0.0.1:18000/health"
```

成功响应：

```json
{
  "status": "ok",
  "service": "CosyVoice3Pro Web Gateway",
  "version": "1.9.0",
  "gatewayReady": true,
  "tritonReady": true,
  "models": {
    "ttsReady": true,
    "streamingTtsReady": true,
    "speakerRegistryReady": true
  }
}
```

全部就绪时返回 HTTP `200`；任一核心服务未就绪时返回 HTTP `503`，
同时 `status` 为 `unavailable`。

## 4. 注册或更新声纹

```text
POST /register
```

使用已存在的 `speakerId` 会原子更新该声纹。

### 4.1 参数

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `speakerId` | string | 是 | 声纹 ID；兼容别名 `speaker_id` |
| `reference_text` | string | 是 | 音频中实际说出的准确文本，最多 4096 字；兼容 `referenceText` |
| `prompt` | string | 否 | 默认声音画像，默认空，最多 512 字 |
| `audio` | file | 二选一 | 上传的提示音频；兼容字段 `prompt_audio` |
| `audio_url` | string | 二选一 | 公开音频地址；兼容字段 `audioUrl` |

`audio` 与 `audio_url` 必须且只能提供一个。音频约束：

- 时长 0.5～30 秒，建议使用 3～10 秒清晰单人声
- 文件或下载内容最大 32 MiB
- 支持 FFmpeg 可解码的 WAV、MP3、M4A、AAC、FLAC、OGG、Opus、WebM
- 服务端自动转换为 16kHz 单声道

### 4.2 上传文件

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/register" \
  -F "speakerId=narrator_upload_01" \
  -F "audio=@./reference.wav;type=audio/wav" \
  -F "reference_text=这是参考音频中实际说出的内容。" \
  -F "prompt=请用成熟、稳重、亲切的语气说话。"
```

### 4.3 使用音频 URL

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "speakerId": "narrator_url_01",
    "audio_url": "https://example.com/reference.mp3",
    "reference_text": "这是参考音频中实际说出的内容。",
    "prompt": "请用温柔、舒缓的语气说话。"
  }'
```

URL 仅允许公开的 HTTP/HTTPS 地址。服务端禁止访问回环、内网、链路本地
和云元数据地址，每次重定向都会重新校验 DNS 与实际连接 IP。

### 4.4 成功响应

```json
{
  "status": "ok",
  "speakerId": "narrator_url_01",
  "speakerVersion": "示例版本号",
  "source": "url",
  "speaker": {
    "speakerId": "narrator_url_01",
    "speakerVersion": "示例版本号",
    "referenceText": "这是参考音频中实际说出的内容。",
    "prompt": "请用温柔、舒缓的语气说话。",
    "sampleRate": 16000,
    "samples": 55680,
    "durationSeconds": 3.48,
    "registeredAt": 1785290000
  }
}
```

`source` 为 `upload` 或 `url`。响应中的 `metadata` 字段为早期
`/register` 客户端保留，新客户端应读取 `speaker`。

## 5. 查询声纹

### 5.1 查询全部

```bash
curl --fail-with-body \
  "http://127.0.0.1:18000/speakers"
```

响应：

```json
{
  "status": "ok",
  "count": 1,
  "speakers": [
    {
      "speakerId": "narrator_01",
      "speakerVersion": "示例版本号",
      "referenceText": "这是参考音频中实际说出的内容。",
      "prompt": "请用成熟、稳重的语气说话。",
      "sampleRate": 16000,
      "samples": 55680,
      "durationSeconds": 3.48,
      "registeredAt": 1785290000
    }
  ]
}
```

### 5.2 查询单个

```bash
speaker_id="narrator_01"

curl --fail-with-body \
  "http://127.0.0.1:18000/speakers/${speaker_id}"
```

成功响应：

```json
{
  "status": "ok",
  "speaker": {
    "speakerId": "narrator_01",
    "speakerVersion": "示例版本号",
    "referenceText": "这是参考音频中实际说出的内容。",
    "prompt": "请用成熟、稳重的语气说话。",
    "sampleRate": 16000,
    "samples": 55680,
    "durationSeconds": 3.48,
    "registeredAt": 1785290000
  }
}
```

Speaker 不存在时返回 HTTP `404`。

## 6. 删除声纹

```bash
speaker_id="narrator_01"

curl --fail-with-body \
  -X DELETE \
  "http://127.0.0.1:18000/speakers/${speaker_id}"
```

成功响应：

```json
{
  "status": "ok",
  "speakerId": "narrator_01",
  "speakerVersion": "示例版本号",
  "deleted": true
}
```

Speaker 不存在时返回 HTTP `404`。

## 7. 文字转语音

```text
POST /tts/
Content-Type: multipart/form-data
```

接口直接返回音频流。它支持三种声音来源，优先级为：

1. `prompt_audio + prompt_text`：本次即时克隆
2. `speakerId`：使用已注册声纹
3. `tts_style`：使用内置声音

同时传入多个来源时，优先级高的来源生效。

### 7.1 通用参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `text` | string | 是 | 无 | 需要合成的文本 |
| `prompt` | string | 否 | 空 | 本次画像覆盖，最长 512 字 |
| `language` | string | 否 | `zh` | 兼容字段；模型自动识别语言 |
| `speed` | enum | 否 | `balanced` | `low`、`balanced`、`fast` |
| `volume` | enum | 否 | `middle` | `small`、`middle`、`large` |
| `output_format` | enum | 否 | `mp3` | `pcm`、`mp3`、`wav`、`aac`、`m4a`、`opus`、`ogg`、`flac`、`webm` |
| `max_chars` | int | 否 | `80` | 长文本每段最大字符数，必须大于 0 |

声音来源参数：

| 参数 | 场景 | 说明 |
| --- | --- | --- |
| `speakerId` | 注册声纹 | `/register` 创建的 Speaker ID；兼容 `speaker_id` |
| `prompt_audio` | 即时克隆 | 本次上传的参考音频，0.5～30 秒 |
| `prompt_text` | 即时克隆 | `prompt_audio` 中实际说出的准确文本 |
| `tts_style` | 内置声音 | 编号 `1`～`4`，默认 `1` |

### 7.2 使用注册声纹

不传 `prompt` 或传空字符串时，使用注册时保存的默认画像：

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/tts/" \
  -F "text=你好，这是已注册声纹的语音合成。" \
  -F "speakerId=narrator_01" \
  -F "speed=balanced" \
  -F "volume=middle" \
  -F "output_format=mp3" \
  --output registered.mp3
```

非空 `prompt` 只覆盖本次画像，不修改注册数据：

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/tts/" \
  -F "text=太好了，我们今天完成了新的升级！" \
  -F "speakerId=narrator_01" \
  -F "prompt=请非常开心、兴奋地说话。" \
  -F "speed=fast" \
  -F "volume=large" \
  -F "output_format=wav" \
  --output happy.wav
```

### 7.3 即时提示音频

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/tts/" \
  -F "text=你好，这是即时声音克隆测试。" \
  -F "prompt_audio=@./reference.wav;type=audio/wav" \
  -F "prompt_text=参考音频中实际说出的内容。" \
  -F "prompt=请用成熟、稳重的语气说话。" \
  -F "output_format=m4a" \
  --output raw-prompt.m4a
```

### 7.4 内置声音

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/tts/" \
  -F "text=你好，这是内置声音测试。" \
  -F "tts_style=1" \
  -F "output_format=mp3" \
  --output builtin.mp3
```

内置映射：

| `tts_style` | Speaker ID |
| --- | --- |
| `1` | `common_speaker_1` |
| `2` | `common_speaker_2` |
| `3` | `common_speaker_3` |
| `4` | `common_speaker_4` |

未知编号按兼容行为回退到 `common_speaker_1`。

### 7.5 TTS 响应

成功时返回所选格式的 16kHz 单声道音频流：

```text
Content-Type: 对应的音频 MIME 类型
Content-Disposition: inline; filename="tts.输出格式"
X-CosyVoice-Mode: tts_style | speaker_id | prompt_audio
X-CosyVoice-Speaker: 实际 Speaker ID 或 raw_prompt
X-CosyVoice-Prompt-Override: true | false
X-CosyVoice-Segments: 长文本分段数量
X-CosyVoice-Inference-Ms: Triton 推理阶段耗时（毫秒）
X-CosyVoice-Encode-Ms: 音频后处理和编码耗时（毫秒）
Server-Timing: inference、encode 与 total 分阶段耗时
```

建议始终使用 curl 的 `--output` 保存响应。

## 8. SSE 在线流式合成

```text
POST /tts/stream
Accept: text/event-stream
Content-Type: multipart/form-data
```

接口不等待完整音频：Speech LLM 产生足够的语音 Token 后，Flow 与 Vocoder
立即生成首段波形，Gateway 持续做语速、音量和 16kHz 重采样，再通过 SSE
返回 Base64 PCM。它支持与 `/tts/` 相同的三种声音来源和 `prompt` 覆盖规则。

### 8.1 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `text` | string | 是 | 无 | 合成文本，在线模式最多 1000 字符 |
| `speakerId` | string | 否 | 无 | 已注册声纹；兼容 `speaker_id` |
| `prompt` | string | 否 | 空 | 本次画像覆盖，最多 512 字 |
| `speed` | enum | 否 | `balanced` | `low`、`balanced`、`fast` |
| `volume` | enum | 否 | `middle` | `small`、`middle`、`large` |
| `tts_style` | int | 否 | `1` | 未传 Speaker 和参考音频时使用内置声音 |
| `prompt_audio` | file | 否 | 无 | 即时克隆参考音频 |
| `prompt_text` | string | 条件必填 | 无 | 上传 `prompt_audio` 时必填 |

流式输出固定为 16kHz、单声道、little-endian signed 16-bit PCM，不能通过
`output_format` 修改。`max_chars` 也不适用于在线模式，因为拆成多个完整请求
会破坏连续的首包与播放语义。

### 8.2 查看原始 SSE

`-N/--no-buffer` 很重要，否则 curl 可能缓冲事件：

```bash
curl --fail-with-body -N --no-buffer \
  -X POST "http://127.0.0.1:18000/tts/stream" \
  -H "Accept: text/event-stream" \
  -F "text=你好，这段声音会在生成时立即开始播放。" \
  -F "speakerId=narrator_01" \
  -F "prompt=请用自然、清晰的语气说话。" \
  -F "speed=balanced" \
  -F "volume=middle"
```

Linux 上可将 SSE 音频事件直接送入 `ffplay` 在线播放（需要 `jq`、
`coreutils` 和 FFmpeg）：

```bash
curl --fail-with-body -sS -N --no-buffer \
  -X POST "http://127.0.0.1:18000/tts/stream" \
  -F "text=你好，这是 curl 驱动的 SSE 在线播报。" \
  -F "speakerId=narrator_01" \
  -F "speed=balanced" \
  -F "volume=middle" \
| awk '/^data: / { sub(/^data: /, ""); print }' \
| jq -r 'select(.audio != null) | .audio' \
| base64 --decode \
| ffplay -loglevel error -nodisp -autoexit \
    -f s16le -ar 16000 -ac 1 -
```

### 8.3 SSE 事件

```text
event: meta
data: {"requestId":"...","encoding":"pcm_s16le","sampleRate":16000,"channels":1,"concurrencyLimit":16,"queueTimeoutSeconds":15}

event: queue
data: {"requestId":"...","queueMs":1.8,"concurrencyLimit":16}

id: 0
event: audio
data: {"seq":0,"samples":3200,"audio":"Base64 PCM..."}

event: done
data: {"chunks":18,"samples":57600,"durationSeconds":3.6,"firstAudioMs":620.4,"queueMs":1.8,"tritonFirstAudioMs":351.2,"postprocessFirstAudioMs":269.2,"inferenceFirstAudioMs":618.6,"totalMs":910.2}
```

事件含义：

- `meta`：一次请求的声音来源、编码、采样率、语速和音量
- `queue`：取得服务端流式并发槽后的等待时间和当前并发上限
- `audio`：可立即播放的 PCM 分块，`seq` 从 0 递增
- `done`：总分块、音频时长、服务端首包、排队、Triton 首段、后处理首段和
  总生成耗时；所有 `*Ms` 字段单位均为毫秒
- `error`：响应开始后的推理错误；收到后应停止播放并展示 `detail`。排队超过
  生产超时时返回 `code=STREAM_BUSY`、`queueMs` 和 `retryAfterSeconds`，客户端
  应按建议退避重试
- `: keep-alive`：长时间没有音频时的 SSE 注释心跳

由于接口为带表单请求体的 `POST`，浏览器应使用 `fetch()` 加
`response.body.getReader()`，而不是只支持 GET 的原生 `EventSource`。
客户端断开连接时，Gateway 会取消对应 gRPC 推理。FFmpeg 延迟到 Triton
首块到达后才创建，并在断连时立即终止，不会让排队请求形成后处理进程风暴。

### 8.4 即时参考音频流式合成

```bash
curl --fail-with-body -N --no-buffer \
  -X POST "http://127.0.0.1:18000/tts/stream" \
  -F "text=这是不注册声纹的即时流式克隆。" \
  -F "prompt_audio=@./reference.wav;type=audio/wav" \
  -F "prompt_text=参考音频中实际说出的准确文本。" \
  -F "prompt=请用温柔、舒缓的语气说话。"
```

## 9. Web 管理后台

浏览器访问：

```text
http://服务器地址:18000/
```

页面通过本文件中的对外 API 完成健康检查、声纹查询、注册、删除、完整音频
合成和 SSE 在线播报，不直接调用内部 Triton `/v2/*`。

## 10. 安全建议

对外 API 默认没有应用层登录认证。公网开放前应在反向代理或 API Gateway
中配置：

- TLS
- 身份认证与权限控制
- 注册、删除和 TTS 的请求限流
- 请求体与访问日志策略
- Speaker ID 的业务级命名空间或租户隔离
