# CosyVoice3Pro `/tts/` 兼容接口

## 1. 接口地址

```text
POST http://服务器地址:18000/tts/
```

该接口保留旧版 CosyVoice `/tts/` 调用方式，使用
`multipart/form-data` 请求并直接返回音频流。它复用 CosyVoice3Pro
中已经注册的内置声纹，推理时不会重复上传或提取提示音频。

## 2. 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `text` | string | 是 | 无 | 需要合成的文本 |
| `language` | string | 否 | `zh` | 兼容旧客户端保留；CosyVoice3 自动处理支持的语言 |
| `speed` | enum | 否 | `balanced` | 语速：`low`、`balanced`、`fast` |
| `volume` | enum | 否 | `middle` | 音量：`small`、`middle`、`large` |
| `output_format` | enum | 否 | `mp3` | `pcm`、`mp3`、`wav`、`aac`、`m4a`、`opus`、`ogg`、`flac`、`webm` |
| `max_chars` | int | 否 | `80` | 长文本分段时每段最大字符数，必须大于 0 |
| `tts_style` | int | 否 | `1` | 内置说话人编号 |

内置说话人映射：

| `tts_style` | CosyVoice3Pro Speaker ID |
| --- | --- |
| `1` | `common_speaker_1` |
| `2` | `common_speaker_2` |
| `3` | `common_speaker_3` |
| `4` | `common_speaker_4` |

为兼容旧版行为，传入未知的 `tts_style` 编号时会回退到
`common_speaker_1`。

## 3. 可直接运行的 curl

### 3.1 MP3

```bash
address=127.0.0.1

curl --fail-with-body \
  -X POST "http://${address}:18000/tts/" \
  -F "text=你好，这是一个 CosyVoice3Pro 接口测试。Nice to meet you!" \
  -F "language=zh" \
  -F "tts_style=1" \
  -F "speed=balanced" \
  -F "volume=middle" \
  -F "output_format=mp3" \
  -F "max_chars=80" \
  --output tts_output_builtin.mp3
```

### 3.2 WAV

```bash
address=127.0.0.1

curl --fail-with-body \
  -X POST "http://${address}:18000/tts/" \
  -F "text=你好，这是二号内置说话人的 WAV 测试。" \
  -F "tts_style=2" \
  -F "speed=fast" \
  -F "volume=large" \
  -F "output_format=wav" \
  --output tts_output_builtin.wav
```

### 3.3 使用默认参数

只有 `text` 是必填参数：

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/tts/" \
  -F "text=你好，这是使用全部默认参数的测试。" \
  --output tts_output_default.mp3
```

## 4. 响应

成功时返回所选格式的音频二进制流，同时包含：

```text
Content-Type: 对应的音频 MIME 类型
Content-Disposition: inline; filename="tts.输出格式"
X-CosyVoice-Speaker: 实际使用的 Speaker ID
X-CosyVoice-Segments: 长文本分段数量
```

建议始终使用 `--output` 保存结果；否则终端会直接显示二进制内容。

常见错误：

| HTTP 状态码 | 说明 |
| --- | --- |
| `400` | `text` 为空或请求体无法解析 |
| `413` | 请求体超过 4 MiB |
| `415` | 请求 Content-Type 不受支持 |
| `422` | 枚举参数、`max_chars` 或 `tts_style` 类型错误 |
| `502` | Triton 推理失败或响应内容无效 |
| `503` | Triton 服务不可用 |

## 5. 长文本处理

当文本长度超过 `max_chars` 时，服务优先按照中英文标点分段。过长且没有
标点的片段会按字符数硬切分。各段可并发推理，响应前会按原文本顺序拼接，
然后统一应用语速、音量和输出编码。
