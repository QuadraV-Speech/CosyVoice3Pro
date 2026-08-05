# CosyVoice3Pro 内部高级 API

本文档面向平台运维、模型调试和需要直接使用 Triton Tensor 协议的高级
集成方。普通业务开发请使用
[CosyVoice3Pro 对外 API](public-api.md)。

## 1. 边界与地址

内部高级 API：

```text
http://127.0.0.1:18000/v2/
```

Gateway 将外部 `/v2/*` 原样代理至容器内部 Triton `18100`。内部 API
返回 Triton 标准 Tensor JSON；TTS 模型返回 24kHz FP32 waveform，不做
音频编码和后处理。

| 地址 | 用途 |
| --- | --- |
| `/v2/health/ready` | Triton 健康检查 |
| `/v2/models/CosyVoice3Pro/ready` | TTS 模型状态 |
| `/v2/models/CosyVoice3Pro/infer` | 高级 TTS Tensor 推理 |
| `/v2/models/CosyVoice3ProStreaming/ready` | Decoupled 流式模型状态 |
| `/v2/models/CosyVoice3ProSpeakerRegistry/ready` | Registry 模型状态 |
| `/v2/models/CosyVoice3ProSpeakerRegistry/infer` | Registry Tensor 操作 |
| `/admin/api/info` | Gateway 内部信息 |

除非需要 Tensor 级控制，否则不要让业务代码依赖这些路径。

## 2. 健康与模型检查

```bash
curl -f "http://127.0.0.1:18000/v2/health/ready"
curl -f "http://127.0.0.1:18000/v2/models/CosyVoice3Pro/ready"
curl -f "http://127.0.0.1:18000/v2/models/CosyVoice3ProStreaming/ready"
curl -f "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/ready"
```

Gateway 内部信息：

```bash
curl --fail-with-body \
  "http://127.0.0.1:18000/admin/api/info"
```

## 3. Triton Tensor 约定

字符串使用 `BYTES`，常用 Shape 为 `[1,1]`：

```json
{
  "name": "speaker_id",
  "shape": [1, 1],
  "datatype": "BYTES",
  "data": ["narrator_01"]
}
```

16kHz 单声道参考音频使用：

- `reference_wav`：`FP32 [1,N]`
- `reference_wav_len`：`INT32 [1,1]`

Speaker ID 规则与对外 API 相同：长度 1～128，只允许字母、数字、
下划线、中划线和点，且不能包含 `..`。

## 4. Speaker Registry

模型地址：

```text
POST /v2/models/CosyVoice3ProSpeakerRegistry/infer
```

`operation` 支持：

| 操作 | 说明 |
| --- | --- |
| `register` | 注册或原子更新声纹 |
| `inspect` | 查询一个声纹 |
| `list` | 列出全部声纹 |
| `delete` | 删除声纹 |

### 4.1 注册输入

| 名称 | 类型 | Shape | 必填 | 说明 |
| --- | --- | --- | --- | --- |
| `operation` | BYTES | `[1,1]` | 是 | 固定为 `register` |
| `speaker_id` | BYTES | `[1,1]` | 是 | Speaker ID |
| `reference_wav` | FP32 | `[1,N]` | 是 | 16kHz 单声道音频 |
| `reference_wav_len` | INT32 | `[1,1]` | 是 | 有效采样点数 |
| `reference_text` | BYTES | `[1,1]` | 是 | 音频实际文本 |
| `prompt` | BYTES | `[1,1]` | 否 | 注册默认画像 |

参考音频时长为 0.5～30 秒。推荐使用客户端生成 Tensor JSON：

```bash
python scripts/client.py build-register-json \
  --speaker-id narrator_01 \
  --audio ./reference.wav \
  --reference-text "这是参考音频中实际说出的内容。" \
  --prompt "请用成熟、稳重的语气说话。" \
  > /tmp/register.json

curl --fail-with-body \
  -X POST \
  "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/infer" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/register.json
```

注册会一次性提取并保存：

- Prompt speech tokens
- Prompt Mel 特征
- CAMPPlus speaker embedding
- 参考文本和默认 Prompt 画像

后续 `speaker_id` 推理不会重复上传或提取参考音频。

### 4.2 查询

```bash
curl --fail-with-body \
  -X POST \
  "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "operation",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["inspect"]
      },
      {
        "name": "speaker_id",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["narrator_01"]
      }
    ]
  }'
```

### 4.3 列表

```bash
curl --fail-with-body \
  -X POST \
  "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "operation",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["list"]
      }
    ]
  }'
```

### 4.4 删除

```bash
curl --fail-with-body \
  -X POST \
  "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "operation",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["delete"]
      },
      {
        "name": "speaker_id",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["narrator_01"]
      }
    ]
  }'
```

Registry 返回标准 Triton `outputs`，其中：

- `status`：`ok` 或 `not_found`
- `message`：JSON 字符串
- `speaker_version`：声纹版本

## 5. 使用 Speaker ID 高级推理

模型地址：

```text
POST /v2/models/CosyVoice3Pro/infer
```

输入：

| 名称 | 类型 | Shape | 必填 | 说明 |
| --- | --- | --- | --- | --- |
| `speaker_id` | BYTES | `[1,1]` | 是 | 已注册 Speaker ID |
| `prompt` | BYTES | `[1,1]` | 否 | 本次画像覆盖，空值使用注册默认画像 |
| `target_text` | BYTES | `[1,1]` | 是 | 需要合成的文本 |

```bash
curl --fail-with-body \
  -X POST \
  "http://127.0.0.1:18000/v2/models/CosyVoice3Pro/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "speaker_id",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["narrator_01"]
      },
      {
        "name": "prompt",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [""]
      },
      {
        "name": "target_text",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["你好，这是高级 Tensor 推理测试。"]
      }
    ]
  }' \
  --output triton-response.json
```

`prompt` 为空或省略时使用注册默认画像；非空时仅覆盖本次请求。

## 6. 原始 Prompt 高级推理

未注册声纹时可以直接传参考音频：

| 名称 | 类型 | Shape | 必填 |
| --- | --- | --- | --- |
| `reference_wav` | FP32 | `[1,N]` | 是 |
| `reference_wav_len` | INT32 | `[1,1]` | 是 |
| `reference_text` | BYTES | `[1,1]` | 是 |
| `prompt` | BYTES | `[1,1]` | 否 |
| `target_text` | BYTES | `[1,1]` | 是 |

`reference_text` 必须准确对应参考音频。`prompt` 可以为空，也可以指定本次
画像。业务调用建议改用对外 `/tts/` 的 `prompt_audio`，无需手工构造
FP32 Tensor。

## 7. 指令字段兼容

CosyVoice3Pro 内部支持：

- `prompt`：推荐字段
- `instruct_text`：兼容别名

当两个字段同时出现且内容不同，请求会失败。Prompt 最长 512 字。

注册默认画像会与参考文本组成内部格式：

```text
You are a helpful assistant. <画像><|endofprompt|><参考音频文本>
```

业务调用方不需要自行拼接该格式。

## 8. waveform 输出

`CosyVoice3Pro` 的 Triton 输出：

- 名称：`waveform`
- 类型：FP32
- 采样率：24kHz
- 声道：单声道

保存 WAV：

```python
import json

import numpy as np
import soundfile as sf

with open("triton-response.json", encoding="utf-8") as file_obj:
    response = json.load(file_obj)

waveform = next(
    output["data"]
    for output in response["outputs"]
    if output["name"] == "waveform"
)
sf.write(
    "output.wav",
    np.asarray(waveform, dtype=np.float32),
    24000,
    subtype="PCM_16",
)
```

需要 MP3、M4A 或长文本处理时，应使用对外 `/tts/`；需要带语速、音量
后处理的在线分块时，使用对外 `/tts/stream`。

### 8.1 Decoupled 流式模型

`CosyVoice3ProStreaming` 与离线模型共用同一份 BLS 实现，但配置了
`model_transaction_policy { decoupled: true }`。它通过 Triton 双向 gRPC
流持续返回多份 `waveform`，不能使用普通 `/v2/.../infer` HTTP 请求消费。

业务客户端不需要直接构造 Triton gRPC Tensor。Gateway 已将它封装为
`POST /tts/stream` SSE，并负责断连取消、PCM 编码、重采样、语速和音量。
直连 gRPC 适合模型调试，输入 Tensor 与本章离线模型完全一致，输出为多段
24kHz FP32 `waveform`。

Gateway 生命周期内复用一个异步 gRPC Channel，但每个请求保持独立的
Decoupled response iterator。Flow 与 Vocoder 子请求使用同一 `request_id`，
并按流式分块序号设置 Triton 优先级：首段优先于后续段，避免高并发时新请求
的首包被旧请求尾段长期阻塞。Vocoder 仍保持 Batch 1；实际 A/B 表明单卡上
盲目增加实例或强制动态 Batch 会增加上下文竞争。

官方基准同样通过 Triton gRPC 测量从提交请求到第一段非空 waveform 的时间。
本仓库的 `scripts/benchmark_streaming.py` 可分别压测直连 gRPC 和 Public SSE，
后者还会统计 Gateway 的 `queueMs`。

## 9. Speaker 缓存与持久化

Registry 默认将声纹保存到：

```text
/workspace/cosyvoice_speaker_store
```

生产环境通过 `COSYVOICE_SPEAKER_STORE_DIR` 挂载宿主机目录。单个 Speaker
使用原子 `.npz` 文件保存，`CosyVoice3Pro` 进程内维护按
`speaker_id + speaker_version` 标识的 LRU 缓存。

更新同一 Speaker 后版本变化，旧缓存不会被继续使用。删除 Speaker 后新的
推理请求会返回不存在。

备份：

```bash
bash manage.sh backup
```

## 10. 内部错误与回滚

Triton 错误使用标准格式：

```json
{
  "error": "具体错误信息"
}
```

常见原因：

- Tensor 名称、类型或 Shape 错误
- Speaker 不存在
- Prompt 字段冲突
- 参考音频时长或采样非法
- GPU、TensorRT-LLM 或下游模型不可用

仓库仍保留上游 `cosyvoice3` 模型和原始 Prompt 调用方式。需要临时关闭
Gateway、让 Triton 直接监听外部 `18000` 时：

```bash
COSYVOICE_WEB_GATEWAY_ENABLED=false \
  bash manage.sh restart
```

此模式下对外 `/health`、`/register`、`/speakers`、`/tts/`、
`/tts/stream` 和 Web 页面均不可用，仅保留 Triton `/v2/*`。
