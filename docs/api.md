# CosyVoice3Pro Triton Speaker Registry 接口文档

## 1. 概述

Triton HTTP 地址：

```text
http://127.0.0.1:18000
```

同一端口的 Web 管理后台：

```text
http://127.0.0.1:18000/
```

浏览器后台说明见 [CosyVoice3Pro 同端口 Web 管理后台](web-admin.md)。

旧版内置说话人 `POST /tts/` 兼容接口继续保留，完整参数和可直接运行的
curl 见 [`/tts/` 兼容接口文档](tts-api.md)。

本次升级新增两个能力：

1. `CosyVoice3ProSpeakerRegistry`：注册、查询、列出和删除说话人。
2. `CosyVoice3Pro`：支持传
   `speaker_id + target_text + 可选 prompt` 完成推理。

原来的 `cosyvoice3` 模型和
`reference_wav + reference_text + target_text` 调用方式继续保留。

注册时会一次性提取并保存：

- Prompt speech tokens
- Prompt Mel 特征
- CAMPPlus speaker embedding
- 默认 `prompt` 画像和提示音频文本

后续推理不再上传提示音频，也不会重复执行 Prompt 特征提取。

以当前 3.48 秒示例音频和 Triton JSON 协议计算，原始 Prompt 请求约为
`911501` bytes，`speaker_id` 请求约为 `169` bytes，请求体减少约
`99.98%`。

## 2. 服务检查

### 2.1 Triton 健康检查

```bash
curl -f http://127.0.0.1:18000/v2/health/ready
```

HTTP `200` 表示 Triton Ready。

### 2.2 模型检查

```bash
curl -f http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/ready
curl -f http://127.0.0.1:18000/v2/models/CosyVoice3Pro/ready
```

## 3. Speaker ID 规则

`speaker_id` 必须满足：

- 长度为 1～128
- 只允许字母、数字、下划线、中划线和点
- 不允许包含 `..`

合法示例：

```text
common_speaker_1
user-1001.voice-a
```

## 4. 注册或更新说话人

Triton 模型地址：

```text
POST /v2/models/CosyVoice3ProSpeakerRegistry/infer
```

注册操作的输入张量：

| 名称 | 类型 | Shape | 必填 | 说明 |
| --- | --- | --- | --- | --- |
| `operation` | BYTES | `[1,1]` | 是 | 固定为 `register` |
| `speaker_id` | BYTES | `[1,1]` | 是 | 说话人 ID |
| `reference_wav` | FP32 | `[1,N]` | 是 | 16kHz 单声道浮点音频 |
| `reference_wav_len` | INT32 | `[1,1]` | 是 | 有效采样点数 |
| `reference_text` | BYTES | `[1,1]` | 是 | 提示音频实际说出的文本 |
| `prompt` | BYTES | `[1,1]` | 否 | Speaker 默认画像；默认空字符串 |

音频约束：

- 采样率：16kHz
- 声道数：单声道
- 最短：0.5 秒
- 最长：30 秒
- 建议使用 3～10 秒清晰人声

推荐使用附带的客户端。客户端会通过 FFmpeg 自动将 WAV、MP3、M4A 等音频转换成 16kHz 单声道：

```bash
python \
  scripts/client.py \
  register \
  --speaker-id common_speaker_1 \
  --audio /path/to/reference.wav \
  --reference-text "希望你以后能够做的比我还好呦。" \
  --prompt "请用成熟、稳重、亲切的语气说话。"
```

使用 `curl` 注册时，需要先把音频转换成 Triton FP32 Tensor JSON。下面的命令可以直接执行：

```bash
python \
  scripts/client.py \
  build-register-json \
  --speaker-id common_speaker_1 \
  --audio /path/to/reference.wav \
  --reference-text "希望你以后能够做的比我还好呦。" \
  --prompt "请用成熟、稳重、亲切的语气说话。" \
  > /tmp/cosyvoice3pro-register.json

curl -sS \
  -X POST "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/infer" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/cosyvoice3pro-register.json
```

这里不能使用 `curl -F` 直接上传文件，因为 `18000` 暴露的是 Triton
Inference HTTP 协议，不是 multipart 文件上传接口。

客户端 `register` 命令解析后的成功响应示例：

```json
{
  "status": "ok",
  "speaker_version": "示例版本号",
  "message": {
    "format_version": 2,
    "speaker_id": "common_speaker_1",
    "speaker_version": "示例版本号",
    "reference_text": "You are a helpful assistant. 请用成熟、稳重、亲切的语气说话。<|endofprompt|>希望你以后能够做的比我还好呦。",
    "reference_transcript": "希望你以后能够做的比我还好呦。",
    "prompt": "请用成熟、稳重、亲切的语气说话。",
    "sample_rate": 16000,
    "samples": 55680,
    "duration_seconds": 3.48,
    "registered_at": 1785225275
  }
}
```

再次使用同一个 `speaker_id` 注册会原子替换旧版本。`speaker_version`
由 ID、默认画像、提示文本和有效音频内容生成。

直接使用 `curl` 时，外层响应是标准 Triton `outputs` 格式，其中
`message.data[0]` 是 JSON 字符串。

## 5. 使用 Speaker ID 推理

Triton 模型地址：

```text
POST /v2/models/CosyVoice3Pro/infer
```

输入张量：

| 名称 | 类型 | Shape | 必填 | 说明 |
| --- | --- | --- | --- | --- |
| `speaker_id` | BYTES | `[1,1]` | 是 | 已注册的说话人 ID |
| `prompt` | BYTES | `[1,1]` | 否 | 本次推理的画像覆盖；默认空字符串 |
| `target_text` | BYTES | `[1,1]` | 是 | 需要合成的文本 |

`prompt` 可以完全省略，也可以显式传 `""`。两者效果相同：
使用注册 Speaker 的默认画像。只有非空 `prompt` 才会覆盖默认画像，
且只对本次请求生效。

直接调用 Triton：

```bash
curl -sS \
  -X POST "http://127.0.0.1:18000/v2/models/CosyVoice3Pro/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "speaker_id",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["common_speaker_1"]
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
        "data": ["你好，这是注册说话人的语音合成测试。"]
      }
    ]
  }' \
  -o triton_response.json
```

Triton 返回 `waveform` FP32 数组，采样率为 24kHz。将上述 curl
响应转换成 WAV：

```bash
python - <<'PY'
import json
import numpy as np
import soundfile as sf

with open("triton_response.json", encoding="utf-8") as file_obj:
    response = json.load(file_obj)

waveform = next(
    output["data"]
    for output in response["outputs"]
    if output["name"] == "waveform"
)
sf.write(
    "cosyvoice3pro-output.wav",
    np.asarray(waveform, dtype=np.float32),
    24000,
    subtype="PCM_16",
)
PY
```

也可以使用客户端直接保存 WAV：

```bash
python \
  scripts/client.py \
  infer \
  --speaker-id common_speaker_1 \
  --text "你好，这是注册说话人的语音合成测试。" \
  --output output.wav
```

客户端输出示例：

```json
{
  "status": "ok",
  "output": "output.wav",
  "sample_rate": 24000,
  "samples": 98880,
  "duration_seconds": 4.12,
  "speaker_id": "common_speaker_1"
}
```

## 6. 查询说话人

```bash
curl -sS \
  -X POST "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/infer" \
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
        "data": ["common_speaker_1"]
      }
    ]
  }'
```

也可以使用客户端：

```bash
python \
  scripts/client.py \
  inspect \
  --speaker-id common_speaker_1
```

不存在时返回：

```json
{
  "status": "not_found",
  "speaker_version": "",
  "message": {
    "speaker_id": "unknown_speaker",
    "exists": false
  }
}
```

## 7. 列出说话人

```bash
curl -sS \
  -X POST "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/infer" \
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

客户端：

```bash
python \
  scripts/client.py list
```

## 8. 删除说话人

```bash
curl -sS \
  -X POST "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/infer" \
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
        "data": ["speaker_to_delete"]
      }
    ]
  }'
```

客户端：

```bash
python \
  scripts/client.py \
  delete \
  --speaker-id speaker_to_delete
```

删除后，即使某个 BLS 实例曾经缓存过该 Speaker，也不会继续使用旧缓存。

## 9. 原始 Prompt 兼容接口

原始接口仍然可用，输入为：

| 名称 | 类型 | Shape |
| --- | --- | --- |
| `reference_wav` | FP32 | `[1,N]` |
| `reference_wav_len` | INT32 | `[1,1]` |
| `reference_text` | BYTES | `[1,1]` |
| `prompt` | BYTES | `[1,1]`，可选，默认空 |
| `target_text` | BYTES | `[1,1]` |

兼容性测试命令：

```bash
python \
  scripts/client.py \
  infer-raw \
  --audio /path/to/reference.wav \
  --reference-text "希望你以后能够做的比我还好呦。" \
  --prompt "" \
  --text "这是原始提示音频兼容性测试。" \
  --output raw-prompt-output.wav
```

`CosyVoice3Pro` 的原始 Prompt 兼容路径已改为使用
“有效音频内容 + reference_text”的 SHA-256，不再只按提示文本缓存。
原始 `cosyvoice3` 模型保持升级前行为。

## 10. 指令控制

### 10.1 默认画像和单次覆盖规则

注册接口和推理接口都支持可选 `prompt`，但作用不同：

- 注册 `prompt`：保存为 Speaker 的默认画像。
- 推理 `prompt` 为空或省略：使用注册时的默认画像。
- 推理 `prompt` 非空：覆盖默认画像，只对当前请求生效。
- 覆盖不会修改注册文件，也不会重新提取声纹、Token、Mel 或
  Speaker Embedding。

例如注册时设置：

```text
请用成熟、稳重、亲切的语气说话。
```

后续推理不传 `prompt` 时自动使用该画像；某次请求传：

```text
请非常开心、兴奋地说话。
```

则该次请求使用开心画像，下一次空 `prompt` 仍回到成熟稳重的默认画像。

| 推理请求 | 实际画像 |
| --- | --- |
| 不传 `prompt` | 注册时的默认画像 |
| `"prompt": ""` | 注册时的默认画像 |
| `"prompt": "请非常开心地说话。"` | 仅本次覆盖为开心画像 |

`prompt` 最大长度为 512 个字符。普通自然语言即可，服务会自动补齐
CosyVoice3 所需的系统前缀和 `<|endofprompt|>` 标记。

### 10.2 注册默认画像

推荐直接使用客户端注册：

```bash
python \
  scripts/client.py \
  register \
  --speaker-id common_speaker_1 \
  --audio /path/to/reference.wav \
  --reference-text "希望你以后能够做的比我还好呦。" \
  --prompt "请用成熟、稳重、亲切的语气说话。"
```

需要纯 curl 时，先生成包含音频 FP32 数组的 JSON：

```bash
python \
  scripts/client.py \
  build-register-json \
  --speaker-id common_speaker_1 \
  --audio /path/to/reference.wav \
  --reference-text "希望你以后能够做的比我还好呦。" \
  --prompt "请用成熟、稳重、亲切的语气说话。" \
  > /tmp/cosyvoice3pro-register.json

curl -sS \
  -X POST "http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/infer" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/cosyvoice3pro-register.json
```

### 10.3 使用注册默认画像 curl

`prompt` 显式传空字符串，也可以删除整个 `prompt` 输入项：

```bash
curl -sS \
  -X POST "http://127.0.0.1:18000/v2/models/CosyVoice3Pro/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "speaker_id",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["common_speaker_1"]
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
        "data": ["你好，这是使用注册默认画像的语音。"]
      }
    ]
  }' \
  -o default-prompt-response.json
```

### 10.4 非空 prompt 覆盖默认画像 curl

```bash
curl -sS \
  -X POST "http://127.0.0.1:18000/v2/models/CosyVoice3Pro/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "speaker_id",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["common_speaker_1"]
      },
      {
        "name": "prompt",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["请非常开心、兴奋地说话。"]
      },
      {
        "name": "target_text",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["太好了，我们今天终于完成了这个项目！"]
      }
    ]
  }' \
  -o override-prompt-response.json
```

### 10.5 客户端覆盖命令

```bash
python \
  scripts/client.py \
  infer \
  --speaker-id common_speaker_1 \
  --prompt "请用温柔、舒缓的语气说话。" \
  --text "夜深了，早点休息吧。" \
  --output gentle-output.wav
```

不传 `--prompt` 时，客户端默认传空字符串。

### 10.6 字段兼容性

旧推理字段 `instruct_text` 仍可使用，语义等同于 `prompt`，用于兼容
已经接入的客户端。新接口统一推荐使用 `prompt`。单次请求不能同时传
非空 `prompt` 和非空 `instruct_text`。

旧的注册方式也继续兼容：如果 `reference_text` 已经包含完整的
`You are a helpful assistant. ...<|endofprompt|>`，且没有单独传
`prompt`，服务会从其中识别默认画像。新调用建议把提示音频原文放在
`reference_text`，把画像单独放在 `prompt`。

## 11. 错误处理

Triton 模型执行错误通常返回 HTTP `500`：

```json
{
  "error": "speaker_id is not registered: unknown_speaker"
}
```

常见错误：

- `speaker_id is not registered`：Speaker 尚未注册或已删除。
- `reference_wav must be at least 0.5 seconds`：提示音频过短。
- `reference_wav must not exceed 30 seconds`：提示音频过长。
- `speaker_id must be ...`：Speaker ID 包含非法字符。
- `reference_text must not be empty`：注册时没有提供提示音频文本。
- `prompt must not exceed 512 characters`：注册或单次推理画像过长。
- `prompt must not contain text after <|endofprompt|>`：完整格式的
  指令在结束标记后仍带有文本。
- `provide only one of prompt or instruct_text`：推理请求同时传了两个
  非空的画像字段。

## 12. 缓存与持久化

当前模型使用：

```text
/workspace/cosyvoice_speaker_store
```

作为所有 10 个 `CosyVoice3Pro` BLS 实例共享的注册目录。每个实例维护最多 64 个 GPU-ready Speaker LRU 缓存，并根据注册文件的 inode、修改时间和大小自动失效。

当前已经运行的容器没有宿主机挂载，因此：

- 重启 Triton：数据保留
- 重启 Docker 容器：数据保留
- 删除并重建容器：数据丢失

`manage.sh install` 会将宿主机 Speaker 目录挂载到：

```text
data/speakers
```

新建容器后，删除或重建容器也不会丢失注册数据。可以通过环境变量指定其他宿主机目录：

```bash
COSYVOICE_SPEAKER_STORE_DIR=/data/cosyvoice-speakers \
  bash manage.sh install
```

## 13. 回滚

本次升级前的模型仓库备份位于容器内：

```text
/workspace/CosyVoice/runtime/triton_trtllm/model_repo_cosyvoice3_copy.backup-speaker-v1
```

回滚时应先停止 Triton，再用该目录恢复 `model_repo_cosyvoice3_copy`，最后重新启动服务。
