# CosyVoice3Pro：把官方 CosyVoice3 变成可直接接入业务的语音服务

> 官方 CosyVoice3 提供优秀的语音生成能力，CosyVoice3Pro 解决声纹复用、
> API 接入、音频交付、并发优化和日常运维。

项目地址：
[github.com/QuadraV-Speech/CosyVoice3Pro](https://github.com/QuadraV-Speech/CosyVoice3Pro)

## 为什么要做 CosyVoice3Pro？

CosyVoice3 已经具备多语言、跨语言声音克隆、指令控制和流式推理能力。
但把模型 Demo 接入真实业务时，开发者通常还要处理这些问题：

- 同一个说话人反复合成时，是否每次都要上传参考音频？
- Prompt Speech Tokens、Mel 和 Speaker Embedding 能否提前提取并复用？
- 业务系统是否必须理解 Triton Tensor 和 gRPC 协议？
- 如何管理多个 Speaker，并支持注册、查询、更新和删除？
- 长文本分段、语速、音量、MP3 等音频格式由谁处理？
- 如何提供一个能注册声音、试听和下载的管理页面？
- A100 上的并发参数应该如何配置？

CosyVoice3Pro 的答案很直接：

> **参考音频只注册一次，后续推理只传 `speakerId + text`。**

## Pro 到底增加了什么？

CosyVoice3Pro 没有替换官方模型。它继续使用
`Fun-CosyVoice3-0.5B-2512`、NVIDIA Triton 和 TensorRT-LLM，在其上增加
一层面向应用开发者的生产服务能力。

| 维度 | 官方 CosyVoice3 Triton Runtime | CosyVoice3Pro |
| --- | --- | --- |
| 模型核心 | CosyVoice3 + TensorRT-LLM | 相同官方模型与推理核心 |
| 业务接口 | Triton V2 / gRPC Tensor | REST、表单和音频流 |
| 声纹复用 | 默认携带参考音频；可选进程内缓存 | 持久化 Speaker Registry |
| Speaker 管理 | 无业务级 CRUD | 注册、更新、列表、详情、删除 |
| Prompt | 客户端组织参考文本和指令 | 注册默认画像，请求级临时覆盖 |
| 注册方式 | 客户端准备音频 Tensor | 上传文件或公开音频 URL |
| 音频交付 | 返回模型波形 | 分段、语速、音量和 9 种输出格式 |
| 管理后台 | Triton Runtime 不包含业务后台 | 18000 同端口 Web 工作台 |
| 并发调优 | 手工配置实例与 KV 参数 | 自动选择 `balanced` / `throughput` |
| 高级接口 | Streaming、Metrics | 原样保留 |

一句话概括：

> **CosyVoice3Pro = 官方推理核心 + 可复用声纹 + 开发者 API + 可交付音频 +
> Web 与生产运维。**

## 声纹为什么能够复用？

注册 Speaker 时，服务会一次性提取并保存：

```text
Prompt Speech Tokens
Prompt Mel Features
CAMPPlus Speaker Embedding
Reference Transcript
Default Prompt Persona
```

后续请求通过 `speakerId` 读取这些特征，不再重复上传和解析参考音频。

这不仅减少了请求体积，也让 Speaker 成为一个可以管理、备份和跨请求复用的
业务实体。

## 默认画像与单次 Prompt 覆盖

注册声音时可以同时保存默认 Prompt：

```text
请用成熟、稳重、亲切的语气说话。
```

推理时的规则非常简单：

| 请求方式 | 实际行为 |
| --- | --- |
| 不传 `prompt` | 使用 Speaker 默认画像 |
| `prompt=""` | 使用 Speaker 默认画像 |
| 非空 `prompt` | 只覆盖本次请求 |

因此，同一个 Speaker 可以有稳定的默认风格，也能在某一次任务中临时切换成
“激动”“温柔”或“严肃”等表达方式。

## 一个端口完成全部工作

```text
 Browser / SDK / curl
          │
          │ :18000
          ▼
 ┌─────────────────────────────────────────────┐
 │          CosyVoice3Pro Gateway              │
 │                                             │
 │ Public API                                  │
 │ /health · /register · /speakers · /tts/     │
 │       ▲                                     │
 │       └──────── Web 工作台使用同一组 API     │
 │                                             │
 │ Advanced API                                │
 │ /v2/* ───────────────► Triton HTTP :18100   │
 └──────────────────────────┬──────────────────┘
                            ├── CosyVoice3Pro
                            ├── Speaker Registry
                            └── upstream cosyvoice3

 gRPC :18001 · Metrics :18002
```

业务开发者使用普通 Public API；模型工程和平台运维仍然可以使用 Triton
HTTP、gRPC 和 Prometheus Metrics。

![CosyVoice3Pro Web 工作台](https://raw.githubusercontent.com/QuadraV-Speech/CosyVoice3Pro/master/docs/assets/web-demo.gif)

## 三步跑通

### 1. 安装

环境需要 Linux、Docker、NVIDIA Driver、NVIDIA Container Runtime 和
CUDA GPU。

```bash
git clone https://github.com/QuadraV-Speech/CosyVoice3Pro.git
cd CosyVoice3Pro
COSYVOICE_GPU_ID=0 bash manage.sh install
```

检查服务：

```bash
curl --fail-with-body http://127.0.0.1:18000/health
```

### 2. 注册一个 Speaker

准备一段 3～10 秒、单人、清晰的参考音频，并填写准确文本：

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/register" \
  -F "speakerId=narrator_01" \
  -F "audio=@./reference.wav;type=audio/wav" \
  -F "reference_text=这是参考音频中实际说出的内容。" \
  -F "prompt=请用成熟、稳重、亲切的语气说话。"
```

也可以直接使用公开音频 URL：

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "speakerId": "narrator_01",
    "audio_url": "https://your-domain.example/reference.mp3",
    "reference_text": "这是参考音频中实际说出的内容。",
    "prompt": "请用成熟、稳重、亲切的语气说话。"
  }'
```

### 3. 只用 Speaker ID 合成

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/tts/" \
  -F "text=你好，这是通过已注册声纹生成的语音。" \
  -F "speakerId=narrator_01" \
  -F "speed=balanced" \
  -F "volume=middle" \
  -F "output_format=mp3" \
  --output output.mp3
```

临时覆盖默认画像，只需要多传一个非空 `prompt`：

```bash
-F "prompt=请用充满活力、略带兴奋的语气说话。"
```

## 音频后处理也交给服务端

Public `/tts/` 支持：

- `low`、`balanced`、`fast` 三档语速；
- `small`、`middle`、`large` 三档音量；
- 长文本自动分段与受控并发；
- PCM、MP3、WAV、AAC、M4A、Opus、OGG、FLAC、WebM；
- 内置声音、注册 Speaker 和即时 Prompt Audio 三种声音来源。

调用方拿到的是可以直接播放、下载或交付下游系统的音频，而不是还需要自行
处理的模型中间结果。

## A100 实测：优化不只停留在接口层

项目采用与官方 `client_grpc.py` 一致的系统 RTF 口径：

```text
系统 RTF = 整组测试墙钟时间 / 全部输出音频总时长
```

在同一台 A100-SXM4-80GB、同一模型、同一 Engine、同一请求链路下，仅切换
服务 Profile：

| 配置 | 成功率 | P50 | P95 | 系统 RTF | 音频吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方默认核心参数复现 | 48/48 | 3.67s | 4.41s | 0.0391 | 25.61x |
| CosyVoice3Pro `throughput` | 48/48 | 3.40s | 4.22s | **0.0329** | **30.42x** |

这组受控测试中：

- 系统 RTF 降低 **15.8%**；
- 音频吞吐提升 **18.8%**；
- 48 个请求全部成功。

需要强调：第一行是使用上游默认核心参数的同链路 A100 复现，并不是
FunAudioLLM 发布的官方 A100 数据。

## 官方 L20 基线

官方 CosyVoice3 Triton 文档公开的是单卡 L20 数据：

| 官方模式 | 并发 / Batch | 官方结果 |
| --- | ---: | --- |
| 流式首包 | 并发 4 | Avg 750.42 ms；P50 740.31；P95 977.55；P99 1002.37 |
| 离线流水线 | Batch 1 | RTF 0.1091 |
| 离线流水线 | Batch 2 | RTF 0.0822 |
| 离线流水线 | Batch 4 | RTF 0.0630 |
| 离线流水线 | Batch 8 | RTF 0.0562 |
| 离线流水线 | Batch 16 | RTF 0.0501 |

来源：
[官方 CosyVoice3 Triton Runtime](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/README.Cosyvoice3.md)

L20 的流式首包、离线批处理与本文 A100 端到端 HTTP 并发并不是相同
工作负载，因此不直接比较倍数。

## 适合哪些场景？

- 有声内容、小说和视频旁白；
- 游戏角色、数字人和虚拟主播；
- 客服、通知与营销语音；
- 多租户 Speaker 管理平台；
- 需要私有化部署的语音生成业务；
- 希望保留 Triton 高级能力，又不想让业务方处理 Tensor 协议的团队。

## 使用前需要知道

- CosyVoice3Pro 是社区服务化增强项目，并非 FunAudioLLM 官方发行版；
- Public `/tts/` 当前返回完整音频，官方 decoupled streaming 仍通过高级
  Triton 接口保留；
- 服务默认没有应用层登录认证，公网部署需要在反向代理或负载均衡层增加
  TLS、鉴权、限流和来源限制；
- 性能会随 GPU、文本长度、Speaker 和采样结果变化，应在目标环境复测；
- 模型权重、基础镜像和上游组件仍遵循各自的许可证与使用条款。

## 最后

CosyVoice3 已经解决了“能不能生成高质量语音”的问题。

CosyVoice3Pro 希望继续解决：

> **怎样让声音可以注册、复用、管理、交付，并稳定地接入真实业务。**

如果这个方向对你有帮助，欢迎：

- 给项目一个 Star；
- 提交实际 GPU 的 Benchmark 数据；
- 反馈 API、音质、并发和部署问题；
- 参与完善流式 Public API、鉴权和更多生产能力。

项目地址：
[github.com/QuadraV-Speech/CosyVoice3Pro](https://github.com/QuadraV-Speech/CosyVoice3Pro)

详细文档：

- [对外开发者 API](https://github.com/QuadraV-Speech/CosyVoice3Pro/blob/master/docs/public-api.md)
- [性能基准与复现](https://github.com/QuadraV-Speech/CosyVoice3Pro/blob/master/docs/benchmark.md)
- [部署与运维](https://github.com/QuadraV-Speech/CosyVoice3Pro/blob/master/docs/web-admin.md)

---

推荐标签：`CosyVoice3` `TTS` `声音克隆` `语音合成` `NVIDIA Triton`
`TensorRT-LLM` `开源` `AI`
