# CosyVoice3Pro 性能基准

本页记录完整音频 Public API `/tts/`、在线 Public SSE `/tts/stream` 和直连
Triton gRPC 的实测结果。每张表会明确计时边界，不把模型内部指标与端到端
结果混在一起。

## 测试环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB |
| NVIDIA Driver | 550.127.08 |
| Triton 镜像 | `soar97/triton-cosyvoice:25.06` |
| 上游 CosyVoice commit | `074ca6d` |
| Gateway | 离线基准 `1.6.1`；流式生产基准 `1.9.0` |
| 模式 | 官方 raw prompt 与生产 registered Speaker 分开统计 |
| Profile | `streaming`：Pro BLS 2、Streaming BLS 2、token2wav 2、vocoder 4 |
| 测试日期 | 离线 2026-07-29；流式 2026-08-05 |

流式关键结果的机器可读快照见
[`benchmark-streaming-a100-2026-08-05.json`](benchmark-streaming-a100-2026-08-05.json)。

测试文本：

```text
你好，欢迎使用 CosyVoice3Pro。注册一次声纹，后续请求只需要传入说话人编号和需要合成的文本。
```

## 与官方一致的统计口径

主指标采用上游
[`client_grpc.py`](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/client_grpc.py#L792-L807)
的聚合算法：

- `系统 RTF = 整组测试墙钟时间 / 全部输出音频总时长`，越低越好。
- `音频吞吐 = 1 / 系统 RTF`，表示每秒墙钟时间生成多少秒音频。
- Average、P50、P90、P95、P99 是从发出 HTTP 请求到完整 WAV
  响应接收完成的端到端延迟。

JSON 结果中的 `request_rtf_average` 是逐请求
`请求延迟 / 该请求音频时长` 的平均值；`rtf_average` 是为旧报告消费者保留
的同义字段。它们会随并发排队上升，不能代替官方口径的系统 RTF，因此不再放入
主结果表。

## 性能 Profile

`COSYVOICE_PERFORMANCE_PROFILE=auto` 会读取容器可见 GPU 显存。显存不小于
70000 MiB 时使用实测的 `streaming`，否则使用 `balanced`。离线请求为主时
显式选择 `throughput`：

| Profile | LLM KV | Pro BLS | Streaming BLS | Legacy BLS | token2wav | vocoder | `/tts/` 并发 | SSE 并发 | 单请求分段 | CUDA 预热 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced` | 0.60 | 10 | 2 | 2 | 1 | 1 | 10 | 4 | 2 | 否 |
| `throughput` | 0.50 | 12 | 2 | 2 | 2 | 2 | 12 | 10 | 2 | 是 |
| `streaming` | 0.50 | 2 | 2 | 1 | 2 | 4 | 4 | 16 | 2 | 是 |

双 `token2wav` 但单 `vocoder` 只会把队列从声学模型转移到声码器，因此
`throughput` Profile 同时扩展两个阶段。模型生成步数、采样参数和音频后处理
保持不变。

流式并发上限和模型实例数刻意解耦。Decoupled BLS 在等待 LLM、Flow 和
Vocoder 子请求时仍能承载多个流；盲目增加 BLS/Flow 会因 GPU 上下文竞争
降低吞吐。受控 A/B 最终选择 2 个 Streaming BLS、2 个 Flow、4 个 Vocoder，
并释放离线 BLS 占用。若使用独占 GPU，可通过环境变量重新执行 A/B。

### 声学动态 Batch（实验）

CosyVoice3Pro 已实现离线 Flow 与 Vocoder 的 Triton 动态组批：BLS 先按固定
Token/Mel bucket 补齐请求，同时传递真实长度；后端组批推理后再按真实长度
拆分输出。Flow 还会把业务 Batch `B` 展开为 CFG Batch `2B`，使用基于官方
选择性混合精度 ONNX 构建的动态 TensorRT engine。流式请求自动走 Batch 1
兼容路径。

这项能力默认关闭。A100-SXM4-80GB 的受控验证中，Flow Batch 2/4 的动态
engine 执行上下文和单请求耗时都明显高于两个静态 Batch 1 engine；真实 HTTP
请求又会因 LLM 完成时间错开而难以稳定组批。强行开启会增加显存并降低端到端
吞吐。官方 L20 的“离线流水线 Batch”与这里的独立 HTTP 请求动态组批不是同一
工作负载，不能直接套用。

需要在自己的固定流量上实验时可显式开启：

```bash
COSYVOICE_PERFORMANCE_PROFILE=throughput \
COSYVOICE_FLOW_BATCH_SIZE=2 \
COSYVOICE_FLOW_BATCH_QUEUE_DELAY_US=5000 \
COSYVOICE_VOCODER_BATCH_SIZE=4 \
COSYVOICE_VOCODER_BATCH_QUEUE_DELAY_US=2000 \
  bash manage.sh restart
```

首次启用 Flow Batch 会在模型目录生成对应 GPU 的动态 TensorRT engine；后续
重启直接复用。建议同时对照 `/v2/models/token2wav/stats` 和
`/v2/models/vocoder/stats` 中的 `batch_stats`、队列时间、端到端系统 RTF 与
显存，确认 Batch 2/4 确实发生且带来正收益。恢复安全默认值：

```bash
COSYVOICE_FLOW_BATCH_SIZE=1 \
COSYVOICE_VOCODER_BATCH_SIZE=1 \
  bash manage.sh restart
```

## A100 受控 A/B

这是“上游默认核心参数”和 CosyVoice3Pro `throughput` Profile 在同一台
A100 上的复测。它用于隔离本项目配置优化带来的变化，不冒充 FunAudioLLM
发布的 A100 结果。

| 变量 | 两组固定值 |
| --- | --- |
| GPU / Driver | A100-SXM4-80GB / 550.127.08 |
| 镜像 / 上游代码 | `25.06` / `074ca6d` |
| 模型 | `Fun-CosyVoice3-0.5B-2512`，同一份 TensorRT engine |
| API / 推理模式 | `/tts/`，完整响应、非流式 |
| Speaker / Prompt | `common_speaker_1` / 空 |
| 文本 | 本页顶部固定测试文本 |
| 后处理 | `balanced` 语速、`middle` 音量、WAV、16 kHz、单声道 |
| 请求 | 12 并发、48 个请求、正式测量前 12 个预热请求 |

| 变化项 | 上游默认核心参数复现 | `throughput` |
| --- | ---: | ---: |
| LLM KV fraction | 0.40 | 0.50 |
| Pro BLS | 10 | 12 |
| token2wav / vocoder | 1 / 1 | 2 / 2 |
| Gateway 全局推理并发 | 10 | 12 |
| CUDA 上下文预热 | 否 | 是 |

| 配置 | 成功/请求 | 系统 RTF | Average | P50 | P90 | P95 | P99 | 音频吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 上游默认核心参数复现 | 48/48 | 0.0391 | 3.64s | 3.67s | 4.03s | 4.41s | 5.63s | 25.61x |
| `throughput` | 48/48 | **0.0329** | **3.36s** | **3.40s** | **3.80s** | **4.22s** | **4.44s** | **30.42x** |

在这组受控测试中，系统 RTF 降低 15.8%，音频吞吐提升 18.8%，P50
降低 7.2%，P95 降低 4.2%。语音 Token 采样具有随机性，输出音频时长
仍会小幅波动；系统 RTF 已按实际音频总时长归一化，单次测量仍应结合多轮
结果判断。

## 高并发压力测试

每组 48 个请求：

| 并发任务 | 成功/请求 | 系统 RTF | Average | P50 | P90 | P95 | P99 | 音频吞吐 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 48/48 | **0.0329** | 3.36s | 3.40s | 3.80s | 4.22s | 4.44s | 30.42x |
| 16 | 48/48 | **0.0322** | 4.29s | 4.41s | 5.17s | 5.42s | 5.88s | 31.06x |
| 24 | 48/48 | **0.0331** | 6.15s | 6.72s | 7.84s | 8.18s | 9.47s | 30.19x |

24 并发时吞吐已经接近平台，延迟会继续上升。在线低延迟业务建议客户端
并发控制在 12～16；批量离线任务可以提高到 24。

长文本公平性回归同时提交一个 6 分段请求和 8 个短请求。短请求 8/8
成功，P50 1.48 秒、最大 1.96 秒；长请求 5.49 秒完成。单个长请求不会
再占满所有全局推理槽。

这是指定软硬件环境的一次端到端测量，不代表所有 GPU、文本和声音都能得到
相同结果。

## 流式 gRPC / SSE 高并发

流式 TTFA（Time To First Audio）采用官方 `client_grpc.py` 的计时边界：在
提交 Triton `stream_infer` 前开始计时，收到第一段非空 `waveform` 时停止。
`scripts/benchmark_official_streaming.py` 进一步复现官方客户端拓扑、数据与输入
padding；`scripts/benchmark_streaming.py` 测量开发者实际使用的 registered
Speaker 和 Public SSE。

### 官方同数据、同客户端口径

使用 `yuekai/seed_tts_cosy2` 的 `wenetspeech4tts` 26 条数据；每个并发任务
持有一条同步 gRPC stream，连续处理一个数据分片；每次请求携带原始 16 kHz
参考音频和文本，并使用官方 10 秒 padding。A100 测试期间同卡保留约
15.2 GiB 的既有服务，因此不是独占 A100 峰值。

| 环境 | 并发 | Prompt 状态 | TTFA Avg | P50 | P95 | P99 | 系统 RTF | 音频吞吐 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方单 L20 | 4 | 官方报告 | 750.42 ms | 740.31 ms | 977.55 ms | 1002.37 ms | 未发布 | 未发布 |
| Pro A100 | 4 | raw prompt 冷/混合缓存 | 944.71 ms | 685.28 ms | 2482.50 ms | 2576.83 ms | 0.1139 | 8.78x |
| Pro A100 | 4 | raw prompt 稳态缓存 | **622.25 ms** | **627.55 ms** | **906.20 ms** | **926.93 ms** | 0.1030 | 9.71x |

首轮尾延迟来自参考音频 tokenizer/embedding 排队，证明生产请求应优先注册
Speaker，而不是反复上传音频。A100 与 L20 硬件不同；稳态数值只说明当前
部署达到了官方量级，不能据此宣称纯软件相对官方提升。

### Registered Speaker 同机 A/B

两组使用相同 A100、官方 26 条 target text、`common_speaker_1` 和相同模型
采样参数。旧配置为 Pro/Streaming/Legacy BLS `12/2/2`、Flow/Vocoder
`2/2`；最终 `streaming` 配置为 `2/2/1`、`2/4`。并发 16 的最终值取 3 次
重复测量中位数，其余为单轮结果。

| gRPC 并发 | 成功 | TTFA Avg（旧 → 新） | TTFA P95（旧 → 新） | 音频吞吐（旧 → 新） |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 26/26 | 478.77 → 479.46 ms | 630.50 → 687.40 ms | 12.13x → 11.86x |
| 8 | 26/26 | 861.47 → **727.16 ms** | 1233.93 → **1025.13 ms** | 13.73x → 13.69x |
| 16 | 26/26 | 1830.47 → **1533.39 ms** | 3050.10 → **2189.18 ms** | 15.01x → **15.23x** |
| 26 | 26/26 | 2998.25 → **2308.09 ms** | 4961.77 → **3671.97 ms** | 15.80x → **16.07x** |

并发 16 的 TTFA 平均降低 16.2%、P95 降低 28.2%；并发 26 分别降低
23.0% 和 26.0%。低并发没有收益，因此此配置定位为流式高并发生产 Profile。
Triton 统计显示旧配置的主要瓶颈是 Vocoder 队列；扩到 4 个后瓶颈转移到
Flow。继续增加第 3 个 Flow 会因 GPU 上下文竞争反而恶化。首块从 15 改成
12 token 也会让请求更早同步争抢 Flow，未采用。

### Public SSE 生产验收

SSE 包含 Gateway 限流、FFmpeg 语速/音量处理、24kHz→16kHz 重采样和 Base64
传输。最终代码在 Triton 首块到达后才启动 FFmpeg，并提供 15 秒排队超时：

| SSE 并发 | 请求 | 成功 | TTFA Avg | TTFA P95 | Queue P95 | 系统 RTF | 音频吞吐 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 32/32 | 872.38 ms | 1100.75 ms | 3.44 ms | 0.0644 | 15.53x |
| 16 | 100 | **100/100** | 1881.25 ms | **2281.99 ms** | 5.41 ms | **0.0586** | **17.05x** |

断连回归同时发起 24 路请求并在 0.5 秒关闭客户端：3 秒后 FFmpeg 进程数
保持 `0 → 0`；取消会继续传播到 Triton BLS，在下一个 Flow/Vocoder 阶段边界
停止 GPU 工作。随后正常请求成功，服务健康状态为 Ready。

## 官方发布基线与 A100 边界

截至上游 commit `074ca6d`，FunAudioLLM 的
[CosyVoice3 Triton 文档](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/README.Cosyvoice3.md#benchmark-with-client-server-mode)
只发布了**单卡 L20** 结果，没有发布 A100 结果：

| 官方硬件 / 模式 | 并发或 Batch | 官方结果 |
| --- | ---: | --- |
| L20，流式首包 | 并发 4 | Average 750.42 ms；P50 740.31 ms；P90 941.05 ms；P95 977.55 ms；P99 1002.37 ms |
| L20，离线流水线 | Batch 1 | RTF 0.1091 |
| L20，离线流水线 | Batch 2 | RTF 0.0822 |
| L20，离线流水线 | Batch 4 | RTF 0.0630 |
| L20，离线流水线 | Batch 8 | RTF 0.0562 |
| L20，离线流水线 | Batch 16 | RTF 0.0501 |

本页的“上游默认核心参数复现”就是补充的 A100 同机基线：它使用上游
[`run_cosyvoice3.sh`](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/run_cosyvoice3.sh)
中的 KV 0.4、BLS 10、单 token2wav/单 vocoder 核心配置，但保留
CosyVoice3Pro Public API、Speaker Registry 和完整 WAV 响应，以便与优化组
严格控制接口和业务链路。官方 L20 的流式首包、离线批处理与本项目 A100
端到端 HTTP 并发不是同一工作负载，只能作为上游参考，不能据此宣称硬件或
软件倍数提升。

## 复现

服务启动并存在 `common_speaker_1` 后执行：

```bash
python3 scripts/benchmark.py \
  --url http://127.0.0.1:18000 \
  --speaker-id common_speaker_1 \
  --concurrency 12 16 24 \
  --requests 48 \
  --warmup 12
```

自定义测试：

```bash
python3 scripts/benchmark.py \
  --speaker-id your_speaker_id \
  --text "需要测试的文本" \
  --concurrency 1 2 4 8 \
  --requests 20 \
  --format json
```

Benchmark 工具会读取服务实际返回的 WAV 数据长度计算音频时长，并兼容
流式 WAV 中未知 `data` chunk 长度的情况。JSON 报告内会记录官方统计口径
名称、来源链接和公式。

复现流式直连 gRPC 与 Public SSE：

```bash
python3 scripts/benchmark_streaming.py \
  --transport both \
  --grpc-url 127.0.0.1:18001 \
  --sse-url http://127.0.0.1:18000/tts/stream \
  --speaker-id common_speaker_1 \
  --concurrency 1,2,4,8,16 \
  --requests 16 \
  --warmup 2 \
  --output-json streaming-benchmark.json
```

`grpc` 结果直接对齐官方的“提交 gRPC → 第一段非空 waveform”边界；`sse`
结果是开发者实际使用 18000 接口时看到的首段，并额外统计 `queueMs`。官方
L20 使用 `yuekai/seed_tts_cosy2` 原始参考音频数据集。严格复现数据、10 秒
padding、同步持久 stream 和任务分片：

```bash
curl -fL \
  "https://huggingface.co/datasets/yuekai/seed_tts_cosy2/resolve/main/data/wenetspeech4tts-00000-of-00001.parquet" \
  -o wenetspeech4tts.parquet

python3 scripts/benchmark_official_streaming.py \
  --server-url 127.0.0.1:18001 \
  --model CosyVoice3ProStreaming \
  --dataset-parquet wenetspeech4tts.parquet \
  --concurrency 4 \
  --output-json official-c4.json
```

生产 registered Speaker 路径保留同一组 target text 和客户端拓扑：

```bash
python3 scripts/benchmark_official_streaming.py \
  --dataset-parquet wenetspeech4tts.parquet \
  --speaker-id common_speaker_1 \
  --concurrency 4,8,16,26 \
  --output-json registered-speaker.json
```

复现上游默认核心参数 A100 基线：

```bash
COSYVOICE_PERFORMANCE_PROFILE=balanced \
COSYVOICE_KV_CACHE_FRACTION=0.4 \
COSYVOICE_PRO_BLS_INSTANCES=10 \
COSYVOICE_TOKEN2WAV_INSTANCES=1 \
COSYVOICE_VOCODER_INSTANCES=1 \
COSYVOICE_TTS_INFERENCE_CONCURRENCY=10 \
COSYVOICE_PRO_EAGER_CUDA_INIT=false \
  bash manage.sh restart

python3 scripts/benchmark.py \
  --url http://127.0.0.1:18000 \
  --speaker-id common_speaker_1 \
  --concurrency 12 \
  --requests 48 \
  --warmup 12

# 恢复 auto；80 GB GPU 会回到 streaming
bash manage.sh restart
```

## 切换与微调

使用保守配置：

```bash
COSYVOICE_PERFORMANCE_PROFILE=balanced \
  bash manage.sh restart
```

80 GB GPU 流式生产配置：

```bash
COSYVOICE_PERFORMANCE_PROFILE=streaming \
  bash manage.sh restart
```

离线吞吐配置：

```bash
COSYVOICE_PERFORMANCE_PROFILE=throughput \
  bash manage.sh restart
```

也可以覆盖单项参数：

```bash
COSYVOICE_PERFORMANCE_PROFILE=throughput \
COSYVOICE_TTS_INFERENCE_CONCURRENCY=16 \
COSYVOICE_TTS_SEGMENT_CONCURRENCY=2 \
  bash manage.sh restart
```

可调变量包括：

- `COSYVOICE_KV_CACHE_FRACTION`
- `COSYVOICE_PRO_BLS_INSTANCES`
- `COSYVOICE_STREAMING_BLS_INSTANCES`
- `COSYVOICE_LEGACY_BLS_INSTANCES`
- `COSYVOICE_TOKEN2WAV_INSTANCES`
- `COSYVOICE_VOCODER_INSTANCES`
- `COSYVOICE_FLOW_BATCH_SIZE`（`1`、`2`、`4`、`8`，默认 `1`）
- `COSYVOICE_FLOW_BATCH_QUEUE_DELAY_US`
- `COSYVOICE_VOCODER_BATCH_SIZE`（`1`、`2`、`4`、`8`，默认 `1`）
- `COSYVOICE_VOCODER_BATCH_QUEUE_DELAY_US`
- `COSYVOICE_TTS_INFERENCE_CONCURRENCY`
- `COSYVOICE_TTS_SEGMENT_CONCURRENCY`
- `COSYVOICE_TTS_STREAMING_CONCURRENCY`
- `COSYVOICE_TTS_STREAM_TIMEOUT_SECONDS`
- `COSYVOICE_TTS_STREAM_QUEUE_TIMEOUT_SECONDS`
- `COSYVOICE_STREAMING_FIRST_CHUNK_TOKENS`（`5`～`25`，实测默认 `15`）
- `COSYVOICE_STREAMING_CHUNK_GROWTH_OFFSET`（`0` 保留低延迟分块节奏，`1`
  减少高并发下的后续 Flow/Vocoder 调用）
- `COSYVOICE_PRO_EAGER_CUDA_INIT`

实例数会显著影响显存。小于 80 GB 的 GPU 应先使用 `balanced`，每次只增加
一个实例并观察 `nvidia-smi`、`nv_inference_queue_duration_us` 和失败
计数。

`/tts/` 响应包含分阶段耗时：

```text
X-CosyVoice-Inference-Ms
X-CosyVoice-Encode-Ms
Server-Timing
```

浏览器开发者工具或 `curl -D -` 可以直接查看这些响应头。
