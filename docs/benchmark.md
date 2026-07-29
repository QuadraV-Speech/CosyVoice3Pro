# CosyVoice3Pro 性能基准

本页记录通过 Public API `/tts/` 得到的端到端实测结果。计时包含 Web
Gateway、Speaker Registry 读取、CosyVoice3Pro 推理、音频后处理与 WAV
响应传输，不是只统计模型内部耗时。

## 测试环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB |
| NVIDIA Driver | 550.127.08 |
| Triton 镜像 | `soar97/triton-cosyvoice:25.06` |
| 上游 CosyVoice commit | `074ca6d` |
| Gateway | CosyVoice3Pro Web Gateway `1.6.0` |
| 模式 | 已注册 Speaker、WAV、16 kHz、单声道 |
| Profile | `throughput`：Pro BLS 12、token2wav 2、vocoder 2 |
| 测试日期 | 2026-07-29 |

测试文本：

```text
你好，欢迎使用 CosyVoice3Pro。注册一次声纹，后续请求只需要传入说话人编号和需要合成的文本。
```

## 性能 Profile

`COSYVOICE_PERFORMANCE_PROFILE=auto` 会读取容器可见 GPU 显存。显存不小于
70000 MiB 时使用 `throughput`，否则使用 `balanced`：

| Profile | LLM KV | Pro BLS | Legacy BLS | token2wav | vocoder | Gateway | 单请求分段 | CUDA 预热 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced` | 0.60 | 10 | 2 | 1 | 1 | 10 | 2 | 否 |
| `throughput` | 0.50 | 12 | 2 | 2 | 2 | 12 | 2 | 是 |

双 `token2wav` 但单 `vocoder` 只会把队列从声学模型转移到声码器，因此
throughput profile 同时扩展两个阶段。模型生成步数、采样参数和音频后处理
保持不变。

## 优化前后 A/B

两组都使用相同 A100、文本、Speaker、24 个请求和 12 并发。由于语音
Token 采样具有随机性，输出音频时长会小幅波动；应结合延迟、RTF、QPS
和音频吞吐判断。

| 配置 | 成功/请求 | P50 | P95 | 平均 RTF | 音频吞吐 | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 优化前 | 24/24 | 3.58s | 4.35s | 0.350 | 28.25x | 2.91 |
| `throughput` | 24/24 | **3.17s** | **3.89s** | **0.336** | **29.80x** | **3.15** |

在这组测试中，P50 降低 11.5%，P95 降低 10.7%，QPS 提升 8.1%，
音频吞吐提升 5.5%。

Triton 指标同时显示，单 token2wav 基线的累计排队时间为 12.51 秒。
扩展 token2wav 后必须同步扩展 vocoder，否则队列会转移到 vocoder；
最终双流水线配置在同轮 74 个 Pro 请求中，token2wav 和 vocoder 的
失败、取消和拒绝计数均为 0。

## 高并发压力测试

每组 48 个请求：

| 并发 | 成功/请求 | P50 | P95 | 平均 RTF | 音频吞吐 | QPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 48/48 | 4.19s | 5.72s | 0.427 | 31.36x | 3.18 |
| 24 | 48/48 | 6.13s | 7.38s | 0.593 | 31.61x | 3.35 |

24 并发时吞吐已经接近平台，延迟会继续上升。在线低延迟业务建议客户端
并发控制在 12～16；批量离线任务可以提高到 24。

长文本公平性回归同时提交一个 6 分段请求和 8 个短请求。短请求 8/8
成功，P50 1.48 秒、最大 1.96 秒；长请求 5.49 秒完成。单个长请求不会
再占满所有全局推理槽。

- `RTF = 请求耗时 / 输出音频时长`，越低越好。
- `音频吞吐 = 全部输出音频总时长 / 整组测试墙钟时间`，越高越好。
- 这是指定软硬件环境的一次端到端测量，不代表所有 GPU、文本和声音都能得到
  相同结果。

## 复现

服务启动并存在 `common_speaker_1` 后执行：

```bash
python3 scripts/benchmark.py \
  --url http://127.0.0.1:18000 \
  --speaker-id common_speaker_1 \
  --concurrency 12 16 24 \
  --requests 48 \
  --warmup 2
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
流式 WAV 中未知 `data` chunk 长度的情况。

## 切换与微调

使用保守配置：

```bash
COSYVOICE_PERFORMANCE_PROFILE=balanced \
  bash manage.sh restart
```

80GB GPU 吞吐配置：

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
- `COSYVOICE_LEGACY_BLS_INSTANCES`
- `COSYVOICE_TOKEN2WAV_INSTANCES`
- `COSYVOICE_VOCODER_INSTANCES`
- `COSYVOICE_TTS_INFERENCE_CONCURRENCY`
- `COSYVOICE_TTS_SEGMENT_CONCURRENCY`
- `COSYVOICE_PRO_EAGER_CUDA_INIT`

实例数会显著影响显存。小于 80GB 的 GPU 应先使用 `balanced`，每次只增加
一个实例并观察 `nvidia-smi`、`nv_inference_queue_duration_us` 和失败
计数。

`/tts/` 响应包含分阶段耗时：

```text
X-CosyVoice-Inference-Ms
X-CosyVoice-Encode-Ms
Server-Timing
```

浏览器开发者工具或 `curl -D -` 可以直接查看这些响应头。
