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
| Gateway | CosyVoice3Pro Web Gateway `1.5.0` |
| 模式 | 已注册 Speaker、WAV、16 kHz、单声道 |
| 预热 | 1 次请求 |
| 每组请求数 | 8 |
| 测试日期 | 2026-07-29 |

测试文本：

```text
你好，欢迎使用 CosyVoice3Pro。注册一次声纹，后续请求只需要传入说话人编号和需要合成的文本。
```

## 实测结果

| 并发 | 成功/请求 | P50 延迟 | P95 延迟 | 平均音频 | 平均 RTF | 音频吞吐 | QPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8/8 | 1.37s | 1.55s | 9.43s | 0.148 | 6.76x | 0.72 |
| 4 | 8/8 | 1.81s | 2.06s | 9.07s | 0.200 | 18.26x | 2.01 |

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
  --concurrency 1 4 \
  --requests 8 \
  --warmup 1
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
