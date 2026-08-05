<div align="center">

# CosyVoice3Pro

### Production-ready CosyVoice serving

**Register once. Speak many times.**

High-performance voice cloning powered by NVIDIA Triton and TensorRT-LLM,<br>
with a reusable Speaker Registry, developer-friendly API, and Web console.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/QuadraV-Speech/CosyVoice3Pro/actions/workflows/ci.yml/badge.svg)](https://github.com/QuadraV-Speech/CosyVoice3Pro/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/QuadraV-Speech/CosyVoice3Pro)](https://github.com/QuadraV-Speech/CosyVoice3Pro/releases)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton-76B900?logo=nvidia&logoColor=white)](https://github.com/triton-inference-server/server)
[![TensorRT--LLM](https://img.shields.io/badge/TensorRT--LLM-Accelerated-76B900)](https://github.com/NVIDIA/TensorRT-LLM)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![HTTP API](https://img.shields.io/badge/HTTP_API-%3A18000-7C3AED)](#public-api)
[![A100 system RTF](https://img.shields.io/badge/A100_system_RTF-0.0322-C8F45D)](docs/benchmark.md)

[中文](README.md) ·
[Quick Start](#quick-start) ·
[Benchmark](#measured-performance) ·
[Public API](docs/public-api.md) ·
[Operations](docs/web-admin.md)

</div>

---

> [!IMPORTANT]
> **CosyVoice3Pro = official CosyVoice3 inference + Speaker Registry +
> developer API + audio delivery + Web and production operations.**

Register reference audio once, then send only `speakerId + text`. A non-empty
request `prompt` temporarily overrides the speaker's default persona.

## Official CosyVoice3 vs. CosyVoice3Pro

| Dimension | Official CosyVoice3 Triton Runtime | CosyVoice3Pro |
| --- | --- | --- |
| Model | `Fun-CosyVoice3-0.5B-2512` | **Same official model and TensorRT-LLM core** |
| API | Triton V2 / gRPC Tensors | **REST, forms, audio streams, curl-ready** |
| Voices | Reference audio by default; optional in-process cache | **Persistent multi-speaker Registry with CRUD** |
| Prompt | Client assembles reference text/instructions | **Stored default with request override** |
| I/O | Client prepares Tensors and processes waveform | **File/URL registration, chunking, speed, volume, nine formats** |
| Operations | Native Triton capabilities | **Same-port Web, health, timing headers, automatic profiles** |
| Streaming | Advanced gRPC Decoupled calls | **Public SSE, browser playback while generating, curl-ready** |

CosyVoice3Pro is a community serving enhancement, not an official FunAudioLLM
distribution.

<div align="center">
  <a href="docs/assets/web-console.png">
    <img src="docs/assets/web-demo.gif" alt="Real CosyVoice3Pro Web console workflow" width="100%">
  </a>
  <sub>Real service workflow: select a speaker → enter a persona and text → synthesize, preview, and download</sub>
</div>

## Measured performance

System RTF follows the upstream definition:
`profile wall time / total synthesized audio duration`.

### Controlled A100 comparison

Same hardware, model, engine, application path, and requests; only the service
profile changes:

| A100-SXM4-80GB configuration | Success | P50 | P95 | System RTF | Audio throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reproduced upstream default core settings | 48/48 | 3.67s | 4.41s | 0.0391 | 25.61x |
| **CosyVoice3Pro `throughput`** | **48/48** | **3.40s** | **4.22s** | **0.0329** | **30.42x** |

System RTF falls **15.8%** and audio throughput rises **18.8%**. The first row
is a same-path reproduction, not an officially published A100 number.

### Official L20 baseline

From the
[official CosyVoice3 Triton documentation](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/README.Cosyvoice3.md):

| Official mode | Concurrency / Batch | Published result |
| --- | ---: | --- |
| Streaming first chunk | Concurrency 4 | Avg 750.42 ms; P50 740.31; P90 941.05; P95 977.55; P99 1002.37 |
| Offline pipeline | Batch 1 | RTF 0.1091 |
| Offline pipeline | Batch 2 | RTF 0.0822 |
| Offline pipeline | Batch 4 | RTF 0.0630 |
| Offline pipeline | Batch 8 | RTF 0.0562 |
| Offline pipeline | Batch 16 | RTF 0.0501 |

Upstream publishes no A100 result. The L20 streaming/offline workloads are not
directly comparable with Pro's end-to-end A100 HTTP benchmark.

### Pro streaming concurrency

The direct gRPC timer follows the upstream first-nonempty-waveform boundary;
each profile uses 16 requests with one registered speaker and fixed text:

| A100 gRPC concurrency | Success | TTFA Avg | P50 | P95 | System RTF | Audio throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 16/16 | 598.99 ms | 455.81 ms | 1137.02 ms | 0.0799 | 12.52x |
| 8 | 16/16 | 1008.40 ms | 996.94 ms | 1396.22 ms | 0.0657 | 15.23x |
| 16 | 16/16 | 2048.85 ms | 2053.21 ms | 3011.77 ms | **0.0581** | **17.21x** |

At concurrency 16, audio throughput improves **33.0%** and average full
generation latency falls **26.0%** versus the pre-optimization baseline. Public
SSE accepts 10 concurrent streams by default and reports queue time separately.

### Pro full-audio concurrency

| GPU | Concurrency | Success | P50 | P95 | System RTF | Audio throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100-SXM4-80GB | 12 | 48/48 | 3.40s | 4.22s | **0.0329** | 30.42x |
| A100-SXM4-80GB | 16 | 48/48 | 4.41s | 5.42s | **0.0322** | 31.06x |
| A100-SXM4-80GB | 24 | 48/48 | 6.72s | 8.18s | **0.0331** | 30.19x |

See the full [benchmark methodology and reproduction command](docs/benchmark.md).

## Architecture

```text
 Browser / SDK / curl
          │
          │ :18000
          ▼
 ┌─────────────────────────────────────────────┐
 │          CosyVoice3Pro Gateway              │
 │                                             │
 │ Public API                                  │
 │ /health · /register · /speakers             │
 │ /tts/ · /tts/stream (SSE)                   │
 │       ▲                                     │
 │       └──────── Web console uses the same API│
 │                                             │
 │ Advanced API                                │
 │ /v2/* ───────────────► Triton HTTP :18100   │
 └──────────────────────────┬──────────────────┘
                            ├── CosyVoice3Pro
                            ├── CosyVoice3ProStreaming
                            ├── Speaker Registry
                            └── upstream cosyvoice3

 gRPC :18001 · Metrics :18002
```

Port `18100` is only used by the Gateway inside the container. Application
developers should use the Public API. Triton Tensor APIs are intended for model
engineering and platform operations.

## Quick start

Requires Linux, Docker, NVIDIA Driver, NVIDIA Container Runtime, and a CUDA
GPU. Initial installation downloads source, images, and models.

```bash
git clone https://github.com/QuadraV-Speech/CosyVoice3Pro.git
cd CosyVoice3Pro
COSYVOICE_GPU_ID=0 bash manage.sh install
```

```bash
curl --fail-with-body http://127.0.0.1:18000/health
```

Web console: `http://SERVER_IP:18000/`

## Public API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health |
| `POST` | `/register` | Register or update a speaker using a file or URL |
| `GET` | `/speakers` | List speakers |
| `GET` | `/speakers/{speakerId}` | Inspect one speaker |
| `DELETE` | `/speakers/{speakerId}` | Delete one speaker |
| `POST` | `/tts/` | Synthesize and return processed audio |
| `POST` | `/tts/stream` | Generate and play incrementally over SSE |
| `GET` | `/` | Web console |

### Register from an audio URL

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "speakerId": "narrator_01",
    "audio_url": "https://example.com/reference.mp3",
    "reference_text": "The exact transcript of the reference audio.",
    "prompt": "Speak in a mature, composed, and friendly tone."
  }'
```

Audio upload through `multipart/form-data` is also supported.

### Synthesize with a registered speaker

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/tts/" \
  -F "text=Hello from the unified CosyVoice3Pro speech API." \
  -F "speakerId=narrator_01" \
  -F "prompt=Speak clearly and naturally." \
  -F "speed=balanced" \
  -F "volume=middle" \
  -F "output_format=mp3" \
  --output output.mp3
```

Prompt resolution:

| Request | Behavior |
| --- | --- |
| `prompt` omitted | Use the speaker's stored default persona |
| `prompt=""` | Use the speaker's stored default persona |
| Non-empty `prompt` | Override the persona for this request only |

See the full [Public API documentation](docs/public-api.md) for file upload,
built-in voices, raw prompt audio, speaker deletion, and all parameters.

### Online SSE playback

```bash
curl --fail-with-body -N --no-buffer \
  -X POST "http://127.0.0.1:18000/tts/stream" \
  -F "text=This audio is returned while it is being generated." \
  -F "speakerId=common_speaker_1" \
  -F "prompt=Speak naturally and clearly."
```

The response contains `meta → audio × N → done` events. Each `audio` event
carries Base64-encoded 16 kHz mono PCM. The Web console decodes and schedules
chunks immediately; see the [Public API documentation](docs/public-api.md)
for a curl-to-ffplay example.

## Operations

```bash
bash manage.sh start
bash manage.sh stop
bash manage.sh restart
bash manage.sh status
bash manage.sh logs
bash manage.sh backup
```

Speakers default to `data/speakers/`; `bash manage.sh backup` creates a manual
backup. See [operations](docs/web-admin.md) for profiles and environment
variables.

Advanced interfaces remain available: Triton HTTP `/v2/*`, gRPC `18001`, and
Metrics `18002`.

## Security

No application-level authentication is enabled by default. Public deployments
should add TLS, authentication, rate limits, source restrictions, and separate
Speaker Registry backups.

## Contributing

Bug reports, documentation, compatibility results, benchmark data, and pull
requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting
a change.

## License

Original CosyVoice3Pro code and documentation are licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

Model weights, base images, CosyVoice, NVIDIA Triton, TensorRT-LLM, and other
upstream components retain their respective original licenses and terms.

## Documentation

- [Public API](docs/public-api.md)
- [Advanced Triton API](docs/advanced-api.md)
- [Benchmark](docs/benchmark.md)
- [Web Gateway operations](docs/web-admin.md)
- [Changelog](CHANGELOG.md)
- [Security Policy](SECURITY.md)

## Acknowledgements

CosyVoice3Pro is built on
[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice),
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server),
and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

Model weights, base images, and upstream components remain subject to their
respective original licenses and terms of use.
