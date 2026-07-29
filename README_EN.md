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
[![A100 RTF](https://img.shields.io/badge/A100_RTF-0.148-C8F45D)](docs/benchmark.md)

[中文](README.md) ·
[Quick Start](#quick-start) ·
[Benchmark](#measured-performance) ·
[Public API](docs/public-api.md) ·
[Operations](docs/web-admin.md)

</div>

---

> **What makes it different:** prompt-audio feature extraction is decoupled
> from every inference request. Register a reference voice once, then synthesize
> with only `speakerId + text`—no repeated Prompt Audio uploads.

A default instruction/persona can be stored during registration. A non-empty
`prompt` in a TTS request overrides that persona for the current request only.
Health checks, speaker registration, lookup, deletion, and TTS are exposed as
plain HTTP APIs without requiring clients to build Triton Tensor JSON.

<div align="center">
  <a href="docs/assets/web-console.png">
    <img src="docs/assets/web-demo.gif" alt="Real CosyVoice3Pro Web console workflow" width="100%">
  </a>
  <sub>Real service workflow: select a speaker → enter a persona and text → synthesize, preview, and download</sub>
</div>

> [!NOTE]
> CosyVoice3Pro is a community deployment project built on top of CosyVoice.
> It is not an official FunAudioLLM distribution. “Pro” refers to the serving
> and production-oriented capabilities added by this project.

## Why CosyVoice3Pro?

| Capability | Raw zero-shot workflow | CosyVoice3Pro |
| --- | --- | --- |
| Prompt audio | Uploaded on every request | Register once, then use `speakerId` |
| Voice features | Extracted repeatedly | Persisted and loaded on demand |
| Default persona | Sent by every client | Stored with the speaker |
| Per-request style | Client-specific glue code | Non-empty `prompt` overrides once |
| Voice sources | Separate workflows | Built-in, registered, and instant cloning |
| Audio processing | Implemented by clients | Speed, volume, chunking, and encoding |
| Management | Built separately | Registry API and Web console included |

## Measured performance

The following end-to-end measurement includes the Web Gateway, Speaker
Registry lookup, model inference, audio post-processing, and WAV response
transfer.

| GPU | Concurrency | Success | P50 | P95 | Average RTF | Audio throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100-SXM4-80GB | 1 | 8/8 | 1.37s | 1.55s | **0.148** | 6.76x |
| A100-SXM4-80GB | 4 | 8/8 | 1.81s | 2.06s | **0.200** | 18.26x |

See the [benchmark methodology and reproduction command](docs/benchmark.md).
Results vary with GPU, text, voice, and deployment configuration.

## Features

- Persistent Speaker Registry with register, inspect, list, update, and delete.
- Reusable prompt speech tokens, mel features, and CAMPPlus speaker embedding.
- Default speaker persona with per-request instruction override.
- Public HTTP API using regular JSON, forms, and audio streams.
- One `/tts/` endpoint for built-in voices, registered speakers, and raw prompt
  audio.
- Speaker registration from an uploaded audio file or a public audio URL.
- Long-text chunking, speed and volume control, and nine output formats.
- Web console for speaker management, synthesis, preview, and download.
- Triton `/v2/*`, gRPC, and Prometheus metrics retained for advanced use.

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
 │ /health · /register · /speakers · /tts/     │
 │       ▲                                     │
 │       └──────── Web console uses the same API│
 │                                             │
 │ Advanced API                                │
 │ /v2/* ───────────────► Triton HTTP :18100   │
 └──────────────────────────┬──────────────────┘
                            ├── CosyVoice3Pro
                            ├── Speaker Registry
                            └── upstream cosyvoice3

 gRPC :18001 · Metrics :18002
```

Port `18100` is only used by the Gateway inside the container. Application
developers should use the Public API. Triton Tensor APIs are intended for model
engineering and platform operations.

## Quick start

### Requirements

- Linux
- Docker
- NVIDIA Driver and NVIDIA Container Runtime
- CUDA-capable NVIDIA GPU
- Network access for the initial image, source, and model downloads

### Install

```bash
git clone https://github.com/QuadraV-Speech/CosyVoice3Pro.git
cd CosyVoice3Pro
COSYVOICE_GPU_ID=0 bash manage.sh install
```

The installer creates the container, prepares CosyVoice and the TensorRT-LLM
engine, deploys the CosyVoice3Pro models and Speaker Registry, installs audio
encoding dependencies, and starts the same-port Web Gateway.

### Start and verify

```bash
bash manage.sh start

curl --fail-with-body \
  http://127.0.0.1:18000/health
```

Open the Web console at:

```text
http://SERVER_IP:18000/
```

## Public API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health |
| `POST` | `/register` | Register or update a speaker using a file or URL |
| `GET` | `/speakers` | List speakers |
| `GET` | `/speakers/{speakerId}` | Inspect one speaker |
| `DELETE` | `/speakers/{speakerId}` | Delete one speaker |
| `POST` | `/tts/` | Synthesize and return processed audio |
| `GET` | `/` | Web console |

The complete parameter and response reference is available in the
[Public API documentation](docs/public-api.md).

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

Audio file upload is also supported through `multipart/form-data`.

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

## Advanced APIs

| Endpoint | Purpose |
| --- | --- |
| `http://HOST:18000/v2/` | Triton HTTP API |
| `HOST:18001` | Triton gRPC |
| `http://HOST:18002/metrics` | Prometheus metrics |

See the [Advanced API documentation](docs/advanced-api.md) for Tensor
contracts, raw prompt inference, and internal Registry operations.

## Benchmark your deployment

```bash
python3 scripts/benchmark.py \
  --url http://127.0.0.1:18000 \
  --speaker-id common_speaker_1 \
  --concurrency 1 4 \
  --requests 8 \
  --warmup 1
```

The tool reports P50/P95 latency, audio duration, RTF, audio throughput, and
QPS using real WAV responses from the Public API.

## Operations

```bash
bash manage.sh start
bash manage.sh stop
bash manage.sh restart
bash manage.sh status
bash manage.sh logs
bash manage.sh backup
```

| Variable | Default | Description |
| --- | --- | --- |
| `COSYVOICE_GPU_ID` | `3` | Host GPU assigned to Docker |
| `COSYVOICE_GIT_PROXY` | Current proxy or empty | Proxy used to fetch upstream code |
| `COSYVOICE_SPEAKER_STORE_DIR` | `data/speakers` | Persistent speaker storage |
| `COSYVOICE_WEB_GATEWAY_ENABLED` | `true` | Enable the same-port Web Gateway |

## Security

The Web console, speaker registration, TTS, and Triton APIs do not include
application-level authentication by default. A production deployment should
add TLS, authentication and authorization, source-IP restrictions, request
size limits, rate limits, and independent Speaker Registry backups at the
reverse-proxy or load-balancer layer.

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
