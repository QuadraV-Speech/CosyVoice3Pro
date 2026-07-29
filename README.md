<div align="center">

# CosyVoice3Pro

### Production-ready CosyVoice serving

**Register once. Speak many times.**

基于 NVIDIA Triton Inference Server 与 TensorRT-LLM 的<br>
高性能语音克隆服务、可复用 Speaker Registry、开发者友好 API 与 Web 工作台

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/QuadraV-Speech/CosyVoice3Pro/actions/workflows/ci.yml/badge.svg)](https://github.com/QuadraV-Speech/CosyVoice3Pro/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/QuadraV-Speech/CosyVoice3Pro)](https://github.com/QuadraV-Speech/CosyVoice3Pro/releases)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton-76B900?logo=nvidia&logoColor=white)](https://github.com/triton-inference-server/server)
[![TensorRT--LLM](https://img.shields.io/badge/TensorRT--LLM-Accelerated-76B900)](https://github.com/NVIDIA/TensorRT-LLM)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![API](https://img.shields.io/badge/HTTP_API-%3A18000-7C3AED)](#api-入口)
[![A100 system RTF](https://img.shields.io/badge/A100_system_RTF-0.0322-C8F45D)](docs/benchmark.md)

[English](README_EN.md) ·
[快速开始](#快速开始) ·
[实测性能](#实测性能) ·
[Web 后台](#web-管理后台) ·
[对外 API](docs/public-api.md) ·
[部署运维](docs/web-admin.md) ·
[参与贡献](#参与贡献)

</div>

---

> [!IMPORTANT]
> **官方 CosyVoice3 解决“高质量生成”，CosyVoice3Pro 解决“如何把它稳定、
> 高效地提供给业务”。** Pro 完整保留官方模型与 Triton 高级接口，并增加
> 可持久化 Speaker Registry、开发者友好 REST API、音频后处理、Web 工作台
> 和面向 A100 的并发 Profile。

最直接的变化是声纹与推理解耦：参考音频注册一次，后续请求只传
`speakerId + text`。注册时还可以保存默认 Prompt 画像；单次请求传入非空
`prompt` 即可临时覆盖，无需重复上传音频或在每个客户端拼装 Tensor。

## 官方 CosyVoice3 与 CosyVoice3Pro

以下对比针对
[官方 CosyVoice3 Triton Runtime](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/runtime/triton_trtllm/README.Cosyvoice3.md)。
Pro 使用相同的 `Fun-CosyVoice3-0.5B-2512` 模型与 TensorRT-LLM
推理核心，优势集中在生产服务层，而不是声称修改了官方模型能力。

| 维度 | 官方 CosyVoice3 Triton Runtime | CosyVoice3Pro |
| --- | --- | --- |
| 模型与音质 | 官方 CosyVoice3 模型 | **完全继承官方模型** |
| 业务调用 | Triton V2 / gRPC Tensor，需要组织参考音频与文本输入 | **普通 REST、表单和音频流；curl 即可调用** |
| 声纹复用 | 默认携带参考音频；可选进程内缓存，但没有持久化 Speaker 实体 | **注册一次，持久化特征，后续只传 `speakerId`** |
| 多 Speaker 管理 | 没有面向业务的查增删 API | **注册、更新、列表、详情、删除完整闭环** |
| 跨重启复用 | 进程内缓存随服务结束 | **Speaker Registry 持久化并按需加载** |
| Prompt 画像 | 客户端随请求组织参考文本/指令 | **注册默认画像，非空请求 Prompt 单次覆盖** |
| 注册来源 | 客户端准备音频 Tensor | **上传文件或公开音频 URL** |
| 统一 TTS | 不同推理方式由客户端组织 | **内置声音、Speaker ID、即时克隆统一 `/tts/`** |
| 音频交付 | 返回模型波形，业务自行处理 | **语速、音量、长文本分段及 9 种输出格式** |
| Web 管理 | Triton Runtime 不含同端口业务后台 | **18000 同端口 Web 工作台，且只调用 Public API** |
| 并发调优 | 手工调整实例与 KV 参数 | **按 GPU 显存自动选择 `balanced` / `throughput`** |
| 可观测性 | Triton 原生指标 | **保留 Metrics，并增加健康检查与分阶段耗时响应头** |
| 流式能力 | 官方 decoupled streaming | 高级 Triton 接口保留；Public `/tts/` 当前返回完整音频 |

一句话概括：**CosyVoice3Pro = 官方 CosyVoice3 推理核心 + 可复用声纹层 +
面向开发者的 API + 可直接交付的音频 + 生产运维能力。**

<div align="center">
  <a href="docs/assets/web-console.png">
    <img src="docs/assets/web-demo.gif" alt="CosyVoice3Pro Web 声音工作台真实操作演示" width="100%">
  </a>
  <sub>真实服务演示：选择 Speaker → 输入画像与文本 → 生成、试听并下载</sub>
</div>

> [!NOTE]
> CosyVoice3Pro 是基于上游 CosyVoice 构建的社区部署项目，并非
> FunAudioLLM 官方发行版。“Pro”指本项目增加的服务化与工程能力。

## 实测性能

端到端测试包含 Web Gateway、Speaker Registry、模型推理、后处理和 WAV
响应传输，不是只统计模型内部耗时。系统 RTF 使用官方聚合口径：
`整组墙钟时间 / 全部输出音频总时长`。

### 官方默认配置与 Pro Profile

同一台 A100、同一模型与 Engine、同一业务链路、文本、Speaker、后处理、
12 并发、48 个请求和 12 次预热，仅改变服务 Profile：

| A100-SXM4-80GB 配置 | 成功率 | P50 | P95 | 系统 RTF | 音频吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方默认核心参数复现 | 48/48 | 3.67s | 4.41s | 0.0391 | 25.61x |
| **CosyVoice3Pro `throughput`** | **48/48** | **3.40s** | **4.22s** | **0.0329** | **30.42x** |

Pro Profile 的系统 RTF 降低 **15.8%**，音频吞吐提升 **18.8%**。
这里的“官方默认核心参数复现”仍使用相同 Pro Public API，以严格控制业务
链路；它不是 FunAudioLLM 发布的 A100 官方数字。

### Pro 高并发扩展

| 环境 | 并发 | 成功率 | P50 | P95 | 系统 RTF | 音频吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100-SXM4-80GB | 12 | 48/48 | 3.40s | 4.22s | **0.0329** | 30.42x |
| A100-SXM4-80GB | 16 | 48/48 | 4.41s | 5.42s | **0.0322** | 31.06x |
| A100-SXM4-80GB | 24 | 48/48 | 6.72s | 8.18s | **0.0331** | 30.19x |

测试条件、指标解释和复现命令见
[性能基准文档](docs/benchmark.md)。其中包含变量受控的上游默认配置 A100
复测，以及官方发布的 L20 基线；上游目前没有发布 A100 性能数字。不同
GPU、文本和声音的结果会有所差异。

## 核心能力

- **Speaker Registry**：注册、查询、列出、更新和删除说话人。
- **声纹解耦**：持久化 Prompt Speech Tokens、Mel 特征和 Speaker
  Embedding。
- **Prompt 画像**：注册默认画像，并支持单次请求覆盖。
- **开发者友好 API**：普通 JSON、表单和音频流，无需了解 Tensor 协议。
- **声纹查增删**：注册/更新、列表、单个查询和删除接口完整覆盖。
- **统一 TTS API**：支持内置声音、注册声纹和即时克隆。
- **双来源注册**：支持上传音频文件和公开音频 URL。
- **音频后处理**：支持长文本分段、语速、音量以及九种输出格式。
- **Web 管理后台**：上传参考音频、管理 Speaker、配置后处理、试听和下载。
- **高级接口保留**：平台和模型工程可继续使用 Triton `/v2/*`。

## 系统架构

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
 │       └──────── Web Admin 使用同一组 API     │
 │                                             │
 │ Advanced API                                │
 │ /v2/* ───────────────► Triton HTTP :18100   │
 └──────────────────────────┬──────────────────┘
                            ├── CosyVoice3Pro
                            ├── Speaker Registry
                            └── upstream cosyvoice3

 gRPC :18001 · Metrics :18002
```

`18100` 仅供容器内部 Gateway 访问。业务开发优先使用 Public API；模型
调试和平台运维才需要 Advanced API。

## API 入口

### 对外 API

| 方法 | 地址 | 用途 |
| --- | --- | --- |
| `GET` | `http://HOST:18000/health` | 服务健康检查 |
| `POST` | `http://HOST:18000/register` | 上传音频或 URL 注册/更新声纹 |
| `GET` | `http://HOST:18000/speakers` | 查询全部声纹 |
| `GET` | `http://HOST:18000/speakers/{speakerId}` | 查询单个声纹 |
| `DELETE` | `http://HOST:18000/speakers/{speakerId}` | 删除声纹 |
| `POST` | `http://HOST:18000/tts/` | 生成处理后的音频 |
| `GET` | `http://HOST:18000/` | Web 管理后台 |

完整参数、响应和 curl 见
[对外 API 文档](docs/public-api.md)。

### 内部高级 API

| 地址 | 用途 |
| --- | --- |
| `http://HOST:18000/v2/` | Triton HTTP API |
| `HOST:18001` | Triton gRPC |
| `http://HOST:18002/metrics` | Prometheus Metrics |

Tensor 协议、模型输入和 Registry 内部操作见
[内部高级 API 文档](docs/advanced-api.md)。

## 快速开始

### 环境要求

- Linux
- Docker
- NVIDIA Driver 与 NVIDIA Container Runtime
- 支持 CUDA 的 NVIDIA GPU
- 首次安装时可访问 GitHub、镜像源和模型下载源

### 1. 克隆项目

```bash
git clone https://github.com/QuadraV-Speech/CosyVoice3Pro.git
cd CosyVoice3Pro
```

### 2. 首次安装

通过 `COSYVOICE_GPU_ID` 指定宿主机 GPU：

```bash
COSYVOICE_GPU_ID=0 bash manage.sh install
```

安装过程会：

1. 创建 `cosyvoice-server` 容器；
2. 准备上游 CosyVoice 与 TensorRT-LLM Engine；
3. 部署 CosyVoice3Pro 和 Speaker Registry 模型；
4. 安装音频编码依赖；
5. 部署同端口 Web Gateway。

### 3. 启动服务

```bash
bash manage.sh start
```

### 4. 检查状态

```bash
curl --fail-with-body \
  http://127.0.0.1:18000/health
```

## 30 秒调用示例

### 使用音频 URL 注册声纹

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "speakerId": "narrator_01",
    "audio_url": "https://example.com/reference.mp3",
    "reference_text": "这是参考音频中实际说出的内容。",
    "prompt": "请用成熟、稳重、亲切的语气说话。"
  }'
```

也可以通过 `multipart/form-data` 直接上传音频。完整参数、文件上传 curl
和返回格式见 [对外 API 文档](docs/public-api.md#4-注册或更新声纹)。

### 查询声纹

```bash
curl --fail-with-body \
  "http://127.0.0.1:18000/speakers"

curl --fail-with-body \
  "http://127.0.0.1:18000/speakers/narrator_01"
```

### 直接生成音频

`/tts/` 可以使用已注册 Speaker，并直接返回处理后的音频：

```bash
curl --fail-with-body \
  -X POST "http://127.0.0.1:18000/tts/" \
  -F "text=你好，这是 CosyVoice3Pro 统一语音接口。" \
  -F "speakerId=common_speaker_1" \
  -F "prompt=请用成熟、稳重、亲切的语气说话。" \
  -F "speed=balanced" \
  -F "volume=middle" \
  -F "output_format=mp3" \
  --output output.mp3
```

同一接口也支持 `tts_style` 内置声音或直接上传 `prompt_audio`。完整参数和
curl 示例见 [对外 API 文档](docs/public-api.md#7-文字转语音)。

Prompt 解析规则：

| 推理请求 | 实际行为 |
| --- | --- |
| 不传 `prompt` | 使用 Speaker 注册时保存的默认画像 |
| `prompt=""` | 使用 Speaker 注册时保存的默认画像 |
| 非空 `prompt` | 只覆盖本次请求，不修改默认画像 |

### 删除声纹

```bash
curl --fail-with-body \
  -X DELETE \
  "http://127.0.0.1:18000/speakers/narrator_01"
```

## Web 管理后台

服务启动后访问：

```text
http://服务器IP:18000/
```

管理后台支持：

- Speaker 列表、搜索和状态检查
- WAV、MP3、M4A 等参考音频上传
- 本地音频或公开音频 URL 注册
- 参考音频文本与默认 Prompt 画像配置
- Speaker 注册、更新和删除
- 默认画像与单次覆盖画像推理
- 语速、音量、输出格式和长文本分段配置
- 处理后音频的在线试听和下载

详细说明见 [`docs/web-admin.md`](docs/web-admin.md)。

## Speaker 数据

注册时会一次性提取并持久化：

```text
Prompt Speech Tokens
Prompt Mel Features
CAMPPlus Speaker Embedding
Reference Transcript
Default Prompt Persona
```

默认宿主机存储目录：

```text
data/speakers/
```

该目录已从 Git 中排除。手动备份：

```bash
bash manage.sh backup
```

## 项目结构

```text
CosyVoice3Pro/
├── gateway/
│   ├── app.py                         # Web Gateway 与 Triton 反向代理
│   ├── legacy_tts.py                  # 统一 TTS 与音频后处理
│   ├── speaker_registration.py        # 文件/URL 声纹注册 API
│   └── web/                           # Web 管理后台
├── models/
│   ├── CosyVoice3Pro/                 # Speaker ID / Raw Prompt 推理
│   └── CosyVoice3ProSpeakerRegistry/  # Speaker 注册与持久化
├── scripts/
│   ├── client.py                      # 注册、查询和推理客户端
│   └── benchmark.py                   # Public API 性能基准工具
├── docs/
│   ├── public-api.md                  # 对外开发者 API
│   ├── advanced-api.md                # 内部 Triton 高级 API
│   ├── benchmark.md                   # 实测性能与复现方法
│   └── web-admin.md                   # Gateway 部署与运维
├── data/
│   └── speakers/                      # 本地声纹数据，不提交
├── manage.sh                          # 安装、启停、状态和备份
└── requirements.txt
```

## 服务管理

```bash
bash manage.sh start
bash manage.sh stop
bash manage.sh restart
bash manage.sh status
bash manage.sh logs
bash manage.sh backup
```

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `COSYVOICE_GPU_ID` | `3` | Docker 使用的宿主机 GPU 编号 |
| `COSYVOICE_GIT_PROXY` | 当前代理或空 | 拉取上游仓库时使用的代理 |
| `COSYVOICE_SPEAKER_STORE_DIR` | `data/speakers` | Speaker 持久化目录 |
| `COSYVOICE_WEB_GATEWAY_ENABLED` | `true` | 是否启用同端口 Gateway |
| `COSYVOICE_PERFORMANCE_PROFILE` | `auto` | 自动按显存选择 `balanced` 或 `throughput` |
| `COSYVOICE_KV_CACHE_FRACTION` | Profile 决定 | TensorRT-LLM KV Cache 显存比例 |
| `COSYVOICE_PRO_BLS_INSTANCES` | Profile 决定 | CosyVoice3Pro 编排实例数 |
| `COSYVOICE_TOKEN2WAV_INSTANCES` | Profile 决定 | 声学模型实例数 |
| `COSYVOICE_VOCODER_INSTANCES` | Profile 决定 | 声码器实例数 |
| `COSYVOICE_TTS_INFERENCE_CONCURRENCY` | Profile 决定 | Gateway 全局推理并发上限 |
| `COSYVOICE_TTS_SEGMENT_CONCURRENCY` | `2` | 单个长文本可同时占用的分段槽数 |
| `COSYVOICE_PRO_EAGER_CUDA_INIT` | Profile 决定 | Ready 前预热 Pro 实例 CUDA 上下文 |

如果 Git 代理仅监听宿主机 `127.0.0.1`，管理脚本会通过 Docker 网关建立
临时转发。

`auto` 在 80GB GPU 上启用双 `token2wav`、双 `vocoder` 的吞吐配置，
其他 GPU 保持单实例保守配置。修改性能参数后执行 `bash manage.sh restart`
生效；配置细节和 A/B 数据见[性能基准文档](docs/benchmark.md)。

## 兼容性

- Web 页面只调用对外 `/health`、`/register`、`/speakers` 和 `/tts/`。
- 保留上游 `cosyvoice3` Triton 模型，方便旧调用回滚。
- 保留原始 `reference_wav + reference_text + target_text` 推理。
- `instruct_text` 仍可作为 `prompt` 的兼容别名。
- 旧版 `tts_style` 请求可以继续调用 `/tts/`。
- 暂时关闭 Gateway 时，Triton 可直接监听外部 `18000`：

```bash
COSYVOICE_WEB_GATEWAY_ENABLED=false bash manage.sh restart
```

恢复 Gateway：

```bash
bash manage.sh restart
```

## 安全提示

Web 管理后台、声纹注册、TTS 与 Triton API 默认不包含应用层登录认证。
部署到生产环境时，建议在外层负载均衡、反向代理或防火墙中配置：

- TLS
- 身份认证与访问控制
- 来源 IP 限制
- 请求体大小和速率限制
- Speaker 数据目录的独立备份

## 文档

- [对外开发者 API](docs/public-api.md)
- [内部 Triton 高级 API](docs/advanced-api.md)
- [性能基准与复现](docs/benchmark.md)
- [Web Gateway 部署与运维](docs/web-admin.md)
- [版本变更记录](CHANGELOG.md)
- [安全策略](SECURITY.md)

## 开源许可

CosyVoice3Pro 的原创代码与文档使用
[Apache License 2.0](LICENSE)，归属声明见 [NOTICE](NOTICE)。

模型权重、基础镜像、CosyVoice、NVIDIA Triton、TensorRT-LLM 及其他上游
组件不因本仓库许可证而重新授权，仍分别受其原始许可证和使用条款约束。

## 参与贡献

欢迎提交 Bug、文档改进、GPU 兼容性结果、Benchmark 数据和 Pull Request。
提交前请阅读 [贡献指南](CONTRIBUTING.md)。仓库已经提供结构化 Bug/Feature
模板，界面变化请附截图，性能变化请附可复现命令和前后数据。

## 致谢与上游

CosyVoice3Pro 基于
[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)、
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
与 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 构建。

模型权重、基础镜像及上游组件分别受其原始许可证和使用条款约束。
