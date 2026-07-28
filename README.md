<div align="center">

# CosyVoice3Pro

### Register once. Speak many times.

基于 NVIDIA Triton Inference Server 与 TensorRT-LLM 的<br>
高性能语音克隆服务、Speaker Registry 与 Web 管理后台

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton-76B900?logo=nvidia&logoColor=white)](https://github.com/triton-inference-server/server)
[![TensorRT--LLM](https://img.shields.io/badge/TensorRT--LLM-Accelerated-76B900)](https://github.com/NVIDIA/TensorRT-LLM)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![API](https://img.shields.io/badge/HTTP_API-%3A18000-7C3AED)](#api-入口)

[快速开始](#快速开始) ·
[Web 后台](#web-管理后台) ·
[API 文档](docs/api.md) ·
[部署运维](docs/web-admin.md)

</div>

---

CosyVoice3Pro 将提示音频的特征提取从每次推理中解耦出来：参考音频只需
注册一次，后续请求只传 `speaker_id + text` 即可完成语音合成。注册时还可
保存默认 Prompt 画像，请求中的非空 `prompt` 会仅对本次推理覆盖默认画像。

Web 管理后台与 Triton HTTP API 统一由 `18000` 端口提供。

## 为什么使用 CosyVoice3Pro

| 能力 | 传统零样本调用 | CosyVoice3Pro |
| --- | --- | --- |
| 提示音频 | 每次请求重复上传 | 注册一次，后续只传 `speaker_id` |
| 声纹特征 | 每次重复提取 | 持久化并按需加载 |
| 默认说话风格 | 客户端每次携带 | 注册时保存默认 Prompt 画像 |
| 临时风格覆盖 | 需要自行拼装 | 请求传非空 `prompt` 即可 |
| 管理能力 | 需要额外开发 | 内置注册、查询、更新、删除和 Web 后台 |

## 核心能力

- **Speaker Registry**：注册、查询、列出、更新和删除说话人。
- **声纹解耦**：持久化 Prompt Speech Tokens、Mel 特征和 Speaker
  Embedding。
- **Prompt 画像**：注册默认画像，并支持单次请求覆盖。
- **双推理模式**：支持 `speaker_id` 推理，也兼容
  `reference_wav + reference_text` 原始调用。
- **Web 管理后台**：上传参考音频、管理 Speaker、在线合成、试听和下载。
- **Triton 兼容代理**：外部 `/v2/*` 地址不变，Gateway 转发至容器内部
  Triton。

## 系统架构

```text
                       :18000
 Browser / SDK / curl ───────┬───────────────────────────────┐
                             │                               │
                             ▼                               │
                  ┌──────────────────────┐                   │
                  │ CosyVoice3Pro Gateway│                   │
                  └──────┬────────┬──────┘                   │
                         │        │                          │
                       / │        │ /v2/*                    │
                         ▼        ▼                          │
                  Web Admin   Triton HTTP :18100             │
                                  │                          │
                                  ├── CosyVoice3Pro          │
                                  ├── Speaker Registry       │
                                  └── upstream cosyvoice3    │
                                                             │
                        gRPC :18001 · Metrics :18002 ◀────────┘
```

`18100` 仅供容器内部 Gateway 访问。外部 Triton HTTP 与 Web 后台统一
使用 `18000`。

## API 入口

| 地址 | 用途 |
| --- | --- |
| `http://HOST:18000/` | Web 管理后台 |
| `http://HOST:18000/v2/` | Triton HTTP API |
| `HOST:18001` | Triton gRPC |
| `http://HOST:18002/metrics` | Prometheus Metrics |

## 快速开始

### 环境要求

- Linux
- Docker
- NVIDIA Driver 与 NVIDIA Container Runtime
- 支持 CUDA 的 NVIDIA GPU
- 首次安装时可访问 GitHub、镜像源和模型下载源

### 1. 克隆项目

```bash
git clone git@github.com:QuadraV-Speech/CosyVoice3Pro.git
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
curl -f http://127.0.0.1:18000/v2/health/ready
curl -f http://127.0.0.1:18000/v2/models/CosyVoice3Pro/ready
curl -f http://127.0.0.1:18000/v2/models/CosyVoice3ProSpeakerRegistry/ready
curl -sS http://127.0.0.1:18000/admin/api/info
```

## 30 秒调用示例

### 注册自己的 Speaker

安装本地客户端依赖：

```bash
python -m pip install -r requirements.txt
```

参考音频建议为 3～10 秒清晰、无背景音乐的单人声。客户端会自动通过
FFmpeg 转换为 16kHz 单声道。

```bash
python scripts/client.py register \
  --speaker-id narrator_female_01 \
  --audio /path/to/reference.wav \
  --reference-text "欢迎使用 CosyVoice3Pro 语音服务。" \
  --prompt "请用成熟、稳重、亲切的语气说话。"
```

### 使用 Speaker ID 合成

使用注册时保存的默认画像：

```bash
python scripts/client.py infer \
  --speaker-id narrator_female_01 \
  --text "你好，这是默认画像的语音合成。" \
  --output default.wav
```

仅本次请求覆盖画像：

```bash
python scripts/client.py infer \
  --speaker-id narrator_female_01 \
  --prompt "请非常开心、兴奋地说话。" \
  --text "太好了，我们完成了新的服务升级！" \
  --output happy.wav
```

Prompt 解析规则：

| 推理请求 | 实际行为 |
| --- | --- |
| 不传 `prompt` | 使用 Speaker 注册时保存的默认画像 |
| `prompt=""` | 使用 Speaker 注册时保存的默认画像 |
| 非空 `prompt` | 只覆盖本次请求，不修改默认画像 |

## Web 管理后台

服务启动后访问：

```text
http://服务器IP:18000/
```

管理后台支持：

- Speaker 列表、搜索和状态检查
- WAV、MP3、M4A 等参考音频上传
- 参考音频文本与默认 Prompt 画像配置
- Speaker 注册、更新和删除
- 默认画像与单次覆盖画像推理
- 24kHz WAV 在线试听和下载

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
│   └── web/                           # Web 管理后台
├── models/
│   ├── CosyVoice3Pro/                 # Speaker ID / Raw Prompt 推理
│   └── CosyVoice3ProSpeakerRegistry/  # Speaker 注册与持久化
├── scripts/
│   └── client.py                      # 注册、查询和推理客户端
├── docs/
│   ├── api.md                         # Triton API 与 curl
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

如果 Git 代理仅监听宿主机 `127.0.0.1`，管理脚本会通过 Docker 网关建立
临时转发。

## 兼容性

- 保留上游 `cosyvoice3` Triton 模型，方便旧调用回滚。
- 保留原始 `reference_wav + reference_text + target_text` 推理。
- `instruct_text` 仍可作为 `prompt` 的兼容别名。
- 暂时关闭 Gateway 时，Triton 可直接监听外部 `18000`：

```bash
COSYVOICE_WEB_GATEWAY_ENABLED=false bash manage.sh restart
```

恢复 Gateway：

```bash
bash manage.sh restart
```

## 安全提示

Web 管理后台和 Triton API 默认不包含应用层登录认证。部署到生产环境时，
建议在外层负载均衡、反向代理或防火墙中配置：

- TLS
- 身份认证与访问控制
- 来源 IP 限制
- 请求体大小和速率限制
- Speaker 数据目录的独立备份

## 文档

- [Triton Speaker Registry 与推理 API](docs/api.md)
- [Web Gateway 部署与运维](docs/web-admin.md)

## 致谢与上游

CosyVoice3Pro 基于
[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)、
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
与 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 构建。

模型权重、基础镜像及上游组件分别受其原始许可证和使用条款约束。
