#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###################################
# 基础配置
###################################

CONTAINER_NAME="cosyvoice-server"
IMAGE_NAME="docker.1panel.live/soar97/triton-cosyvoice:25.06"

GPU_ID="${COSYVOICE_GPU_ID:-3}"

HOST_HTTP_PORT=18000
HOST_GRPC_PORT=18001
HOST_METRICS_PORT=18002
TRITON_INTERNAL_HTTP_PORT=18100
WEB_GATEWAY_ENABLED="${COSYVOICE_WEB_GATEWAY_ENABLED:-true}"

WORKSPACE_DIR="/workspace"
COSYVOICE_DIR="${WORKSPACE_DIR}/CosyVoice"
TRITON_DIR="${COSYVOICE_DIR}/runtime/triton_trtllm"
TRITON_MODEL_OVERRIDES_DIR="${SCRIPT_DIR}/models"
WEB_GATEWAY_SOURCE_DIR="${SCRIPT_DIR}/gateway"
CONTAINER_WEB_GATEWAY_DIR="/workspace/CosyVoice3Pro/gateway"

HOST_SPEAKER_STORE_DIR="${COSYVOICE_SPEAKER_STORE_DIR:-${SCRIPT_DIR}/data/speakers}"
CONTAINER_SPEAKER_STORE_DIR="/workspace/cosyvoice_speaker_store"

LOG_FILE="/tmp/cosyvoice_triton.log"
GATEWAY_LOG_FILE="/tmp/cosyvoice_gateway.log"

DECOUPLED_MODE="False"

# Performance profile. "auto" keeps the conservative single acoustic-model
# instance on smaller GPUs and selects the measured streaming-first profile on
# 80 GB GPUs. Use "throughput" explicitly for offline-heavy workloads.
PERFORMANCE_PROFILE="${COSYVOICE_PERFORMANCE_PROFILE:-auto}"
PERFORMANCE_CONFIG_RESOLVED="false"
KV_CACHE_FREE_GPU_MEMORY_FRACTION=""
PRO_BLS_INSTANCE_COUNT=""
STREAMING_BLS_INSTANCE_COUNT=""
LEGACY_BLS_INSTANCE_COUNT=""
TOKEN2WAV_INSTANCE_COUNT=""
VOCODER_INSTANCE_COUNT=""
INFERENCE_CONCURRENCY=""
SEGMENT_CONCURRENCY=""
STREAMING_CONCURRENCY=""
STREAMING_TIMEOUT_SECONDS="${COSYVOICE_TTS_STREAM_TIMEOUT_SECONDS:-300}"
STREAMING_QUEUE_TIMEOUT_SECONDS="${COSYVOICE_TTS_STREAM_QUEUE_TIMEOUT_SECONDS:-15}"
STREAMING_FIRST_CHUNK_TOKENS=""
STREAMING_CHUNK_GROWTH_OFFSET=""
EAGER_CUDA_INIT=""
FLOW_BATCH_SIZE=""
FLOW_BATCH_QUEUE_DELAY_US=""
FLOW_BATCHING_ENABLED=""
FLOW_PREFERRED_BATCH_SIZES=""
VOCODER_BATCH_SIZE=""
VOCODER_BATCH_QUEUE_DELAY_US=""
VOCODER_BATCHING_ENABLED=""
VOCODER_PREFERRED_BATCH_SIZES=""

# Git 克隆代理。优先使用专用变量，否则沿用当前 shell 的代理配置。
GIT_PROXY_URL="${COSYVOICE_GIT_PROXY:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}"
PROXY_RELAY_PORT="${COSYVOICE_PROXY_RELAY_PORT:-17897}"
STARTUP_TIMEOUT_SECONDS=180
HEALTH_URL="http://127.0.0.1:${HOST_HTTP_PORT}/v2/health/ready"

###################################
# 颜色配置
###################################

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO] $*${NC}"
}

log_ok() {
    echo -e "${GREEN}[OK] $*${NC}"
}

log_warn() {
    echo -e "${YELLOW}[WARN] $*${NC}"
}

log_err() {
    echo -e "${RED}[ERROR] $*${NC}"
}

log_step() {
    echo -e "${CYAN}========== $* ==========${NC}"
}

positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

decimal_fraction() {
    [[ "$1" =~ ^0\.[0-9]+$ ]] &&
        awk -v value="$1" 'BEGIN { exit !(value > 0 && value < 1) }'
}

resolve_performance_config() {
    if [ "${PERFORMANCE_CONFIG_RESOLVED}" = "true" ]; then
        return
    fi

    local resolved_profile="${PERFORMANCE_PROFILE}"
    local gpu_memory_mb=0
    if [ "${resolved_profile}" = "auto" ] && container_running; then
        gpu_memory_mb="$(
            docker exec "${CONTAINER_NAME}" /bin/bash -lc \
                "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1" \
                2>/dev/null | tr -d '[:space:]'
        )"
        if ! positive_integer "${gpu_memory_mb:-0}"; then
            gpu_memory_mb=0
        fi
        if [ "${gpu_memory_mb}" -ge 70000 ]; then
            resolved_profile="streaming"
        else
            resolved_profile="balanced"
        fi
    fi

    case "${resolved_profile}" in
        streaming)
            KV_CACHE_FREE_GPU_MEMORY_FRACTION="${COSYVOICE_KV_CACHE_FRACTION:-0.50}"
            PRO_BLS_INSTANCE_COUNT="${COSYVOICE_PRO_BLS_INSTANCES:-2}"
            STREAMING_BLS_INSTANCE_COUNT="${COSYVOICE_STREAMING_BLS_INSTANCES:-2}"
            LEGACY_BLS_INSTANCE_COUNT="${COSYVOICE_LEGACY_BLS_INSTANCES:-1}"
            TOKEN2WAV_INSTANCE_COUNT="${COSYVOICE_TOKEN2WAV_INSTANCES:-2}"
            VOCODER_INSTANCE_COUNT="${COSYVOICE_VOCODER_INSTANCES:-4}"
            INFERENCE_CONCURRENCY="${COSYVOICE_TTS_INFERENCE_CONCURRENCY:-4}"
            SEGMENT_CONCURRENCY="${COSYVOICE_TTS_SEGMENT_CONCURRENCY:-2}"
            STREAMING_CONCURRENCY="${COSYVOICE_TTS_STREAMING_CONCURRENCY:-16}"
            STREAMING_FIRST_CHUNK_TOKENS="${COSYVOICE_STREAMING_FIRST_CHUNK_TOKENS:-15}"
            STREAMING_CHUNK_GROWTH_OFFSET="${COSYVOICE_STREAMING_CHUNK_GROWTH_OFFSET:-1}"
            EAGER_CUDA_INIT="${COSYVOICE_PRO_EAGER_CUDA_INIT:-true}"
            FLOW_BATCH_SIZE="${COSYVOICE_FLOW_BATCH_SIZE:-${COSYVOICE_ACOUSTIC_BATCH_SIZE:-1}}"
            FLOW_BATCH_QUEUE_DELAY_US="${COSYVOICE_FLOW_BATCH_QUEUE_DELAY_US:-${COSYVOICE_ACOUSTIC_BATCH_QUEUE_DELAY_US:-0}}"
            VOCODER_BATCH_SIZE="${COSYVOICE_VOCODER_BATCH_SIZE:-${COSYVOICE_ACOUSTIC_BATCH_SIZE:-1}}"
            VOCODER_BATCH_QUEUE_DELAY_US="${COSYVOICE_VOCODER_BATCH_QUEUE_DELAY_US:-${COSYVOICE_ACOUSTIC_BATCH_QUEUE_DELAY_US:-0}}"
            ;;
        throughput)
            KV_CACHE_FREE_GPU_MEMORY_FRACTION="${COSYVOICE_KV_CACHE_FRACTION:-0.50}"
            PRO_BLS_INSTANCE_COUNT="${COSYVOICE_PRO_BLS_INSTANCES:-12}"
            STREAMING_BLS_INSTANCE_COUNT="${COSYVOICE_STREAMING_BLS_INSTANCES:-2}"
            LEGACY_BLS_INSTANCE_COUNT="${COSYVOICE_LEGACY_BLS_INSTANCES:-2}"
            TOKEN2WAV_INSTANCE_COUNT="${COSYVOICE_TOKEN2WAV_INSTANCES:-2}"
            VOCODER_INSTANCE_COUNT="${COSYVOICE_VOCODER_INSTANCES:-2}"
            INFERENCE_CONCURRENCY="${COSYVOICE_TTS_INFERENCE_CONCURRENCY:-12}"
            SEGMENT_CONCURRENCY="${COSYVOICE_TTS_SEGMENT_CONCURRENCY:-2}"
            STREAMING_CONCURRENCY="${COSYVOICE_TTS_STREAMING_CONCURRENCY:-10}"
            STREAMING_FIRST_CHUNK_TOKENS="${COSYVOICE_STREAMING_FIRST_CHUNK_TOKENS:-15}"
            STREAMING_CHUNK_GROWTH_OFFSET="${COSYVOICE_STREAMING_CHUNK_GROWTH_OFFSET:-1}"
            EAGER_CUDA_INIT="${COSYVOICE_PRO_EAGER_CUDA_INIT:-true}"
            FLOW_BATCH_SIZE="${COSYVOICE_FLOW_BATCH_SIZE:-${COSYVOICE_ACOUSTIC_BATCH_SIZE:-1}}"
            FLOW_BATCH_QUEUE_DELAY_US="${COSYVOICE_FLOW_BATCH_QUEUE_DELAY_US:-${COSYVOICE_ACOUSTIC_BATCH_QUEUE_DELAY_US:-0}}"
            VOCODER_BATCH_SIZE="${COSYVOICE_VOCODER_BATCH_SIZE:-${COSYVOICE_ACOUSTIC_BATCH_SIZE:-1}}"
            VOCODER_BATCH_QUEUE_DELAY_US="${COSYVOICE_VOCODER_BATCH_QUEUE_DELAY_US:-${COSYVOICE_ACOUSTIC_BATCH_QUEUE_DELAY_US:-0}}"
            ;;
        balanced)
            KV_CACHE_FREE_GPU_MEMORY_FRACTION="${COSYVOICE_KV_CACHE_FRACTION:-0.60}"
            PRO_BLS_INSTANCE_COUNT="${COSYVOICE_PRO_BLS_INSTANCES:-10}"
            STREAMING_BLS_INSTANCE_COUNT="${COSYVOICE_STREAMING_BLS_INSTANCES:-2}"
            LEGACY_BLS_INSTANCE_COUNT="${COSYVOICE_LEGACY_BLS_INSTANCES:-2}"
            TOKEN2WAV_INSTANCE_COUNT="${COSYVOICE_TOKEN2WAV_INSTANCES:-1}"
            VOCODER_INSTANCE_COUNT="${COSYVOICE_VOCODER_INSTANCES:-1}"
            INFERENCE_CONCURRENCY="${COSYVOICE_TTS_INFERENCE_CONCURRENCY:-10}"
            SEGMENT_CONCURRENCY="${COSYVOICE_TTS_SEGMENT_CONCURRENCY:-2}"
            STREAMING_CONCURRENCY="${COSYVOICE_TTS_STREAMING_CONCURRENCY:-4}"
            STREAMING_FIRST_CHUNK_TOKENS="${COSYVOICE_STREAMING_FIRST_CHUNK_TOKENS:-15}"
            STREAMING_CHUNK_GROWTH_OFFSET="${COSYVOICE_STREAMING_CHUNK_GROWTH_OFFSET:-0}"
            EAGER_CUDA_INIT="${COSYVOICE_PRO_EAGER_CUDA_INIT:-false}"
            FLOW_BATCH_SIZE="${COSYVOICE_FLOW_BATCH_SIZE:-${COSYVOICE_ACOUSTIC_BATCH_SIZE:-1}}"
            FLOW_BATCH_QUEUE_DELAY_US="${COSYVOICE_FLOW_BATCH_QUEUE_DELAY_US:-${COSYVOICE_ACOUSTIC_BATCH_QUEUE_DELAY_US:-0}}"
            VOCODER_BATCH_SIZE="${COSYVOICE_VOCODER_BATCH_SIZE:-${COSYVOICE_ACOUSTIC_BATCH_SIZE:-1}}"
            VOCODER_BATCH_QUEUE_DELAY_US="${COSYVOICE_VOCODER_BATCH_QUEUE_DELAY_US:-${COSYVOICE_ACOUSTIC_BATCH_QUEUE_DELAY_US:-0}}"
            ;;
        *)
            log_err "COSYVOICE_PERFORMANCE_PROFILE 仅支持 auto、balanced、throughput、streaming"
            exit 1
            ;;
    esac

    if ! decimal_fraction "${KV_CACHE_FREE_GPU_MEMORY_FRACTION}"; then
        log_err "COSYVOICE_KV_CACHE_FRACTION 必须是 0 到 1 之间的小数"
        exit 1
    fi
    local value
    for value in \
        "${PRO_BLS_INSTANCE_COUNT}" \
        "${STREAMING_BLS_INSTANCE_COUNT}" \
        "${LEGACY_BLS_INSTANCE_COUNT}" \
        "${TOKEN2WAV_INSTANCE_COUNT}" \
        "${VOCODER_INSTANCE_COUNT}" \
        "${INFERENCE_CONCURRENCY}" \
        "${SEGMENT_CONCURRENCY}" \
        "${STREAMING_CONCURRENCY}" \
        "${STREAMING_TIMEOUT_SECONDS}" \
        "${STREAMING_QUEUE_TIMEOUT_SECONDS}"; do
        if ! positive_integer "${value}"; then
            log_err "性能实例数和并发数必须是正整数"
            exit 1
        fi
    done
    if [[ "${STREAMING_CHUNK_GROWTH_OFFSET}" != "0" &&
          "${STREAMING_CHUNK_GROWTH_OFFSET}" != "1" ]]; then
        log_err "COSYVOICE_STREAMING_CHUNK_GROWTH_OFFSET 仅支持 0 或 1"
        exit 1
    fi
    if ! positive_integer "${STREAMING_FIRST_CHUNK_TOKENS}" ||
       [ "${STREAMING_FIRST_CHUNK_TOKENS}" -lt 5 ] ||
       [ "${STREAMING_FIRST_CHUNK_TOKENS}" -gt 25 ]; then
        log_err "COSYVOICE_STREAMING_FIRST_CHUNK_TOKENS 必须是 5 到 25 的整数"
        exit 1
    fi

    case "${FLOW_BATCH_SIZE}" in
        1)
            FLOW_BATCHING_ENABLED="false"
            FLOW_PREFERRED_BATCH_SIZES="1"
            ;;
        2)
            FLOW_BATCHING_ENABLED="true"
            FLOW_PREFERRED_BATCH_SIZES="2"
            ;;
        4)
            FLOW_BATCHING_ENABLED="true"
            FLOW_PREFERRED_BATCH_SIZES="2, 4"
            ;;
        8)
            FLOW_BATCHING_ENABLED="true"
            FLOW_PREFERRED_BATCH_SIZES="2, 4, 8"
            ;;
        *)
            log_err "COSYVOICE_FLOW_BATCH_SIZE 仅支持 1、2、4、8"
            exit 1
            ;;
    esac
    case "${VOCODER_BATCH_SIZE}" in
        1)
            VOCODER_BATCHING_ENABLED="false"
            VOCODER_PREFERRED_BATCH_SIZES="1"
            ;;
        2)
            VOCODER_BATCHING_ENABLED="true"
            VOCODER_PREFERRED_BATCH_SIZES="2"
            ;;
        4)
            VOCODER_BATCHING_ENABLED="true"
            VOCODER_PREFERRED_BATCH_SIZES="2, 4"
            ;;
        8)
            VOCODER_BATCHING_ENABLED="true"
            VOCODER_PREFERRED_BATCH_SIZES="2, 4, 8"
            ;;
        *)
            log_err "COSYVOICE_VOCODER_BATCH_SIZE 仅支持 1、2、4、8"
            exit 1
            ;;
    esac
    if ! [[ "${FLOW_BATCH_QUEUE_DELAY_US}" =~ ^[0-9]+$ ]] ||
       ! [[ "${VOCODER_BATCH_QUEUE_DELAY_US}" =~ ^[0-9]+$ ]]; then
        log_err "Flow/Vocoder Batch queue delay 必须是非负整数"
        exit 1
    fi
    if [ "${FLOW_BATCH_SIZE}" -eq 1 ]; then
        FLOW_BATCH_QUEUE_DELAY_US=0
    fi
    if [ "${VOCODER_BATCH_SIZE}" -eq 1 ]; then
        VOCODER_BATCH_QUEUE_DELAY_US=0
    fi

    PERFORMANCE_PROFILE="${resolved_profile}"
    case "${EAGER_CUDA_INIT,,}" in
        true|false)
            EAGER_CUDA_INIT="${EAGER_CUDA_INIT,,}"
            ;;
        *)
            log_err "COSYVOICE_PRO_EAGER_CUDA_INIT 仅支持 true 或 false"
            exit 1
            ;;
    esac
    PERFORMANCE_CONFIG_RESOLVED="true"
}

###################################
# 工具函数
###################################

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"
}

container_running() {
    docker ps --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"
}

container_gpu_healthy() {
    docker exec "${CONTAINER_NAME}" /bin/bash -lc "nvidia-smi >/dev/null 2>&1"
}

service_process_running() {
    # Bracket the first character so pgrep does not match its own command line.
    if [ "${WEB_GATEWAY_ENABLED}" = "true" ]; then
        container_running && docker exec "${CONTAINER_NAME}" /bin/bash -lc \
            "pgrep -f '[t]ritonserver --model-repository' >/dev/null 2>&1 && \
             pgrep -f '[t]rtllm-serve serve' >/dev/null 2>&1 && \
             pgrep -f '[u]vicorn app:app' >/dev/null 2>&1"
    else
        container_running && docker exec "${CONTAINER_NAME}" /bin/bash -lc \
            "pgrep -f '[t]ritonserver --model-repository' >/dev/null 2>&1 && \
             pgrep -f '[t]rtllm-serve serve' >/dev/null 2>&1"
    fi
}

health_ready() {
    curl -fsS "${HEALTH_URL}" >/dev/null 2>&1
}

ensure_container_exists() {
    if ! container_exists; then
        log_err "容器不存在：${CONTAINER_NAME}"
        log_err "请先执行：$0 install"
        exit 1
    fi
}

ensure_container_running() {
    ensure_container_exists

    if ! container_running; then
        log_warn "容器未运行，正在启动：${CONTAINER_NAME}"
        docker start "${CONTAINER_NAME}" >/dev/null
        log_ok "容器已启动"
    fi
}

ensure_container_gpu_healthy() {
    ensure_container_running

    if container_gpu_healthy; then
        log_ok "容器 GPU/NVML 正常"
        return
    fi

    log_warn "容器内 GPU/NVML 不可用，尝试重启容器刷新 NVIDIA runtime"
    docker restart "${CONTAINER_NAME}" >/dev/null
    sleep 2

    if ! container_gpu_healthy; then
        log_err "容器重启后 GPU/NVML 仍不可用"
        log_err "请检查宿主机 NVIDIA driver / nvidia-container-runtime"
        docker exec "${CONTAINER_NAME}" /bin/bash -lc "nvidia-smi" || true
        exit 1
    fi

    log_ok "容器 GPU/NVML 已恢复"
}

backup_speaker_store() {
    if ! container_exists; then
        log_warn "容器不存在，跳过 Speaker 数据备份"
        return
    fi

    mkdir -p "${HOST_SPEAKER_STORE_DIR}"
    if docker cp \
        "${CONTAINER_NAME}:${CONTAINER_SPEAKER_STORE_DIR}/." \
        "${HOST_SPEAKER_STORE_DIR}/"; then
        log_ok "Speaker 数据已备份：${HOST_SPEAKER_STORE_DIR}"
    else
        log_warn "容器中没有可备份的 Speaker 数据"
    fi
}

exec_in_container() {
    docker exec "${CONTAINER_NAME}" /bin/bash -c "$1"
}

CONTAINER_GIT_PROXY=""
PROXY_RELAY_PID=""

stop_proxy_relay() {
    if [ -n "${PROXY_RELAY_PID}" ] && kill -0 "${PROXY_RELAY_PID}" 2>/dev/null; then
        kill "${PROXY_RELAY_PID}" 2>/dev/null || true
        wait "${PROXY_RELAY_PID}" 2>/dev/null || true
    fi

    PROXY_RELAY_PID=""
}

prepare_git_proxy() {
    CONTAINER_GIT_PROXY=""

    if [ -z "${GIT_PROXY_URL}" ]; then
        log_info "未配置 Git 代理，使用直连"
        return
    fi

    # 容器中的 127.0.0.1 指向容器自身。若代理只监听宿主机回环地址，
    # 临时在当前容器的 Docker 网关上建立 TCP 转发。
    if [[ "${GIT_PROXY_URL}" =~ ^(https?://)([^/@]+@)?(127\.0\.0\.1|localhost):([0-9]+)(/.*)?$ ]]; then
        local proxy_scheme="${BASH_REMATCH[1]}"
        local proxy_auth="${BASH_REMATCH[2]}"
        local proxy_port="${BASH_REMATCH[4]}"
        local proxy_path="${BASH_REMATCH[5]}"
        local docker_gateway

        if ! command -v ncat >/dev/null 2>&1; then
            log_err "代理监听在本机回环地址，但未安装 ncat，容器无法访问该代理"
            log_err "请安装 ncat，或将 COSYVOICE_GIT_PROXY 设置为容器可访问的代理地址"
            return 1
        fi

        if ! [[ "${PROXY_RELAY_PORT}" =~ ^[0-9]+$ ]] ||
            [ "${PROXY_RELAY_PORT}" -lt 1 ] ||
            [ "${PROXY_RELAY_PORT}" -gt 65535 ]; then
            log_err "无效的代理转发端口：${PROXY_RELAY_PORT}"
            return 1
        fi

        docker_gateway="$(docker inspect \
            --format '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}' \
            "${CONTAINER_NAME}")"

        if [ -z "${docker_gateway}" ]; then
            log_err "无法获取容器的 Docker 网关地址"
            return 1
        fi

        ncat -l "${docker_gateway}" "${PROXY_RELAY_PORT}" \
            --keep-open \
            --sh-exec "ncat 127.0.0.1 ${proxy_port}" \
            >/tmp/cosyvoice_git_proxy_relay.log 2>&1 &
        PROXY_RELAY_PID=$!

        sleep 1
        if ! kill -0 "${PROXY_RELAY_PID}" 2>/dev/null; then
            log_err "代理转发启动失败，详情见 /tmp/cosyvoice_git_proxy_relay.log"
            PROXY_RELAY_PID=""
            return 1
        fi

        CONTAINER_GIT_PROXY="${proxy_scheme}${proxy_auth}${docker_gateway}:${PROXY_RELAY_PORT}${proxy_path}"
        log_ok "已为容器建立临时 Git 代理转发"
    else
        CONTAINER_GIT_PROXY="${GIT_PROXY_URL}"
        log_info "使用 COSYVOICE_GIT_PROXY/系统代理克隆仓库"
    fi
}

wait_for_ready() {
    log_info "等待 Triton ready，超时 ${STARTUP_TIMEOUT_SECONDS}s"

    for _ in $(seq 1 "${STARTUP_TIMEOUT_SECONDS}"); do
        if health_ready; then
            log_ok "Triton 已 ready：${HEALTH_URL}"
            return
        fi

        if ! service_process_running; then
            log_err "服务进程已退出，最近日志如下："
            docker exec "${CONTAINER_NAME}" /bin/bash -lc "tail -n 120 ${LOG_FILE}" || true
            exit 1
        fi

        sleep 1
    done

    log_err "等待 Triton ready 超时，最近日志如下："
    docker exec "${CONTAINER_NAME}" /bin/bash -lc "tail -n 120 ${LOG_FILE}" || true
    exit 1
}

###################################
# 安装步骤
###################################

install_create_container() {
    log_step "创建容器"

    if container_exists; then
        backup_speaker_store
    fi
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    mkdir -p "${HOST_SPEAKER_STORE_DIR}"

    docker run -dit \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        --gpus "device=${GPU_ID}" \
        --ipc=host \
        --shm-size=8g \
        -p "${HOST_HTTP_PORT}:18000" \
        -p "${HOST_GRPC_PORT}:18001" \
        -p "${HOST_METRICS_PORT}:18002" \
        -v "${HOST_SPEAKER_STORE_DIR}:${CONTAINER_SPEAKER_STORE_DIR}" \
        "${IMAGE_NAME}" \
        /bin/bash

    log_ok "容器创建完成：${CONTAINER_NAME}"
}

install_clone_repo() {
    log_step "克隆 CosyVoice 仓库"

    prepare_git_proxy
    trap stop_proxy_relay EXIT

    local proxy_args=()
    if [ -n "${CONTAINER_GIT_PROXY}" ]; then
        proxy_args=(
            -e "HTTP_PROXY=${CONTAINER_GIT_PROXY}"
            -e "HTTPS_PROXY=${CONTAINER_GIT_PROXY}"
            -e "http_proxy=${CONTAINER_GIT_PROXY}"
            -e "https_proxy=${CONTAINER_GIT_PROXY}"
        )
    fi

    local clone_status=0
    docker exec "${proxy_args[@]}" "${CONTAINER_NAME}" /bin/bash -c "
set -e

export GIT_TERMINAL_PROMPT=0

cd ${WORKSPACE_DIR}

if [ -d CosyVoice/.git ] && git -C CosyVoice rev-parse --verify HEAD >/dev/null 2>&1; then
    echo 'CosyVoice already cloned, skip clone'
else
    if [ -e CosyVoice ]; then
        echo 'Removing incomplete CosyVoice checkout'
        rm -rf CosyVoice
    fi

    git -c http.version=HTTP/1.1 clone --progress \
        https://github.com/FunAudioLLM/CosyVoice.git
fi

cd ${COSYVOICE_DIR}
git -c http.version=HTTP/1.1 submodule update --init --recursive --progress
" || clone_status=$?

    stop_proxy_relay
    trap - EXIT

    if [ "${clone_status}" -ne 0 ]; then
        log_err "CosyVoice 仓库克隆失败"
        return "${clone_status}"
    fi

    log_ok "仓库准备完成"
}

install_modify_script() {
    log_step "修改 run_cosyvoice3.sh 配置"
    resolve_performance_config

    local triton_http_port=18000
    if [ "${WEB_GATEWAY_ENABLED}" = "true" ]; then
        triton_http_port="${TRITON_INTERNAL_HTTP_PORT}"
    fi

    exec_in_container "
set -e

cd ${TRITON_DIR}

if [ ! -f run_cosyvoice3.sh ]; then
    echo 'run_cosyvoice3.sh not found'
    exit 1
fi

sed -i -E 's/DECOUPLED_MODE=(True|False)/DECOUPLED_MODE=${DECOUPLED_MODE}/g' run_cosyvoice3.sh

sed -i -E 's/--kv_cache_free_gpu_memory_fraction[[:space:]]+[0-9.]+/--kv_cache_free_gpu_memory_fraction ${KV_CACHE_FREE_GPU_MEMORY_FRACTION}/g' run_cosyvoice3.sh

sed -i -E 's/--http-port[[:space:]]+[0-9]+/--http-port ${triton_http_port}/g' run_cosyvoice3.sh

echo 'current config:'
grep -n 'DECOUPLED_MODE=' run_cosyvoice3.sh || true
grep -n 'kv_cache_free_gpu_memory_fraction' run_cosyvoice3.sh || true
grep -n 'tritonserver --model-repository' run_cosyvoice3.sh || true
"

    log_ok "脚本配置修改完成"
}

install_compile_triton_model() {
    log_step "编译 Triton 模型"

    exec_in_container "
set -e

cd ${TRITON_DIR}

export HF_ENDPOINT=https://hf-mirror.com

bash run_cosyvoice3.sh 0 2
"

    log_ok "Triton 模型编译完成"
}

install_model_overrides() {
    log_step "部署 CosyVoice3Pro 和声学 Batch 模型"
    resolve_performance_config

    local cosyvoice_model_dir="${TRITON_MODEL_OVERRIDES_DIR}/CosyVoice3Pro"
    local streaming_model_dir="${TRITON_MODEL_OVERRIDES_DIR}/CosyVoice3ProStreaming"
    local registry_model_dir="${TRITON_MODEL_OVERRIDES_DIR}/CosyVoice3ProSpeakerRegistry"
    local token2wav_model_dir="${TRITON_MODEL_OVERRIDES_DIR}/token2wav"
    local vocoder_model_dir="${TRITON_MODEL_OVERRIDES_DIR}/vocoder"

    if [ ! -f "${cosyvoice_model_dir}/config.pbtxt" ] ||
       [ ! -f "${cosyvoice_model_dir}/1/model.py" ] ||
       [ ! -f "${streaming_model_dir}/config.pbtxt" ] ||
       [ ! -f "${streaming_model_dir}/1/model.py" ] ||
       [ ! -f "${registry_model_dir}/config.pbtxt" ] ||
       [ ! -f "${registry_model_dir}/1/model.py" ] ||
       [ ! -f "${token2wav_model_dir}/config.pbtxt" ] ||
       [ ! -f "${token2wav_model_dir}/1/model.py" ] ||
       [ ! -f "${vocoder_model_dir}/config.pbtxt" ] ||
       [ ! -f "${vocoder_model_dir}/1/model.py" ]; then
        log_err "Triton 模型覆盖文件不完整：${TRITON_MODEL_OVERRIDES_DIR}"
        exit 1
    fi

    docker exec "${CONTAINER_NAME}" /bin/bash -lc "
set -e
mkdir -p \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3Pro/1' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3ProStreaming/1' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3ProSpeakerRegistry/1' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/token2wav/1' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/vocoder/1' \
  '${CONTAINER_SPEAKER_STORE_DIR}'
"

    docker cp \
        "${cosyvoice_model_dir}/." \
        "${CONTAINER_NAME}:${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3Pro/"
    docker cp \
        "${streaming_model_dir}/." \
        "${CONTAINER_NAME}:${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3ProStreaming/"
    docker cp \
        "${registry_model_dir}/." \
        "${CONTAINER_NAME}:${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3ProSpeakerRegistry/"
    docker cp \
        "${token2wav_model_dir}/." \
        "${CONTAINER_NAME}:${TRITON_DIR}/model_repo_cosyvoice3_copy/token2wav/"
    docker cp \
        "${vocoder_model_dir}/." \
        "${CONTAINER_NAME}:${TRITON_DIR}/model_repo_cosyvoice3_copy/vocoder/"

    exec_in_container "
set -e
sed -i -E '0,/count:[[:space:]]*[0-9]+/s//count: ${PRO_BLS_INSTANCE_COUNT}/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3Pro/config.pbtxt'
sed -i -E '0,/count:[[:space:]]*[0-9]+/s//count: ${STREAMING_BLS_INSTANCE_COUNT}/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3ProStreaming/config.pbtxt'
sed -i -E '/key:[[:space:]]*\"streaming_chunk_growth_offset\"/,/}/s/string_value:[[:space:]]*\"[01]\"/string_value: \"${STREAMING_CHUNK_GROWTH_OFFSET}\"/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3ProStreaming/config.pbtxt'
sed -i -E '/key:[[:space:]]*\"streaming_first_chunk_tokens\"/,/}/s/string_value:[[:space:]]*\"[0-9]+\"/string_value: \"${STREAMING_FIRST_CHUNK_TOKENS}\"/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3ProStreaming/config.pbtxt'
sed -i -E '/key:[[:space:]]*\"eager_cuda_init\"/,/}/s/string_value:[[:space:]]*\"(true|false)\"/string_value: \"${EAGER_CUDA_INIT}\"/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3Pro/config.pbtxt'
sed -i -E '/key:[[:space:]]*\"eager_cuda_init\"/,/}/s/string_value:[[:space:]]*\"(true|false)\"/string_value: \"${EAGER_CUDA_INIT}\"/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3ProStreaming/config.pbtxt'
sed -i -E '/key:[[:space:]]*\"flow_batching_enabled\"/,/}/s/string_value:[[:space:]]*\"(true|false)\"/string_value: \"${FLOW_BATCHING_ENABLED}\"/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3Pro/config.pbtxt'
sed -i -E '/key:[[:space:]]*\"vocoder_batching_enabled\"/,/}/s/string_value:[[:space:]]*\"(true|false)\"/string_value: \"${VOCODER_BATCHING_ENABLED}\"/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3Pro/config.pbtxt'
sed -i -E '0,/count:[[:space:]]*[0-9]+/s//count: ${LEGACY_BLS_INSTANCE_COUNT}/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/cosyvoice3/config.pbtxt'
token_config='${TRITON_DIR}/model_repo_cosyvoice3_copy/token2wav/config.pbtxt'
sed -i -E 's/max_batch_size:[[:space:]]*[0-9]+/max_batch_size: ${FLOW_BATCH_SIZE}/' \"\${token_config}\"
sed -i -E 's/preferred_batch_size:[[:space:]]*\\[[^]]*\\]/preferred_batch_size: [${FLOW_PREFERRED_BATCH_SIZES}]/' \"\${token_config}\"
sed -i -E 's/max_queue_delay_microseconds:[[:space:]]*[0-9]+/max_queue_delay_microseconds: ${FLOW_BATCH_QUEUE_DELAY_US}/' \"\${token_config}\"
vocoder_config='${TRITON_DIR}/model_repo_cosyvoice3_copy/vocoder/config.pbtxt'
sed -i -E 's/max_batch_size:[[:space:]]*[0-9]+/max_batch_size: ${VOCODER_BATCH_SIZE}/' \"\${vocoder_config}\"
sed -i -E 's/preferred_batch_size:[[:space:]]*\\[[^]]*\\]/preferred_batch_size: [${VOCODER_PREFERRED_BATCH_SIZES}]/' \"\${vocoder_config}\"
sed -i -E 's/max_queue_delay_microseconds:[[:space:]]*[0-9]+/max_queue_delay_microseconds: ${VOCODER_BATCH_QUEUE_DELAY_US}/' \"\${vocoder_config}\"
sed -i -E 's/flow\\.decoder\\.estimator\\.autocast_fp16\\.dynamic_batch\\.[0-9]+\\.plan/flow.decoder.estimator.autocast_fp16.dynamic_batch.${FLOW_BATCH_SIZE}.plan/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/token2wav/config.pbtxt'
sed -i -E '0,/count:[[:space:]]*[0-9]+/s//count: ${TOKEN2WAV_INSTANCE_COUNT}/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/token2wav/config.pbtxt'
sed -i -E '0,/count:[[:space:]]*[0-9]+/s//count: ${VOCODER_INSTANCE_COUNT}/' \
  '${TRITON_DIR}/model_repo_cosyvoice3_copy/vocoder/config.pbtxt'
"

    log_ok "模型覆盖文件部署完成"
    log_info "实例配置：Pro BLS=${PRO_BLS_INSTANCE_COUNT}，Streaming BLS=${STREAMING_BLS_INSTANCE_COUNT}，Legacy BLS=${LEGACY_BLS_INSTANCE_COUNT}，token2wav=${TOKEN2WAV_INSTANCE_COUNT}，vocoder=${VOCODER_INSTANCE_COUNT}，eager CUDA=${EAGER_CUDA_INIT}"
    log_info "流式首块=${STREAMING_FIRST_CHUNK_TOKENS} tokens，后续块增长偏移=${STREAMING_CHUNK_GROWTH_OFFSET}"
    log_info "Flow Batch：max=${FLOW_BATCH_SIZE}，preferred=[${FLOW_PREFERRED_BATCH_SIZES}]，queue=${FLOW_BATCH_QUEUE_DELAY_US}us"
    log_info "Vocoder Batch：max=${VOCODER_BATCH_SIZE}，preferred=[${VOCODER_PREFERRED_BATCH_SIZES}]，queue=${VOCODER_BATCH_QUEUE_DELAY_US}us"
    local mounted_store
    mounted_store="$(docker inspect "${CONTAINER_NAME}" --format \
        '{{range .Mounts}}{{if eq .Destination "/workspace/cosyvoice_speaker_store"}}{{.Source}}{{end}}{{end}}')"
    if [ -n "${mounted_store}" ]; then
        log_info "Speaker 持久化目录：${mounted_store}"
    else
        log_warn "当前容器未挂载 Speaker 目录；数据可跨重启保留，但删除容器后会丢失"
        log_info "容器内 Speaker 目录：${CONTAINER_SPEAKER_STORE_DIR}"
    fi
}

prepare_acoustic_batch_assets() {
    resolve_performance_config

    if [ "${FLOW_BATCH_SIZE}" -le 1 ]; then
        log_info "Flow Batch=1，跳过动态 Flow engine 构建"
        return
    fi

    local prepare_script="${SCRIPT_DIR}/scripts/prepare_flow_batch.py"
    if [ ! -f "${prepare_script}" ]; then
        log_err "缺少动态 Flow 构建脚本：${prepare_script}"
        exit 1
    fi

    log_step "准备动态 Batch Flow TensorRT engine"
    docker exec "${CONTAINER_NAME}" /bin/bash -lc \
        "mkdir -p '${CONTAINER_WEB_GATEWAY_DIR}/scripts'"
    docker cp \
        "${prepare_script}" \
        "${CONTAINER_NAME}:${CONTAINER_WEB_GATEWAY_DIR}/scripts/prepare_flow_batch.py"

    local opt_batch_size="${FLOW_BATCH_SIZE}"
    if [ "${opt_batch_size}" -gt 4 ]; then
        opt_batch_size=4
    fi
    exec_in_container "
set -e
cd '${COSYVOICE_DIR}'
export PYTHONPATH='${COSYVOICE_DIR}/third_party/Matcha-TTS':\${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=0 python3 \
  '${CONTAINER_WEB_GATEWAY_DIR}/scripts/prepare_flow_batch.py' \
  --model-dir '${TRITON_DIR}/Fun-CosyVoice3-0.5B-2512' \
  --max-batch-size '${FLOW_BATCH_SIZE}' \
  --opt-batch-size '${opt_batch_size}'
"
    log_ok "动态 Batch Flow engine 已就绪"
}

install_web_gateway() {
    log_step "部署 CosyVoice3Pro Web Gateway"

    if [ ! -f "${WEB_GATEWAY_SOURCE_DIR}/app.py" ] ||
       [ ! -f "${WEB_GATEWAY_SOURCE_DIR}/legacy_tts.py" ] ||
       [ ! -f "${WEB_GATEWAY_SOURCE_DIR}/streaming_tts.py" ] ||
       [ ! -f "${WEB_GATEWAY_SOURCE_DIR}/speaker_registration.py" ] ||
       [ ! -f "${WEB_GATEWAY_SOURCE_DIR}/tts_utils.py" ] ||
       [ ! -f "${WEB_GATEWAY_SOURCE_DIR}/web/index.html" ] ||
       [ ! -f "${WEB_GATEWAY_SOURCE_DIR}/web/styles.css" ] ||
       [ ! -f "${WEB_GATEWAY_SOURCE_DIR}/web/app.js" ]; then
        log_err "Web Gateway 文件不完整：${WEB_GATEWAY_SOURCE_DIR}"
        exit 1
    fi

    if ! docker exec "${CONTAINER_NAME}" /bin/bash -lc \
        "command -v ffmpeg >/dev/null 2>&1"; then
        log_info "安装 Web Gateway 音频编码依赖 FFmpeg"
        docker exec "${CONTAINER_NAME}" /bin/bash -lc "
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ffmpeg
"
    fi

    docker exec "${CONTAINER_NAME}" /bin/bash -lc "
set -e
mkdir -p '${CONTAINER_WEB_GATEWAY_DIR}/web'
"

    docker cp \
        "${WEB_GATEWAY_SOURCE_DIR}/." \
        "${CONTAINER_NAME}:${CONTAINER_WEB_GATEWAY_DIR}/"

    log_ok "Web Gateway 部署完成"
}

###################################
# 服务管理
###################################

install_service() {
    log_step "安装 CosyVoice Triton 服务"

    log_info "镜像：${IMAGE_NAME}"
    log_info "容器：${CONTAINER_NAME}"
    log_info "GPU：${GPU_ID}"
    log_info "HTTP 端口：${HOST_HTTP_PORT}"
    log_info "gRPC 端口：${HOST_GRPC_PORT}"
    log_info "Metrics 端口：${HOST_METRICS_PORT}"
    log_info "DECOUPLED_MODE：${DECOUPLED_MODE}"
    log_info "WEB_GATEWAY_ENABLED：${WEB_GATEWAY_ENABLED}"

    log_step "拉取镜像"
    docker pull "${IMAGE_NAME}"

    install_create_container
    resolve_performance_config
    log_info "PERFORMANCE_PROFILE：${PERFORMANCE_PROFILE}"
    log_info "KV_CACHE_FREE_GPU_MEMORY_FRACTION：${KV_CACHE_FREE_GPU_MEMORY_FRACTION}"
    install_clone_repo
    install_modify_script
    install_compile_triton_model
    prepare_acoustic_batch_assets
    install_model_overrides
    install_web_gateway

    log_ok "安装完成"
    log_info "启动服务：$0 start"
}

start_service() {

    log_step "启动 CosyVoice Triton 服务"

    ensure_container_running
    ensure_container_gpu_healthy
    resolve_performance_config

    if service_process_running; then
        log_warn "检测到服务进程已存在"
        wait_for_ready
        return
    fi

    install_modify_script
    prepare_acoustic_batch_assets
    install_model_overrides
    if [ "${WEB_GATEWAY_ENABLED}" = "true" ]; then
        install_web_gateway
    fi

    if [ "${WEB_GATEWAY_ENABLED}" = "true" ]; then
        docker exec -d "${CONTAINER_NAME}" /bin/bash -c "
set -e

cd ${TRITON_DIR}

mkdir -p \$(dirname ${LOG_FILE})
touch ${LOG_FILE}
touch ${GATEWAY_LOG_FILE}


nohup bash run_cosyvoice3.sh 3 3 > ${LOG_FILE} 2>&1 &

cd ${CONTAINER_WEB_GATEWAY_DIR}
COSYVOICE_TRITON_UPSTREAM=http://127.0.0.1:${TRITON_INTERNAL_HTTP_PORT} \
COSYVOICE_TRITON_GRPC_UPSTREAM=127.0.0.1:${HOST_GRPC_PORT} \
COSYVOICE_TTS_INFERENCE_CONCURRENCY=${INFERENCE_CONCURRENCY} \
COSYVOICE_TTS_SEGMENT_CONCURRENCY=${SEGMENT_CONCURRENCY} \
COSYVOICE_TTS_STREAMING_CONCURRENCY=${STREAMING_CONCURRENCY} \
COSYVOICE_TTS_STREAM_TIMEOUT_SECONDS=${STREAMING_TIMEOUT_SECONDS} \
COSYVOICE_TTS_STREAM_QUEUE_TIMEOUT_SECONDS=${STREAMING_QUEUE_TIMEOUT_SECONDS} \
  nohup python3 -m uvicorn app:app \
    --host 0.0.0.0 \
    --port 18000 \
    --no-server-header \
    > ${GATEWAY_LOG_FILE} 2>&1 &
"
    else
        docker exec -d "${CONTAINER_NAME}" /bin/bash -c "
set -e

cd ${TRITON_DIR}

mkdir -p \$(dirname ${LOG_FILE})
touch ${LOG_FILE}

nohup bash run_cosyvoice3.sh 3 3 > ${LOG_FILE} 2>&1 &
"
    fi

    sleep 2

    log_ok "服务启动命令已提交"

    log_info "日志文件：${LOG_FILE}"
    log_info "网关日志：${GATEWAY_LOG_FILE}"

    log_info "查看日志："
    echo "    $0 logs"

    log_info "健康检查："
    echo "    ${HEALTH_URL}"

    wait_for_ready
}

stop_service() {
    log_step "停止 CosyVoice Triton 服务"

    ensure_container_running

    exec_in_container "
ps -ef | grep -E 'tritonserver|trtllm|uvicorn app:app|cosyvoice' | grep -v grep | awk '{print \$2}' | xargs -r kill -9
"

    log_ok "服务已停止"
}

restart_service() {
    log_step "重启 CosyVoice Triton 服务"

    stop_service
    start_service

    log_ok "服务已重启"
}

show_logs() {
    ensure_container_running

    log_step "查看服务日志"

    docker exec -it "${CONTAINER_NAME}" /bin/bash -c "
touch ${LOG_FILE}
touch ${GATEWAY_LOG_FILE}
tail -f ${LOG_FILE} ${GATEWAY_LOG_FILE}
"
}

show_status() {
    log_step "容器状态"

    docker ps -a --filter "name=^/${CONTAINER_NAME}$" || true

    echo ""
    log_step "容器 GPU"

    if container_exists && container_running; then
        if ! docker exec "${CONTAINER_NAME}" /bin/bash -lc "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits"; then
            log_warn "容器内 GPU/NVML 不可用，可执行：$0 start 自动尝试修复"
        fi
    else
        log_warn "容器未运行"
    fi

    echo ""
    log_step "生效中的性能配置"

    if container_exists && container_running; then
        docker exec "${CONTAINER_NAME}" /bin/bash -lc "
for model in CosyVoice3Pro CosyVoice3ProStreaming cosyvoice3 token2wav vocoder; do
    config='${TRITON_DIR}/model_repo_cosyvoice3_copy/'\"\${model}\"'/config.pbtxt'
    count=\$(sed -n '/instance_group/,/]/p' \"\${config}\" 2>/dev/null |
        sed -n -E 's/.*count:[[:space:]]*([0-9]+).*/\1/p' | head -n 1)
    max_batch=\$(sed -n -E 's/^max_batch_size:[[:space:]]*([0-9]+).*/\1/p' \"\${config}\" | head -n 1)
    queue_delay=\$(sed -n -E 's/.*max_queue_delay_microseconds:[[:space:]]*([0-9]+).*/\1/p' \"\${config}\" | head -n 1)
    printf '%-30s instances=%s max_batch=%s queue_us=%s\n' \
        \"\${model}\" \"\${count:-unknown}\" \"\${max_batch:-0}\" \"\${queue_delay:--}\"
done
sed -n '/key:[[:space:]]*\"eager_cuda_init\"/,/}/p' \
    '${TRITON_DIR}/model_repo_cosyvoice3_copy/CosyVoice3Pro/config.pbtxt' |
    grep 'string_value' | head -n 1 || true
ps -ef | grep '[t]rtllm-serve serve' | grep -oE -- \
    '--kv_cache_free_gpu_memory_fraction[[:space:]]+[0-9.]+' | head -n 1 || true
gateway_pid=\$(pgrep -f '[u]vicorn app:app' | head -n 1)
if [ -n \"\${gateway_pid}\" ]; then
    tr '\0' '\n' < \"/proc/\${gateway_pid}/environ\" |
        grep -E '^COSYVOICE_(TTS_.*CONCURRENCY|TRITON_GRPC_UPSTREAM)=' | sort || true
fi
"
    else
        log_warn "容器未运行"
    fi

    echo ""
    log_step "服务进程"

    if container_exists; then
        docker exec "${CONTAINER_NAME}" /bin/bash -c "
ps -ef | grep -E 'tritonserver|trtllm|uvicorn app:app' | grep -v grep || true
" || true
    else
        log_warn "容器不存在"
    fi

    echo ""
    log_step "Triton Health"

    if health_ready; then
        log_ok "READY：${HEALTH_URL}"
    else
        log_warn "NOT READY：${HEALTH_URL}"
    fi
}

remove_service() {
    log_step "删除容器"

    backup_speaker_store
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

    log_ok "容器已删除：${CONTAINER_NAME}"
}

show_usage() {
    echo ""
    echo -e "${CYAN}Usage:${NC}"
    echo "  $0 install    安装：拉取镜像、创建容器、克隆仓库、编译模型"
    echo "  $0 start      启动 Triton 服务"
    echo "  $0 stop       停止 Triton 服务"
    echo "  $0 restart    重启 Triton 服务"
    echo "  $0 logs       查看服务日志"
    echo "  $0 status     查看容器、进程、健康状态"
    echo "  $0 backup     备份 Speaker 注册数据到宿主机"
    echo "  $0 remove     删除容器"
    echo ""
}

###################################
# 主入口
###################################

case "$1" in
    install)
        install_service
        ;;

    start)
        start_service
        ;;

    stop)
        stop_service
        ;;

    restart)
        restart_service
        ;;

    logs)
        show_logs
        ;;

    status)
        show_status
        ;;

    backup)
        backup_speaker_store
        ;;

    remove)
        remove_service
        ;;

    *)
        show_usage
        ;;
esac
