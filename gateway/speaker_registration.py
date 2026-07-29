import asyncio
import ipaddress
import json
import logging
import socket
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request

from legacy_tts import (
    MAX_FORM_BYTES,
    MAX_PROMPT_LENGTH,
    _decode_prompt_audio,
    _read_form_data,
    _speaker_id_field,
)


router = APIRouter(tags=["speaker-registration"])
logger = logging.getLogger(__name__)

REGISTRY_MODEL = "CosyVoice3ProSpeakerRegistry"
MAX_AUDIO_URL_LENGTH = 2048
MAX_REFERENCE_TEXT_LENGTH = 4096
MAX_REDIRECTS = 3
DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_registry_semaphore = asyncio.Semaphore(2)


def _aliased_field(fields, primary, alias):
    primary_value = fields.get(primary, "").strip()
    alias_value = fields.get(alias, "").strip()
    if primary_value and alias_value and primary_value != alias_value:
        raise HTTPException(
            status_code=422,
            detail=f"{primary} 与 {alias} 不能设置为不同值",
        )
    return primary_value or alias_value


def _resolve_addresses(hostname, port):
    return {
        item[4][0]
        for item in socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    }


async def _validate_public_audio_url(url):
    if not url or len(url) > MAX_AUDIO_URL_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"audio_url 不能为空且不能超过 {MAX_AUDIO_URL_LENGTH} 个字符",
        )

    try:
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="audio_url 格式非法") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise HTTPException(
            status_code=422,
            detail="audio_url 仅支持 http 或 https",
        )
    if not parsed.hostname:
        raise HTTPException(status_code=422, detail="audio_url 缺少主机名")
    if parsed.username is not None or parsed.password is not None:
        raise HTTPException(
            status_code=422,
            detail="audio_url 不能包含用户名或密码",
        )

    try:
        addresses = await asyncio.to_thread(
            _resolve_addresses,
            parsed.hostname,
            port,
        )
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=422,
            detail="audio_url 主机名无法解析",
        ) from exc

    if not addresses:
        raise HTTPException(
            status_code=422,
            detail="audio_url 主机名没有可用地址",
        )
    for address in addresses:
        try:
            ip_address = ipaddress.ip_address(address)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="audio_url 主机地址非法",
            ) from exc
        if not ip_address.is_global:
            raise HTTPException(
                status_code=422,
                detail="audio_url 不允许访问内网、回环或链路本地地址",
            )
    return parsed


def _validate_connected_peer(response):
    network_stream = response.extensions.get("network_stream")
    server_address = (
        network_stream.get_extra_info("server_addr")
        if network_stream is not None
        else None
    )
    if not server_address:
        raise HTTPException(
            status_code=400,
            detail="audio_url 无法验证远端地址",
        )
    address = server_address[0] if isinstance(
        server_address, tuple) else server_address
    try:
        ip_address = ipaddress.ip_address(address)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="audio_url 远端地址非法",
        ) from exc
    if not ip_address.is_global:
        raise HTTPException(
            status_code=422,
            detail="audio_url 实际连接到了非公网地址",
        )


async def _download_audio(request, audio_url):
    current_url = audio_url
    for redirect_count in range(MAX_REDIRECTS + 1):
        parsed = await _validate_public_audio_url(current_url)
        try:
            async with request.app.state.http_client.stream(
                "GET",
                current_url,
                headers={
                    "Accept": "audio/*,application/octet-stream;q=0.8",
                    "Accept-Encoding": "identity",
                    "User-Agent": "CosyVoice3Pro/1.3 audio-url-register",
                },
                follow_redirects=False,
                timeout=DOWNLOAD_TIMEOUT,
            ) as response:
                _validate_connected_peer(response)
                if response.status_code in REDIRECT_STATUSES:
                    location = response.headers.get("location", "").strip()
                    if not location:
                        raise HTTPException(
                            status_code=400,
                            detail="audio_url 重定向缺少 Location",
                        )
                    if redirect_count >= MAX_REDIRECTS:
                        raise HTTPException(
                            status_code=400,
                            detail=f"audio_url 重定向不能超过 {MAX_REDIRECTS} 次",
                        )
                    current_url = urljoin(current_url, location)
                    continue

                if response.status_code < 200 or response.status_code >= 300:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "audio_url 下载失败，远端返回 HTTP "
                            f"{response.status_code}"
                        ),
                    )

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > MAX_FORM_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail="audio_url 音频不能超过 32 MiB",
                            )
                    except ValueError:
                        pass

                chunks = []
                total_bytes = 0
                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > MAX_FORM_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="audio_url 音频不能超过 32 MiB",
                        )
                    chunks.append(chunk)
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            logger.warning(
                "audio URL download failed host=%s error=%s",
                parsed.hostname,
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=400,
                detail="audio_url 下载失败，无法连接远程音频",
            ) from exc

        audio_bytes = b"".join(chunks)
        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="audio_url 返回了空内容",
            )
        return audio_bytes

    raise HTTPException(status_code=400, detail="audio_url 下载失败")


def _uploaded_audio(files):
    audio = files.get("audio")
    prompt_audio = files.get("prompt_audio")
    if audio is not None and prompt_audio is not None:
        raise HTTPException(
            status_code=422,
            detail="audio 与 prompt_audio 只能上传一个",
        )
    return audio or prompt_audio


def _registry_payload(speaker_id, waveform, reference_text, prompt):
    sample_count = int(waveform.size)
    return {
        "inputs": [
            {
                "name": "operation",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": ["register"],
            },
            {
                "name": "speaker_id",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [speaker_id],
            },
            {
                "name": "reference_wav",
                "shape": [1, sample_count],
                "datatype": "FP32",
                "data": [waveform.tolist()],
            },
            {
                "name": "reference_wav_len",
                "shape": [1, 1],
                "datatype": "INT32",
                "data": [[sample_count]],
            },
            {
                "name": "reference_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [reference_text],
            },
            {
                "name": "prompt",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [prompt],
            },
        ]
    }


def _registry_operation_payload(operation, speaker_id=""):
    inputs = [
        {
            "name": "operation",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [operation],
        },
    ]
    if speaker_id:
        inputs.append(
            {
                "name": "speaker_id",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [speaker_id],
            }
        )
    return {"inputs": inputs}


def _first_output(outputs, name):
    values = outputs.get(name) or []
    return values[0] if values else ""


async def _call_registry(request, payload, timeout=30):
    upstream = request.app.state.triton_upstream
    url = f"{upstream}/v2/models/{REGISTRY_MODEL}/infer"
    try:
        async with _registry_semaphore:
            response = await request.app.state.http_client.post(
                url,
                params={"request_id": f"registry-{uuid4().hex}"},
                json=payload,
                timeout=timeout,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="Speaker Registry 服务不可用",
        ) from exc

    if response.status_code != 200:
        try:
            upstream_error = response.json().get("error", response.text)
        except (ValueError, AttributeError):
            upstream_error = response.text
        status_code = (
            503
            if "No CUDA GPUs are available" in upstream_error
            else 502
        )
        raise HTTPException(
            status_code=status_code,
            detail=f"Speaker Registry 请求失败：{upstream_error}",
        )

    try:
        outputs = {
            output["name"]: output.get("data", [])
            for output in response.json().get("outputs", [])
        }
        status = str(_first_output(outputs, "status"))
        speaker_version = str(_first_output(outputs, "speaker_version"))
        message_value = _first_output(outputs, "message")
        message = json.loads(message_value) if message_value else {}
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Speaker Registry 返回了非法响应",
        ) from exc

    if not isinstance(message, dict):
        raise HTTPException(
            status_code=502,
            detail="Speaker Registry 返回了非法 message",
        )
    return status, speaker_version, message


async def _register_with_triton(
    request,
    speaker_id,
    waveform,
    reference_text,
    prompt,
):
    status, speaker_version, metadata = await _call_registry(
        request,
        _registry_payload(
            speaker_id,
            waveform,
            reference_text,
            prompt,
        ),
        timeout=180,
    )
    if status != "ok":
        raise HTTPException(
            status_code=502,
            detail="Speaker Registry 未确认注册成功",
        )
    return speaker_version, metadata


def _public_speaker(metadata):
    return {
        "speakerId": metadata.get("speaker_id", ""),
        "speakerVersion": metadata.get("speaker_version", ""),
        "referenceText": metadata.get("reference_transcript", ""),
        "prompt": metadata.get("prompt", ""),
        "sampleRate": metadata.get("sample_rate"),
        "samples": metadata.get("samples"),
        "durationSeconds": metadata.get("duration_seconds"),
        "registeredAt": metadata.get("registered_at"),
    }


def _path_speaker_id(speaker_id):
    result = _speaker_id_field({"speakerId": speaker_id})
    if not result:
        raise HTTPException(status_code=400, detail="speakerId 不能为空")
    return result


@router.post("/register")
async def register_speaker(request: Request):
    fields, files = await _read_form_data(request)
    speaker_id = _speaker_id_field(fields)
    if not speaker_id:
        raise HTTPException(status_code=400, detail="speakerId 不能为空")

    reference_text = _aliased_field(
        fields,
        "reference_text",
        "referenceText",
    )
    if not reference_text:
        raise HTTPException(status_code=400, detail="reference_text 不能为空")
    if len(reference_text) > MAX_REFERENCE_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                "reference_text 不能超过 "
                f"{MAX_REFERENCE_TEXT_LENGTH} 个字符"
            ),
        )

    prompt = fields.get("prompt", "").strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"prompt 不能超过 {MAX_PROMPT_LENGTH} 个字符",
        )

    audio_url = _aliased_field(fields, "audio_url", "audioUrl")
    uploaded_audio = _uploaded_audio(files)
    if bool(audio_url) == bool(uploaded_audio):
        raise HTTPException(
            status_code=422,
            detail="audio 文件与 audio_url 必须且只能提供一个",
        )

    if uploaded_audio is not None:
        audio_bytes = uploaded_audio["content"]
        source = "upload"
    else:
        audio_bytes = await _download_audio(request, audio_url)
        source = "url"

    try:
        waveform = await asyncio.to_thread(
            _decode_prompt_audio,
            audio_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    speaker_version, metadata = await _register_with_triton(
        request,
        speaker_id,
        waveform,
        reference_text,
        prompt,
    )
    logger.info(
        "speaker registered speaker=%s source=%s duration=%s version=%s",
        speaker_id,
        source,
        metadata.get("duration_seconds"),
        speaker_version,
    )
    return {
        "status": "ok",
        "speakerId": speaker_id,
        "speakerVersion": speaker_version,
        "source": source,
        "speaker": _public_speaker(metadata),
        # Retained for clients of the initial /register release.
        "metadata": metadata,
    }


@router.get("/speakers")
async def list_speakers(request: Request):
    status, _, message = await _call_registry(
        request,
        _registry_operation_payload("list"),
    )
    if status != "ok":
        raise HTTPException(
            status_code=502,
            detail="Speaker Registry 未能列出声纹",
        )
    speakers = message.get("speakers", [])
    if not isinstance(speakers, list):
        raise HTTPException(
            status_code=502,
            detail="Speaker Registry 返回了非法声纹列表",
        )
    public_speakers = [
        _public_speaker(speaker)
        for speaker in speakers
        if isinstance(speaker, dict)
    ]
    return {
        "status": "ok",
        "count": len(public_speakers),
        "speakers": public_speakers,
    }


@router.get("/speakers/{speaker_id}")
async def inspect_speaker(speaker_id: str, request: Request):
    speaker_id = _path_speaker_id(speaker_id)
    status, speaker_version, message = await _call_registry(
        request,
        _registry_operation_payload("inspect", speaker_id),
    )
    if status == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"speakerId 不存在：{speaker_id}",
        )
    if status != "ok":
        raise HTTPException(
            status_code=502,
            detail="Speaker Registry 未能查询声纹",
        )
    message.setdefault("speaker_id", speaker_id)
    message.setdefault("speaker_version", speaker_version)
    return {
        "status": "ok",
        "speaker": _public_speaker(message),
    }


@router.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str, request: Request):
    speaker_id = _path_speaker_id(speaker_id)
    status, speaker_version, message = await _call_registry(
        request,
        _registry_operation_payload("delete", speaker_id),
    )
    if status == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"speakerId 不存在：{speaker_id}",
        )
    if status != "ok" or not message.get("deleted"):
        raise HTTPException(
            status_code=502,
            detail="Speaker Registry 未确认删除成功",
        )
    return {
        "status": "ok",
        "speakerId": speaker_id,
        "speakerVersion": speaker_version,
        "deleted": True,
    }
