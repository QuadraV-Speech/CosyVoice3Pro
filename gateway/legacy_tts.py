import asyncio
import json
import logging
import re
import shutil
import subprocess
import time
from email.parser import BytesParser
from email.policy import default as email_policy
from typing import Dict, Tuple
from urllib.parse import parse_qs
from uuid import uuid4

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

try:
    from .tts_utils import positive_env, split_text
except ImportError:
    from tts_utils import positive_env, split_text


router = APIRouter(tags=["legacy-tts"])
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
OUTPUT_SAMPLE_RATE = 16000
REFERENCE_SAMPLE_RATE = 16000
MIN_PROMPT_SECONDS = 0.5
MAX_PROMPT_SECONDS = 30
MAX_PROMPT_LENGTH = 512
MAX_FORM_BYTES = 32 * 1024 * 1024
INFERENCE_CONCURRENCY = positive_env(
    "COSYVOICE_TTS_INFERENCE_CONCURRENCY", 10)
SEGMENT_CONCURRENCY = positive_env(
    "COSYVOICE_TTS_SEGMENT_CONCURRENCY", 2)

TTS_STYLES = {
    1: "common_speaker_1",
    2: "common_speaker_2",
    3: "common_speaker_3",
    4: "common_speaker_4",
}

SPEED_MAP = {
    "low": 0.85,
    "balanced": 1.0,
    "fast": 1.15,
}

VOLUME_MAP = {
    "small": 0.8,
    "middle": 1.0,
    "large": 1.2,
}

MEDIA_TYPES = {
    "pcm": "application/octet-stream",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "m4a": "audio/mp4",
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "webm": "audio/webm",
}

_SPEAKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)


def _decode_form_value(raw_value: bytes, charset: str) -> str:
    try:
        return raw_value.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return raw_value.decode("utf-8", errors="replace")


async def _read_form_data(
    request: Request,
) -> Tuple[Dict[str, str], Dict[str, dict]]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FORM_BYTES:
                raise HTTPException(status_code=413, detail="请求体过大")
        except ValueError:
            raise HTTPException(status_code=400, detail="Content-Length 非法")

    body = await request.body()
    if len(body) > MAX_FORM_BYTES:
        raise HTTPException(status_code=413, detail="请求体过大")

    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type == "multipart/form-data":
        message = BytesParser(policy=email_policy).parsebytes(
            b"Content-Type: "
            + content_type.encode("latin-1")
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + body
        )
        if not message.is_multipart():
            raise HTTPException(status_code=400, detail="multipart 请求体非法")

        fields = {}
        files = {}
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            value = part.get_payload(decode=True) or b""
            filename = part.get_param(
                "filename", header="content-disposition")
            if filename is not None:
                files[name] = {
                    "filename": filename,
                    "content_type": part.get_content_type(),
                    "content": value,
                }
            else:
                fields[name] = _decode_form_value(
                    value, part.get_content_charset() or "utf-8")
        return fields, files

    if media_type == "application/x-www-form-urlencoded":
        parsed = parse_qs(
            body.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=False,
        )
        return {
            name: values[-1] for name, values in parsed.items()
        }, {}

    if media_type == "application/json":
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="JSON 请求体非法") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON 请求体必须是对象")
        fields = {
            str(name): "" if value is None else str(value)
            for name, value in payload.items()
        }
        return fields, {}

    raise HTTPException(
        status_code=415,
        detail=(
            "仅支持 multipart/form-data、"
            "application/x-www-form-urlencoded 或 application/json"
        ),
    )


def _enum_field(fields, name, default, choices):
    value = fields.get(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(choices)
        raise HTTPException(
            status_code=422,
            detail=f"{name} 必须是以下值之一：{allowed}",
        )
    return value


def _int_field(fields, name, default):
    raw_value = fields.get(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"{name} 必须是整数") from exc


def _decode_prompt_audio(audio_bytes):
    if not audio_bytes:
        raise ValueError("prompt_audio 不能为空")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("服务端未安装 FFmpeg")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "f32le",
        "-codec:a", "pcm_f32le",
        "-ar", str(REFERENCE_SAMPLE_RATE),
        "-ac", "1",
        "pipe:1",
    ]
    try:
        process = subprocess.run(
            command,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("prompt_audio 解码超时") from exc

    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"prompt_audio 无法解码：{detail}")
    try:
        waveform = np.frombuffer(process.stdout, dtype=np.float32).copy()
    except ValueError as exc:
        raise ValueError("prompt_audio 解码结果非法") from exc

    duration = waveform.size / REFERENCE_SAMPLE_RATE
    if duration < MIN_PROMPT_SECONDS or duration > MAX_PROMPT_SECONDS:
        raise ValueError(
            "prompt_audio 时长必须在 "
            f"{MIN_PROMPT_SECONDS}～{MAX_PROMPT_SECONDS} 秒之间"
        )
    if not np.all(np.isfinite(waveform)):
        raise ValueError("prompt_audio 包含非法采样值")
    return waveform


def _speaker_id_field(fields):
    camel_case = fields.get("speakerId", "").strip()
    snake_case = fields.get("speaker_id", "").strip()
    if camel_case and snake_case and camel_case != snake_case:
        raise HTTPException(
            status_code=422,
            detail="speakerId 与 speaker_id 不能设置为不同值",
        )
    speaker_id = camel_case or snake_case
    if speaker_id and (
        not _SPEAKER_ID_PATTERN.fullmatch(speaker_id)
        or ".." in speaker_id
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "speakerId 仅允许字母、数字、下划线、中划线和点，"
                "长度 1～128，且不能包含 '..'"
            ),
        )
    return speaker_id


async def _infer_segment(
    request: Request,
    text: str,
    request_id: str,
    prompt: str,
    speaker_id: str = "",
    reference_samples=None,
    reference_text: str = "",
):
    inputs = []
    if speaker_id:
        inputs.append(
            {
                "name": "speaker_id",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [speaker_id],
            }
        )
    else:
        sample_count = len(reference_samples[0])
        inputs.extend([
            {
                "name": "reference_wav",
                "shape": [1, sample_count],
                "datatype": "FP32",
                "data": reference_samples,
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
        ])

    inputs.extend([
        {
            "name": "prompt",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [prompt],
        },
        {
            "name": "target_text",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        },
    ])
    payload = {"inputs": inputs}
    upstream = request.app.state.triton_upstream
    url = f"{upstream}/v2/models/CosyVoice3Pro/infer"

    try:
        async with _inference_semaphore:
            response = await request.app.state.http_client.post(
                url,
                params={"request_id": request_id},
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Triton 服务不可用：{exc}",
        ) from exc

    if response.status_code != 200:
        try:
            upstream_error = response.json().get("error", response.text)
        except (ValueError, AttributeError):
            upstream_error = response.text
        raise HTTPException(
            status_code=502,
            detail=f"Triton 推理失败：{upstream_error}",
        )

    try:
        outputs = {
            output["name"]: output.get("data", [])
            for output in response.json().get("outputs", [])
        }
        waveform = np.asarray(outputs["waveform"], dtype=np.float32).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Triton 响应中缺少有效 waveform",
        ) from exc

    if waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise HTTPException(
            status_code=502,
            detail="Triton 返回了空音频或非法采样值",
        )
    return waveform


def _encode_audio(waveform, speed, volume, output_format):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("服务端未安装 FFmpeg")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "f32le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-i", "pipe:0",
        "-af", f"volume={VOLUME_MAP[volume]},atempo={SPEED_MAP[speed]}",
        "-ar", str(OUTPUT_SAMPLE_RATE),
        "-ac", "1",
    ]

    if output_format == "pcm":
        command += ["-codec:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
    elif output_format == "mp3":
        command += [
            "-codec:a", "libmp3lame", "-b:a", "128k",
            "-q:a", "2", "-f", "mp3", "pipe:1",
        ]
    elif output_format == "wav":
        command += ["-codec:a", "pcm_s16le", "-f", "wav", "pipe:1"]
    elif output_format == "aac":
        command += [
            "-codec:a", "aac", "-b:a", "128k",
            "-f", "adts", "pipe:1",
        ]
    elif output_format == "m4a":
        command += [
            "-codec:a", "aac", "-b:a", "128k",
            "-movflags", "frag_keyframe+empty_moov",
            "-f", "mp4", "pipe:1",
        ]
    elif output_format == "opus":
        command += [
            "-codec:a", "libopus", "-b:a", "128k",
            "-application", "voip", "-f", "opus", "pipe:1",
        ]
    elif output_format == "ogg":
        command += [
            "-codec:a", "libvorbis", "-q:a", "5",
            "-f", "ogg", "pipe:1",
        ]
    elif output_format == "flac":
        command += [
            "-codec:a", "flac", "-compression_level", "5",
            "-f", "flac", "pipe:1",
        ]
    elif output_format == "webm":
        command += [
            "-codec:a", "libopus", "-b:a", "128k",
            "-application", "voip", "-f", "webm", "pipe:1",
        ]

    process = subprocess.run(
        command,
        input=np.ascontiguousarray(waveform, dtype=np.float32).tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg 编码失败：{detail}")
    if not process.stdout:
        raise RuntimeError("FFmpeg 返回了空音频")
    return process.stdout


@router.post("/tts/")
async def legacy_tts(request: Request):
    request_started_at = time.perf_counter()
    fields, files = await _read_form_data(request)
    text = fields.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    # language is retained for wire compatibility. CosyVoice3 detects and
    # handles Chinese, English, Japanese and other supported text directly.
    fields.get("language", "zh")
    speed = _enum_field(fields, "speed", "balanced", SPEED_MAP)
    volume = _enum_field(fields, "volume", "middle", VOLUME_MAP)
    output_format = _enum_field(
        fields, "output_format", "mp3", MEDIA_TYPES)
    max_chars = _int_field(fields, "max_chars", 80)
    if max_chars <= 0:
        raise HTTPException(status_code=422, detail="max_chars 必须大于 0")

    prompt = fields.get("prompt", "").strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"prompt 不能超过 {MAX_PROMPT_LENGTH} 个字符",
        )

    requested_speaker_id = _speaker_id_field(fields)
    prompt_audio = files.get("prompt_audio")
    reference_samples = None
    reference_text = ""
    tts_style = None

    if prompt_audio is not None:
        reference_text = fields.get("prompt_text", "").strip()
        if not reference_text:
            raise HTTPException(
                status_code=400,
                detail="上传 prompt_audio 时 prompt_text 不能为空",
            )
        try:
            reference_waveform = await asyncio.to_thread(
                _decode_prompt_audio,
                prompt_audio["content"],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        reference_samples = [reference_waveform.tolist()]
        speaker_id = ""
        resolved_speaker = "raw_prompt"
        mode = "prompt_audio"
    elif requested_speaker_id:
        speaker_id = requested_speaker_id
        resolved_speaker = speaker_id
        mode = "speaker_id"
    else:
        tts_style = _int_field(fields, "tts_style", 1)
        # Preserve the old endpoint's fallback behavior for unknown style IDs.
        if tts_style not in TTS_STYLES:
            tts_style = 1
        speaker_id = TTS_STYLES[tts_style]
        resolved_speaker = speaker_id
        mode = "tts_style"

    segments = split_text(text, max_chars)
    if not segments:
        raise HTTPException(status_code=400, detail="text 不能为空")

    logger.info(
        "tts requested mode=%s style=%s speaker=%s prompt_override=%s "
        "chars=%s segments=%s format=%s speed=%s volume=%s",
        mode,
        tts_style,
        resolved_speaker,
        bool(prompt),
        len(text),
        len(segments),
        output_format,
        speed,
        volume,
    )

    request_group_id = uuid4().hex
    segment_semaphore = asyncio.Semaphore(
        min(SEGMENT_CONCURRENCY, len(segments)))

    async def infer_segment(index, segment):
        # A large text request must not occupy every global inference slot.
        # Keeping this limit per request improves fairness and P95 latency when
        # several clients synthesize segmented text concurrently.
        async with segment_semaphore:
            return await _infer_segment(
                request=request,
                text=segment,
                request_id=f"{request_group_id}-{index}",
                prompt=prompt,
                speaker_id=speaker_id,
                reference_samples=reference_samples,
                reference_text=reference_text,
            )

    inference_started_at = time.perf_counter()
    waveforms = await asyncio.gather(*[
        infer_segment(index, segment)
        for index, segment in enumerate(segments)
    ])
    inference_finished_at = time.perf_counter()
    waveform = np.concatenate(waveforms)

    try:
        audio_bytes = await asyncio.to_thread(
            _encode_audio,
            waveform,
            speed,
            volume,
            output_format,
        )
    except RuntimeError as exc:
        logger.exception("legacy tts audio encoding failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    request_finished_at = time.perf_counter()

    inference_ms = (
        inference_finished_at - inference_started_at) * 1000
    encode_ms = (request_finished_at - inference_finished_at) * 1000
    total_ms = (request_finished_at - request_started_at) * 1000

    return Response(
        content=audio_bytes,
        media_type=MEDIA_TYPES[output_format],
        headers={
            "Content-Disposition": f'inline; filename="tts.{output_format}"',
            "X-CosyVoice-Mode": mode,
            "X-CosyVoice-Speaker": resolved_speaker,
            "X-CosyVoice-Prompt-Override": str(bool(prompt)).lower(),
            "X-CosyVoice-Segments": str(len(segments)),
            "X-CosyVoice-Inference-Ms": f"{inference_ms:.1f}",
            "X-CosyVoice-Encode-Ms": f"{encode_ms:.1f}",
            "Server-Timing": (
                f'inference;dur={inference_ms:.1f}, '
                f'encode;dur={encode_ms:.1f}, total;dur={total_ms:.1f}'
            ),
        },
    )
