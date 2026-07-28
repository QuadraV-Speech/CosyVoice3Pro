import asyncio
import json
import logging
import re
import shutil
import subprocess
from email.parser import BytesParser
from email.policy import default as email_policy
from typing import Dict
from urllib.parse import parse_qs

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response


router = APIRouter(tags=["legacy-tts"])
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
OUTPUT_SAMPLE_RATE = 16000
MAX_FORM_BYTES = 4 * 1024 * 1024
INFERENCE_CONCURRENCY = 10

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

_PUNCTUATION_PARTS = re.compile(r".+?[。！？!?；;，,：:.]+|.+$", re.DOTALL)
_inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)


def _decode_form_value(raw_value: bytes, charset: str) -> str:
    try:
        return raw_value.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return raw_value.decode("utf-8", errors="replace")


async def _read_form_fields(request: Request) -> Dict[str, str]:
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
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            if not name or part.get_filename() is not None:
                continue
            value = part.get_payload(decode=True) or b""
            fields[name] = _decode_form_value(
                value, part.get_content_charset() or "utf-8")
        return fields

    if media_type == "application/x-www-form-urlencoded":
        parsed = parse_qs(
            body.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=False,
        )
        return {name: values[-1] for name, values in parsed.items()}

    if media_type == "application/json":
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="JSON 请求体非法") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON 请求体必须是对象")
        return {
            str(name): "" if value is None else str(value)
            for name, value in payload.items()
        }

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


def _split_text(text: str, max_chars: int):
    text = text.strip()
    if not text:
        return []

    segments = []
    buffer = ""

    def append_hard_split(value):
        for index in range(0, len(value), max_chars):
            segment = value[index:index + max_chars].rstrip("，,").strip()
            if segment:
                segments.append(segment)

    for part in _PUNCTUATION_PARTS.findall(text):
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if buffer:
                segments.append(buffer.rstrip("，,"))
                buffer = ""
            append_hard_split(part)
        elif len(buffer) + len(part) <= max_chars:
            buffer += part
        else:
            if buffer:
                segments.append(buffer.rstrip("，,"))
            buffer = part

    if buffer:
        segments.append(buffer.rstrip("，,"))
    return segments


async def _infer_segment(request: Request, speaker_id: str, text: str, index: int):
    payload = {
        "inputs": [
            {
                "name": "speaker_id",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [speaker_id],
            },
            {
                "name": "target_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [text],
            },
        ]
    }
    upstream = request.app.state.triton_upstream
    url = f"{upstream}/v2/models/CosyVoice3Pro/infer"

    try:
        async with _inference_semaphore:
            response = await request.app.state.http_client.post(
                url,
                params={"request_id": str(index)},
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
    fields = await _read_form_fields(request)
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

    tts_style = _int_field(fields, "tts_style", 1)
    # Preserve the old endpoint's fallback behavior for unknown style IDs.
    if tts_style not in TTS_STYLES:
        tts_style = 1
    speaker_id = TTS_STYLES[tts_style]

    segments = _split_text(text, max_chars)
    if not segments:
        raise HTTPException(status_code=400, detail="text 不能为空")

    logger.info(
        "legacy tts requested style=%s speaker=%s chars=%s segments=%s "
        "format=%s speed=%s volume=%s",
        tts_style,
        speaker_id,
        len(text),
        len(segments),
        output_format,
        speed,
        volume,
    )

    waveforms = await asyncio.gather(*[
        _infer_segment(request, speaker_id, segment, index)
        for index, segment in enumerate(segments)
    ])
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

    return Response(
        content=audio_bytes,
        media_type=MEDIA_TYPES[output_format],
        headers={
            "Content-Disposition": f'inline; filename="tts.{output_format}"',
            "X-CosyVoice-Speaker": speaker_id,
            "X-CosyVoice-Segments": str(len(segments)),
        },
    )
