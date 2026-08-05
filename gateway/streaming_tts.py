import asyncio
import base64
import contextlib
import json
import logging
import os
import shutil
import time
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

try:
    import tritonclient.grpc.aio as grpcclient
except ImportError:  # Keep lightweight tooling and unit tests importable.
    grpcclient = None

try:
    from .legacy_tts import (
        MAX_PROMPT_LENGTH,
        OUTPUT_SAMPLE_RATE,
        SAMPLE_RATE,
        SPEED_MAP,
        TTS_STYLES,
        VOLUME_MAP,
        _decode_prompt_audio,
        _enum_field,
        _int_field,
        _read_form_data,
        _speaker_id_field,
    )
    from .tts_utils import positive_env
except ImportError:
    from legacy_tts import (
        MAX_PROMPT_LENGTH,
        OUTPUT_SAMPLE_RATE,
        SAMPLE_RATE,
        SPEED_MAP,
        TTS_STYLES,
        VOLUME_MAP,
        _decode_prompt_audio,
        _enum_field,
        _int_field,
        _read_form_data,
        _speaker_id_field,
    )
    from tts_utils import positive_env


router = APIRouter(tags=["streaming-tts"])
logger = logging.getLogger(__name__)

STREAMING_MODEL = "CosyVoice3ProStreaming"
TRITON_GRPC_UPSTREAM = os.environ.get(
    "COSYVOICE_TRITON_GRPC_UPSTREAM", "127.0.0.1:18001")
STREAMING_CONCURRENCY = positive_env(
    "COSYVOICE_TTS_STREAMING_CONCURRENCY", 2)
STREAM_TIMEOUT_SECONDS = positive_env(
    "COSYVOICE_TTS_STREAM_TIMEOUT_SECONDS", 300)
MAX_STREAM_TEXT_LENGTH = 1000
PCM_READ_BYTES = 8192
PCM_EVENT_BYTES = 6400  # 200 ms at 16 kHz mono PCM16.
SSE_KEEPALIVE_SECONDS = 10

_streaming_semaphore = asyncio.Semaphore(STREAMING_CONCURRENCY)


def _sse_event(event, payload, event_id=None):
    """Serialize one SSE event without allowing multiline field injection."""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(
        "data: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return "\n".join(lines) + "\n\n"


def _grpc_input(name, value, datatype):
    triton_input = grpcclient.InferInput(name, list(value.shape), datatype)
    triton_input.set_data_from_numpy(value)
    return triton_input


def _build_triton_inputs(
    text,
    prompt,
    speaker_id,
    reference_waveform=None,
    reference_text="",
):
    inputs = []
    if speaker_id:
        inputs.append(_grpc_input(
            "speaker_id",
            np.asarray([[speaker_id]], dtype=object),
            "BYTES",
        ))
    else:
        waveform = np.ascontiguousarray(
            reference_waveform, dtype=np.float32).reshape(1, -1)
        inputs.extend([
            _grpc_input("reference_wav", waveform, "FP32"),
            _grpc_input(
                "reference_wav_len",
                np.asarray([[waveform.shape[1]]], dtype=np.int32),
                "INT32",
            ),
            _grpc_input(
                "reference_text",
                np.asarray([[reference_text]], dtype=object),
                "BYTES",
            ),
        ])

    inputs.extend([
        _grpc_input(
            "prompt", np.asarray([[prompt]], dtype=object), "BYTES"),
        _grpc_input(
            "target_text", np.asarray([[text]], dtype=object), "BYTES"),
    ])
    return inputs


async def _triton_waveforms(
    text,
    prompt,
    speaker_id,
    reference_waveform,
    reference_text,
    request_id,
):
    client = grpcclient.InferenceServerClient(
        url=TRITON_GRPC_UPSTREAM,
        verbose=False,
    )
    response_iterator = None

    async def request_iterator():
        yield {
            "model_name": STREAMING_MODEL,
            "inputs": _build_triton_inputs(
                text=text,
                prompt=prompt,
                speaker_id=speaker_id,
                reference_waveform=reference_waveform,
                reference_text=reference_text,
            ),
            "outputs": [grpcclient.InferRequestedOutput("waveform")],
            "request_id": request_id,
        }

    try:
        response_iterator = client.stream_infer(
            request_iterator(),
            stream_timeout=STREAM_TIMEOUT_SECONDS,
        )
        async for result, error in response_iterator:
            if error is not None:
                raise RuntimeError(str(error))
            if result is None:
                continue
            waveform = result.as_numpy("waveform")
            if waveform is None:
                continue
            waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if waveform.size == 0:
                continue
            if not np.all(np.isfinite(waveform)):
                raise RuntimeError("Triton 返回了非法音频采样值")
            yield waveform
    finally:
        if response_iterator is not None:
            response_iterator.cancel()
        await client.close()


async def _close_stdin(stdin):
    if stdin is None or stdin.is_closing():
        return
    stdin.close()
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        await stdin.wait_closed()


async def _feed_ffmpeg(
    process,
    request,
    inference,
    stats,
):
    cancelled = False
    try:
        async for waveform in inference:
            if await request.is_disconnected():
                stats["disconnected"] = True
                return
            stats["triton_chunks"] += 1
            stats["input_samples"] += int(waveform.size)
            process.stdin.write(
                np.ascontiguousarray(waveform, dtype=np.float32).tobytes()
            )
            await process.stdin.drain()
        if stats["triton_chunks"] == 0:
            raise RuntimeError("Triton 未返回任何音频分块")
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        # On request cancellation the generator's unified cleanup terminates
        # FFmpeg. Closing its pipe here as well can dispatch pipe-lost twice in
        # Python's subprocess protocol.
        if not cancelled:
            await _close_stdin(process.stdin)


async def _terminate_process(process):
    if process is None or process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def _stream_sse(
    request,
    request_id,
    text,
    prompt,
    speed,
    volume,
    mode,
    resolved_speaker,
    speaker_id,
    reference_waveform,
    reference_text,
    started_at,
):
    process = None
    producer = None
    read_task = None
    slot_acquired = False
    stats = {
        "triton_chunks": 0,
        "input_samples": 0,
        "output_samples": 0,
        "sse_chunks": 0,
        "first_audio_ms": None,
        "disconnected": False,
    }

    def audio_event(chunk):
        if stats["first_audio_ms"] is None:
            stats["first_audio_ms"] = (
                time.perf_counter() - started_at) * 1000
        sequence = stats["sse_chunks"]
        sample_count = len(chunk) // 2
        stats["sse_chunks"] += 1
        stats["output_samples"] += sample_count
        return _sse_event("audio", {
            "seq": sequence,
            "samples": sample_count,
            "audio": base64.b64encode(chunk).decode("ascii"),
        }, event_id=sequence)

    yield _sse_event("meta", {
        "requestId": request_id,
        "model": STREAMING_MODEL,
        "mode": mode,
        "speakerId": resolved_speaker,
        "promptOverride": bool(prompt),
        "encoding": "pcm_s16le",
        "sampleRate": OUTPUT_SAMPLE_RATE,
        "channels": 1,
        "speed": speed,
        "volume": volume,
    })

    try:
        await _streaming_semaphore.acquire()
        slot_acquired = True
        if await request.is_disconnected():
            return
        process = await asyncio.create_subprocess_exec(
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
            "-codec:a", "pcm_s16le",
            "-f", "s16le",
            "-flush_packets", "1",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        inference = _triton_waveforms(
            text=text,
            prompt=prompt,
            speaker_id=speaker_id,
            reference_waveform=reference_waveform,
            reference_text=reference_text,
            request_id=request_id,
        )
        producer = asyncio.create_task(
            _feed_ffmpeg(process, request, inference, stats)
        )

        remainder = b""
        pending_pcm = b""
        while True:
            if read_task is None:
                read_task = asyncio.create_task(
                    process.stdout.read(PCM_READ_BYTES)
                )
            done, _ = await asyncio.wait(
                {read_task}, timeout=SSE_KEEPALIVE_SECONDS)
            if not done:
                yield ": keep-alive\n\n"
                continue

            chunk = read_task.result()
            read_task = None
            if not chunk:
                break

            chunk = remainder + chunk
            complete_bytes = len(chunk) - (len(chunk) % 2)
            remainder = chunk[complete_bytes:]
            chunk = chunk[:complete_bytes]
            if not chunk:
                continue
            pending_pcm += chunk
            while len(pending_pcm) >= PCM_EVENT_BYTES:
                event_chunk = pending_pcm[:PCM_EVENT_BYTES]
                pending_pcm = pending_pcm[PCM_EVENT_BYTES:]
                yield audio_event(event_chunk)

        await producer
        producer = None
        stderr = (await process.stderr.read()).decode(
            "utf-8", errors="replace").strip()
        return_code = await process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg 流式后处理失败：{stderr}")
        if remainder:
            raise RuntimeError("FFmpeg 返回了不完整的 PCM 采样")
        if stats["disconnected"]:
            return
        if pending_pcm:
            yield audio_event(pending_pcm)
        if stats["output_samples"] == 0:
            raise RuntimeError("流式后处理未返回音频")

        total_ms = (time.perf_counter() - started_at) * 1000
        yield _sse_event("done", {
            "requestId": request_id,
            "chunks": stats["sse_chunks"],
            "samples": stats["output_samples"],
            "durationSeconds": (
                stats["output_samples"] / OUTPUT_SAMPLE_RATE),
            "firstAudioMs": round(stats["first_audio_ms"], 1),
            "totalMs": round(total_ms, 1),
        })
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "streaming tts failed request_id=%s speaker=%s",
            request_id,
            resolved_speaker,
        )
        yield _sse_event("error", {
            "requestId": request_id,
            "detail": str(exc),
        })
    finally:
        if read_task is not None:
            read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await read_task
        if producer is not None:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer
        await _terminate_process(process)
        if slot_acquired:
            _streaming_semaphore.release()


@router.post("/tts/stream")
async def streaming_tts(request: Request):
    started_at = time.perf_counter()
    if grpcclient is None:
        raise HTTPException(
            status_code=503,
            detail="服务端未安装 Triton gRPC Client",
        )
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=503, detail="服务端未安装 FFmpeg")

    fields, files = await _read_form_data(request)
    text = fields.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    if len(text) > MAX_STREAM_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"流式 text 不能超过 {MAX_STREAM_TEXT_LENGTH} 个字符",
        )

    fields.get("language", "zh")
    speed = _enum_field(fields, "speed", "balanced", SPEED_MAP)
    volume = _enum_field(fields, "volume", "middle", VOLUME_MAP)
    prompt = fields.get("prompt", "").strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"prompt 不能超过 {MAX_PROMPT_LENGTH} 个字符",
        )

    requested_speaker_id = _speaker_id_field(fields)
    prompt_audio = files.get("prompt_audio")
    reference_waveform = None
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
        speaker_id = ""
        resolved_speaker = "raw_prompt"
        mode = "prompt_audio"
    elif requested_speaker_id:
        speaker_id = requested_speaker_id
        resolved_speaker = speaker_id
        mode = "speaker_id"
    else:
        tts_style = _int_field(fields, "tts_style", 1)
        if tts_style not in TTS_STYLES:
            tts_style = 1
        speaker_id = TTS_STYLES[tts_style]
        resolved_speaker = speaker_id
        mode = "tts_style"

    request_id = uuid4().hex
    logger.info(
        "streaming tts requested request_id=%s mode=%s style=%s speaker=%s "
        "prompt_override=%s chars=%s speed=%s volume=%s",
        request_id,
        mode,
        tts_style,
        resolved_speaker,
        bool(prompt),
        len(text),
        speed,
        volume,
    )

    return StreamingResponse(
        _stream_sse(
            request=request,
            request_id=request_id,
            text=text,
            prompt=prompt,
            speed=speed,
            volume=volume,
            mode=mode,
            resolved_speaker=resolved_speaker,
            speaker_id=speaker_id,
            reference_waveform=reference_waveform,
            reference_text=reference_text,
            started_at=started_at,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-CosyVoice-Request-Id": request_id,
            "X-CosyVoice-Stream-Encoding": "pcm_s16le",
            "X-CosyVoice-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
        },
    )
