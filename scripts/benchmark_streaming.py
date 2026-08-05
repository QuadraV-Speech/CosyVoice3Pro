#!/usr/bin/env python3
"""Benchmark CosyVoice3Pro streaming over direct Triton gRPC or Public SSE.

The gRPC TTFA definition matches the upstream CosyVoice3 client: start the
clock immediately before submitting stream inference and stop it on the first
non-empty waveform response. Input preparation and warmup are outside timing.
"""

import argparse
import asyncio
import base64
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np


DEFAULT_TEXT = "你好，这是 CosyVoice3Pro 高并发流式语音合成基准测试。"


@dataclass
class RequestResult:
    success: bool
    ttfa_ms: float = 0.0
    total_ms: float = 0.0
    audio_seconds: float = 0.0
    chunks: int = 0
    queue_ms: float = 0.0
    error: str = ""


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(transport, concurrency, results, wall_seconds):
    successful = [result for result in results if result.success]
    ttfa = [result.ttfa_ms for result in successful]
    total = [result.total_ms for result in successful]
    queue = [result.queue_ms for result in successful]
    audio_seconds = sum(result.audio_seconds for result in successful)
    return {
        "transport": transport,
        "concurrency": concurrency,
        "requests": len(results),
        "success": len(successful),
        "failed": len(results) - len(successful),
        "wallSeconds": round(wall_seconds, 4),
        "requestsPerSecond": round(
            len(successful) / wall_seconds if wall_seconds else 0, 4),
        "audioSeconds": round(audio_seconds, 4),
        "audioThroughput": round(
            audio_seconds / wall_seconds if wall_seconds else 0, 4),
        "systemRtf": round(
            wall_seconds / audio_seconds if audio_seconds else 0, 6),
        "ttfaMs": {
            "average": round(sum(ttfa) / len(ttfa), 2) if ttfa else 0,
            "p50": round(percentile(ttfa, 50), 2),
            "p90": round(percentile(ttfa, 90), 2),
            "p95": round(percentile(ttfa, 95), 2),
            "p99": round(percentile(ttfa, 99), 2),
            "max": round(max(ttfa), 2) if ttfa else 0,
        },
        "totalLatencyMs": {
            "average": round(sum(total) / len(total), 2) if total else 0,
            "p50": round(percentile(total, 50), 2),
            "p95": round(percentile(total, 95), 2),
            "p99": round(percentile(total, 99), 2),
            "max": round(max(total), 2) if total else 0,
        },
        "gatewayQueueMs": {
            "average": round(sum(queue) / len(queue), 2) if queue else 0,
            "p95": round(percentile(queue, 95), 2),
            "max": round(max(queue), 2) if queue else 0,
        },
        "errors": [result.error for result in results if not result.success][:5],
    }


def grpc_inputs(grpcclient, speaker_id, text, prompt):
    def make_input(name, values, datatype):
        value = np.asarray(values, dtype=(object if datatype == "BYTES" else None))
        triton_input = grpcclient.InferInput(name, list(value.shape), datatype)
        triton_input.set_data_from_numpy(value)
        return triton_input

    return [
        make_input("speaker_id", [[speaker_id]], "BYTES"),
        make_input("prompt", [[prompt]], "BYTES"),
        make_input("target_text", [[text]], "BYTES"),
    ]


async def grpc_request(client, grpcclient, model, speaker_id, text, prompt):
    request_id = uuid4().hex

    async def request_iterator():
        yield {
            "model_name": model,
            "inputs": grpc_inputs(grpcclient, speaker_id, text, prompt),
            "outputs": [grpcclient.InferRequestedOutput("waveform")],
            "request_id": request_id,
        }

    started = time.perf_counter()
    first_audio = None
    samples = 0
    chunks = 0
    response_iterator = client.stream_infer(request_iterator(), stream_timeout=300)
    try:
        async for result, error in response_iterator:
            if error is not None:
                raise RuntimeError(str(error))
            if result is None:
                continue
            waveform = result.as_numpy("waveform")
            if waveform is None or waveform.size == 0:
                continue
            if first_audio is None:
                first_audio = time.perf_counter()
            samples += int(waveform.size)
            chunks += 1
    except Exception as exc:
        return RequestResult(success=False, error=str(exc))
    finally:
        response_iterator.cancel()

    finished = time.perf_counter()
    if first_audio is None or samples == 0:
        return RequestResult(success=False, error="gRPC returned no audio")
    return RequestResult(
        success=True,
        ttfa_ms=(first_audio - started) * 1000,
        total_ms=(finished - started) * 1000,
        audio_seconds=samples / 24000,
        chunks=chunks,
    )


async def iter_sse(response):
    event_name = "message"
    data_lines = []
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif not line and data_lines:
            yield event_name, json.loads("\n".join(data_lines))
            event_name = "message"
            data_lines = []


async def sse_request(client, url, speaker_id, text, prompt):
    started = time.perf_counter()
    first_audio = None
    samples = 0
    chunks = 0
    queue_ms = 0.0
    try:
        async with client.stream(
            "POST",
            url,
            data={
                "text": text,
                "speakerId": speaker_id,
                "prompt": prompt,
                "speed": "balanced",
                "volume": "middle",
            },
            headers={"Accept": "text/event-stream"},
            timeout=None,
        ) as response:
            response.raise_for_status()
            async for event_name, payload in iter_sse(response):
                if event_name == "queue":
                    queue_ms = float(payload.get("queueMs", 0))
                elif event_name == "audio":
                    chunk = base64.b64decode(payload["audio"], validate=True)
                    if first_audio is None:
                        first_audio = time.perf_counter()
                    samples += len(chunk) // 2
                    chunks += 1
                elif event_name == "error":
                    raise RuntimeError(payload.get("detail", "SSE stream failed"))
                elif event_name == "done":
                    break
    except Exception as exc:
        return RequestResult(success=False, error=str(exc))

    finished = time.perf_counter()
    if first_audio is None or samples == 0:
        return RequestResult(success=False, error="SSE returned no audio")
    return RequestResult(
        success=True,
        ttfa_ms=(first_audio - started) * 1000,
        total_ms=(finished - started) * 1000,
        audio_seconds=samples / 16000,
        chunks=chunks,
        queue_ms=queue_ms,
    )


async def run_workers(concurrency, request_count, request_factory):
    queue = asyncio.Queue()
    for index in range(request_count):
        queue.put_nowait(index)
    results = []

    async def worker():
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await request_factory(index))
            queue.task_done()

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results, time.perf_counter() - started


def parse_concurrency(value):
    result = []
    for item in value.split(","):
        concurrency = int(item.strip())
        if concurrency <= 0:
            raise argparse.ArgumentTypeError("concurrency must be positive")
        result.append(concurrency)
    return result


async def main_async(args):
    reports = []
    transports = (
        ["grpc", "sse"] if args.transport == "both" else [args.transport]
    )

    for transport in transports:
        if transport == "grpc":
            try:
                import tritonclient.grpc.aio as grpcclient
            except ImportError as exc:
                raise RuntimeError(
                    "gRPC benchmark requires tritonclient[grpc]") from exc
            resource = grpcclient.InferenceServerClient(
                url=args.grpc_url, verbose=False)

            async def request_factory(index):
                return await grpc_request(
                    resource,
                    grpcclient,
                    args.model,
                    args.speaker_id,
                    args.text,
                    args.prompt,
                )
        else:
            resource = httpx.AsyncClient(
                trust_env=False,
                limits=httpx.Limits(
                    max_connections=max(args.concurrency),
                    max_keepalive_connections=max(args.concurrency),
                ),
            )

            async def request_factory(index):
                return await sse_request(
                    resource,
                    args.sse_url,
                    args.speaker_id,
                    args.text,
                    args.prompt,
                )

        try:
            for _ in range(args.warmup):
                warmup = await request_factory(-1)
                if not warmup.success:
                    raise RuntimeError(
                        f"{transport} warmup failed: {warmup.error}")

            for concurrency in args.concurrency:
                results, wall_seconds = await run_workers(
                    concurrency,
                    args.requests,
                    request_factory,
                )
                report = summarize(
                    transport, concurrency, results, wall_seconds)
                reports.append(report)
                print(json.dumps(report, ensure_ascii=False))
        finally:
            if transport == "grpc":
                await resource.close()
            else:
                await resource.aclose()

    output = {
        "standard": "upstream-grpc-submit-to-first-nonempty-waveform",
        "metricSource": (
            "https://github.com/QwenAudio/CosyVoice/blob/main/"
            "runtime/triton_trtllm/client_grpc.py"
        ),
        "grpcClientMode": "shared-async-channel",
        "host": platform.node(),
        "speakerId": args.speaker_id,
        "text": args.text,
        "requestsPerProfile": args.requests,
        "warmup": args.warmup,
        "profiles": reports,
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport", choices=["grpc", "sse", "both"], default="grpc")
    parser.add_argument("--grpc-url", default="127.0.0.1:18001")
    parser.add_argument(
        "--sse-url", default="http://127.0.0.1:18000/tts/stream")
    parser.add_argument("--model", default="CosyVoice3ProStreaming")
    parser.add_argument("--speaker-id", default="common_speaker_1")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--concurrency", type=parse_concurrency, default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.requests <= 0 or args.warmup < 0:
        parser.error("requests must be positive and warmup cannot be negative")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
