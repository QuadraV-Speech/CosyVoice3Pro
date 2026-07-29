#!/usr/bin/env python3
"""Benchmark the public CosyVoice3Pro TTS API with real audio responses."""

import argparse
import io
import json
import math
import statistics
import struct
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


DEFAULT_TEXT = (
    "你好，欢迎使用 CosyVoice3Pro。注册一次声纹，"
    "后续请求只需要传入说话人编号和需要合成的文本。"
)
OFFICIAL_BENCHMARK_SOURCE = (
    "https://github.com/FunAudioLLM/CosyVoice/blob/"
    "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/"
    "runtime/triton_trtllm/client_grpc.py"
)


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def wav_duration(content):
    with wave.open(io.BytesIO(content), "rb") as audio:
        frame_rate = audio.getframerate()
        block_align = audio.getnchannels() * audio.getsampwidth()

    # Streaming WAV encoders commonly use 0xffffffff as the data-chunk size
    # because the final byte count is unknown when the header is emitted.
    # Read the actual available data bytes instead of trusting getnframes().
    cursor = 12
    while cursor + 8 <= len(content):
        chunk_id = content[cursor:cursor + 4]
        declared_size = struct.unpack_from("<I", content, cursor + 4)[0]
        data_start = cursor + 8
        if chunk_id == b"data":
            actual_size = min(declared_size, len(content) - data_start)
            return actual_size / (frame_rate * block_align)
        cursor = data_start + declared_size + (declared_size % 2)
    raise RuntimeError("TTS API returned a WAV without a data chunk")


def synthesize(args):
    started_at = time.perf_counter()
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        f"{args.url.rstrip('/')}/tts/",
        data={
            "text": args.text,
            "speakerId": args.speaker_id,
            "prompt": args.prompt,
            "speed": "balanced",
            "volume": "middle",
            "output_format": "wav",
            "max_chars": "80",
        },
        timeout=args.timeout,
    )
    latency = time.perf_counter() - started_at
    response.raise_for_status()
    duration = wav_duration(response.content)
    if duration <= 0:
        raise RuntimeError("TTS API returned an empty WAV")
    return {
        "latency_seconds": latency,
        "audio_seconds": duration,
        "rtf": latency / duration,
        "bytes": len(response.content),
    }


def summarize_results(
    results,
    errors,
    concurrency,
    request_count,
    wall_seconds,
):
    if not results:
        raise RuntimeError(
            f"all requests failed at concurrency {concurrency}: "
            + "; ".join(errors[:3])
        )

    latencies = [item["latency_seconds"] for item in results]
    audio_durations = [item["audio_seconds"] for item in results]
    request_rtfs = [item["rtf"] for item in results]
    total_audio_seconds = sum(audio_durations)
    system_rtf = wall_seconds / total_audio_seconds
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "successful": len(results),
        "failed": len(errors),
        "latency_min_seconds": min(latencies),
        "latency_average_seconds": statistics.mean(latencies),
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p90_seconds": percentile(latencies, 90),
        "latency_p95_seconds": percentile(latencies, 95),
        "latency_p99_seconds": percentile(latencies, 99),
        "latency_max_seconds": max(latencies),
        "audio_average_seconds": statistics.mean(audio_durations),
        "total_audio_seconds": total_audio_seconds,
        "system_rtf": system_rtf,
        "request_rtf_average": statistics.mean(request_rtfs),
        # Retained for consumers of reports generated before v1.6.1. This is
        # per-request RTF, not the official aggregate/system RTF.
        "rtf_average": statistics.mean(request_rtfs),
        "audio_throughput_x": 1 / system_rtf,
        "requests_per_second": len(results) / wall_seconds,
        "wall_seconds": wall_seconds,
        "response_average_bytes": round(
            statistics.mean(item["bytes"] for item in results)
        ),
        "errors": errors,
    }


def benchmark_profile(args, concurrency):
    started_at = time.perf_counter()
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(synthesize, args)
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(str(exc))
    wall_seconds = time.perf_counter() - started_at

    return summarize_results(
        results,
        errors,
        concurrency,
        args.requests,
        wall_seconds,
    )


def markdown_report(report):
    print(
        "| 并发任务 | 成功/请求 | 系统 RTF | Average (ms) | "
        "P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | 音频吞吐 |"
    )
    print(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for item in report["profiles"]:
        print(
            "| {concurrency} | {successful}/{requests} | {system_rtf:.4f} | "
            "{average:.2f} | {p50:.2f} | {p90:.2f} | {p95:.2f} | "
            "{p99:.2f} | {audio_throughput_x:.2f}x |".format(
                average=item["latency_average_seconds"] * 1000,
                p50=item["latency_p50_seconds"] * 1000,
                p90=item["latency_p90_seconds"] * 1000,
                p95=item["latency_p95_seconds"] * 1000,
                p99=item["latency_p99_seconds"] * 1000,
                **item,
            )
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark the public CosyVoice3Pro /tts/ API",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:18000",
        help="Web Gateway base URL",
    )
    parser.add_argument(
        "--speaker-id",
        default="common_speaker_1",
        help="registered speaker used for all requests",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4],
        help="one or more concurrency profiles",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=8,
        help="request count for each concurrency profile",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="warm-up requests before measurement",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.requests < 1 or args.warmup < 0:
        raise SystemExit("--requests must be positive and --warmup cannot be negative")
    if any(value < 1 for value in args.concurrency):
        raise SystemExit("--concurrency values must be positive")

    for _ in range(args.warmup):
        synthesize(args)

    report = {
        "url": args.url,
        "speaker_id": args.speaker_id,
        "text": args.text,
        "warmup_requests": args.warmup,
        "metric_standard": {
            "name": "FunAudioLLM CosyVoice Triton benchmark",
            "source": OFFICIAL_BENCHMARK_SOURCE,
            "system_rtf": "wall_seconds / total_audio_seconds",
            "latency": "complete HTTP response latency",
        },
        "profiles": [
            benchmark_profile(args, concurrency)
            for concurrency in args.concurrency
        ],
    }
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        markdown_report(report)


if __name__ == "__main__":
    main()
