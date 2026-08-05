#!/usr/bin/env python3
"""Run a CosyVoice3Pro streaming benchmark aligned with upstream CosyVoice.

The benchmark intentionally follows the upstream CosyVoice3 Triton client:

* one persistent synchronous gRPC stream per concurrent task;
* raw 16 kHz reference audio and its transcript for every request;
* the same 10-second input padding strategy;
* TTFA measured from ``async_stream_infer`` submission to the first response;
* contiguous dataset shards, processed sequentially by each task.

Unlike the upstream client, audio files are not written during the timed run.
This avoids filesystem noise and does not change per-request TTFA/latency.
"""

import argparse
import json
import math
import platform
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DatasetItem:
    item_id: str
    waveform: np.ndarray
    reference_text: str
    target_text: str


@dataclass
class RequestResult:
    item_id: str
    success: bool
    ttfa_ms: float = 0.0
    second_chunk_ms: float = 0.0
    total_ms: float = 0.0
    audio_seconds: float = 0.0
    chunks: int = 0
    error: str = ""


class ResponseState:
    def __init__(self):
        self.responses = queue.Queue()
        self.started = 0.0
        self.first_response = 0.0
        self.second_response = 0.0

    def callback(self, result, error):
        now = time.perf_counter()
        if error is None:
            if self.first_response == 0.0:
                self.first_response = now
            elif self.second_response == 0.0:
                self.second_response = now
        self.responses.put((result, error))


class PersistentStreamWorker:
    def __init__(
        self, grpcclient, server_url, model, padding_seconds, speaker_id,
    ):
        self.grpcclient = grpcclient
        self.model = model
        self.padding_seconds = padding_seconds
        self.speaker_id = speaker_id
        self.client = grpcclient.InferenceServerClient(
            url=server_url, verbose=False)
        self.current_state = None
        self.client.start_stream(callback=self._callback)

    def _callback(self, result, error):
        state = self.current_state
        if state is not None:
            state.callback(result, error)

    def close(self):
        self.client.stop_stream()
        self.client.close()

    def infer(self, item):
        state = ResponseState()
        inputs, outputs = prepare_request(
            self.grpcclient,
            item,
            self.padding_seconds,
            self.speaker_id,
        )
        request_id = str(uuid.uuid4())
        self.current_state = state
        state.started = time.perf_counter()
        try:
            self.client.async_stream_infer(
                self.model,
                inputs,
                request_id=request_id,
                outputs=outputs,
                enable_empty_final_response=True,
            )

            samples = 0
            chunks = 0
            while True:
                result, error = state.responses.get(timeout=300)
                if error is not None:
                    raise RuntimeError(str(error))
                response = result.get_response()
                final_parameter = response.parameters.get(
                    "triton_final_response")
                if final_parameter is not None and final_parameter.bool_param:
                    break
                waveform = result.as_numpy("waveform")
                if waveform is not None and waveform.size:
                    samples += int(waveform.size)
                    chunks += 1

            finished = time.perf_counter()
            if state.first_response == 0.0 or samples == 0:
                raise RuntimeError("stream completed without audio")
            return RequestResult(
                item_id=item.item_id,
                success=True,
                ttfa_ms=(state.first_response - state.started) * 1000,
                second_chunk_ms=(
                    (state.second_response - state.first_response) * 1000
                    if state.second_response else 0.0
                ),
                total_ms=(finished - state.started) * 1000,
                audio_seconds=samples / 24000,
                chunks=chunks,
            )
        except Exception as exc:
            return RequestResult(
                item_id=item.item_id, success=False, error=str(exc))
        finally:
            self.current_state = None


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


def latency_summary(values):
    if not values:
        return {
            "average": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "average": round(sum(values) / len(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p90": round(percentile(values, 90), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "max": round(max(values), 2),
    }


def load_parquet(path, limit=0, repeat=1):
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=path, split="train")
    row_count = min(limit, len(dataset)) if limit else len(dataset)
    items = []
    for repeat_index in range(repeat):
        for index in range(row_count):
            row = dataset[index]
            prompt_audio = row["prompt_audio"]
            waveform = np.asarray(prompt_audio["array"], dtype=np.float32)
            sample_rate = int(prompt_audio["sampling_rate"])
            if waveform.ndim != 1:
                waveform = waveform.reshape(-1)
            if sample_rate != 16000:
                from scipy.signal import resample

                sample_count = int(len(waveform) * 16000 / sample_rate)
                waveform = resample(waveform, sample_count).astype(np.float32)
            item_id = str(row["id"])
            if repeat > 1:
                item_id = f"{item_id}-repeat-{repeat_index}"
            items.append(DatasetItem(
                item_id=item_id,
                waveform=np.ascontiguousarray(waveform),
                reference_text=str(row["prompt_text"]),
                target_text=str(row["target_text"]),
            ))
    return items, len(dataset)


def prepare_request(grpcclient, item, padding_seconds, speaker_id=""):
    from tritonclient.utils import np_to_triton_dtype

    def tensor(name, value, datatype=None):
        value = np.asarray(value)
        triton_input = grpcclient.InferInput(
            name,
            list(value.shape),
            datatype or np_to_triton_dtype(value.dtype),
        )
        triton_input.set_data_from_numpy(value)
        return triton_input

    target_text = tensor(
        "target_text",
        np.array([[item.target_text]], dtype=object),
        "BYTES",
    )
    if speaker_id:
        inputs = [
            tensor(
                "speaker_id",
                np.array([[speaker_id]], dtype=object),
                "BYTES",
            ),
            target_text,
        ]
    else:
        waveform = item.waveform
        lengths = np.array([[len(waveform)]], dtype=np.int32)
        if padding_seconds:
            duration = len(waveform) / 16000
            if item.reference_text:
                estimated_target_duration = (
                    duration / len(item.reference_text) * len(item.target_text))
            else:
                estimated_target_duration = duration
            required_samples = int(
                padding_seconds
                * 16000
                * (
                    int((estimated_target_duration + duration)
                        // padding_seconds)
                    + 1
                )
            )
            samples = np.zeros((1, required_samples), dtype=np.float32)
            samples[0, :len(waveform)] = waveform
        else:
            samples = waveform.reshape(1, -1).astype(np.float32)
        inputs = [
            tensor("reference_wav", samples),
            tensor("reference_wav_len", lengths),
            tensor(
                "reference_text",
                np.array([[item.reference_text]], dtype=object),
                "BYTES",
            ),
            target_text,
        ]
    return inputs, [grpcclient.InferRequestedOutput("waveform")]


def split_contiguous(items, task_count):
    task_count = min(task_count, len(items))
    quotient, remainder = divmod(len(items), task_count)
    shards = []
    start = 0
    for index in range(task_count):
        size = quotient + (1 if index < remainder else 0)
        shards.append(items[start:start + size])
        start += size
    return shards


def stats_snapshot(grpcclient, server_url):
    client = grpcclient.InferenceServerClient(url=server_url, verbose=False)
    try:
        return client.get_inference_statistics(model_name="", as_json=True)
    finally:
        client.close()


def stats_delta(before, after):
    before_by_name = {
        item["name"]: item for item in before.get("model_stats", [])
    }
    result = {}
    for current in after.get("model_stats", []):
        name = current["name"]
        previous = before_by_name.get(name, {})
        inference_count = int(current.get("inference_count", 0)) - int(
            previous.get("inference_count", 0))
        execution_count = int(current.get("execution_count", 0)) - int(
            previous.get("execution_count", 0))
        if inference_count == 0 and execution_count == 0:
            continue
        current_stats = current.get("inference_stats", {})
        previous_stats = previous.get("inference_stats", {})

        def delta_ms(key):
            return round((
                int(current_stats.get(key, {}).get("ns", 0))
                - int(previous_stats.get(key, {}).get("ns", 0))
            ) / 1e6, 2)

        result[name] = {
            "inferenceCount": inference_count,
            "executionCount": execution_count,
            "queueMs": delta_ms("queue"),
            "computeInputMs": delta_ms("compute_input"),
            "computeInferMs": delta_ms("compute_infer"),
            "computeOutputMs": delta_ms("compute_output"),
        }
    return result


def run_profile(grpcclient, args, items, concurrency):
    shards = split_contiguous(items, concurrency)
    barrier = threading.Barrier(len(shards))

    def worker(shard):
        stream = PersistentStreamWorker(
            grpcclient,
            args.server_url,
            args.model,
            args.padding_seconds,
            args.speaker_id,
        )
        results = []
        try:
            barrier.wait(timeout=30)
            for item in shard:
                results.append(stream.infer(item))
        finally:
            stream.close()
        return results

    stats_before = stats_snapshot(grpcclient, args.server_url)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = [executor.submit(worker, shard) for shard in shards]
        results = [
            result
            for future in futures
            for result in future.result()
        ]
    wall_seconds = time.perf_counter() - started
    stats_after = stats_snapshot(grpcclient, args.server_url)

    successful = [item for item in results if item.success]
    audio_seconds = sum(item.audio_seconds for item in successful)
    return {
        "concurrency": concurrency,
        "effectiveConcurrency": len(shards),
        "requests": len(results),
        "success": len(successful),
        "failed": len(results) - len(successful),
        "wallSeconds": round(wall_seconds, 4),
        "requestsPerSecond": round(
            len(successful) / wall_seconds if wall_seconds else 0.0, 4),
        "audioSeconds": round(audio_seconds, 4),
        "audioThroughput": round(
            audio_seconds / wall_seconds if wall_seconds else 0.0, 4),
        "systemRtf": round(
            wall_seconds / audio_seconds if audio_seconds else 0.0, 6),
        "ttfaMs": latency_summary(
            [item.ttfa_ms for item in successful]),
        "secondChunkGapMs": latency_summary(
            [item.second_chunk_ms for item in successful
             if item.second_chunk_ms > 0]),
        "totalLatencyMs": latency_summary(
            [item.total_ms for item in successful]),
        "averageChunks": (
            round(
                sum(item.chunks for item in successful) / len(successful),
                2,
            )
            if successful else 0.0
        ),
        "tritonStatsDelta": stats_delta(stats_before, stats_after),
        "errors": [item.error for item in results if not item.success][:5],
    }


def parse_concurrency(value):
    result = [int(item.strip()) for item in value.split(",")]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("concurrency must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--server-url", default="127.0.0.1:18001")
    parser.add_argument("--model", default="CosyVoice3ProStreaming")
    parser.add_argument("--dataset-parquet", required=True)
    parser.add_argument("--dataset-name", default="yuekai/seed_tts_cosy2")
    parser.add_argument("--split-name", default="wenetspeech4tts")
    parser.add_argument(
        "--speaker-id",
        default="",
        help="Use a registered speaker while retaining official texts/topology",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--concurrency", type=parse_concurrency, default=[4])
    parser.add_argument("--padding-seconds", type=int, default=10)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.limit < 0 or args.repeat <= 0 or args.padding_seconds < 0:
        parser.error("limit/padding cannot be negative and repeat must be positive")

    import tritonclient.grpc as grpcclient

    items, source_rows = load_parquet(
        args.dataset_parquet, limit=args.limit, repeat=args.repeat)
    reports = []
    for concurrency in args.concurrency:
        report = run_profile(grpcclient, args, items, concurrency)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    output = {
        "standard": "upstream-cosyvoice3-triton-grpc-streaming",
        "metricSource": (
            "https://github.com/QwenAudio/CosyVoice/blob/main/"
            "runtime/triton_trtllm/client_grpc.py"
        ),
        "clientTopology": "persistent-sync-grpc-stream-per-task",
        "ttfaBoundary": "async_stream_infer-to-first-response",
        "rawPromptPerRequest": not bool(args.speaker_id),
        "registeredSpeakerId": args.speaker_id,
        "inputPaddingSeconds": (
            0 if args.speaker_id else args.padding_seconds),
        "audioWriteDuringRun": False,
        "host": platform.node(),
        "dataset": args.dataset_name,
        "split": args.split_name,
        "sourceRows": source_rows,
        "evaluatedRequests": len(items),
        "repeat": args.repeat,
        "model": args.model,
        "profiles": reports,
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
