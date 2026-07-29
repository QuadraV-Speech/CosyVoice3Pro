#!/usr/bin/env python3
import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

SYNTHESIS_MODEL = "CosyVoice3Pro"
REGISTRY_MODEL = "CosyVoice3ProSpeakerRegistry"


def triton_input(name, shape, datatype, data):
    return {
        "name": name,
        "shape": shape,
        "datatype": datatype,
        "data": data,
    }


def post_infer(base_url, model_name, inputs, timeout):
    url = f"{base_url.rstrip('/')}/v2/models/{model_name}/infer"
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        url,
        json={"inputs": inputs},
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"Triton returned HTTP {response.status_code}: {response.text[:2000]}")
    return response.json()


def output_map(payload):
    return {
        item["name"]: item.get("data", [])
        for item in payload.get("outputs", [])
    }


def first_output_value(outputs, name, default=""):
    values = outputs.get(name) or []
    return values[0] if values else default


def audio_bytes_to_wav_bytes(audio_bytes, rate=16000):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        "-ar", str(rate),
        "-ac", "1",
        "pipe:1",
    ]
    process = subprocess.run(
        command,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio conversion failed: "
            + process.stderr.decode(errors="ignore"))
    return process.stdout


def load_reference_audio(path):
    source = Path(path)
    wav_bytes = audio_bytes_to_wav_bytes(source.read_bytes(), rate=16000)
    waveform, sample_rate = sf.read(
        io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise RuntimeError(f"unexpected converted sample rate: {sample_rate}")
    waveform = waveform[:, 0]
    if waveform.size == 0:
        raise RuntimeError("reference audio is empty")
    return np.ascontiguousarray(waveform.reshape(1, -1), dtype=np.float32)


def registry_inputs(operation, speaker_id=""):
    inputs = [
        triton_input("operation", [1, 1], "BYTES", [operation]),
    ]
    if speaker_id:
        inputs.append(
            triton_input("speaker_id", [1, 1], "BYTES", [speaker_id]))
    return inputs


def print_registry_response(payload):
    outputs = output_map(payload)
    message = first_output_value(outputs, "message")
    try:
        message = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        pass
    result = {
        "status": first_output_value(outputs, "status"),
        "speaker_version": first_output_value(outputs, "speaker_version"),
        "message": message,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_register_inputs(args):
    waveform = load_reference_audio(args.audio)
    inputs = registry_inputs("register", args.speaker_id)
    inputs.extend([
        triton_input(
            "reference_wav",
            list(waveform.shape),
            "FP32",
            waveform.tolist(),
        ),
        triton_input(
            "reference_wav_len",
            [1, 1],
            "INT32",
            [[int(waveform.shape[1])]],
        ),
        triton_input(
            "reference_text",
            [1, 1],
            "BYTES",
            [args.reference_text],
        ),
        triton_input(
            "prompt",
            [1, 1],
            "BYTES",
            [args.prompt],
        ),
    ])
    return inputs


def command_register(args):
    inputs = build_register_inputs(args)
    payload = post_infer(
        args.url, REGISTRY_MODEL, inputs, args.timeout)
    print_registry_response(payload)


def command_build_register_json(args):
    print(json.dumps(
        {"inputs": build_register_inputs(args)},
        ensure_ascii=False,
        separators=(",", ":"),
    ))


def command_inspect(args):
    payload = post_infer(
        args.url,
        REGISTRY_MODEL,
        registry_inputs("inspect", args.speaker_id),
        args.timeout,
    )
    print_registry_response(payload)


def command_list(args):
    payload = post_infer(
        args.url,
        REGISTRY_MODEL,
        registry_inputs("list"),
        args.timeout,
    )
    print_registry_response(payload)


def command_delete(args):
    payload = post_infer(
        args.url,
        REGISTRY_MODEL,
        registry_inputs("delete", args.speaker_id),
        args.timeout,
    )
    print_registry_response(payload)


def save_inference_output(payload, output, details):
    outputs = output_map(payload)
    waveform_data = outputs.get("waveform")
    if not waveform_data:
        raise RuntimeError("Triton response does not contain waveform data")
    waveform = np.asarray(waveform_data, dtype=np.float32).reshape(-1)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, 24000, subtype="PCM_16")
    result = {
        "status": "ok",
        "output": str(output_path),
        "sample_rate": 24000,
        "samples": int(waveform.size),
        "duration_seconds": round(waveform.size / 24000, 3),
    }
    result.update(details)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_infer(args):
    inputs = [
        triton_input("speaker_id", [1, 1], "BYTES", [args.speaker_id]),
        triton_input(
            "prompt", [1, 1], "BYTES", [args.prompt]),
        triton_input("target_text", [1, 1], "BYTES", [args.text]),
    ]
    payload = post_infer(args.url, SYNTHESIS_MODEL, inputs, args.timeout)
    save_inference_output(
        payload,
        args.output,
        {
            "speaker_id": args.speaker_id,
            "prompt": args.prompt,
        },
    )


def command_infer_raw(args):
    waveform = load_reference_audio(args.audio)
    inputs = [
        triton_input(
            "reference_wav",
            list(waveform.shape),
            "FP32",
            waveform.tolist(),
        ),
        triton_input(
            "reference_wav_len",
            [1, 1],
            "INT32",
            [[int(waveform.shape[1])]],
        ),
        triton_input(
            "reference_text",
            [1, 1],
            "BYTES",
            [args.reference_text],
        ),
        triton_input(
            "prompt", [1, 1], "BYTES", [args.prompt]),
        triton_input("target_text", [1, 1], "BYTES", [args.text]),
    ]
    payload = post_infer(args.url, SYNTHESIS_MODEL, inputs, args.timeout)
    save_inference_output(
        payload,
        args.output,
        {
            "mode": "raw_prompt",
            "prompt": args.prompt,
        },
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="CosyVoice3Pro Triton speaker registry client")
    parser.add_argument(
        "--url", default="http://127.0.0.1:18000",
        help="Triton HTTP base URL")
    parser.add_argument(
        "--timeout", type=float, default=180,
        help="request timeout in seconds")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser(
        "register", help="register or replace a speaker")
    register_parser.add_argument("--speaker-id", required=True)
    register_parser.add_argument("--audio", required=True)
    register_parser.add_argument("--reference-text", required=True)
    register_parser.add_argument(
        "--prompt", default="",
        help="optional default speaker persona")
    register_parser.set_defaults(handler=command_register)

    build_register_parser = subparsers.add_parser(
        "build-register-json",
        help="write a Triton register request JSON to stdout",
    )
    build_register_parser.add_argument("--speaker-id", required=True)
    build_register_parser.add_argument("--audio", required=True)
    build_register_parser.add_argument("--reference-text", required=True)
    build_register_parser.add_argument(
        "--prompt", default="",
        help="optional default speaker persona")
    build_register_parser.set_defaults(handler=command_build_register_json)

    infer_parser = subparsers.add_parser(
        "infer", help="synthesize with a registered speaker")
    infer_parser.add_argument("--speaker-id", required=True)
    infer_parser.add_argument("--text", required=True)
    infer_parser.add_argument(
        "--prompt", "--instruct-text", dest="prompt", default="",
        help="optional per-request persona override")
    infer_parser.add_argument("--output", required=True)
    infer_parser.set_defaults(handler=command_infer)

    infer_raw_parser = subparsers.add_parser(
        "infer-raw", help="synthesize with a raw prompt (legacy-compatible)")
    infer_raw_parser.add_argument("--audio", required=True)
    infer_raw_parser.add_argument("--reference-text", required=True)
    infer_raw_parser.add_argument("--text", required=True)
    infer_raw_parser.add_argument(
        "--prompt", "--instruct-text", dest="prompt", default="",
        help="optional per-request persona override")
    infer_raw_parser.add_argument("--output", required=True)
    infer_raw_parser.set_defaults(handler=command_infer_raw)

    inspect_parser = subparsers.add_parser(
        "inspect", help="inspect one registered speaker")
    inspect_parser.add_argument("--speaker-id", required=True)
    inspect_parser.set_defaults(handler=command_inspect)

    list_parser = subparsers.add_parser(
        "list", help="list registered speakers")
    list_parser.set_defaults(handler=command_list)

    delete_parser = subparsers.add_parser(
        "delete", help="delete one registered speaker")
    delete_parser.add_argument("--speaker-id", required=True)
    delete_parser.set_defaults(handler=command_delete)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
