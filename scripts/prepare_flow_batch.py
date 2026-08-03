#!/usr/bin/env python3
"""Build the CosyVoice3 dynamic-batch Flow estimator.

The official CosyVoice3 estimator ONNX fixes the classifier-free-guidance
dimension metadata to two. Its mixed-precision graph itself is batch-agnostic,
so a business batch B can use classifier-free-guidance batch 2*B after the
input metadata is made dynamic. This preserves the official selective FP16/
FP32 precision policy; forcing the whole estimator to FP16 can produce NaNs.
Generated artifacts stay beside the downloaded model and are not stored here.
"""

import argparse
from pathlib import Path
import sys
import time

INPUT_NAMES = ["x", "mask", "mu", "t", "spks", "cond"]
OUTPUT_NAMES = ["estimator_out"]


def log(message):
    print(f"[flow-batch] {message}", flush=True)


def validate_dynamic_onnx(path):
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    inputs = {item.name: item for item in model.graph.input}
    for name in INPUT_NAMES:
        if name not in inputs:
            raise RuntimeError(f"dynamic ONNX is missing input {name}")
        first_dim = inputs[name].type.tensor_type.shape.dim[0]
        if not first_dim.dim_param:
            raise RuntimeError(
                f"dynamic ONNX input {name} has a static batch dimension")


def make_batch_dynamic(source_path, output_path):
    import onnx

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    log(f"making official mixed-precision ONNX batch-dynamic: {output_path}")
    started = time.perf_counter()
    model = onnx.load(str(source_path), load_external_data=False)
    inputs = {item.name: item for item in model.graph.input}
    for name in INPUT_NAMES:
        if name not in inputs:
            raise RuntimeError(f"official mixed-precision ONNX misses {name}")
        first_dim = inputs[name].type.tensor_type.shape.dim[0]
        first_dim.ClearField("dim_value")
        first_dim.dim_param = "cfg_batch"
    outputs = {item.name: item for item in model.graph.output}
    if OUTPUT_NAMES[0] not in outputs:
        raise RuntimeError("official mixed-precision ONNX misses estimator_out")
    output_batch = outputs[OUTPUT_NAMES[0]].type.tensor_type.shape.dim[0]
    output_batch.ClearField("dim_value")
    output_batch.dim_param = "cfg_batch"
    onnx.save(model, str(temporary))
    validate_dynamic_onnx(temporary)
    temporary.replace(output_path)
    log(f"dynamic ONNX prepared in {time.perf_counter() - started:.1f}s")


def profile_shapes(business_batch, frames):
    cfg_batch = business_batch * 2
    return {
        "x": (cfg_batch, 80, frames),
        "mask": (cfg_batch, 1, frames),
        "mu": (cfg_batch, 80, frames),
        "t": (cfg_batch,),
        "spks": (cfg_batch, 80),
        "cond": (cfg_batch, 80, frames),
    }


def build_plan(onnx_path, plan_path, opt_batch, max_batch):
    import tensorrt as trt

    temporary = plan_path.with_suffix(plan_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    log(
        f"building FP16 TensorRT plan: opt_batch={opt_batch}, "
        f"max_batch={max_batch}")
    started = time.perf_counter()
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    # B=4 (CFG batch 8) needs a larger build-time tactic workspace than the
    # official static B=1 engine. This does not become persistent runtime VRAM.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)

    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            errors = "\n".join(
                str(parser.get_error(index))
                for index in range(parser.num_errors)
            )
            raise RuntimeError(f"failed to parse dynamic Flow ONNX:\n{errors}")

    profile = builder.create_optimization_profile()
    minimum = profile_shapes(1, 4)
    optimum = profile_shapes(opt_batch, 500)
    maximum = profile_shapes(max_batch, 3000)
    for name in INPUT_NAMES:
        profile.set_shape(name, minimum[name], optimum[name], maximum[name])
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the dynamic Flow plan")
    with open(temporary, "wb") as plan_file:
        plan_file.write(serialized)
    temporary.replace(plan_path)
    log(f"TensorRT build completed in {time.perf_counter() - started:.1f}s")


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--max-batch-size", type=int, default=4)
    result.add_argument("--opt-batch-size", type=int, default=4)
    result.add_argument("--force-onnx", action="store_true")
    result.add_argument("--force-build", action="store_true")
    return result


def main():
    import torch

    args = parser().parse_args()
    if args.max_batch_size < 2:
        log("max batch is 1; no dynamic Flow asset is required")
        return
    if not 1 <= args.opt_batch_size <= args.max_batch_size:
        raise SystemExit("opt batch size must be between 1 and max batch size")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to export/build the Flow batch plan")

    model_dir = args.model_dir.resolve()
    source_path = model_dir / "flow.decoder.estimator.autocast_fp16.onnx"
    onnx_path = model_dir / (
        "flow.decoder.estimator.autocast_fp16.dynamic_batch.onnx")
    plan_path = model_dir / (
        "flow.decoder.estimator.autocast_fp16.dynamic_batch."
        f"{args.max_batch_size}.plan")

    if not source_path.exists():
        raise SystemExit(f"official mixed-precision ONNX not found: {source_path}")
    if args.force_onnx:
        onnx_path.unlink(missing_ok=True)
    if args.force_build:
        plan_path.unlink(missing_ok=True)

    if not onnx_path.exists():
        make_batch_dynamic(source_path, onnx_path)
    else:
        validate_dynamic_onnx(onnx_path)
        log(f"reusing dynamic ONNX: {onnx_path}")

    if not plan_path.exists():
        build_plan(
            onnx_path,
            plan_path,
            opt_batch=args.opt_batch_size,
            max_batch=args.max_batch_size,
        )
    else:
        log(f"reusing TensorRT plan: {plan_path}")

    log("dynamic Flow assets are ready")


if __name__ == "__main__":
    sys.exit(main())
