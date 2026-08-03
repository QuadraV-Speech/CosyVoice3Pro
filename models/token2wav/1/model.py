# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 CosyVoice3Pro contributors.
# SPDX-License-Identifier: Apache-2.0

"""Offline-batched CosyVoice3 Flow backend.

The upstream CosyVoice3 inference path fixes the business batch to one and the
classifier-free-guidance batch to two. This backend preserves that path as a
fallback, while allowing Triton's dynamic batcher to collate compatible
offline requests and run Flow with a real business batch.
"""

import json
import logging
import os
import queue
import types

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils

from cosyvoice.utils.mask import make_pad_mask


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
torch.set_num_threads(1)


class TrtContextWrapper:
    def __init__(self, trt_engine, trt_concurrent=1, device="cuda:0"):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        self.device = device
        for _ in range(trt_concurrent):
            context = trt_engine.create_execution_context()
            stream = torch.cuda.stream(torch.cuda.Stream(torch.device(device)))
            if context is None:
                raise RuntimeError("failed to create TensorRT execution context")
            self.trt_context_pool.put([context, stream])

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])


@torch.inference_mode()
def dynamic_decoder_forward(decoder, mu, mask, n_timesteps, temperature=1.0,
                            spks=None, cond=None, streaming=False):
    """Expand CosyVoice3's deterministic noise template to the real batch."""
    business_batch = mu.size(0)
    noise = decoder.rand_noise[:, :, :mu.size(2)].to(
        device=mu.device, dtype=mu.dtype)
    noise = noise.expand(business_batch, -1, -1).clone() * temperature
    t_span = torch.linspace(
        0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    if decoder.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    return decoder.solve_euler(
        noise,
        t_span=t_span,
        mu=mu,
        mask=mask,
        spks=spks,
        cond=cond,
        streaming=streaming,
    ), None


def dynamic_solve_euler(decoder, x, t_span, mu, mask, spks, cond,
                        streaming=False):
    """Batch-safe classifier-free-guidance Euler solver."""
    t = t_span[0].unsqueeze(dim=0)
    dt = t_span[1] - t_span[0]
    business_batch = x.size(0)
    cfg_batch = business_batch * 2
    frames = x.size(2)

    x_in = torch.zeros(
        [cfg_batch, 80, frames], device=x.device, dtype=spks.dtype)
    mask_in = torch.zeros(
        [cfg_batch, 1, frames], device=x.device, dtype=spks.dtype)
    mu_in = torch.zeros_like(x_in)
    t_in = torch.zeros([cfg_batch], device=x.device, dtype=spks.dtype)
    spks_in = torch.zeros(
        [cfg_batch, 80], device=x.device, dtype=spks.dtype)
    cond_in = torch.zeros_like(x_in)

    for step in range(1, len(t_span)):
        # Conditional and unconditional halves use the same noise and mask.
        x_in[:business_batch] = x
        x_in[business_batch:] = x
        mask_in[:business_batch] = mask
        mask_in[business_batch:] = mask
        mu_in.zero_()
        mu_in[:business_batch] = mu
        t_in[:] = t
        spks_in.zero_()
        spks_in[:business_batch] = spks
        cond_in.zero_()
        cond_in[:business_batch] = cond

        prediction = decoder.forward_estimator(
            x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming)
        conditional, unconditional = torch.split(
            prediction, [business_batch, business_batch], dim=0)
        prediction = (
            (1.0 + decoder.inference_cfg_rate) * conditional
            - decoder.inference_cfg_rate * unconditional
        )
        x = x + dt * prediction
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    return x.float()


def dynamic_forward_estimator(decoder, x, mask, mu, t, spks, cond,
                              streaming=False):
    """Run either PyTorch or TensorRT estimator with a dynamic CFG batch."""
    if isinstance(decoder.estimator, torch.nn.Module):
        return decoder.estimator(
            x, mask, mu, t, spks, cond, streaming=streaming)

    [context, stream], engine = decoder.estimator.acquire_estimator()
    cfg_batch = x.size(0)
    frames = x.size(2)
    try:
        torch.cuda.current_stream().synchronize()
        with stream:
            context.set_input_shape("x", (cfg_batch, 80, frames))
            context.set_input_shape("mask", (cfg_batch, 1, frames))
            context.set_input_shape("mu", (cfg_batch, 80, frames))
            context.set_input_shape("t", (cfg_batch,))
            context.set_input_shape("spks", (cfg_batch, 80))
            context.set_input_shape("cond", (cfg_batch, 80, frames))
            pointers = [
                x.contiguous().data_ptr(),
                mask.contiguous().data_ptr(),
                mu.contiguous().data_ptr(),
                t.contiguous().data_ptr(),
                spks.contiguous().data_ptr(),
                cond.contiguous().data_ptr(),
                x.data_ptr(),
            ]
            for index, pointer in enumerate(pointers):
                context.set_tensor_address(engine.get_tensor_name(index), pointer)
            if not context.execute_async_v3(
                    torch.cuda.current_stream().cuda_stream):
                raise RuntimeError("TensorRT Flow estimator execution failed")
            torch.cuda.current_stream().synchronize()
        return x
    finally:
        decoder.estimator.release_estimator(context, stream)


class TritonPythonModel:
    """CosyVoice3 Flow backend with offline dynamic batching."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        parameters = self.model_config["parameters"]
        model_params = {
            key: value["string_value"] for key, value in parameters.items()
        }
        self.max_batch_size = max(
            1, int(self.model_config.get("max_batch_size", 1)))
        self.device = torch.device("cuda")
        model_dir = model_params["model_dir"]

        with open(os.path.join(model_dir, "cosyvoice3.yaml"), "r") as config:
            configs = load_hyperpyyaml(config, overrides={
                "qwen_pretrain_path": os.path.join(
                    model_dir, "CosyVoice-BlankEN")
            })
        self.flow = configs["flow"]
        self.flow.half()
        self.flow.load_state_dict(
            torch.load(
                os.path.join(model_dir, "flow.pt"),
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        self.flow.to(self.device).eval()
        self.token_mel_ratio = self.flow.token_mel_ratio

        dynamic_plan = os.path.join(
            model_dir,
            model_params.get(
                "dynamic_batch_plan",
                f"flow.decoder.estimator.autocast_fp16.dynamic_batch."
                f"{self.max_batch_size}.plan",
            ),
        )
        fallback_plan = os.path.join(
            model_dir,
            f"flow.decoder.estimator.autocast_fp16."
            f"{torch.cuda.current_device()}.plan",
        )
        self.batch_capable = os.path.isfile(dynamic_plan)
        plan_path = dynamic_plan if self.batch_capable else fallback_plan
        if not os.path.isfile(plan_path):
            raise RuntimeError(f"Flow TensorRT plan not found: {plan_path}")

        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(plan_path, "rb") as plan_file:
            engine = trt.Runtime(
                trt.Logger(trt.Logger.INFO)
            ).deserialize_cuda_engine(plan_file.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize Flow plan: {plan_path}")
        self.flow.decoder.estimator = TrtContextWrapper(
            engine, trt_concurrent=1, device=str(self.device))

        # Patch only this loaded decoder instance; upstream source files remain
        # untouched and the B=1 fallback uses equivalent math.
        self.flow.decoder.solve_euler = types.MethodType(
            dynamic_solve_euler, self.flow.decoder)
        self.flow.decoder.forward_estimator = types.MethodType(
            dynamic_forward_estimator, self.flow.decoder)
        self.flow.decoder.forward = types.MethodType(
            dynamic_decoder_forward, self.flow.decoder)

        logger.info(
            "token2wav initialized: max_batch=%s, batch_capable=%s, plan=%s",
            self.max_batch_size,
            self.batch_capable,
            plan_path,
        )

    @staticmethod
    def _tensor(request, name, required=True):
        value = pb_utils.get_input_tensor_by_name(request, name)
        if value is None and required:
            raise ValueError(f"missing required input: {name}")
        return value

    @staticmethod
    def _length(request, name, fallback):
        value = pb_utils.get_input_tensor_by_name(request, name)
        if value is None:
            return int(fallback)
        return int(value.as_numpy().reshape(-1)[0])

    def _read_request(self, request):
        target = torch.utils.dlpack.from_dlpack(
            self._tensor(request, "target_speech_tokens").to_dlpack()
        ).to(self.device)
        prompt_token = torch.utils.dlpack.from_dlpack(
            self._tensor(request, "prompt_speech_tokens").to_dlpack()
        ).to(self.device)
        prompt_feat = torch.utils.dlpack.from_dlpack(
            self._tensor(request, "prompt_speech_feat").to_dlpack()
        ).to(self.device)
        embedding = torch.utils.dlpack.from_dlpack(
            self._tensor(request, "prompt_spk_embedding").to_dlpack()
        ).to(self.device)

        if target.dim() == 1:
            target = target.unsqueeze(0)
        if prompt_token.dim() == 1:
            prompt_token = prompt_token.unsqueeze(0)
        if prompt_feat.dim() == 2:
            prompt_feat = prompt_feat.unsqueeze(0)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if any(value.shape[0] != 1 for value in (
                target, prompt_token, prompt_feat, embedding)):
            raise ValueError("token2wav expects one sample per Triton request")

        target_len = self._length(
            request, "target_speech_tokens_len", target.shape[1])
        prompt_token_len = self._length(
            request, "prompt_speech_tokens_len", prompt_token.shape[1])
        prompt_feat_len = self._length(
            request, "prompt_speech_feat_len", prompt_feat.shape[1])
        if not 0 < target_len <= target.shape[1]:
            raise ValueError("invalid target_speech_tokens_len")
        if not 0 < prompt_token_len <= prompt_token.shape[1]:
            raise ValueError("invalid prompt_speech_tokens_len")
        if not 0 < prompt_feat_len <= prompt_feat.shape[1]:
            raise ValueError("invalid prompt_speech_feat_len")
        if prompt_feat.shape[2] != self.flow.output_size:
            raise ValueError("prompt_speech_feat must have 80 channels")

        token_offset_pb = self._tensor(request, "token_offset", required=False)
        finalize_pb = self._tensor(request, "finalize", required=False)
        token_offset = (
            int(token_offset_pb.as_numpy().reshape(-1)[0])
            if token_offset_pb is not None else None
        )
        finalize = (
            bool(finalize_pb.as_numpy().reshape(-1)[0])
            if finalize_pb is not None else True
        )
        return {
            "target": target[:, :target_len].contiguous(),
            "target_len": target_len,
            "prompt_token": prompt_token[:, :prompt_token_len].contiguous(),
            "prompt_token_len": prompt_token_len,
            "prompt_feat": prompt_feat[:, :prompt_feat_len].contiguous(),
            "prompt_feat_len": prompt_feat_len,
            "embedding": embedding.contiguous(),
            "token_offset": token_offset,
            "finalize": finalize,
        }

    @staticmethod
    def _offline(sample):
        return sample["finalize"] and sample["token_offset"] is None

    def _infer_one(self, sample):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            mel, _ = self.flow.inference(
                token=sample["target"],
                token_len=torch.tensor(
                    [sample["target_len"]],
                    dtype=torch.int32,
                    device=self.device,
                ),
                prompt_token=sample["prompt_token"],
                prompt_token_len=torch.tensor(
                    [sample["prompt_token_len"]],
                    dtype=torch.int32,
                    device=self.device,
                ),
                prompt_feat=sample["prompt_feat"],
                prompt_feat_len=torch.tensor(
                    [sample["prompt_feat_len"]],
                    dtype=torch.int32,
                    device=self.device,
                ),
                embedding=sample["embedding"],
                streaming=not sample["finalize"],
                finalize=sample["finalize"],
            )
        if sample["token_offset"] is not None:
            mel = mel[:, :, sample["token_offset"] * self.token_mel_ratio:]
        return [mel.squeeze(0).float()]

    def _infer_batch(self, samples):
        batch_size = len(samples)
        combined_lengths = torch.tensor(
            [
                sample["prompt_token_len"] + sample["target_len"]
                for sample in samples
            ],
            dtype=torch.int64,
            device=self.device,
        )
        max_tokens = int(combined_lengths.max().item())
        combined = torch.zeros(
            [batch_size, max_tokens],
            dtype=torch.int32,
            device=self.device,
        )
        for index, sample in enumerate(samples):
            prompt_len = sample["prompt_token_len"]
            target_len = sample["target_len"]
            combined[index, :prompt_len] = sample["prompt_token"][0]
            combined[index, prompt_len:prompt_len + target_len] = (
                sample["target"][0])

        embeddings = torch.cat(
            [sample["embedding"] for sample in samples], dim=0)
        embeddings = F.normalize(embeddings, dim=1)
        embeddings = self.flow.spk_embed_affine_layer(embeddings)

        token_mask = (~make_pad_mask(
            combined_lengths, max_len=max_tokens
        )).unsqueeze(-1).to(embeddings)
        encoded = self.flow.input_embedding(
            torch.clamp(combined, min=0)) * token_mask
        encoded = self.flow.pre_lookahead_layer(encoded)
        encoded = encoded.repeat_interleave(
            self.token_mel_ratio, dim=1)

        total_mel_lengths = combined_lengths * self.token_mel_ratio
        max_mel_frames = encoded.shape[1]
        conditions = torch.zeros(
            [batch_size, max_mel_frames, self.flow.output_size],
            dtype=encoded.dtype,
            device=self.device,
        )
        for index, sample in enumerate(samples):
            prompt_frames = sample["prompt_feat_len"]
            if prompt_frames > int(total_mel_lengths[index].item()):
                raise ValueError(
                    "prompt Mel is longer than combined Token condition")
            conditions[index, :prompt_frames] = sample["prompt_feat"][0]

        mel_mask = (~make_pad_mask(
            total_mel_lengths, max_len=max_mel_frames
        )).to(encoded)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            features, _ = self.flow.decoder(
                mu=encoded.transpose(1, 2).contiguous(),
                mask=mel_mask.unsqueeze(1),
                spks=embeddings,
                cond=conditions.transpose(1, 2).contiguous(),
                n_timesteps=10,
                streaming=False,
            )

        outputs = []
        for index, sample in enumerate(samples):
            start = sample["prompt_feat_len"]
            end = int(total_mel_lengths[index].item())
            outputs.append(features[index, :, start:end].float())
        return outputs

    @staticmethod
    def _response(mel):
        # Python backend DLPack currently requires the variable-length output
        # to be materialized on CPU before constructing individual responses.
        mel = mel.contiguous().cpu()
        tensor = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel))
        return pb_utils.InferenceResponse(output_tensors=[tensor])

    def execute(self, requests):
        try:
            samples = [self._read_request(request) for request in requests]
        except Exception as exc:
            return [
                pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(exc)))
                for _ in requests
            ]

        use_batch = (
            self.batch_capable
            and len(samples) > 1
            and len(samples) <= self.max_batch_size
            and all(self._offline(sample) for sample in samples)
        )
        if use_batch:
            try:
                outputs = self._infer_batch(samples)
                return [self._response(output) for output in outputs]
            except Exception as exc:
                logger.exception(
                    "batched Flow failed; falling back to B=1: %s", exc)

        responses = []
        for sample in samples:
            try:
                responses.append(self._response(self._infer_one(sample)[0]))
            except Exception as exc:
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(exc))))
        return responses
