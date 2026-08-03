# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 CosyVoice3Pro contributors.
# SPDX-License-Identifier: Apache-2.0

"""Offline-batched Causal HiFT vocoder backend."""

import json
import logging
import os

import torch
from hyperpyyaml import load_hyperpyyaml
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
torch.set_num_threads(1)


class TritonPythonModel:
    """Batch compatible wrapper around CosyVoice3 CausalHiFTGenerator."""

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
        self.hift = configs["hift"]
        state = {
            key.replace("generator.", ""): value
            for key, value in torch.load(
                os.path.join(model_dir, "hift.pt"),
                map_location="cpu",
                weights_only=True,
            ).items()
        }
        self.hift.load_state_dict(state, strict=True)
        self.hift.to(self.device).eval()
        self.samples_per_mel_frame = int(
            torch.tensor(self.hift.upsample_rates).prod().item()
            * self.hift.istft_params["hop_len"]
        )
        logger.info(
            "vocoder initialized: max_batch=%s, samples_per_mel=%s",
            self.max_batch_size,
            self.samples_per_mel_frame,
        )

    @staticmethod
    def _tensor(request, name, required=True):
        value = pb_utils.get_input_tensor_by_name(request, name)
        if value is None and required:
            raise ValueError(f"missing required input: {name}")
        return value

    def _read_request(self, request):
        mel = torch.utils.dlpack.from_dlpack(
            self._tensor(request, "mel").to_dlpack()
        ).to(self.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.dim() != 3 or mel.shape[0] != 1 or mel.shape[1] != 80:
            raise ValueError("mel must have shape [1, 80, frames]")

        mel_len_pb = self._tensor(request, "mel_len", required=False)
        mel_len = (
            int(mel_len_pb.as_numpy().reshape(-1)[0])
            if mel_len_pb is not None else int(mel.shape[2])
        )
        if not 0 < mel_len <= mel.shape[2]:
            raise ValueError("invalid mel_len")
        finalize = bool(
            self._tensor(request, "finalize").as_numpy().reshape(-1)[0])
        return {
            "mel": mel.contiguous(),
            "mel_len": mel_len,
            "finalize": finalize,
        }

    def _infer_one(self, sample):
        mel = sample["mel"][:, :, :sample["mel_len"]].contiguous()
        with torch.no_grad():
            speech, _ = self.hift.inference(
                speech_feat=mel, finalize=sample["finalize"])
        return speech.reshape(1, -1).float()

    def _infer_batch(self, samples):
        padded_frames = samples[0]["mel"].shape[2]
        if any(sample["mel"].shape[2] != padded_frames
               for sample in samples):
            raise ValueError("batched vocoder requests must share a Mel bucket")
        mel_batch = torch.cat([sample["mel"] for sample in samples], dim=0)
        with torch.no_grad():
            speech_batch, _ = self.hift.inference(
                speech_feat=mel_batch, finalize=True)
        speech_batch = speech_batch.reshape(len(samples), -1).float()

        outputs = []
        for index, sample in enumerate(samples):
            valid_samples = sample["mel_len"] * self.samples_per_mel_frame
            if valid_samples > speech_batch.shape[1]:
                raise RuntimeError(
                    "vocoder output is shorter than expected from mel_len")
            outputs.append(
                speech_batch[index:index + 1, :valid_samples].contiguous())
        return outputs

    @staticmethod
    def _response(speech):
        tensor = pb_utils.Tensor.from_dlpack(
            "tts_speech", to_dlpack(speech.contiguous()))
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
            len(samples) > 1
            and len(samples) <= self.max_batch_size
            and all(sample["finalize"] for sample in samples)
        )
        if use_batch:
            try:
                outputs = self._infer_batch(samples)
                return [self._response(output) for output in outputs]
            except Exception as exc:
                logger.exception(
                    "batched vocoder failed; falling back to B=1: %s", exc)

        responses = []
        for sample in samples:
            try:
                responses.append(self._response(self._infer_one(sample)))
            except Exception as exc:
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(exc))))
        return responses
