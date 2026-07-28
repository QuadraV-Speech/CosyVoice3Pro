import hashlib
import json
import os
import re
import tempfile
import time
from functools import partial

import numpy as np
import torch
import torchaudio
import triton_python_backend_utils as pb_utils
from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram
from torch.utils.dlpack import to_dlpack


torch.set_num_threads(1)

SAMPLE_RATE = 16000
SPEAKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
INSTRUCTION_PREFIX = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"
DEFAULT_INSTRUCTION = f"{INSTRUCTION_PREFIX}{END_OF_PROMPT}"
MAX_PROMPT_LENGTH = 512

mel_spectrogram = partial(
    matcha_mel_spectrogram,
    n_fft=1920,
    num_mels=80,
    sampling_rate=24000,
    hop_size=480,
    win_size=1920,
    fmin=0,
    fmax=None,
    center=False,
)


class TritonPythonModel:
    """Persistent prompt-feature registry for CosyVoice3Pro speakers."""

    def initialize(self, args):
        self.logger = pb_utils.Logger
        model_config = json.loads(args["model_config"])
        parameters = model_config.get("parameters", {})
        model_params = {
            key: value["string_value"] for key, value in parameters.items()
        }

        self.device = torch.device("cuda")
        self.speaker_store_dir = model_params.get(
            "speaker_store_dir", "/workspace/cosyvoice_speaker_store")
        self.max_reference_seconds = max(
            1.0, float(model_params.get("max_reference_seconds", "30")))
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=SAMPLE_RATE, new_freq=24000).to(self.device)
        os.makedirs(self.speaker_store_dir, exist_ok=True)
        self.logger.log_info(
            f"Speaker registry initialized store={self.speaker_store_dir}, "
            f"max_reference_seconds={self.max_reference_seconds}")

    @staticmethod
    def _input_tensor(request, name):
        return pb_utils.get_input_tensor_by_name(request, name)

    @classmethod
    def _decode_string(cls, request, name, required=False):
        tensor = cls._input_tensor(request, name)
        if tensor is None:
            if required:
                raise ValueError(f"{name} is required")
            return ""
        values = tensor.as_numpy().reshape(-1)
        if values.size != 1:
            raise ValueError(f"{name} must contain exactly one value")
        value = values[0]
        result = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        result = result.strip()
        if required and not result:
            raise ValueError(f"{name} must not be empty")
        return result

    @classmethod
    def _compose_reference_text(cls, reference_text, prompt):
        reference_text = reference_text.strip()
        prompt = prompt.strip()
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"prompt must not exceed {MAX_PROMPT_LENGTH} characters")

        existing_instruction, separator, reference_transcript = (
            reference_text.partition(END_OF_PROMPT))
        if not separator:
            reference_transcript = reference_text
            existing_instruction = INSTRUCTION_PREFIX
        if not reference_transcript.strip():
            raise ValueError(
                "reference_text must contain the prompt audio transcript")

        if prompt:
            if END_OF_PROMPT in prompt:
                instruction, _, trailing_text = prompt.partition(END_OF_PROMPT)
                if trailing_text.strip():
                    raise ValueError(
                        f"prompt must not contain text after {END_OF_PROMPT}")
                instruction = instruction.strip()
            else:
                instruction = prompt
            if not instruction:
                instruction = existing_instruction.strip() or INSTRUCTION_PREFIX
            if not instruction.startswith(INSTRUCTION_PREFIX):
                instruction = f"{INSTRUCTION_PREFIX} {instruction}"
        else:
            instruction = existing_instruction.strip() or INSTRUCTION_PREFIX

        composed = (
            f"{instruction}{END_OF_PROMPT}{reference_transcript.strip()}")
        return (
            composed,
            cls._persona_from_reference_text(composed),
            reference_transcript.strip(),
        )

    @staticmethod
    def _persona_from_reference_text(reference_text):
        instruction, separator, _ = reference_text.partition(END_OF_PROMPT)
        if not separator:
            return ""
        instruction = instruction.strip()
        if instruction.startswith(INSTRUCTION_PREFIX):
            instruction = instruction[len(INSTRUCTION_PREFIX):].strip()
        return instruction

    @staticmethod
    def _validate_speaker_id(speaker_id):
        if not SPEAKER_ID_PATTERN.fullmatch(speaker_id) or ".." in speaker_id:
            raise ValueError(
                "speaker_id must be 1-128 characters using letters, digits, "
                "underscore, dash, or dot; '..' is not allowed")

    def _speaker_path(self, speaker_id):
        self._validate_speaker_id(speaker_id)
        return os.path.join(self.speaker_store_dir, f"{speaker_id}.npz")

    @staticmethod
    def _string_tensor(name, value):
        return pb_utils.Tensor(
            name, np.array([[str(value).encode("utf-8")]], dtype=object))

    def _response(self, status, message, speaker_version=""):
        if not isinstance(message, str):
            message = json.dumps(
                message, ensure_ascii=False, separators=(",", ":"))
        return pb_utils.InferenceResponse(output_tensors=[
            self._string_tensor("status", status),
            self._string_tensor("message", message),
            self._string_tensor("speaker_version", speaker_version),
        ])

    @staticmethod
    def _extract_metadata(data):
        if "metadata_json" not in data:
            return {}
        raw = data["metadata_json"].item()
        return json.loads(str(raw))

    def _read_metadata(self, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = self._extract_metadata(data)
            if not metadata:
                metadata = {
                    "speaker_id": str(data["speaker_id"].item()),
                    "speaker_version": str(data["speaker_version"].item()),
                    "reference_text": str(data["reference_text"].item()),
                }
            stored_reference_text = str(data["reference_text"].item())
            metadata.setdefault(
                "prompt",
                self._persona_from_reference_text(stored_reference_text),
            )
            _, separator, transcript = stored_reference_text.partition(
                END_OF_PROMPT)
            metadata.setdefault(
                "reference_transcript",
                transcript if separator else stored_reference_text,
            )
        return metadata

    def _forward_audio_tokenizer(self, wav, wav_len):
        inference_request = pb_utils.InferenceRequest(
            model_name="audio_tokenizer",
            requested_output_names=["prompt_speech_tokens"],
            inputs=[wav, wav_len],
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message())
        output = pb_utils.get_output_tensor_by_name(
            inference_response, "prompt_speech_tokens")
        return torch.utils.dlpack.from_dlpack(output.to_dlpack()).cpu()

    def _forward_speaker_embedding(self, wav_tensor):
        reference_wav = pb_utils.Tensor.from_dlpack(
            "reference_wav", to_dlpack(wav_tensor))
        inference_request = pb_utils.InferenceRequest(
            model_name="speaker_embedding",
            requested_output_names=["prompt_spk_embedding"],
            inputs=[reference_wav],
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message())
        output = pb_utils.get_output_tensor_by_name(
            inference_response, "prompt_spk_embedding")
        return torch.utils.dlpack.from_dlpack(output.to_dlpack()).cpu()

    def _extract_speech_feat(self, speech):
        speech_feat = mel_spectrogram(speech).squeeze(dim=0).transpose(0, 1)
        return speech_feat.unsqueeze(dim=0)

    def _atomic_write(self, path, arrays):
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".npz",
            dir=self.speaker_store_dir,
        )
        os.close(fd)
        try:
            np.savez(temp_path, **arrays)
            with open(temp_path, "rb") as file_obj:
                os.fsync(file_obj.fileno())
            os.replace(temp_path, path)
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def _register(self, request):
        speaker_id = self._decode_string(
            request, "speaker_id", required=True)
        self._validate_speaker_id(speaker_id)
        reference_text_raw = self._decode_string(
            request, "reference_text", required=True)
        prompt_raw = self._decode_string(request, "prompt")
        reference_text, default_prompt, reference_transcript = (
            self._compose_reference_text(reference_text_raw, prompt_raw))

        wav = self._input_tensor(request, "reference_wav")
        wav_len = self._input_tensor(request, "reference_wav_len")
        if wav is None or wav_len is None:
            raise ValueError(
                "reference_wav and reference_wav_len are required for register")

        wav_np = wav.as_numpy()
        wav_len_values = wav_len.as_numpy().reshape(-1)
        if wav_np.ndim != 2 or wav_np.shape[0] != 1:
            raise ValueError("reference_wav must have shape [1, samples]")
        if wav_len_values.size != 1:
            raise ValueError("reference_wav_len must contain exactly one value")
        wav_len_value = int(wav_len_values[0])
        if wav_len_value <= 0 or wav_len_value > wav_np.shape[1]:
            raise ValueError("reference_wav_len is outside reference_wav bounds")

        min_samples = SAMPLE_RATE // 2
        max_samples = int(SAMPLE_RATE * self.max_reference_seconds)
        if wav_len_value < min_samples:
            raise ValueError("reference_wav must be at least 0.5 seconds")
        if wav_len_value > max_samples:
            raise ValueError(
                f"reference_wav must not exceed {self.max_reference_seconds:g} seconds")

        wav_data = np.ascontiguousarray(
            wav_np[:, :wav_len_value], dtype=np.float32)
        if not np.isfinite(wav_data).all():
            raise ValueError("reference_wav contains NaN or infinity")
        if float(np.max(np.abs(wav_data))) > 8.0:
            raise ValueError("reference_wav amplitude is outside a safe range")

        prompt_speech_tokens = self._forward_audio_tokenizer(
            wav, wav_len).unsqueeze(0)
        prompt_speech_tokens_for_llm = prompt_speech_tokens.clone()

        wav_tensor_cpu = torch.from_numpy(wav_data)
        prompt_spk_embedding = self._forward_speaker_embedding(
            wav_tensor_cpu)

        wav_tensor_gpu = wav_tensor_cpu.to(self.device)
        prompt_speech_resample = self.resampler(wav_tensor_gpu)
        speech_feat = self._extract_speech_feat(prompt_speech_resample)

        token_len = min(
            int(speech_feat.shape[1] / 2), prompt_speech_tokens.shape[-1])
        if token_len <= 0:
            raise ValueError("reference_wav is too short to extract prompt features")
        prompt_speech_feat = (
            speech_feat[:, :2 * token_len].contiguous().half().cpu())
        prompt_speech_tokens = (
            prompt_speech_tokens[:, :token_len].contiguous().cpu())
        prompt_speech_tokens_for_llm = (
            prompt_speech_tokens_for_llm.contiguous().cpu())
        prompt_spk_embedding = (
            prompt_spk_embedding.contiguous().half().cpu())

        digest = hashlib.sha256()
        digest.update(speaker_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(reference_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(wav_data.tobytes())
        speaker_version = digest.hexdigest()[:16]
        registered_at = int(time.time())
        metadata = {
            "format_version": 2,
            "speaker_id": speaker_id,
            "speaker_version": speaker_version,
            "reference_text": reference_text,
            "reference_transcript": reference_transcript,
            "prompt": default_prompt,
            "sample_rate": SAMPLE_RATE,
            "samples": wav_len_value,
            "duration_seconds": round(wav_len_value / SAMPLE_RATE, 3),
            "registered_at": registered_at,
        }

        arrays = {
            "speaker_id": np.array(speaker_id),
            "speaker_version": np.array(speaker_version),
            "reference_text": np.array(reference_text),
            "metadata_json": np.array(json.dumps(
                metadata, ensure_ascii=False, separators=(",", ":"))),
            "prompt_speech_tokens_for_llm": (
                prompt_speech_tokens_for_llm.numpy().astype(
                    np.int32, copy=False)),
            "prompt_speech_tokens": (
                prompt_speech_tokens.numpy().astype(np.int32, copy=False)),
            "prompt_speech_feat": (
                prompt_speech_feat.numpy().astype(np.float16, copy=False)),
            "prompt_spk_embedding": (
                prompt_spk_embedding.numpy().astype(np.float16, copy=False)),
        }
        self._atomic_write(self._speaker_path(speaker_id), arrays)
        self.logger.log_info(
            f"Registered speaker_id={speaker_id}, version={speaker_version}, "
            f"duration={metadata['duration_seconds']}s")
        return self._response("ok", metadata, speaker_version)

    def _inspect(self, request):
        speaker_id = self._decode_string(
            request, "speaker_id", required=True)
        path = self._speaker_path(speaker_id)
        if not os.path.exists(path):
            return self._response(
                "not_found", {"speaker_id": speaker_id, "exists": False})
        metadata = self._read_metadata(path)
        metadata["exists"] = True
        return self._response(
            "ok", metadata, metadata.get("speaker_version", ""))

    def _delete(self, request):
        speaker_id = self._decode_string(
            request, "speaker_id", required=True)
        path = self._speaker_path(speaker_id)
        if not os.path.exists(path):
            return self._response(
                "not_found", {"speaker_id": speaker_id, "deleted": False})
        metadata = self._read_metadata(path)
        os.unlink(path)
        self.logger.log_info(f"Deleted speaker_id={speaker_id}")
        return self._response(
            "ok",
            {"speaker_id": speaker_id, "deleted": True},
            metadata.get("speaker_version", ""),
        )

    def _list(self):
        speakers = []
        for filename in sorted(os.listdir(self.speaker_store_dir)):
            if not filename.endswith(".npz") or filename.startswith("."):
                continue
            speaker_id = filename[:-4]
            if not SPEAKER_ID_PATTERN.fullmatch(speaker_id) or ".." in speaker_id:
                continue
            path = os.path.join(self.speaker_store_dir, filename)
            try:
                metadata = self._read_metadata(path)
            except Exception as exc:
                metadata = {
                    "speaker_id": speaker_id,
                    "error": str(exc),
                }
            speakers.append(metadata)
        return self._response(
            "ok", {"count": len(speakers), "speakers": speakers})

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                operation = self._decode_string(
                    request, "operation", required=True).lower()
                if operation == "register":
                    response = self._register(request)
                elif operation == "inspect":
                    response = self._inspect(request)
                elif operation == "delete":
                    response = self._delete(request)
                elif operation == "list":
                    response = self._list()
                else:
                    raise ValueError(
                        "operation must be one of: register, inspect, delete, list")
                responses.append(response)
            except Exception as exc:
                self.logger.log_error(f"Speaker registry request failed: {exc}")
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(exc))))
        return responses
