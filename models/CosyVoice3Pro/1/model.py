import json
import hashlib
import os
import re
import time
import asyncio
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils

import httpx
import torchaudio
from functools import partial
from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram


torch.set_num_threads(1)

# CosyVoice3 mel params: fmax=None (Nyquist), center=False
mel_spectrogram = partial(matcha_mel_spectrogram,
    n_fft=1920, num_mels=80, sampling_rate=24000,
    hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)

SPEAKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
INSTRUCTION_PREFIX = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"
DEFAULT_INSTRUCTION = f"{INSTRUCTION_PREFIX}{END_OF_PROMPT}"
MAX_PROMPT_LENGTH = 512


def parse_speech_token_string(response_text):
    """Parse speech tokens from string like '<|s_123|><|s_456|>' into list of int IDs."""
    speech_tokens = response_text.strip().split('><')
    if len(speech_tokens) > 1:
        speech_tokens = ['<' + t if not t.startswith('<') else t for t in speech_tokens]
        speech_tokens = [t + '>' if not t.endswith('>') else t for t in speech_tokens]
    speech_ids = []
    for token_str in speech_tokens:
        match = re.match(r'<\|s_(\d+)\|>', token_str)
        if match:
            speech_ids.append(int(match.group(1)))
    return speech_ids


class TritonPythonModel:
    """CosyVoice3Pro BLS orchestrator for Triton Inference Server.

    Orchestrates: audio_tokenizer, speaker_embedding, remote LLM (httpx),
    token2wav (flow-only), and vocoder (CausalHiFTGenerator).
    Supports both streaming (decoupled) and offline (non-decoupled) modes.
    """

    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])
        parameters = self.model_config['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}

        self.device = torch.device("cuda")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        # Streaming config
        self.token_frame_rate = 25
        self.flow_pre_lookahead_len = 3
        self.token_hop_len = 15
        self.token_mel_ratio = 2
        self.dynamic_chunk_strategy = model_params.get("dynamic_chunk_strategy", "exponential")
        self.logger.log_info(f"CosyVoice3 BLS initialized, decoupled={self.decoupled}, "
                             f"chunk_strategy={self.dynamic_chunk_strategy}")

        # HTTP client for remote LLM (trtllm-serve default port: 8000)
        self.http_client = httpx.AsyncClient()
        self.api_base = model_params.get("llm_api_base", "http://localhost:8000/v1/chat/completions")

        # Each BLS instance owns an LRU of GPU-ready prompt tensors. Registered
        # speakers are persisted in a shared directory so all instances can
        # lazily load the same immutable snapshot.
        self.speaker_store_dir = model_params.get(
            "speaker_store_dir", "/workspace/cosyvoice_speaker_store")
        self.speaker_cache_max_entries = max(
            1, int(model_params.get("speaker_cache_max_entries", "64")))
        os.makedirs(self.speaker_store_dir, exist_ok=True)
        self.speaker_cache = OrderedDict()
        self.eager_cuda_init = (
            model_params.get("eager_cuda_init", "false").lower() == "true")
        legacy_batching = model_params.get(
            "acoustic_batching_enabled", "false").lower() == "true"
        self.flow_batching_enabled = (
            model_params.get(
                "flow_batching_enabled", str(legacy_batching)).lower()
            == "true")
        self.vocoder_batching_enabled = (
            model_params.get(
                "vocoder_batching_enabled", str(legacy_batching)).lower()
            == "true")
        self.acoustic_token_bucket_size = max(
            1, int(model_params.get("acoustic_token_bucket_size", "64")))
        self.acoustic_mel_bucket_size = max(
            1, int(model_params.get("acoustic_mel_bucket_size", "128")))
        if self.eager_cuda_init:
            # Without this, every Python backend instance creates its CUDA
            # context during the first concurrent burst after a restart.
            self._cuda_warmup_tensor = torch.empty(
                1, device=self.device, dtype=torch.float16)
            torch.cuda.synchronize(self.device)
        self.logger.log_info(
            f"CosyVoice3 speaker store={self.speaker_store_dir}, "
            f"cache_max_entries={self.speaker_cache_max_entries}, "
            f"eager_cuda_init={self.eager_cuda_init}, "
            f"flow_batching={self.flow_batching_enabled}, "
            f"vocoder_batching={self.vocoder_batching_enabled}, "
            f"token_bucket={self.acoustic_token_bucket_size}, "
            f"mel_bucket={self.acoustic_mel_bucket_size}")

    @staticmethod
    def _round_up(value, quantum):
        return ((value + quantum - 1) // quantum) * quantum

    @staticmethod
    def _pad_last_dim(tensor, target_length):
        current_length = tensor.shape[-1]
        if current_length > target_length:
            raise ValueError(
                f"cannot pad length {current_length} to smaller target "
                f"{target_length}")
        if current_length == target_length:
            return tensor.contiguous()
        return torch.nn.functional.pad(
            tensor, (0, target_length - current_length)).contiguous()

    @staticmethod
    def _pad_time_dim(tensor, target_length):
        current_length = tensor.shape[1]
        if current_length > target_length:
            raise ValueError(
                f"cannot pad length {current_length} to smaller target "
                f"{target_length}")
        if current_length == target_length:
            return tensor.contiguous()
        return torch.nn.functional.pad(
            tensor, (0, 0, 0, target_length - current_length)).contiguous()

    def _convert_speech_tokens_to_str(self, speech_tokens):
        """Convert speech token IDs tensor/list to string like '<|s_N|>'."""
        if isinstance(speech_tokens, torch.Tensor):
            speech_tokens = speech_tokens.cpu().numpy().flatten().tolist()
        return "".join(f"<|s_{int(tid)}|>" for tid in speech_tokens)

    def _extract_speech_feat(self, speech):
        """Extract mel spectrogram from 24kHz speech for flow prompt."""
        speech_feat = mel_spectrogram(speech).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.unsqueeze(dim=0).to(self.device)
        return speech_feat

    async def forward_llm_streaming(self, target_text, reference_text, prompt_speech_tokens):
        """Async generator: stream LLM tokens via httpx SSE."""
        full_text = f"{reference_text}{target_text}"
        prompt_speech_tokens_str = self._convert_speech_tokens_to_str(prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            "max_tokens": 750,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": True,
        }

        buffer = ""
        async with self.http_client.stream("POST", self.api_base, json=payload, timeout=None) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    line_data = line[len("data: "):].strip()
                    if line_data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(line_data)
                        content = json_data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            buffer += content
                            while True:
                                match = re.search(r"<\|s_(\d+)\|>", buffer)
                                if not match:
                                    break
                                token_num = int(match.group(1))
                                # final_id = token_num + ORIGINAL_VOCAB_SIZE
                                yield token_num
                                buffer = buffer[match.end():]
                    except json.JSONDecodeError:
                        continue

        # Flush remaining tokens
        while True:
            match = re.search(r"<\|s_(\d+)\|>", buffer)
            if not match:
                break
            token_num = int(match.group(1))
            #final_id = token_num + ORIGINAL_VOCAB_SIZE
            yield token_num
            buffer = buffer[match.end():]

    async def forward_llm_offline(self, target_text, reference_text, prompt_speech_tokens):
        """Non-streaming LLM call, returns all speech token IDs at once."""
        full_text = f"{reference_text}{target_text}"
        prompt_speech_tokens_str = self._convert_speech_tokens_to_str(prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            "max_tokens": 750,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": False,
        }
        response = await self.http_client.post(self.api_base, json=payload, timeout=None)
        response.raise_for_status()
        response_json = response.json()
        generated_content = response_json['choices'][0]['message']['content']
        speech_ids = parse_speech_token_string(generated_content)
        # return [sid + ORIGINAL_VOCAB_SIZE for sid in speech_ids]
        return speech_ids

    def forward_audio_tokenizer(self, wav, wav_len):
        """BLS call to audio_tokenizer."""
        inference_request = pb_utils.InferenceRequest(
            model_name='audio_tokenizer',
            requested_output_names=['prompt_speech_tokens'],
            inputs=[wav, wav_len]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_speech_tokens = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_speech_tokens')
        return torch.utils.dlpack.from_dlpack(prompt_speech_tokens.to_dlpack()).cpu()

    def forward_speaker_embedding(self, wav):
        """BLS call to speaker_embedding."""
        inference_request = pb_utils.InferenceRequest(
            model_name='speaker_embedding',
            requested_output_names=['prompt_spk_embedding'],
            inputs=[pb_utils.Tensor.from_dlpack("reference_wav", to_dlpack(wav))]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_spk_embedding = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_spk_embedding')
        return torch.utils.dlpack.from_dlpack(prompt_spk_embedding.to_dlpack())

    async def forward_token2wav(self, target_speech_tokens, prompt_speech_tokens,
                                prompt_speech_feat, prompt_spk_embedding,
                                request_id, token_offset=None, finalize=True,
                                priority=100):
        """Async BLS call to token2wav (flow-only). Returns mel tensor."""
        target_length = int(target_speech_tokens.shape[-1])
        prompt_token_length = int(prompt_speech_tokens.shape[-1])
        prompt_feat_length = int(prompt_speech_feat.shape[1])

        if self.flow_batching_enabled and token_offset is None:
            target_bucket = self._round_up(
                target_length, self.acoustic_token_bucket_size)
            prompt_token_bucket = self._round_up(
                prompt_token_length, self.acoustic_token_bucket_size)
            # Prompt Token and Mel remain aligned at the model's 1:2 ratio.
            prompt_feat_bucket = max(
                prompt_feat_length, prompt_token_bucket * self.token_mel_ratio)
            target_speech_tokens = self._pad_last_dim(
                target_speech_tokens, target_bucket)
            prompt_speech_tokens = self._pad_last_dim(
                prompt_speech_tokens, prompt_token_bucket)
            prompt_speech_feat = self._pad_time_dim(
                prompt_speech_feat, prompt_feat_bucket)

        target_tokens_pb = pb_utils.Tensor.from_dlpack(
            "target_speech_tokens", to_dlpack(target_speech_tokens))
        target_len_pb = pb_utils.Tensor(
            "target_speech_tokens_len",
            np.array([[target_length]], dtype=np.int32))
        prompt_tokens_pb = pb_utils.Tensor.from_dlpack(
            "prompt_speech_tokens", to_dlpack(prompt_speech_tokens))
        prompt_token_len_pb = pb_utils.Tensor(
            "prompt_speech_tokens_len",
            np.array([[prompt_token_length]], dtype=np.int32))
        prompt_feat_pb = pb_utils.Tensor.from_dlpack(
            "prompt_speech_feat", to_dlpack(prompt_speech_feat))
        prompt_feat_len_pb = pb_utils.Tensor(
            "prompt_speech_feat_len",
            np.array([[prompt_feat_length]], dtype=np.int32))
        prompt_emb_pb = pb_utils.Tensor.from_dlpack(
            "prompt_spk_embedding", to_dlpack(prompt_spk_embedding))

        inputs = [
            target_tokens_pb,
            target_len_pb,
            prompt_tokens_pb,
            prompt_token_len_pb,
            prompt_feat_pb,
            prompt_feat_len_pb,
            prompt_emb_pb,
        ]

        if token_offset is not None:
            inputs.append(pb_utils.Tensor("token_offset",
                          np.array([[token_offset]], dtype=np.int32)))
            inputs.append(pb_utils.Tensor("finalize",
                          np.array([[finalize]], dtype=np.bool_)))

        inference_request = pb_utils.InferenceRequest(
            model_name='token2wav',
            requested_output_names=['mel'],
            inputs=inputs,
            request_id=request_id,
            parameters={"priority": priority},
        )

        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        mel = pb_utils.get_output_tensor_by_name(inference_response, 'mel')
        return torch.utils.dlpack.from_dlpack(mel.to_dlpack())

    async def forward_vocoder(self, mel, finalize):
        """Async BLS call to vocoder. Returns speech tensor."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [80, T] -> [1, 80, T]
        mel_length = int(mel.shape[-1])
        if self.vocoder_batching_enabled and finalize:
            mel_bucket = self._round_up(
                mel_length, self.acoustic_mel_bucket_size)
            mel = self._pad_last_dim(mel, mel_bucket)
        mel_pb = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel.float()))
        mel_len_pb = pb_utils.Tensor(
            "mel_len", np.array([[mel_length]], dtype=np.int32))
        finalize_pb = pb_utils.Tensor("finalize",
                      np.array([[finalize]], dtype=np.bool_))

        inference_request = pb_utils.InferenceRequest(
            model_name='vocoder',
            requested_output_names=['tts_speech'],
            inputs=[mel_pb, mel_len_pb, finalize_pb],
        )

        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        speech = pb_utils.get_output_tensor_by_name(inference_response, 'tts_speech')
        return torch.utils.dlpack.from_dlpack(speech.to_dlpack()).cpu()

    @staticmethod
    def _decode_optional_string(request, name):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return ""
        value = tensor.as_numpy().reshape(-1)[0]
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @staticmethod
    def _normalize_reference_text(reference_text):
        reference_text = reference_text.strip()
        if END_OF_PROMPT not in reference_text:
            reference_text = DEFAULT_INSTRUCTION + reference_text
        return reference_text

    @staticmethod
    def _apply_prompt(reference_text, prompt):
        """Override the registered persona without changing cached prompt features."""
        prompt = prompt.strip()
        if not prompt:
            return reference_text
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"prompt must not exceed {MAX_PROMPT_LENGTH} characters")

        _, separator, prompt_transcript = reference_text.partition(END_OF_PROMPT)
        if not separator:
            raise ValueError(
                f"reference_text must contain {END_OF_PROMPT}")

        if END_OF_PROMPT in prompt:
            instruction, _, trailing_text = prompt.partition(END_OF_PROMPT)
            if trailing_text.strip():
                raise ValueError(
                    f"prompt must not contain text after {END_OF_PROMPT}")
            instruction = instruction.strip()
        else:
            instruction = prompt

        if not instruction:
            return reference_text
        if not instruction.startswith(INSTRUCTION_PREFIX):
            instruction = f"{INSTRUCTION_PREFIX} {instruction}"
        return f"{instruction}{END_OF_PROMPT}{prompt_transcript}"

    def _resolve_prompt_override(self, request):
        prompt = self._decode_optional_string(request, "prompt").strip()
        instruct_text = self._decode_optional_string(
            request, "instruct_text").strip()
        if prompt and instruct_text:
            raise ValueError(
                "provide only one of prompt or instruct_text")
        return prompt or instruct_text

    @staticmethod
    def _validate_speaker_id(speaker_id):
        if not SPEAKER_ID_PATTERN.fullmatch(speaker_id) or ".." in speaker_id:
            raise ValueError(
                "speaker_id must be 1-128 characters using letters, digits, "
                "underscore, dash, or dot; '..' is not allowed")

    def _cache_get(self, key):
        cached = self.speaker_cache.get(key)
        if cached is not None:
            self.speaker_cache.move_to_end(key)
        return cached

    def _cache_put(self, key, value):
        self.speaker_cache[key] = value
        self.speaker_cache.move_to_end(key)
        while len(self.speaker_cache) > self.speaker_cache_max_entries:
            self.speaker_cache.popitem(last=False)

    @staticmethod
    def _prompt_tuple(cached):
        return (
            cached["prompt_speech_tokens_for_llm"],
            cached["prompt_speech_tokens"],
            cached["prompt_speech_feat"],
            cached["prompt_spk_embedding"],
            cached["reference_text"],
        )

    def _registered_speaker_path(self, speaker_id):
        self._validate_speaker_id(speaker_id)
        return os.path.join(self.speaker_store_dir, f"{speaker_id}.npz")

    def _load_registered_prompt(self, speaker_id):
        path = self._registered_speaker_path(speaker_id)
        try:
            stat = os.stat(path)
        except FileNotFoundError as exc:
            raise ValueError(f"speaker_id is not registered: {speaker_id}") from exc

        cache_prefix = f"speaker:{speaker_id}:"
        cache_key = (
            f"{cache_prefix}{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_size}")
        cached = self._cache_get(cache_key)
        if cached is not None:
            return self._prompt_tuple(cached)

        try:
            with np.load(path, allow_pickle=False) as data:
                stored_speaker_id = str(data["speaker_id"].item())
                reference_text = str(data["reference_text"].item())
                prompt_speech_tokens_for_llm_np = (
                    data["prompt_speech_tokens_for_llm"].copy())
                prompt_speech_tokens_np = data["prompt_speech_tokens"].copy()
                prompt_speech_feat_np = data["prompt_speech_feat"].copy()
                prompt_spk_embedding_np = data["prompt_spk_embedding"].copy()
        except Exception as exc:
            raise ValueError(
                f"registered speaker data is invalid: {speaker_id}: {exc}") from exc

        if stored_speaker_id != speaker_id:
            raise ValueError(f"registered speaker id mismatch: {speaker_id}")
        if prompt_speech_tokens_for_llm_np.ndim != 2:
            raise ValueError("invalid prompt_speech_tokens_for_llm shape")
        if prompt_speech_tokens_np.ndim != 2:
            raise ValueError("invalid prompt_speech_tokens shape")
        if prompt_speech_feat_np.ndim != 3 or prompt_speech_feat_np.shape[-1] != 80:
            raise ValueError("invalid prompt_speech_feat shape")
        if prompt_spk_embedding_np.ndim != 2:
            raise ValueError("invalid prompt_spk_embedding shape")

        cached = {
            "prompt_speech_tokens_for_llm": torch.from_numpy(
                np.ascontiguousarray(prompt_speech_tokens_for_llm_np)
            ).to(torch.int32),
            "prompt_speech_tokens": torch.from_numpy(
                np.ascontiguousarray(prompt_speech_tokens_np)
            ).to(torch.int32),
            "prompt_speech_feat": torch.from_numpy(
                np.ascontiguousarray(prompt_speech_feat_np)
            ).to(self.device, dtype=torch.float16),
            "prompt_spk_embedding": torch.from_numpy(
                np.ascontiguousarray(prompt_spk_embedding_np)
            ).to(self.device, dtype=torch.float16),
            "reference_text": reference_text,
        }

        # A replacement creates a new cache key. Drop stale snapshots for the
        # same speaker before adding the current one.
        for key in list(self.speaker_cache):
            if key.startswith(cache_prefix) and key != cache_key:
                del self.speaker_cache[key]
        self._cache_put(cache_key, cached)
        return self._prompt_tuple(cached)

    def _prepare_raw_prompt(self, request):
        wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
        wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")
        reference_text = self._normalize_reference_text(
            self._decode_optional_string(request, "reference_text"))

        if wav is None or wav_len is None:
            raise ValueError(
                "provide speaker_id, or provide reference_wav and reference_wav_len")

        wav_np = wav.as_numpy()
        wav_len_values = wav_len.as_numpy().reshape(-1)
        if wav_np.ndim != 2 or wav_np.shape[0] != 1:
            raise ValueError("reference_wav must have shape [1, samples]")
        if wav_len_values.size != 1:
            raise ValueError("reference_wav_len must contain one value")
        wav_len_val = int(wav_len_values[0])
        if wav_len_val <= 0 or wav_len_val > wav_np.shape[1]:
            raise ValueError("reference_wav_len is outside reference_wav bounds")

        raw_hash = hashlib.sha256()
        raw_hash.update(reference_text.encode("utf-8"))
        raw_hash.update(
            np.ascontiguousarray(wav_np[:, :wav_len_val], dtype=np.float32).tobytes())
        cache_key = f"raw:{raw_hash.hexdigest()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return self._prompt_tuple(cached)

        prompt_speech_tokens = self.forward_audio_tokenizer(
            wav, wav_len).unsqueeze(0)

        wav_tensor = torch.from_numpy(wav_np[:, :wav_len_val])
        prompt_spk_embedding = self.forward_speaker_embedding(wav_tensor)

        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=24000)(wav_tensor)
        speech_feat = self._extract_speech_feat(prompt_speech_resample)

        prompt_speech_tokens_for_llm = prompt_speech_tokens.clone()
        token_len = min(
            int(speech_feat.shape[1] / 2), prompt_speech_tokens.shape[-1])
        if token_len <= 0:
            raise ValueError("reference_wav is too short to extract prompt features")
        prompt_speech_feat = speech_feat[:, :2 * token_len].contiguous().half()
        prompt_speech_tokens = prompt_speech_tokens[:, :token_len].contiguous()

        cached = {
            "prompt_speech_tokens_for_llm": prompt_speech_tokens_for_llm,
            "prompt_speech_tokens": prompt_speech_tokens,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_spk_embedding": prompt_spk_embedding,
            "reference_text": reference_text,
        }
        self._cache_put(cache_key, cached)
        return self._prompt_tuple(cached)

    def _prepare_prompt(self, request):
        """Resolve registered speaker features or extract a raw prompt."""
        speaker_id = self._decode_optional_string(request, "speaker_id").strip()
        if speaker_id:
            prompt = self._load_registered_prompt(speaker_id)
        else:
            prompt = self._prepare_raw_prompt(request)

        prompt_override = self._resolve_prompt_override(request)
        return (*prompt[:-1], self._apply_prompt(
            prompt[-1], prompt_override))

    async def _process_request_streaming(self, request):
        """Process a single request in streaming (decoupled) mode."""
        request_id = request.request_id()
        response_sender = request.get_response_sender()

        try:
            prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
                prompt_spk_embedding, reference_text = self._prepare_prompt(request)

            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')

            semantic_token_ids_arr = []
            token_offset = 0
            chunk_index = 0
            this_token_hop_len = self.token_hop_len
            accumulated_mel = None
            speech_offset = 0
            start_time = time.time()

            async for generated_id in self.forward_llm_streaming(
                target_text=target_text,
                reference_text=reference_text,
                prompt_speech_tokens=prompt_speech_tokens_for_llm,
            ):
                semantic_token_ids_arr.append(generated_id)

                while True:
                    pending_num = len(semantic_token_ids_arr) - token_offset
                    if pending_num < this_token_hop_len + self.flow_pre_lookahead_len:
                        break

                    # Prepare tokens for this chunk
                    end_idx = token_offset + this_token_hop_len + self.flow_pre_lookahead_len
                    this_tokens = torch.tensor(
                        semantic_token_ids_arr[:end_idx]
                    ).unsqueeze(0).to(torch.int32).to(self.device)

                    # Call token2wav (flow-only) -> mel_chunk
                    mel_chunk = await self.forward_token2wav(
                        this_tokens, prompt_speech_tokens,
                        prompt_speech_feat, prompt_spk_embedding,
                        request_id, token_offset=token_offset, finalize=False,
                        priority=chunk_index + 1,
                    )

                    # Accumulate mel
                    if mel_chunk.dim() == 2:
                        mel_chunk = mel_chunk.unsqueeze(0)
                    if accumulated_mel is None:
                        accumulated_mel = mel_chunk
                    else:
                        accumulated_mel = torch.cat([accumulated_mel, mel_chunk], dim=2)

                    # Call vocoder
                    speech = await self.forward_vocoder(accumulated_mel, finalize=False)

                    # Extract new speech
                    new_speech = speech[:, speech_offset:]
                    speech_offset += new_speech.shape[1]

                    if new_speech.shape[1] > 0:
                        audio_tensor = pb_utils.Tensor.from_dlpack(
                            "waveform", to_dlpack(new_speech))
                        inference_response = pb_utils.InferenceResponse(
                            output_tensors=[audio_tensor])
                        response_sender.send(inference_response)

                    token_offset += this_token_hop_len

                    # Dynamic chunk strategy
                    if self.dynamic_chunk_strategy == "exponential":
                        this_token_hop_len = self.token_frame_rate * (2 ** chunk_index)
                    elif self.dynamic_chunk_strategy == "time_based":
                        cost_time = time.time() - start_time
                        duration = token_offset / self.token_frame_rate
                        if chunk_index > 0 and cost_time > 0:
                            avg_chunk_time = cost_time / (chunk_index + 1)
                            if avg_chunk_time > 0:
                                multiples = (duration - cost_time) / avg_chunk_time
                                next_pending = len(semantic_token_ids_arr) - token_offset
                                if multiples > 4:
                                    this_token_hop_len = (next_pending // self.token_hop_len + 1) * self.token_hop_len
                                elif multiples > 2:
                                    this_token_hop_len = (next_pending // self.token_hop_len) * self.token_hop_len
                                else:
                                    this_token_hop_len = self.token_hop_len
                                this_token_hop_len = max(self.token_hop_len, this_token_hop_len)

                    chunk_index += 1

            # Final chunk with remaining tokens
            if len(semantic_token_ids_arr) > 0:
                remaining_tokens = torch.tensor(
                    semantic_token_ids_arr
                ).unsqueeze(0).to(torch.int32).to(self.device)

                mel_chunk = await self.forward_token2wav(
                    remaining_tokens, prompt_speech_tokens,
                    prompt_speech_feat, prompt_spk_embedding,
                    request_id, token_offset=token_offset, finalize=True,
                    priority=chunk_index + 1,
                )

                if mel_chunk.dim() == 2:
                    mel_chunk = mel_chunk.unsqueeze(0)
                if accumulated_mel is None:
                    accumulated_mel = mel_chunk
                else:
                    accumulated_mel = torch.cat([accumulated_mel, mel_chunk], dim=2)

                speech = await self.forward_vocoder(accumulated_mel, finalize=True)

                new_speech = speech[:, speech_offset:]
                if new_speech.shape[1] > 0:
                    audio_tensor = pb_utils.Tensor.from_dlpack(
                        "waveform", to_dlpack(new_speech))
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[audio_tensor])
                    response_sender.send(inference_response)

            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        except Exception as e:
            self.logger.log_error(f"Error in streaming request: {e}")
            error_response = pb_utils.InferenceResponse(
                error=pb_utils.TritonError(str(e)))
            response_sender.send(error_response)
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    async def _process_request_offline(self, request):
        """Process a single request in offline (non-decoupled) mode."""
        request_id = request.request_id()

        prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
            prompt_spk_embedding, reference_text = self._prepare_prompt(request)

        target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
        target_text = target_text[0][0].decode('utf-8')

        # Get all speech tokens at once (use full untruncated prompt tokens for LLM)
        all_token_ids = await self.forward_llm_offline(
            target_text=target_text,
            reference_text=reference_text,
            prompt_speech_tokens=prompt_speech_tokens_for_llm,
        )

        if len(all_token_ids) == 0:
            raise pb_utils.TritonModelException("LLM generated no speech tokens")

        all_tokens = torch.tensor(all_token_ids).unsqueeze(0).to(torch.int32).to(self.device)

        # token2wav (no token_offset, finalize=True) -> full mel
        mel = await self.forward_token2wav(
            all_tokens, prompt_speech_tokens,
            prompt_speech_feat, prompt_spk_embedding,
            request_id,
        )

        # vocoder -> full speech
        speech = await self.forward_vocoder(mel, finalize=True)

        audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(speech))
        return pb_utils.InferenceResponse(output_tensors=[audio_tensor])

    async def execute(self, requests):
        if self.decoupled:
            tasks = [
                asyncio.create_task(self._process_request_streaming(request))
                for request in requests
            ]
            await asyncio.gather(*tasks)
            return None
        else:
            responses = []
            for request in requests:
                try:
                    response = await self._process_request_offline(request)
                    responses.append(response)
                except Exception as e:
                    self.logger.log_error(f"Error in offline request: {e}")
                    responses.append(pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))))
            return responses

    def finalize(self):
        self.logger.log_info("Finalizing CosyVoice3 BLS model")
        if hasattr(self, "http_client"):
            asyncio.run(self.http_client.aclose())
