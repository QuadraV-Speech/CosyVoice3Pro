from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class StreamingTtsWiringTest(unittest.TestCase):
    def test_streaming_model_is_decoupled(self):
        config = (
            REPOSITORY_ROOT
            / "models"
            / "CosyVoice3ProStreaming"
            / "config.pbtxt"
        ).read_text(encoding="utf-8")
        self.assertIn('name: "CosyVoice3ProStreaming"', config)
        self.assertIn("decoupled: true", config)
        self.assertIn('name: "waveform"', config)

    def test_public_endpoint_and_web_player_use_sse(self):
        gateway = (
            REPOSITORY_ROOT / "gateway" / "streaming_tts.py"
        ).read_text(encoding="utf-8")
        web = (
            REPOSITORY_ROOT / "gateway" / "web" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn('@router.post("/tts/stream")', gateway)
        self.assertIn('media_type="text/event-stream"', gateway)
        self.assertIn('fetch("/tts/stream"', web)
        self.assertIn("schedulePcm", web)
        self.assertIn('eventName === "queue"', web)

    def test_streaming_gateway_reuses_grpc_channel_and_reports_queue(self):
        gateway = (
            REPOSITORY_ROOT / "gateway" / "streaming_tts.py"
        ).read_text(encoding="utf-8")
        app = (
            REPOSITORY_ROOT / "gateway" / "app.py"
        ).read_text(encoding="utf-8")
        self.assertIn("create_triton_grpc_client", app)
        self.assertIn("app.state.streaming_grpc_client", app)
        self.assertIn('_sse_event("queue"', gateway)
        self.assertIn('"inferenceFirstAudioMs"', gateway)
        self.assertIn('"-probesize", "32"', gateway)
        self.assertIn("STREAM_QUEUE_TIMEOUT_SECONDS", gateway)
        self.assertIn('"code": "STREAM_BUSY"', gateway)
        self.assertIn("CLIENT_DISCONNECT_POLL_SECONDS", gateway)
        self.assertIn("await request.is_disconnected()", gateway)
        self.assertLess(
            gateway.index("first_waveform = await anext(inference)"),
            gateway.index("process = await asyncio.create_subprocess_exec"),
        )

    def test_streaming_profile_scales_acoustic_capacity(self):
        manage = (REPOSITORY_ROOT / "manage.sh").read_text(encoding="utf-8")
        vocoder = (
            REPOSITORY_ROOT / "models" / "vocoder" / "config.pbtxt"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'STREAMING_BLS_INSTANCE_COUNT="${COSYVOICE_STREAMING_BLS_INSTANCES:-2}"',
            manage,
        )
        self.assertIn(
            'VOCODER_INSTANCE_COUNT="${COSYVOICE_VOCODER_INSTANCES:-4}"',
            manage,
        )
        self.assertIn(
            'STREAMING_CONCURRENCY="${COSYVOICE_TTS_STREAMING_CONCURRENCY:-16}"',
            manage,
        )
        self.assertIn('resolved_profile="streaming"', manage)
        self.assertIn("priority_levels: 100", vocoder)

    def test_throughput_profile_reduces_repeated_streaming_work(self):
        manage = (REPOSITORY_ROOT / "manage.sh").read_text(encoding="utf-8")
        streaming_config = (
            REPOSITORY_ROOT
            / "models"
            / "CosyVoice3ProStreaming"
            / "config.pbtxt"
        ).read_text(encoding="utf-8")
        model = (
            REPOSITORY_ROOT / "models" / "CosyVoice3Pro" / "1" / "model.py"
        ).read_text(encoding="utf-8")
        self.assertIn("COSYVOICE_STREAMING_CHUNK_GROWTH_OFFSET:-1", manage)
        self.assertGreaterEqual(
            manage.count('key:[[:space:]]*\\"eager_cuda_init\\"'),
            2,
        )
        self.assertIn('key: "streaming_chunk_growth_offset"', streaming_config)
        self.assertIn('key: "streaming_first_chunk_tokens"', streaming_config)
        self.assertIn("self.streaming_chunk_growth_offset", model)
        self.assertIn("self.token_hop_len = int(", model)
        self.assertIn("response_sender.is_cancelled()", model)


if __name__ == "__main__":
    unittest.main()
