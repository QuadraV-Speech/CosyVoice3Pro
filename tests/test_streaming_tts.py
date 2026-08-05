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


if __name__ == "__main__":
    unittest.main()
