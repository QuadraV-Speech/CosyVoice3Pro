import os
import unittest
from unittest.mock import patch

from gateway.tts_utils import positive_env, split_text


class LegacyTtsPerformanceHelpersTest(unittest.TestCase):
    def test_positive_env_accepts_positive_integer(self):
        with patch.dict(os.environ, {"TEST_TTS_LIMIT": "12"}):
            self.assertEqual(positive_env("TEST_TTS_LIMIT", 2), 12)

    def test_positive_env_rejects_zero(self):
        with patch.dict(os.environ, {"TEST_TTS_LIMIT": "0"}):
            with self.assertRaises(RuntimeError):
                positive_env("TEST_TTS_LIMIT", 2)

    def test_split_text_keeps_all_content(self):
        text = "第一句话。第二句话！第三句话？"
        segments = split_text(text, 8)
        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segments), text)


if __name__ == "__main__":
    unittest.main()
