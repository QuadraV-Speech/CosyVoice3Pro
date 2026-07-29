import io
import struct
import unittest
import wave

from scripts.benchmark import percentile, wav_duration


def wav_bytes(seconds=1, sample_rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * sample_rate * seconds)
    return buffer.getvalue()


class BenchmarkHelpersTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1, 2, 3], 50), 2)
        self.assertAlmostEqual(percentile([1, 2], 95), 1.95)

    def test_wav_duration_reads_regular_header(self):
        self.assertAlmostEqual(wav_duration(wav_bytes(seconds=2)), 2.0)

    def test_wav_duration_accepts_streaming_unknown_data_size(self):
        content = bytearray(wav_bytes(seconds=1))
        data_offset = content.index(b"data")
        struct.pack_into("<I", content, data_offset + 4, 0xFFFFFFFF)
        self.assertAlmostEqual(wav_duration(bytes(content)), 1.0)


if __name__ == "__main__":
    unittest.main()
