import io
import struct
import unittest
import wave

from scripts.benchmark import percentile, summarize_results, wav_duration


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

    def test_summary_uses_official_aggregate_rtf_definition(self):
        results = [
            {
                "latency_seconds": 1.0,
                "audio_seconds": 10.0,
                "rtf": 0.1,
                "bytes": 100,
            },
            {
                "latency_seconds": 2.0,
                "audio_seconds": 10.0,
                "rtf": 0.2,
                "bytes": 200,
            },
        ]
        summary = summarize_results(
            results,
            errors=[],
            concurrency=2,
            request_count=2,
            wall_seconds=2.5,
        )
        self.assertAlmostEqual(summary["system_rtf"], 2.5 / 20)
        self.assertAlmostEqual(summary["audio_throughput_x"], 8.0)
        self.assertAlmostEqual(summary["request_rtf_average"], 0.15)
        self.assertEqual(summary["rtf_average"], summary["request_rtf_average"])
        self.assertAlmostEqual(summary["latency_average_seconds"], 1.5)
        self.assertAlmostEqual(summary["latency_p90_seconds"], 1.9)
        self.assertAlmostEqual(summary["latency_p99_seconds"], 1.99)


if __name__ == "__main__":
    unittest.main()
