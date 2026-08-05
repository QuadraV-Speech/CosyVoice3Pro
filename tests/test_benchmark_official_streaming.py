from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import benchmark_official_streaming as benchmark  # noqa: E402


class OfficialStreamingBenchmarkTest(unittest.TestCase):
    def test_split_contiguous_matches_upstream_task_shards(self):
        shards = benchmark.split_contiguous(list(range(10)), 4)
        self.assertEqual([len(shard) for shard in shards], [3, 3, 2, 2])
        self.assertEqual(
            [item for shard in shards for item in shard],
            list(range(10)),
        )

    def test_latency_summary_reports_upstream_percentiles(self):
        summary = benchmark.latency_summary([100, 200, 300, 400])
        self.assertEqual(summary["average"], 250.0)
        self.assertEqual(summary["p50"], 250.0)
        self.assertEqual(summary["p95"], 385.0)


if __name__ == "__main__":
    unittest.main()
