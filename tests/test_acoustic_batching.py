import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_flow_batch", ROOT / "scripts" / "prepare_flow_batch.py")
PREPARE_FLOW_BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE_FLOW_BATCH)


class AcousticBatchingTest(unittest.TestCase):
    def test_flow_profile_doubles_business_batch_for_cfg(self):
        shapes = PREPARE_FLOW_BATCH.profile_shapes(4, 512)
        self.assertEqual(shapes["x"], (8, 80, 512))
        self.assertEqual(shapes["mask"], (8, 1, 512))
        self.assertEqual(shapes["t"], (8,))
        self.assertEqual(shapes["spks"], (8, 80))

    def test_safe_repository_defaults_keep_acoustic_models_at_batch_one(self):
        token_config = (ROOT / "models" / "token2wav" / "config.pbtxt").read_text()
        vocoder_config = (ROOT / "models" / "vocoder" / "config.pbtxt").read_text()
        pro_config = (
            ROOT / "models" / "CosyVoice3Pro" / "config.pbtxt").read_text()
        self.assertIn("max_batch_size: 1", token_config)
        self.assertIn("max_batch_size: 1", vocoder_config)
        self.assertIn("priority_levels: 100", vocoder_config)
        self.assertIn('key: "flow_batching_enabled"', pro_config)
        self.assertIn('key: "vocoder_batching_enabled"', pro_config)


if __name__ == "__main__":
    unittest.main()
