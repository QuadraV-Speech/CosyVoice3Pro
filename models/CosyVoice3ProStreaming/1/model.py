"""Load the shared CosyVoice3Pro BLS implementation in decoupled mode."""

import importlib.util
from pathlib import Path


source = (
    Path(__file__).resolve().parents[2]
    / "CosyVoice3Pro"
    / "1"
    / "model.py"
)
spec = importlib.util.spec_from_file_location(
    "cosyvoice3pro_shared_model", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

TritonPythonModel = module.TritonPythonModel
