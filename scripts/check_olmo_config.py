"""Verify the Olmo runtime pin preserves published model configuration before a GPU retry.

Run with transformers==5.5.3: python scripts/check_olmo_config.py
"""
import hashlib
import json
import urllib.request
from pathlib import Path

from transformers import Olmo3Config, __version__

ROOT = Path(__file__).resolve().parents[1]
MODEL = "allenai/Olmo-3-7B-Think"


def main():
    with urllib.request.urlopen(f"https://huggingface.co/{MODEL}/resolve/main/config.json", timeout=30) as response:
        source = response.read()
    raw = json.loads(source)
    config = Olmo3Config.from_dict(raw)
    assert config.rope_parameters["rope_theta"] == raw["rope_theta"]
    for key, value in raw["rope_scaling"].items():
        assert config.rope_parameters[key] == value
    assert config.sliding_window == raw["sliding_window"]
    result = {"transformers": __version__, "model": MODEL,
              "source_config_sha256": hashlib.sha256(source).hexdigest(),
              "rope_parameters": config.rope_parameters, "sliding_window": config.sliding_window,
              "layer_count": len(config.layer_types),
              "full_attention_layers": config.layer_types.count("full_attention"), "passed": True}
    dest = ROOT / "kernels/runs/e37-olmo-config-check"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
