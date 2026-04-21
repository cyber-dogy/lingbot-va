#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Update LingBot-VA transformer attn_mode in-place.")
    parser.add_argument("--model-dir", required=True, help="Path to the model directory that contains transformer/config.json")
    parser.add_argument("--mode", required=True, choices=["torch", "flashattn", "flex"], help="Target attention mode")
    args = parser.parse_args()

    config_path = Path(args.model_dir).expanduser() / "transformer" / "config.json"
    if not config_path.is_file():
      raise FileNotFoundError(f"Missing config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config["attn_mode"] = args.mode

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Updated {config_path} -> attn_mode={args.mode}")


if __name__ == "__main__":
    main()
