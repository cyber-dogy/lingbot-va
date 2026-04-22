from __future__ import annotations

from ._bootstrap import bootstrap_local_cli_imports

bootstrap_local_cli_imports()

import argparse
import json

import torch

from lingbot_va.cli.shared import add_config_arguments, apply_cli_overrides
from lingbot_va.config import load_config
from lingbot_va.train.builders import build_dataloaders, build_model
from lingbot_va.train.checkpoints import load_checkpoint_payload
from lingbot_va.train.eval import evaluate_model_on_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all periodic checkpoints.")
    add_config_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    device = torch.device(cfg.device)
    checkpoint_paths = sorted(cfg.periodic_ckpt_dir.glob("epoch_*.pt"))
    if not checkpoint_paths:
        raise SystemExit(f"未找到周期 checkpoint：{cfg.periodic_ckpt_dir}")

    _, _, _, dataloader_valid = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    results = []
    for checkpoint_path in checkpoint_paths:
        payload = load_checkpoint_payload(checkpoint_path)
        model.load_state_dict(payload["model"])
        summary = evaluate_model_on_loader(model, dataloader_valid, device=device)
        summary["checkpoint"] = str(checkpoint_path)
        results.append(summary)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
