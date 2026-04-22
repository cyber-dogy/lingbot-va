from __future__ import annotations

from ._bootstrap import bootstrap_local_cli_imports

bootstrap_local_cli_imports()

import argparse
import json
from pathlib import Path

import torch

from lingbot_va.cli.shared import add_config_arguments, apply_cli_overrides
from lingbot_va.config import load_config
from lingbot_va.train.builders import build_dataloaders, build_model
from lingbot_va.train.checkpoints import load_checkpoint_payload
from lingbot_va.train.eval import evaluate_model_on_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one lingbot_va checkpoint.")
    add_config_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint 路径；为空时默认读取 latest.pt。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    device = torch.device(cfg.device)
    checkpoint_path = cfg.latest_ckpt_path if args.checkpoint is None else args.checkpoint.expanduser().resolve()

    _, _, _, dataloader_valid = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    payload = load_checkpoint_payload(checkpoint_path)
    model.load_state_dict(payload["model"])

    summary = evaluate_model_on_loader(model, dataloader_valid, device=device)
    summary["checkpoint"] = str(checkpoint_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
