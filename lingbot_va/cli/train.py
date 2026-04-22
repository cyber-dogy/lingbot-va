from __future__ import annotations

from ._bootstrap import bootstrap_local_cli_imports

bootstrap_local_cli_imports()

import argparse
import json

from lingbot_va.cli.shared import add_config_arguments, apply_cli_overrides
from lingbot_va.config import load_config
from lingbot_va.train.runner import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standalone lingbot_va line.")
    add_config_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    summary = train_experiment(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
