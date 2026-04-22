from __future__ import annotations

import argparse
import json

from lingbot_va.config import ExperimentConfig, apply_config_overrides, default_config_path


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path()),
        help="配置 JSON 路径。",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="用 JSON 语法覆盖配置，例如 --set train_epochs=1。",
    )


def _parse_override_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return raw


def parse_config_overrides(items: list[str] | None) -> dict[str, object] | None:
    if not items:
        return None
    overrides: dict[str, object] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"无效的 --set 参数：{item!r}，应为 KEY=VALUE。")
        key = key.strip()
        if not key:
            raise SystemExit(f"无效的 --set 参数：{item!r}，KEY 不能为空。")
        overrides[key] = _parse_override_value(value)
    return overrides


def apply_cli_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    return apply_config_overrides(cfg, parse_config_overrides(args.config_overrides))
