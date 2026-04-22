from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any

from .schema import ExperimentConfig


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    return PACKAGE_ROOT / "configs" / "synthetic_smoke.json"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [
        Path.cwd(),
        PACKAGE_ROOT,
        PACKAGE_ROOT / "configs",
    ]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()


def _compose_payload(config_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    extends = payload.get("extends")
    if extends is None:
        return payload
    base_path = _resolve_config_path(Path(config_path.parent) / str(extends))
    base_payload = _compose_payload(base_path, _json_load(base_path))
    merged = dict(base_payload)
    for key, value in payload.items():
        if key == "extends":
            continue
        merged[key] = value
    return merged


def config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    kwargs: dict[str, Any] = {}
    for field in fields(ExperimentConfig):
        if field.name not in payload:
            continue
        kwargs[field.name] = payload[field.name]
    return ExperimentConfig(**kwargs)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def save_config(cfg: ExperimentConfig, path: Path | None = None) -> Path:
    target_path = cfg.config_path if path is None else Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(config_to_dict(cfg), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target_path


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = default_config_path() if path is None else _resolve_config_path(path)
    payload = _compose_payload(config_path, _json_load(config_path))
    return config_from_dict(payload)


def apply_config_overrides(
    cfg: ExperimentConfig,
    overrides: dict[str, Any] | None,
) -> ExperimentConfig:
    if not overrides:
        return cfg
    payload = config_to_dict(cfg)
    unknown_keys = sorted(set(overrides) - set(payload))
    if unknown_keys:
        raise KeyError(f"未知配置字段：{', '.join(unknown_keys)}")
    payload.update(overrides)
    return config_from_dict(payload)
