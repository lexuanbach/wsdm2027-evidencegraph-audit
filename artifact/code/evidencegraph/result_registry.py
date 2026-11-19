"""Immutable, source-hashed envelopes for measured paper display registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import sha256


def source_descriptor(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        display_path = str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        display_path = str(resolved)
    descriptor: dict[str, Any] = {
        "path": display_path,
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if resolved.suffix == ".jsonl":
        with resolved.open("rb") as handle:
            descriptor["rows"] = sum(1 for _ in handle)
    return descriptor


def verified_source_path(descriptor: dict[str, Any], project_root: Path) -> Path:
    """Resolve an embedded descriptor and verify it against the current bytes."""
    if not isinstance(descriptor, dict) or not descriptor.get("path"):
        raise ValueError("missing embedded source descriptor")
    path = Path(str(descriptor["path"]))
    if not path.is_absolute():
        path = project_root / path
    actual = source_descriptor(path, project_root)
    for key in ("path", "sha256", "bytes"):
        if descriptor.get(key) != actual.get(key):
            raise ValueError(f"embedded source descriptor mismatch for {path}: {key}")
    if "rows" in actual and descriptor.get("rows") != actual["rows"]:
        raise ValueError(f"embedded source descriptor mismatch for {path}: rows")
    return path


def complete_registry_envelope(
    display_id: str,
    registry: dict[str, Any],
    source_paths: list[Path],
    *,
    project_root: Path,
) -> dict[str, Any]:
    if not display_id or not registry or not source_paths:
        raise ValueError("complete registries require an ID, data, and raw sources")
    resolved = [path.resolve() for path in source_paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("registry source paths must be unique")
    return {
        "status": "complete",
        "display_id": display_id,
        "sources": [source_descriptor(path, project_root) for path in resolved],
        "registry": registry,
    }


def write_complete_registry(
    display_id: str,
    registry: dict[str, Any],
    source_paths: list[Path],
    output_dir: Path,
    *,
    project_root: Path,
) -> Path:
    envelope = complete_registry_envelope(
        display_id, registry, source_paths, project_root=project_root
    )
    serialized = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{display_id}.json"
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to replace changed measured registry: {output}")
    output.write_text(serialized, encoding="utf-8")
    return output
