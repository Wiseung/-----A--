from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_python_sources(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def build_fingerprints(
    graph_path: Path,
    config_path: Path,
    official_root: Path,
    solver_root: Path,
) -> dict[str, str | None]:
    return {
        "graph_hash": sha256_file(graph_path) if graph_path.is_file() else None,
        "config_hash": sha256_file(config_path) if config_path.is_file() else None,
        "official_code_hash": sha256_python_sources(official_root / "code"),
        "solver_code_hash": sha256_python_sources(solver_root),
    }


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def valid_run_label(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value) is not None


def ensure_run_metadata(path: Path, metadata: dict[str, object]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(f"run identity mismatch in {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_run_metadata(
    experiment_id: str,
    run_id: str,
    fingerprints: dict[str, str | None],
    settings: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "config_hash": fingerprints.get("config_hash"),
        "official_code_hash": fingerprints.get("official_code_hash"),
        "solver_code_hash": fingerprints.get("solver_code_hash"),
        **settings,
    }
