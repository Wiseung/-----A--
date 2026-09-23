from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CODE = ROOT / "2026_official" / "code"
if str(OFFICIAL_CODE) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE))

from contest_io import _read_json  # noqa: E402
from evaluation_validation import validate_graph  # noqa: E402


@dataclass(slots=True)
class Graph:
    path: Path
    raw: dict[str, Any]
    ops: dict[int, dict[str, Any]]
    tensors: dict[int, dict[str, Any]]
    op_inputs: dict[int, set[int]]
    op_outputs: dict[int, set[int]]
    tensor_producers: dict[int, set[int]]
    tensor_consumers: dict[int, set[int]]
    direct_op_edges: list[tuple[int, int, int]]


def parse_graph(raw: dict[str, Any], path: Path | str = "<memory>") -> Graph:
    validate_graph(raw)
    ops = {op["id"]: op for op in raw["ops"]}
    tensors = {tensor["id"]: tensor for tensor in raw["tensors"]}
    op_ids, tensor_ids = set(ops), set(tensors)
    op_inputs = {op_id: set() for op_id in op_ids}
    op_outputs = {op_id: set() for op_id in op_ids}
    tensor_producers: dict[int, set[int]] = defaultdict(set)
    tensor_consumers: dict[int, set[int]] = defaultdict(set)
    direct_op_edges = []

    for edge in raw["edges"]:
        source, target = edge["source"], edge["target"]
        if source in op_ids and target in tensor_ids:
            op_outputs[source].add(target)
            tensor_producers[target].add(source)
        elif source in tensor_ids and target in op_ids:
            op_inputs[target].add(source)
            tensor_consumers[source].add(target)
        elif source in op_ids and target in op_ids:
            direct_op_edges.append((source, target, edge.get("data_size", 0)))

    return Graph(
        path=Path(path), raw=raw, ops=ops, tensors=tensors,
        op_inputs=op_inputs, op_outputs=op_outputs,
        tensor_producers=dict(tensor_producers),
        tensor_consumers=dict(tensor_consumers),
        direct_op_edges=direct_op_edges,
    )


def load_graph(path: Path | str) -> Graph:
    graph_path = Path(path).resolve()
    return parse_graph(_read_json(graph_path), graph_path)


def load_config(official_root: Path | str, config_path: Path | str | None = None) -> dict[str, Any]:
    from evaluation_validation import (
        read_evaluation_config,
        read_required_settings,
    )
    from multicore_cut_evaluate_problem_3 import read_cache_config

    official_root = Path(official_root).resolve()
    config = Path(config_path).resolve() if config_path else official_root / "data" / "config.txt"
    common = read_evaluation_config(str(config))
    scene_a = read_required_settings(
        str(config), "multicore_scene_a",
        ("task_cross_core_wait_cycles", "task_same_core_wait_cycles"),
    )
    scene_b = read_required_settings(
        str(config), "multicore_scene_b", ("cross_core_copy_delay_cycles",)
    )
    problem_3 = read_cache_config(str(config))
    return {
        "path": config,
        **common,
        "scene_a": scene_a,
        "scene_b": scene_b,
        "problem_3": problem_3,
    }