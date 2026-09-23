from __future__ import annotations

import argparse
import csv
import heapq
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CODE = ROOT / "2026_official" / "code"
sys.path.insert(0, str(OFFICIAL_CODE))

from evaluation_validation import validate_graph  # noqa: E402


def profile_graph(graph: dict[str, Any]) -> dict[str, Any]:
    validate_graph(graph)
    tensors = {tensor["id"]: tensor for tensor in graph["tensors"]}
    ops = {op["id"]: op for op in graph["ops"]}
    op_ids = set(ops)
    tensor_ids = set(tensors)
    producers: dict[int, set[int]] = defaultdict(set)
    consumers: dict[int, set[int]] = defaultdict(set)
    predecessors = {op_id: set() for op_id in op_ids}
    successors = {op_id: set() for op_id in op_ids}
    input_tensors: dict[int, set[int]] = defaultdict(set)
    output_tensors: dict[int, set[int]] = defaultdict(set)

    for edge in graph["edges"]:
        source, target = edge["source"], edge["target"]
        if source in op_ids and target in tensor_ids:
            producers[target].add(source)
            output_tensors[source].add(target)
        elif source in tensor_ids and target in op_ids:
            consumers[source].add(target)
            input_tensors[target].add(source)
        elif source in op_ids and target in op_ids and source != target:
            predecessors[target].add(source)
            successors[source].add(target)

    for tensor_id in tensor_ids:
        for producer in producers[tensor_id]:
            for consumer in consumers[tensor_id]:
                if producer != consumer:
                    predecessors[consumer].add(producer)
                    successors[producer].add(consumer)

    indegree = {op_id: len(preds) for op_id, preds in predecessors.items()}
    ready = [op_id for op_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        op_id = heapq.heappop(ready)
        order.append(op_id)
        for child in successors[op_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(op_ids):
        raise ValueError("operation dependency graph is cyclic")

    level: dict[int, int] = {}
    forward: dict[int, int] = {}
    for op_id in order:
        op_cycles = max(1, ops[op_id]["cycles"])
        level[op_id] = 1 + max((level[p] for p in predecessors[op_id]), default=-1)
        forward[op_id] = op_cycles + max(
            (forward[p] for p in predecessors[op_id]), default=0
        )
    backward: dict[int, int] = {}
    for op_id in reversed(order):
        op_cycles = max(1, ops[op_id]["cycles"])
        backward[op_id] = op_cycles + max(
            (backward[s] for s in successors[op_id]), default=0
        )

    compute_ops = {
        op_id for op_id, op in ops.items()
        if op["op"] not in {"COPY_IN", "COPY_OUT"}
    }
    m_ops = [ops[op_id] for op_id in compute_ops if ops[op_id]["pipe"] == "PIPE_M"]
    v_ops = [ops[op_id] for op_id in compute_ops if ops[op_id]["pipe"] == "PIPE_V"]
    pipe_counts = {
        pipe: sum(op["pipe"] == pipe for op in ops.values())
        for pipe in ("PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_V")
    }
    pipe_cycles = {
        pipe: sum(op["cycles"] for op in ops.values() if op["pipe"] == pipe)
        for pipe in ("PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_V")
    }

    ddr_inputs = {
        tensor_id for tensor_id, tensor in tensors.items()
        if tensor["pos"] == "DDR" and not producers[tensor_id]
    }
    ddr_outputs = {
        tensor_id for tensor_id, tensor in tensors.items()
        if tensor["pos"] == "DDR" and producers[tensor_id]
    }
    input_consumers: dict[int, set[int]] = defaultdict(set)
    for tensor_id in ddr_inputs:
        for consumer in consumers[tensor_id]:
            if ops[consumer]["op"] == "COPY_IN":
                for local_tensor in output_tensors[consumer]:
                    input_consumers[tensor_id].update(
                        op_id for op_id in consumers[local_tensor]
                        if op_id in compute_ops
                    )
            elif consumer in compute_ops:
                input_consumers[tensor_id].add(consumer)

    shared_inputs = {
        tensor_id for tensor_id, users in input_consumers.items() if len(users) > 1
    }
    sizes = sorted(tensor["size"] for tensor in tensors.values())
    p50 = statistics.median(sizes) if sizes else 0
    p90_index = max(0, (9 * len(sizes) + 9) // 10 - 1)
    p90 = sizes[p90_index] if sizes else 0

    return {
        "tensor_count": len(tensors),
        "op_count": len(ops),
        "edge_count": len(graph["edges"]),
        "noncopy_op_count": len(compute_ops),
        "ddr_tensor_count": sum(t["pos"] == "DDR" for t in tensors.values()),
        "ddr_input_count": len(ddr_inputs),
        "ddr_input_bytes": sum(tensors[tid]["size"] for tid in ddr_inputs),
        "ddr_output_bytes": sum(tensors[tid]["size"] for tid in ddr_outputs),
        "pipe_m_op_count": len(m_ops),
        "pipe_m_cycles": sum(op["cycles"] for op in m_ops),
        "pipe_v_op_count": len(v_ops),
        "pipe_v_cycles": sum(op["cycles"] for op in v_ops),
        "pipe_mte2_op_count": pipe_counts["PIPE_MTE2"],
        "pipe_mte2_cycles": pipe_cycles["PIPE_MTE2"],
        "pipe_mte3_op_count": pipe_counts["PIPE_MTE3"],
        "pipe_mte3_cycles": pipe_cycles["PIPE_MTE3"],
        "tensor_total_bytes": sum(sizes),
        "max_tensor_bytes": max(sizes, default=0),
        "tensor_size_p50_bytes": p50,
        "tensor_size_p90_bytes": p90,
        "shared_input_count": len(shared_inputs),
        "shared_input_bytes": sum(tensors[tid]["size"] for tid in shared_inputs),
        "topological_level_count": max(level.values(), default=0) + 1,
        "estimated_critical_path_cycles": max(forward.values(), default=0),
        "maximum_forward_path_cycles": max(forward.values(), default=0),
        "maximum_backward_path_cycles": max(backward.values(), default=0),
    }


def markdown_report(rows: list[dict[str, Any]]) -> str:
    fields = [
        "case", "tensor_count", "op_count", "noncopy_op_count", "edge_count",
        "pipe_m_op_count", "pipe_m_cycles", "pipe_v_op_count", "pipe_v_cycles",
        "ddr_input_count", "shared_input_count", "shared_input_bytes",
        "tensor_total_bytes", "max_tensor_bytes", "estimated_critical_path_cycles",
    ]
    lines = [
        "# 2026 正式计算图画像",
        "",
        f"扫描用例数：{len(rows)}。每张图先经官方 `validate_graph` 校验；关键路径是按源码接受的 op 依赖（tensor 桥接及直接 op 边）对 `max(1, cycles)` 求最长路径，不是 evaluator makespan。",
        "",
        "| 指标 | 最小 | 中位数 | 最大 |",
        "|---|---:|---:|---:|",
    ]
    summary_fields = [
        "tensor_count", "op_count", "noncopy_op_count", "edge_count",
        "pipe_m_cycles", "pipe_v_cycles", "ddr_input_bytes",
        "shared_input_count", "shared_input_bytes", "tensor_total_bytes",
        "max_tensor_bytes", "estimated_critical_path_cycles",
    ]
    for field in summary_fields:
        values = [row[field] for row in rows]
        lines.append(
            f"| `{field}` | {min(values)} | {statistics.median(values)} | {max(values)} |"
        )
    lines.extend(["", "## 逐用例", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"])
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile all official graph JSON files.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "2026_official" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()

    rows = []
    files = sorted(args.data_dir.glob("case_*.json"))
    if not files:
        parser.error(f"no case_*.json files found in {args.data_dir}")
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            graph = json.load(stream)
        row = {"case": path.stem, **profile_graph(graph)}
        rows.append(row)
        print(f"profiled {path.name}: {row['op_count']} ops")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "data_profile.csv"
    md_path = args.output_dir / "data_profile.md"
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown_report(rows), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())