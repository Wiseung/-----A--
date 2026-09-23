from __future__ import annotations

from dataclasses import dataclass

from .graph_analysis import GraphAnalysis
from .features import operation_features


@dataclass(slots=True)
class Partition:
    groups: list[list[int]]
    op_to_subgraph: dict[int, int]


def _cut_bytes_by_position(analysis: GraphAnalysis) -> list[int]:
    graph = analysis.graph
    count = len(analysis.topological_order)
    delta = [0] * (count + 1)
    for tensor_id, tensor in graph.tensors.items():
        if tensor["pos"] == "DDR":
            continue
        producer_positions = [
            analysis.topological_index[op_id]
            for op_id in graph.tensor_producers.get(tensor_id, set())
            if op_id in analysis.eligible_ops
        ]
        consumer_positions = [
            analysis.topological_index[op_id]
            for op_id in graph.tensor_consumers.get(tensor_id, set())
            if op_id in analysis.eligible_ops
        ]
        if not producer_positions or not consumer_positions:
            continue
        first, last = min(producer_positions), max(consumer_positions)
        if first < last:
            delta[first + 1] += tensor["size"]
            delta[last + 1] -= tensor["size"]

    op_features = operation_features(analysis)
    input_users: dict[int, list[int]] = {}
    for op_id, features in op_features.items():
        for tensor_id in features["external_ddr_inputs"]:
            input_users.setdefault(tensor_id, []).append(features["topological_index"])
    for tensor_id, positions in input_users.items():
        if len(positions) > 1:
            first, last = min(positions), max(positions)
            delta[first + 1] += graph.tensors[tensor_id]["size"]
            delta[last + 1] -= graph.tensors[tensor_id]["size"]

    cuts = [0] * (count + 1)
    active = 0
    for boundary in range(1, count):
        active += delta[boundary]
        cuts[boundary] = active
    return cuts


def partition_contiguous(analysis: GraphAnalysis, group_count: int, problem: int) -> Partition:
    order = analysis.topological_order
    if not order:
        return Partition([], {})
    group_count = max(1, min(group_count, len(order)))
    if group_count == 1:
        groups = [list(order)]
    else:
        op_features = operation_features(analysis)
        weights = [
            max(1, max(
                op_features[op_id]["cycles"]
                if op_features[op_id]["pipe"] == "PIPE_M" else 0,
                op_features[op_id]["cycles"]
                if op_features[op_id]["pipe"] == "PIPE_V" else 0,
                1,
            ))
            for op_id in order
        ]
        prefix = [0]
        for weight in weights:
            prefix.append(prefix[-1] + weight)
        cuts = _cut_bytes_by_position(analysis)
        max_cut = max(cuts, default=0)
        cut_weight = {1: 0.30, 2: 0.12, 3: 0.05}[problem]
        ends = []
        previous = 0
        total = prefix[-1]
        for group_id in range(1, group_count):
            low = previous + 1
            high = len(order) - (group_count - group_id)
            target = total * group_id / group_count
            end = min(
                range(low, high + 1),
                key=lambda position: (
                    abs(prefix[position] - target) / max(total, 1)
                    + cut_weight * cuts[position] / max(max_cut, 1),
                    position,
                ),
            )
            ends.append(end)
            previous = end
        groups = []
        start = 0
        for end in [*ends, len(order)]:
            groups.append(order[start:end])
            start = end

    op_to_subgraph = {
        op_id: subgraph_id
        for subgraph_id, members in enumerate(groups)
        for op_id in members
    }
    return Partition(groups, op_to_subgraph)


def candidate_group_counts(num_ops: int, ncores: int, problem: int) -> list[int]:
    if not num_ops:
        return [0]
    if ncores == 1:
        return [1]
    multipliers = {1: (1, 2, 4), 2: (2, 4), 3: (2, 4, 6)}[problem]
    return list(dict.fromkeys(min(num_ops, ncores * value) for value in multipliers))