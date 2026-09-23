from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .graph_analysis import GraphAnalysis

if TYPE_CHECKING:
    from .partition import Partition


@dataclass(slots=True)
class SubgraphFeatures:
    subgraph_id: int
    members: list[int]
    work_by_pipe: dict[str, int]
    m_cycles: int
    v_cycles: int
    internal_critical_path: int
    estimated_compute: int
    input_tensors: dict[int, int]
    external_inputs: dict[int, int]
    output_tensors: dict[int, int]
    shared_ddr_inputs: set[int]
    predecessor_subgraphs: set[int]
    successor_subgraphs: set[int]
    boundary_bytes: dict[int, int]
    l1_pressure_est: int
    ub_pressure_est: int
    residence_cost_est: int
    topological_min_index: int
    topological_max_index: int


def graph_features(analysis: GraphAnalysis) -> dict[str, Any]:
    graph = analysis.graph
    compute = analysis.eligible_ops
    by_pipe = {
        pipe: [
            max(1, graph.ops[op_id]["cycles"])
            for op_id in compute if graph.ops[op_id]["pipe"] == pipe
        ]
        for pipe in ("PIPE_M", "PIPE_V", "PIPE_MTE2", "PIPE_MTE3")
    }
    shared_inputs = {
        tensor_id for ids in analysis.external_sources.values() for tensor_id in ids
        if sum(tensor_id in other for other in analysis.external_sources.values()) > 1
    }
    memory_events: dict[str, dict[int, int]] = {
        "L1": defaultdict(int), "UB": defaultdict(int)
    }
    residence_cost = {"L1": 0, "UB": 0}
    for tensor_id, (start, end) in analysis.tensor_lifetimes.items():
        tensor = graph.tensors[tensor_id]
        pos, size = tensor["pos"], tensor["size"]
        memory_events[pos][start] += size
        memory_events[pos][end + 1] -= size
        residence_cost[pos] += size * max(1, end - start + 1)
    memory_peak = {}
    for pos, events in memory_events.items():
        used = peak = 0
        for at in sorted(events):
            used += events[at]
            peak = max(peak, used)
        memory_peak[pos] = peak

    ddr_inputs = {
        tensor_id for ids in analysis.external_sources.values() for tensor_id in ids
    }
    ddr_outputs = {
        tensor_id for tensor_id, tensor in graph.tensors.items()
        if tensor["pos"] == "DDR" and graph.tensor_producers.get(tensor_id)
    }
    tensor_sizes = [tensor["size"] for tensor in graph.tensors.values()]
    return {
        "tensor_count": len(graph.tensors),
        "op_count": len(graph.ops),
        "noncopy_op_count": len(compute),
        "edge_count": len(graph.raw["edges"]),
        "pipe_cycles": {pipe: sum(cycles) for pipe, cycles in by_pipe.items()},
        "pipe_counts": {pipe: len(cycles) for pipe, cycles in by_pipe.items()},
        "critical_path_cycles": max(analysis.forward_path.values(), default=0),
        "total_m_cycles": sum(by_pipe["PIPE_M"]),
        "total_v_cycles": sum(by_pipe["PIPE_V"]),
        "ddr_input_count": len(ddr_inputs),
        "ddr_input_bytes": sum(graph.tensors[t]["size"] for t in ddr_inputs),
        "ddr_output_bytes": sum(graph.tensors[t]["size"] for t in ddr_outputs),
        "shared_input_count": len(shared_inputs),
        "shared_input_bytes": sum(graph.tensors[t]["size"] for t in shared_inputs),
        "tensor_total_bytes": sum(tensor_sizes),
        "max_tensor_bytes": max(tensor_sizes, default=0),
        "l1_peak_est": memory_peak["L1"],
        "ub_peak_est": memory_peak["UB"],
        "l1_residence_cost_est": residence_cost["L1"],
        "ub_residence_cost_est": residence_cost["UB"],
    }


def operation_features(analysis: GraphAnalysis) -> dict[int, dict[str, Any]]:
    graph = analysis.graph
    result = {}
    for op_id in analysis.eligible_ops:
        inputs = graph.op_inputs[op_id]
        outputs = graph.op_outputs[op_id]
        input_sizes = [graph.tensors[t]["size"] for t in inputs]
        output_sizes = [graph.tensors[t]["size"] for t in outputs]
        shared_inputs = sum(
            len(graph.tensor_consumers.get(t, set()) & analysis.eligible_ops) > 1
            for t in inputs
        )
        fanout = sum(
            len(graph.tensor_consumers.get(t, set()) & analysis.eligible_ops)
            for t in outputs
        )
        op = graph.ops[op_id]
        result[op_id] = {
            "topological_index": analysis.topological_index[op_id],
            "topological_level": analysis.topological_level[op_id],
            "forward_critical_path": analysis.forward_path[op_id],
            "backward_critical_path": analysis.backward_path[op_id],
            "cycles": max(1, op["cycles"]),
            "pipe": op["pipe"],
            "input_bytes": sum(input_sizes),
            "output_bytes": sum(output_sizes),
            "max_input_bytes": max(input_sizes, default=0),
            "max_output_bytes": max(output_sizes, default=0),
            "shared_input_count": shared_inputs,
            "output_fanout": fanout,
            "external_ddr_inputs": set(analysis.external_sources.get(op_id, set())),
        }
    return result


def build_subgraph_features(
    analysis: GraphAnalysis, partition: Partition
) -> tuple[list[SubgraphFeatures], dict[tuple[int, int], dict[int | tuple, int]]]:
    graph = analysis.graph
    group_count = len(partition.groups)
    group_inputs: list[dict[int, int]] = [defaultdict(int) for _ in range(group_count)]
    group_external: list[dict[int, int]] = [defaultdict(int) for _ in range(group_count)]
    group_outputs: list[dict[int, int]] = [defaultdict(int) for _ in range(group_count)]
    group_external_users: list[set[int]] = [set() for _ in range(group_count)]
    group_work: list[dict[str, int]] = [defaultdict(int) for _ in range(group_count)]
    group_cp: list[int] = [0] * group_count
    group_lifetimes: list[dict[int, tuple[int, int]]] = [dict() for _ in range(group_count)]
    group_bounds: dict[tuple[int, int], dict[int | tuple, int]] = defaultdict(dict)
    pred_groups = [set() for _ in range(group_count)]
    succ_groups = [set() for _ in range(group_count)]
    members_by_group = partition.groups

    for group_id, members in enumerate(members_by_group):
        member_set = set(members)
        local_rank: dict[int, int] = {}
        for op_id in reversed(analysis.topological_order):
            if op_id not in member_set:
                continue
            op = graph.ops[op_id]
            cycles = max(1, op["cycles"])
            group_work[group_id][op["pipe"]] += cycles
            local_rank[op_id] = cycles + max(
                (local_rank[child] for child in analysis.successors[op_id]
                 if child in member_set),
                default=0,
            )
            for tensor_id in graph.op_inputs[op_id]:
                tensor = graph.tensors[tensor_id]
                if tensor["pos"] != "DDR":
                    group_inputs[group_id][tensor_id] = tensor["size"]
            for tensor_id in graph.op_outputs[op_id]:
                tensor = graph.tensors[tensor_id]
                if tensor["pos"] != "DDR":
                    group_outputs[group_id][tensor_id] = tensor["size"]
        group_cp[group_id] = max(local_rank.values(), default=0)
        for op_id in members:
            for tensor_id in analysis.external_inputs.get(op_id, set()):
                group_external[group_id][tensor_id] = graph.tensors[tensor_id]["size"]
                group_external_users[group_id].add(tensor_id)

    for tensor_id, tensor in graph.tensors.items():
        producer_groups = {
            partition.op_to_subgraph[op_id]
            for op_id in graph.tensor_producers.get(tensor_id, set())
            if op_id in partition.op_to_subgraph
        }
        consumer_groups = {
            partition.op_to_subgraph[op_id]
            for op_id in graph.tensor_consumers.get(tensor_id, set())
            if op_id in partition.op_to_subgraph
        }
        if tensor["pos"] == "DDR":
            continue
        for source_group in producer_groups:
            for target_group in consumer_groups:
                if source_group != target_group:
                    group_bounds[(source_group, target_group)][tensor_id] = tensor["size"]
                    pred_groups[target_group].add(source_group)
                    succ_groups[source_group].add(target_group)
        positions_by_group: dict[int, list[int]] = defaultdict(list)
        for op_id in (
            graph.tensor_producers.get(tensor_id, set())
            | graph.tensor_consumers.get(tensor_id, set())
        ):
            group_id = partition.op_to_subgraph.get(op_id)
            if group_id is not None:
                positions_by_group[group_id].append(analysis.topological_index[op_id])
        for group_id, local_positions in positions_by_group.items():
            group_lifetimes[group_id][tensor_id] = (
                min(local_positions), max(local_positions)
            )

    for source, target, byte_count in graph.direct_op_edges:
        if source not in partition.op_to_subgraph or target not in partition.op_to_subgraph:
            continue
        source_group = partition.op_to_subgraph[source]
        target_group = partition.op_to_subgraph[target]
        if source_group != target_group:
            key = ("direct", source, target)
            group_bounds[(source_group, target_group)][key] = byte_count
            pred_groups[target_group].add(source_group)
            succ_groups[source_group].add(target_group)

    for target, parents in analysis.predecessors.items():
        target_group = partition.op_to_subgraph[target]
        for parent in parents:
            parent_group = partition.op_to_subgraph[parent]
            if parent_group != target_group:
                pred_groups[target_group].add(parent_group)
                succ_groups[parent_group].add(target_group)
                group_bounds.setdefault((parent_group, target_group), {})

    group_features = []
    for group_id, members in enumerate(members_by_group):
        work = dict(group_work[group_id])
        m_cycles = work.get("PIPE_M", 0)
        v_cycles = work.get("PIPE_V", 0)
        estimated = max(group_cp[group_id], *work.values(), 0)
        position_map = [analysis.topological_index[op_id] for op_id in members]
        peaks = {"L1": 0, "UB": 0}
        residence = 0
        for pos in ("L1", "UB"):
            events: dict[int, int] = defaultdict(int)
            for tensor_id, (start, end) in group_lifetimes[group_id].items():
                tensor = graph.tensors[tensor_id]
                if tensor["pos"] == pos:
                    events[start] += tensor["size"]
                    events[end + 1] -= tensor["size"]
                    residence += tensor["size"] * max(1, end - start + 1)
            active = peak = 0
            for step in sorted(events):
                active += events[step]
                peak = max(peak, active)
            peaks[pos] = peak
        external = dict(group_external[group_id])
        group_features.append(SubgraphFeatures(
            subgraph_id=group_id,
            members=list(members),
            work_by_pipe=work,
            m_cycles=m_cycles,
            v_cycles=v_cycles,
            internal_critical_path=group_cp[group_id],
            estimated_compute=estimated,
            input_tensors=dict(group_inputs[group_id]),
            external_inputs=external,
            output_tensors=dict(group_outputs[group_id]),
            shared_ddr_inputs=set(group_external_users[group_id]),
            predecessor_subgraphs=pred_groups[group_id],
            successor_subgraphs=succ_groups[group_id],
            boundary_bytes={
                parent: sum(group_bounds[(parent, group_id)].values())
                for parent in pred_groups[group_id]
            },
            l1_pressure_est=peaks["L1"],
            ub_pressure_est=peaks["UB"],
            residence_cost_est=residence,
            topological_min_index=min(position_map, default=-1),
            topological_max_index=max(position_map, default=-1),
        ))
    return group_features, dict(group_bounds)