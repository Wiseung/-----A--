from __future__ import annotations

import heapq
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

from .features import SubgraphFeatures, build_subgraph_features
from .graph_analysis import GraphAnalysis
from .legality import validate_plan
from .partition import Partition


@dataclass(slots=True)
class PlacementEstimate:
    score: tuple[float, float, int]
    ranking_score: tuple[float, float, float, float, float, int]
    core_id: int
    group_start: float
    group_finish: float
    pipe_ends: dict[str, float]
    pipe_states: list[dict[str, float]]
    op_finish_times: dict[int, float]
    output_ready_times: dict[int, float]
    cache_insert_events: list[tuple[float, int, int, int]]
    cache_access_events: list[tuple[float, int, int, int, bool]]
    external_ready_times: dict[int, float]
    tensor_input_ready: dict[int, float]
    external_input_updates: set[int]
    tensor_target_updates: set[int]
    final_output_updates: set[tuple[int, int]]
    delta_read_bytes: int
    delta_write_bytes: int
    delta_cross_copy_bytes: int
    delta_repeated_input_bytes: int
    active_core_penalty: float
    critical_path_cross_core_penalty: float
    m_v_imbalance_penalty: float


def _input_maps(
    analysis: GraphAnalysis,
    partition: Partition,
    features: list[SubgraphFeatures],
    bounds: dict[tuple[int, int], dict[int | tuple, int]],
) -> tuple[
    list[dict[int, int]],
    list[dict[int | tuple, tuple[set[int], int]]],
    list[int],
]:
    graph = analysis.graph
    external = [dict(item.external_inputs) for item in features]
    internal: list[dict[int | tuple, tuple[set[int], int]]] = [dict() for _ in features]
    outgoing: list[set[int]] = [set() for _ in features]
    outgoing_direct_bytes = [0] * len(features)

    for (source_group, target_group), tensor_sizes in bounds.items():
        for tensor_id, size in tensor_sizes.items():
            source_groups, _ = internal[target_group].setdefault(
                tensor_id, (set(), size)
            )
            source_groups.add(source_group)
            if isinstance(tensor_id, int):
                outgoing[source_group].add(tensor_id)
            else:
                outgoing_direct_bytes[source_group] += size

    for group_id, members in enumerate(partition.groups):
        for op_id in members:
            for tensor_id in graph.op_outputs[op_id]:
                if graph.tensors[tensor_id]["pos"] == "DDR":
                    continue
                has_final_copy = any(
                    graph.ops[consumer]["op"] == "COPY_OUT"
                    for consumer in graph.tensor_consumers.get(tensor_id, set())
                )
                eligible_consumers = (
                    graph.tensor_consumers.get(tensor_id, set()) & analysis.eligible_ops
                )
                if has_final_copy or not eligible_consumers:
                    outgoing[group_id].add(tensor_id)
    outgoing_bytes = [
        outgoing_direct_bytes[group_id] + sum(
            graph.tensors[tensor_id]["size"] for tensor_id in tensor_ids
        )
        for group_id, tensor_ids in enumerate(outgoing)
    ]
    return external, internal, outgoing_bytes


def _boundary_input_sizes(
    group_id: int,
    internal: list[dict[int | tuple, tuple[set[int], int]]],
    features: list[SubgraphFeatures],
) -> dict[int | tuple, tuple[set[int], int]]:
    result = dict(internal[group_id])
    for tensor_id, size in features[group_id].external_inputs.items():
        result[("external", tensor_id)] = (set(), size)
    return result


def _cache_contains_at(
    events: list[tuple[float, int, int, int]],
    query_time: float,
    capacity: int,
    tensor_id: int,
) -> bool:
    cache: OrderedDict[int, int] = OrderedDict()
    cache_bytes = 0
    for available_at, _, inserted_id, size in sorted(events):
        if available_at > query_time:
            break
        if inserted_id in cache or size > capacity:
            continue
        while cache and cache_bytes + size > capacity:
            _, evicted_size = cache.popitem(last=False)
            cache_bytes -= evicted_size
        cache[inserted_id] = size
        cache_bytes += size
    return tensor_id in cache


def _reuse_distance_summary(
    events: list[tuple[float, int, int, int, bool]],
) -> tuple[int, dict[str, int]]:
    ordered = sorted(events)
    last_access: dict[int, int] = {}
    totals: defaultdict[int, int] = defaultdict(int)
    for index, (_, _, tensor_id, _, _) in enumerate(ordered):
        previous = last_access.get(tensor_id)
        if previous is not None:
            seen_misses: set[int] = set()
            distance = 0
            for _, _, other_id, size, cache_hit in ordered[previous + 1:index]:
                if other_id != tensor_id and not cache_hit and other_id not in seen_misses:
                    seen_misses.add(other_id)
                    distance += size
            totals[tensor_id] += distance
        last_access[tensor_id] = index
    by_tensor = {
        str(tensor_id): distance for tensor_id, distance in sorted(totals.items())
    }
    return sum(by_tensor.values()), by_tensor


def _original_copy_bytes(analysis: GraphAnalysis) -> int:
    graph = analysis.graph
    total = 0
    for op_id, op in graph.ops.items():
        if op["op"] == "COPY_IN":
            total += sum(graph.tensors[tensor_id]["size"]
                         for tensor_id in graph.op_outputs[op_id])
        elif op["op"] == "COPY_OUT":
            total += sum(graph.tensors[tensor_id]["size"]
                         for tensor_id in graph.op_inputs[op_id])
    return total


_PLACEMENT_SCORING_WEIGHTS = {
    "baseline": (0.0, 0.0, 0.0),
    "soft": (1.0, 1.0, 1.0),
}


def _relative_load_imbalance(loads: list[int]) -> float:
    total = sum(loads)
    if not loads or total == 0:
        return 0.0
    return max(loads) / (total / len(loads)) - 1.0


def _placement_penalties(
    group_id: int,
    core_id: int,
    ncores: int,
    core_schedules: list[list[int]],
    core_m_cycles: list[int],
    core_v_cycles: list[int],
    features: list[SubgraphFeatures],
    predecessors: list[set[int]],
    assigned_core: dict[int, int],
    critical_groups: set[int],
) -> tuple[float, float, float]:
    active_core_count = sum(bool(schedule) for schedule in core_schedules)
    if not core_schedules[core_id]:
        active_core_count += 1
    active_core_penalty = float(max(0, ncores - active_core_count))

    critical_path_cross_core_penalty = float(sum(
        parent in critical_groups
        and parent in assigned_core
        and assigned_core[parent] != core_id
        for parent in predecessors[group_id]
        if group_id in critical_groups
    ))

    candidate_m_cycles = list(core_m_cycles)
    candidate_v_cycles = list(core_v_cycles)
    candidate_m_cycles[core_id] += features[group_id].m_cycles
    candidate_v_cycles[core_id] += features[group_id].v_cycles
    m_v_imbalance_penalty = (
        _relative_load_imbalance(candidate_m_cycles)
        + _relative_load_imbalance(candidate_v_cycles)
    )
    return (
        active_core_penalty,
        critical_path_cross_core_penalty,
        m_v_imbalance_penalty,
    )


def _placement_ranking_score(
    finish: float,
    transfer_bytes: float,
    core_id: int,
    penalties: tuple[float, float, float],
    weights: tuple[float, float, float],
) -> tuple[float, float, float, float, float, int]:
    active_core_penalty, critical_path_penalty, imbalance_penalty = penalties
    active_weight, critical_weight, imbalance_weight = weights
    return (
        finish,
        critical_weight * critical_path_penalty,
        active_weight * active_core_penalty,
        imbalance_weight * imbalance_penalty,
        transfer_bytes,
        core_id,
    )


def _estimated_partition_copy_bytes(
    analysis: GraphAnalysis,
    partition: Partition,
    owners: dict[int, int],
) -> int:
    graph = analysis.graph
    task_copy_bytes = 0
    for tensor_id, tensor in graph.tensors.items():
        producer_ops = graph.tensor_producers.get(tensor_id, set()) & analysis.eligible_ops
        consumer_ops = graph.tensor_consumers.get(tensor_id, set()) & analysis.eligible_ops
        producer_cores = {
            owners[partition.op_to_subgraph[op_id]] for op_id in producer_ops
        }
        consumer_cores = {
            owners[partition.op_to_subgraph[op_id]] for op_id in consumer_ops
        }
        size = tensor["size"]
        if consumer_cores and not producer_cores:
            task_copy_bytes += size * len(consumer_cores)
        has_original_copy_out = any(
            graph.ops[op_id]["op"] == "COPY_OUT"
            for op_id in graph.tensor_consumers.get(tensor_id, set())
        )
        if producer_cores and (has_original_copy_out or not consumer_cores):
            task_copy_bytes += size * len(producer_cores)
        task_copy_bytes += 2 * size * sum(
            source_core != target_core
            for source_core in producer_cores
            for target_core in consumer_cores
        )

    for source, target, size in graph.direct_op_edges:
        if source not in partition.op_to_subgraph or target not in partition.op_to_subgraph:
            continue
        source_core = owners[partition.op_to_subgraph[source]]
        target_core = owners[partition.op_to_subgraph[target]]
        if source_core != target_core:
            task_copy_bytes += 2 * max(0, int(size))
    return task_copy_bytes - _original_copy_bytes(analysis)


def _schedule_live_range_metrics(
    analysis: GraphAnalysis,
    partition: Partition,
    plan: dict[str, Any],
    problem: int,
    capacity: dict[str, int],
) -> dict[str, Any]:
    graph = analysis.graph
    group_members = {
        group_id: sorted(
            members,
            key=analysis.topological_index.__getitem__,
        )
        for group_id, members in enumerate(partition.groups)
    }
    streams: list[tuple[int, list[int]]] = []
    if problem == 1:
        for core_id, schedule in enumerate(plan["core_schedules"]):
            for group_id in schedule:
                streams.append((core_id, group_members.get(group_id, [])))
    else:
        for core_id, schedule in enumerate(plan["core_schedules"]):
            streams.append((
                core_id,
                [
                    op_id
                    for group_id in schedule
                    for op_id in group_members.get(group_id, [])
                ],
            ))

    core_l1_peak = [0] * len(plan["core_schedules"])
    core_ub_peak = [0] * len(plan["core_schedules"])
    core_l1_overflow = [0] * len(plan["core_schedules"])
    core_ub_overflow = [0] * len(plan["core_schedules"])
    core_l1_residence = [0] * len(plan["core_schedules"])
    core_ub_residence = [0] * len(plan["core_schedules"])

    all_tensors = graph.tensors
    memory_tensors = {
        tensor_id: tensor
        for tensor_id, tensor in all_tensors.items()
        if tensor["pos"] in {"L1", "UB"}
    }
    boundary_tensors: set[int] = set()
    long_lived_tensors: set[int] = set()
    for tensor_id, tensor in all_tensors.items():
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
        crosses_groups = any(
            source_group != target_group
            for source_group in producer_groups
            for target_group in consumer_groups
        )
        if tensor["pos"] == "DDR":
            crosses_groups = len(consumer_groups) > 1 or len(producer_groups) > 1
        if crosses_groups:
            boundary_tensors.add(tensor_id)
        if tensor_id in memory_tensors and len(producer_groups | consumer_groups) > 1:
            long_lived_tensors.add(tensor_id)
    boundary_direct_edge_bytes = sum(
        max(0, int(size))
        for source, target, size in graph.direct_op_edges
        if source in partition.op_to_subgraph
        and target in partition.op_to_subgraph
        and partition.op_to_subgraph[source] != partition.op_to_subgraph[target]
    )

    for core_id, operations in streams:
        if not operations:
            continue
        positions = {op_id: index for index, op_id in enumerate(operations)}
        events: dict[str, dict[int, int]] = {
            "L1": defaultdict(int),
            "UB": defaultdict(int),
        }
        residence = {"L1": 0, "UB": 0}
        local_ops = set(operations)
        for tensor_id, tensor in memory_tensors.items():
            incident_ops = (
                graph.tensor_producers.get(tensor_id, set())
                | graph.tensor_consumers.get(tensor_id, set())
            ) & local_ops
            if not incident_ops:
                continue
            start = min(positions[op_id] for op_id in incident_ops)
            end = max(positions[op_id] for op_id in incident_ops)
            size = max(0, int(tensor["size"]))
            position = tensor["pos"]
            events[position][start] += size
            events[position][end + 1] -= size
            residence[position] += size * max(1, end - start + 1)

        peaks = {"L1": 0, "UB": 0}
        for position in ("L1", "UB"):
            active = 0
            for event_position in sorted(events[position]):
                active += events[position][event_position]
                peaks[position] = max(peaks[position], active)
        core_l1_peak[core_id] = max(core_l1_peak[core_id], peaks["L1"])
        core_ub_peak[core_id] = max(core_ub_peak[core_id], peaks["UB"])
        core_l1_overflow[core_id] = max(
            core_l1_overflow[core_id],
            max(0, peaks["L1"] - int(capacity.get("L1", 0))),
        )
        core_ub_overflow[core_id] = max(
            core_ub_overflow[core_id],
            max(0, peaks["UB"] - int(capacity.get("UB", 0))),
        )
        core_l1_residence[core_id] += residence["L1"]
        core_ub_residence[core_id] += residence["UB"]

    as_dict = lambda values: {
        str(core_id): value for core_id, value in enumerate(values)
    }
    owners = {
        group_id: core_id
        for core_id, schedule in enumerate(plan["core_schedules"])
        for group_id in schedule
    }
    boundary_copy_bytes_est = (
        _estimated_partition_copy_bytes(analysis, partition, owners)
        if problem != 1 else 0
    )
    return {
        "core_l1_peak_est": as_dict(core_l1_peak),
        "core_ub_peak_est": as_dict(core_ub_peak),
        "l1_overflow_bytes_est": as_dict(core_l1_overflow),
        "ub_overflow_bytes_est": as_dict(core_ub_overflow),
        "l1_residence_bytes_est": as_dict(core_l1_residence),
        "ub_residence_bytes_est": as_dict(core_ub_residence),
        "max_core_l1_peak_est": max(core_l1_peak, default=0),
        "max_core_ub_peak_est": max(core_ub_peak, default=0),
        "max_l1_overflow_bytes_est": max(core_l1_overflow, default=0),
        "max_ub_overflow_bytes_est": max(core_ub_overflow, default=0),
        "long_lived_tensor_bytes": sum(
            int(memory_tensors[tensor_id]["size"])
            for tensor_id in long_lived_tensors
        ),
        "boundary_tensor_count": len(boundary_tensors),
        "boundary_tensor_bytes": sum(
            int(all_tensors[tensor_id]["size"])
            for tensor_id in boundary_tensors
        ),
        "boundary_direct_edge_bytes": boundary_direct_edge_bytes,
        "boundary_bytes_total": (
            sum(int(all_tensors[tensor_id]["size"]) for tensor_id in boundary_tensors)
            + boundary_direct_edge_bytes
        ),
        "boundary_copy_bytes_est": boundary_copy_bytes_est,
    }


def schedule_partition(
    analysis: GraphAnalysis,
    partition: Partition,
    ncores: int,
    problem: int,
    config: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    placement_scoring: str = "baseline",
    cache_ordering: str = "fifo",
) -> dict[str, Any]:
    try:
        scoring_weights = _PLACEMENT_SCORING_WEIGHTS[placement_scoring]
    except KeyError as error:
        raise ValueError(f"unknown placement scoring: {placement_scoring}") from error
    if cache_ordering not in {"fifo", "reuse_distance"}:
        raise ValueError(f"unknown cache ordering: {cache_ordering}")
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "placement_scoring": placement_scoring,
            "cache_ordering": cache_ordering,
            "groups": [],
            "estimated_read_copy_bytes": 0,
            "estimated_write_copy_bytes": 0,
            "estimated_cross_copy_bytes": 0,
            "estimated_repeated_input_bytes": 0,
            "cache_accesses": 0,
            "reuse_distance_bytes": 0,
            "reuse_distance_bytes_by_tensor": {},
            "core_l1_peak_est": {},
            "core_ub_peak_est": {},
            "l1_overflow_bytes_est": {},
            "ub_overflow_bytes_est": {},
            "l1_residence_bytes_est": {},
            "ub_residence_bytes_est": {},
            "max_core_l1_peak_est": 0,
            "max_core_ub_peak_est": 0,
            "max_l1_overflow_bytes_est": 0,
            "max_ub_overflow_bytes_est": 0,
            "long_lived_tensor_bytes": 0,
            "boundary_tensor_count": 0,
            "boundary_tensor_bytes": 0,
            "boundary_direct_edge_bytes": 0,
            "boundary_bytes_total": 0,
            "boundary_copy_bytes_est": 0,
        })
    features, bounds = build_subgraph_features(analysis, partition)
    count = len(features)
    if count == 0:
        if diagnostics is not None:
            diagnostics.update({
                "estimated_new_copy_bytes": 0,
                "original_graph_copy_bytes": _original_copy_bytes(analysis),
                "estimated_partition_added_copy_bytes": 0,
            })
        plan = {"node_to_subgraph": {}, "core_schedules": [[] for _ in range(ncores)]}
        validate_plan(analysis.graph, plan, ncores)
        return plan

    external_inputs, internal_inputs, output_copy_bytes = _input_maps(
        analysis, partition, features, bounds
    )
    graph = analysis.graph
    tensor_producer_ops = {
        tensor_id: set(graph.tensor_producers.get(tensor_id, set()))
        & analysis.eligible_ops
        for tensor_id in graph.tensors
    }
    final_output_groups: dict[int, set[int]] = {}
    for tensor_id, producer_ops in tensor_producer_ops.items():
        if not producer_ops:
            continue
        has_final_copy = any(
            graph.ops[consumer]["op"] == "COPY_OUT"
            for consumer in graph.tensor_consumers.get(tensor_id, set())
        )
        eligible_consumers = (
            graph.tensor_consumers.get(tensor_id, set()) & analysis.eligible_ops
        )
        if has_final_copy or not eligible_consumers:
            final_output_groups[tensor_id] = {
                partition.op_to_subgraph[op_id] for op_id in producer_ops
            }
    predecessors = [set(item.predecessor_subgraphs) for item in features]
    successors = [set(item.successor_subgraphs) for item in features]
    boundary_sizes: dict[tuple[int, int], dict[int | tuple, int]] = bounds
    rank = [0] * count
    sync = (
        config["scene_a"]["task_cross_core_wait_cycles"]
        if problem == 1
        else config["scene_b"]["cross_core_copy_delay_cycles"]
    )
    group_indegree = [len(items) for items in predecessors]
    group_ready = [group_id for group_id, degree in enumerate(group_indegree) if degree == 0]
    heapq.heapify(group_ready)
    group_topological_order = []
    while group_ready:
        group_id = heapq.heappop(group_ready)
        group_topological_order.append(group_id)
        for child in successors[group_id]:
            group_indegree[child] -= 1
            if group_indegree[child] == 0:
                heapq.heappush(group_ready, child)
    if len(group_topological_order) != count:
        raise RuntimeError("subgraph dependency graph contains a cycle")

    for group_id in reversed(group_topological_order):
        children = []
        for child in successors[group_id]:
            size = sum(boundary_sizes.get((group_id, child), {}).values())
            copy_time = 2 * size / config["bandwidth"]
            children.append(rank[child] + sync + copy_time)
        rank[group_id] = features[group_id].estimated_compute + max(children, default=0)

    remaining = [len(items) for items in predecessors]
    core_schedules: list[list[int]] = [[] for _ in range(ncores)]
    core_task_end = [0.0] * ncores
    core_pipe_end: list[dict[str, float]] = [defaultdict(float) for _ in range(ncores)]
    core_m_cycles = [0] * ncores
    core_v_cycles = [0] * ncores
    assigned_core: dict[int, int] = {}
    estimated_finish: dict[int, float] = {}
    op_finish_by_id: dict[int, float] = {}
    external_core_users: dict[int, set[int]] = defaultdict(set)
    tensor_target_cores: dict[int, set[int]] = defaultdict(set)
    external_input_ready: dict[tuple[int, int], float] = {}
    tensor_input_ready: dict[tuple[int, int], float] = {}
    final_output_cores: dict[int, set[int]] = defaultdict(set)
    cache_insert_events: list[tuple[float, int, int, int]] = []
    cache_access_events: list[tuple[float, int, int, int, bool]] = []
    cache_capacity = config["problem_3"]["cache_capacity_bytes"]
    cache_bandwidth = config["problem_3"]["cache_bandwidth_bytes_per_cycle"]
    ddr_bandwidth = config["bandwidth"]
    critical_path_length = max(analysis.forward_path.values(), default=0)
    critical_ops = {
        op_id for op_id in analysis.eligible_ops
        if (
            analysis.forward_path[op_id]
            + analysis.backward_path[op_id]
            - max(1, graph.ops[op_id]["cycles"])
            == critical_path_length
        )
    }
    critical_groups = {
        partition.op_to_subgraph[op_id]
        for op_id in critical_ops
    }

    def ready_key(ready_group_id: int) -> tuple[float, float, float, int]:
        shared_bytes = sum(
            graph.tensors[tensor_id]["size"]
            for tensor_id in features[ready_group_id].shared_ddr_inputs
        )
        cache_priority = (
            -float(shared_bytes)
            if problem == 3 and cache_ordering == "reuse_distance"
            else 0.0
        )
        return (
            -rank[ready_group_id],
            cache_priority,
            -features[ready_group_id].estimated_compute,
            ready_group_id,
        )

    ready = [ready_key(group_id) for group_id, degree in enumerate(remaining) if degree == 0]
    heapq.heapify(ready)

    while ready:
        _, _, _, group_id = heapq.heappop(ready)
        best: PlacementEstimate | None = None
        candidate_scores = []
        inputs = _boundary_input_sizes(group_id, internal_inputs, features)
        group_members = set(features[group_id].members)

        for core_id in range(ncores):
            penalties = _placement_penalties(
                group_id,
                core_id,
                ncores,
                core_schedules,
                core_m_cycles,
                core_v_cycles,
                features,
                predecessors,
                assigned_core,
                critical_groups,
            )
            if problem == 1:
                source_ready = 0.0
                for parent in predecessors[group_id]:
                    wait = 0 if assigned_core[parent] == core_id else config[
                        "scene_a"]["task_cross_core_wait_cycles"]
                    source_ready = max(
                        source_ready,
                        estimated_finish[parent] + wait,
                    )
                ddr_read_cycles = sum(inputs[key][1] for key in inputs) / ddr_bandwidth
                group_work = dict(features[group_id].work_by_pipe)
                group_work["PIPE_MTE3"] = group_work.get("PIPE_MTE3", 0) + (
                    output_copy_bytes[group_id] / ddr_bandwidth
                )
                group_work["PIPE_MTE2"] = group_work.get("PIPE_MTE2", 0) + ddr_read_cycles
                start = max(
                    core_task_end[core_id]
                    + (config["scene_a"]["task_same_core_wait_cycles"]
                       if core_schedules[core_id] else 0),
                    source_ready,
                )
                duration = max(
                    features[group_id].internal_critical_path,
                    max(group_work.values(), default=0),
                )
                finish = start + duration
                pipe_ends = {"__task__": finish}
                transfer_bytes = float(sum(inputs[key][1] for key in inputs))
                score = (finish, transfer_bytes, core_id)
                candidate = PlacementEstimate(
                    score=score,
                    ranking_score=_placement_ranking_score(
                        finish,
                        transfer_bytes,
                        core_id,
                        penalties,
                        scoring_weights,
                    ),
                    core_id=core_id,
                    group_start=start,
                    group_finish=finish,
                    pipe_ends=pipe_ends,
                    pipe_states=[dict(state) for state in core_pipe_end],
                    op_finish_times={},
                    output_ready_times={},
                    cache_insert_events=[],
                    cache_access_events=[],
                    external_ready_times={},
                    tensor_input_ready={},
                    external_input_updates=set(),
                    tensor_target_updates=set(),
                    final_output_updates=set(),
                    delta_read_bytes=sum(inputs[key][1] for key in inputs),
                    delta_write_bytes=output_copy_bytes[group_id],
                    delta_cross_copy_bytes=0,
                    delta_repeated_input_bytes=0,
                    active_core_penalty=penalties[0],
                    critical_path_cross_core_penalty=penalties[1],
                    m_v_imbalance_penalty=penalties[2],
                )
            else:
                pipe_states = [dict(state) for state in core_pipe_end]
                op_finish_times: dict[int, float] = {}
                op_start_times: dict[int, float] = {}
                candidate_cache_events: list[tuple[float, int, int, int]] = []
                candidate_cache_accesses: list[tuple[float, int, int, int, bool]] = []
                external_ready_times: dict[int, float] = {}
                tensor_ready_times: dict[int, float] = {}
                external_updates: set[int] = set()
                tensor_updates: set[int] = set()
                final_updates: set[tuple[int, int]] = set()
                input_ready_by_tensor: dict[int, float] = {}
                direct_ready_by_op: dict[int, float] = defaultdict(float)
                deltas = {"read": 0, "write": 0, "cross": 0, "repeated": 0}

                def schedule_copyin(
                    target_core: int,
                    logical_id: int | None,
                    size: int,
                    earliest: float,
                ) -> float:
                    start = max(
                        pipe_states[target_core].get("PIPE_MTE2", 0.0),
                        earliest,
                    )
                    cache_hit = (
                        problem == 3
                        and logical_id is not None
                        and size > 0
                        and _cache_contains_at(
                            cache_insert_events + candidate_cache_events,
                            start,
                            cache_capacity,
                            logical_id,
                        )
                    )
                    bandwidth = cache_bandwidth if cache_hit else ddr_bandwidth
                    end = start + max(1, math.ceil(size / bandwidth))
                    pipe_states[target_core]["PIPE_MTE2"] = end
                    deltas["read"] += size
                    access_order = len(cache_access_events) + len(candidate_cache_accesses)
                    if logical_id is not None:
                        candidate_cache_accesses.append(
                            (end, access_order, logical_id, size, cache_hit)
                        )
                    if (problem == 3 and not cache_hit and logical_id is not None
                            and size <= cache_capacity):
                        event_order = len(cache_insert_events) + len(candidate_cache_events)
                        candidate_cache_events.append((end, event_order, logical_id, size))
                    return end

                def schedule_cross_pair(
                    source_core: int,
                    source_ready_at: float,
                    logical_id: int | None,
                    size: int,
                ) -> float:
                    out_start = max(
                        pipe_states[source_core].get("PIPE_MTE3", 0.0),
                        source_ready_at,
                    )
                    out_end = out_start + max(1, math.ceil(size / ddr_bandwidth))
                    pipe_states[source_core]["PIPE_MTE3"] = out_end
                    deltas["write"] += size
                    deltas["cross"] += 2 * size
                    return schedule_copyin(
                        core_id, logical_id, size, out_end + sync
                    )

                for input_key, (source_groups, size) in sorted(
                    inputs.items(), key=lambda item: repr(item[0])
                ):
                    if (isinstance(input_key, tuple)
                            and input_key[0] == "external"):
                        tensor_id = input_key[1]
                        if core_id in external_core_users[tensor_id]:
                            deltas["repeated"] += size
                            input_ready_by_tensor[tensor_id] = external_input_ready.get(
                                (tensor_id, core_id), 0.0
                            )
                            continue
                        external_ready_times[tensor_id] = schedule_copyin(
                            core_id, tensor_id, size, 0.0
                        )
                        external_updates.add(tensor_id)
                        input_ready_by_tensor[tensor_id] = external_ready_times[tensor_id]
                        continue

                    is_tensor = isinstance(input_key, int)
                    tensor_id = input_key if is_tensor else None
                    if is_tensor and core_id in tensor_target_cores[tensor_id]:
                        deltas["repeated"] += size
                        input_ready_by_tensor[tensor_id] = tensor_input_ready.get(
                            (tensor_id, core_id), 0.0
                        )
                        continue

                    source_cores = {
                        assigned_core[source_group]
                        for source_group in source_groups
                        if source_group in assigned_core
                    }
                    if group_id in source_groups:
                        source_cores.add(core_id)
                    remote_ready = []
                    for source_core in sorted(source_cores - {core_id}):
                        if is_tensor:
                            producer_ops = [
                                op_id for op_id in tensor_producer_ops[tensor_id]
                                if partition.op_to_subgraph[op_id] in source_groups
                                and assigned_core.get(
                                    partition.op_to_subgraph[op_id]
                                ) == source_core
                            ]
                            source_ready_at = max(
                                (op_finish_by_id[op_id] for op_id in producer_ops
                                 if op_id in op_finish_by_id),
                                default=max(
                                    (estimated_finish[source_group]
                                     for source_group in source_groups
                                     if assigned_core.get(source_group) == source_core),
                                    default=0.0,
                                ),
                            )
                            logical_id = tensor_id
                        else:
                            source_op = input_key[1]
                            source_ready_at = op_finish_by_id.get(
                                source_op,
                                estimated_finish.get(next(iter(source_groups), -1), 0.0),
                            )
                            logical_id = None
                        remote_ready.append(schedule_cross_pair(
                            source_core, source_ready_at, logical_id, size
                        ))

                    if is_tensor:
                        tensor_updates.add(tensor_id)
                        input_ready_by_tensor[tensor_id] = max(remote_ready, default=0.0)
                        tensor_ready_times[tensor_id] = input_ready_by_tensor[tensor_id]
                    else:
                        target_op = input_key[2]
                        direct_ready_by_op[target_op] = max(
                            direct_ready_by_op[target_op],
                            max(remote_ready, default=op_finish_by_id.get(input_key[1], 0.0)),
                        )

                for op_id in sorted(
                    group_members, key=analysis.topological_index.__getitem__
                ):
                    parent_ready = max((
                        op_finish_times.get(
                            parent,
                            op_finish_by_id.get(
                                parent,
                                estimated_finish.get(
                                    partition.op_to_subgraph[parent], 0.0
                                ),
                            ),
                        )
                        for parent in analysis.predecessors[op_id]
                    ), default=0.0)
                    tensor_ready = max((
                        input_ready_by_tensor.get(tensor_id, 0.0)
                        for tensor_id in graph.op_inputs[op_id]
                    ), default=0.0)
                    pipe = graph.ops[op_id]["pipe"]
                    start = max(
                        pipe_states[core_id].get(pipe, 0.0),
                        parent_ready,
                        tensor_ready,
                        direct_ready_by_op.get(op_id, 0.0),
                    )
                    end = start + max(1, graph.ops[op_id]["cycles"])
                    pipe_states[core_id][pipe] = end
                    op_start_times[op_id] = start
                    op_finish_times[op_id] = end

                candidate_owners = dict(assigned_core)
                candidate_owners[group_id] = core_id
                for tensor_id, producer_groups in final_output_groups.items():
                    if not producer_groups.issubset(candidate_owners):
                        continue
                    source_cores = {
                        candidate_owners[source_group]
                        for source_group in producer_groups
                    }
                    for source_core in sorted(source_cores):
                        if (source_core in final_output_cores[tensor_id]
                                or (tensor_id, source_core) in final_updates):
                            continue
                        producer_ops = [
                            op_id for op_id in tensor_producer_ops[tensor_id]
                            if candidate_owners[partition.op_to_subgraph[op_id]] == source_core
                        ]
                        source_ready_at = max((
                            op_finish_times.get(op_id, op_finish_by_id.get(op_id, 0.0))
                            for op_id in producer_ops
                        ), default=0.0)
                        out_start = max(
                            pipe_states[source_core].get("PIPE_MTE3", 0.0),
                            source_ready_at,
                        )
                        out_end = out_start + max(
                            1,
                            math.ceil(graph.tensors[tensor_id]["size"] / ddr_bandwidth),
                        )
                        pipe_states[source_core]["PIPE_MTE3"] = out_end
                        deltas["write"] += graph.tensors[tensor_id]["size"]
                        final_updates.add((tensor_id, source_core))

                output_ready_times = {
                    tensor_id: max(
                        op_finish_times[op_id]
                        for op_id in tensor_producer_ops[tensor_id] & group_members
                    )
                    for tensor_id in graph.tensors
                    if tensor_producer_ops[tensor_id] & group_members
                }
                group_start = min(op_start_times.values(), default=0.0)
                group_finish = max(op_finish_times.values(), default=group_start)
                partial_makespan = max(
                    [group_finish, *estimated_finish.values()]
                    + [end for state in pipe_states for end in state.values()]
                )
                transfer_bytes = float(deltas["read"] + deltas["write"])
                score = (partial_makespan, transfer_bytes, core_id)
                candidate = PlacementEstimate(
                    score=score,
                    ranking_score=_placement_ranking_score(
                        partial_makespan,
                        transfer_bytes,
                        core_id,
                        penalties,
                        scoring_weights,
                    ),
                    core_id=core_id,
                    group_start=group_start,
                    group_finish=group_finish,
                    pipe_ends=dict(pipe_states[core_id]),
                    pipe_states=pipe_states,
                    op_finish_times=op_finish_times,
                    output_ready_times=output_ready_times,
                    cache_insert_events=candidate_cache_events,
                    cache_access_events=candidate_cache_accesses,
                    external_ready_times=external_ready_times,
                    tensor_input_ready=tensor_ready_times,
                    external_input_updates=external_updates,
                    tensor_target_updates=tensor_updates,
                    final_output_updates=final_updates,
                    delta_read_bytes=deltas["read"],
                    delta_write_bytes=deltas["write"],
                    delta_cross_copy_bytes=deltas["cross"],
                    delta_repeated_input_bytes=deltas["repeated"],
                    active_core_penalty=penalties[0],
                    critical_path_cross_core_penalty=penalties[1],
                    m_v_imbalance_penalty=penalties[2],
                )

            if best is None or candidate.ranking_score < best.ranking_score:
                best = candidate
            candidate_scores.append({
                "core_id": candidate.core_id,
                "score": candidate.score,
                "ranking_score": candidate.ranking_score,
                "group_start": candidate.group_start,
                "group_finish": candidate.group_finish,
                "delta_read_bytes": candidate.delta_read_bytes,
                "delta_write_bytes": candidate.delta_write_bytes,
                "delta_cross_copy_bytes": candidate.delta_cross_copy_bytes,
                "delta_repeated_input_bytes": candidate.delta_repeated_input_bytes,
                "active_core_penalty": candidate.active_core_penalty,
                "critical_path_cross_core_penalty": candidate.critical_path_cross_core_penalty,
                "m_v_imbalance_penalty": candidate.m_v_imbalance_penalty,
                "pipe_ends": candidate.pipe_ends,
            })

        if best is None:
            raise RuntimeError("no core candidate was available")
        core_id = best.core_id
        assigned_core[group_id] = core_id
        estimated_finish[group_id] = best.group_finish
        core_schedules[core_id].append(group_id)
        core_m_cycles[core_id] += features[group_id].m_cycles
        core_v_cycles[core_id] += features[group_id].v_cycles
        if problem == 1:
            core_task_end[core_id] = estimated_finish[group_id]
        else:
            for state_core, state in enumerate(best.pipe_states):
                core_pipe_end[state_core].clear()
                core_pipe_end[state_core].update(state)
            op_finish_by_id.update(best.op_finish_times)
            for tensor_id, ready_at in best.external_ready_times.items():
                external_input_ready[(tensor_id, core_id)] = ready_at
            for tensor_id, ready_at in best.tensor_input_ready.items():
                tensor_input_ready[(tensor_id, core_id)] = ready_at
            for tensor_id in best.tensor_target_updates:
                tensor_target_cores[tensor_id].add(core_id)
            for tensor_id, source_core in best.final_output_updates:
                final_output_cores[tensor_id].add(source_core)
            cache_insert_events.extend(best.cache_insert_events)
            cache_access_events.extend(best.cache_access_events)

        if diagnostics is not None:
            feature = features[group_id]
            diagnostics["groups"].append({
                "group_id": group_id,
                "members": list(feature.members),
                "work_by_pipe": dict(feature.work_by_pipe),
                "m_cycles": feature.m_cycles,
                "v_cycles": feature.v_cycles,
                "estimated_compute": feature.estimated_compute,
                "internal_critical_path": feature.internal_critical_path,
                "l1_pressure_est": feature.l1_pressure_est,
                "ub_pressure_est": feature.ub_pressure_est,
                "residence_cost_est": feature.residence_cost_est,
                "boundary_bytes": sum(feature.boundary_bytes.values()),
                "rank": rank[group_id],
                "candidate_placements": candidate_scores,
                "chosen_core": core_id,
                "group_start": best.group_start,
                "group_finish": best.group_finish,
                "delta_read_bytes": best.delta_read_bytes,
                "delta_write_bytes": best.delta_write_bytes,
                "delta_cross_copy_bytes": best.delta_cross_copy_bytes,
                "delta_repeated_input_bytes": best.delta_repeated_input_bytes,
            })
            diagnostics["estimated_read_copy_bytes"] += best.delta_read_bytes
            diagnostics["estimated_write_copy_bytes"] += best.delta_write_bytes
            diagnostics["estimated_cross_copy_bytes"] += best.delta_cross_copy_bytes
            diagnostics["estimated_repeated_input_bytes"] += best.delta_repeated_input_bytes
            diagnostics["cache_accesses"] = len(cache_access_events)

        for tensor_id in external_inputs[group_id]:
            external_core_users[tensor_id].add(core_id)

        for child in successors[group_id]:
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(
                    ready,
                    ready_key(child),
                )

    if len(assigned_core) != count:
        raise RuntimeError("scheduler failed to assign all subgraphs")

    plan = {
        "node_to_subgraph": {
            str(op_id): partition.op_to_subgraph[op_id]
            for op_id in sorted(analysis.eligible_ops)
        },
        "core_schedules": core_schedules,
    }

    if diagnostics is not None:
        diagnostics.update(_schedule_live_range_metrics(
            analysis,
            partition,
            plan,
            problem,
            config.get("capacity", {}),
        ))
        reuse_distance_bytes, reuse_distance_by_tensor = _reuse_distance_summary(
            cache_access_events
        )
        diagnostics["reuse_distance_bytes"] = reuse_distance_bytes
        diagnostics["reuse_distance_bytes_by_tensor"] = reuse_distance_by_tensor
        diagnostics["estimated_new_copy_bytes"] = (
            diagnostics["estimated_read_copy_bytes"]
            + diagnostics["estimated_write_copy_bytes"]
        )
        diagnostics["original_graph_copy_bytes"] = _original_copy_bytes(analysis)
        diagnostics["estimated_partition_added_copy_bytes"] = (
            diagnostics["estimated_new_copy_bytes"]
            - diagnostics["original_graph_copy_bytes"]
            if problem != 1 else None
        )
        diagnostics["estimated_partition_added_copy_bytes_recomputed"] = (
            _estimated_partition_copy_bytes(analysis, partition, assigned_core)
            if problem != 1 else None
        )

    validate_plan(analysis.graph, plan, ncores)
    return plan