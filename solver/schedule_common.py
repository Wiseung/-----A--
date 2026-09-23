from __future__ import annotations

import heapq
from collections import OrderedDict, defaultdict
from typing import Any

from .features import SubgraphFeatures, build_subgraph_features
from .graph_analysis import GraphAnalysis
from .legality import validate_plan
from .partition import Partition


def _input_maps(
    analysis: GraphAnalysis,
    partition: Partition,
    features: list[SubgraphFeatures],
    bounds: dict[tuple[int, int], dict[int | tuple, int]],
) -> tuple[
    list[dict[int, int]],
    list[dict[int | tuple, tuple[int, int]]],
    list[set[int]],
    list[int],
]:
    graph = analysis.graph
    external = [dict(item.external_inputs) for item in features]
    internal: list[dict[int | tuple, tuple[int, int]]] = [dict() for _ in features]
    outgoing: list[set[int]] = [set() for _ in features]
    outgoing_direct_bytes = [0] * len(features)

    for (source_group, target_group), tensor_sizes in bounds.items():
        for tensor_id, size in tensor_sizes.items():
            internal[target_group][tensor_id] = (source_group, size)
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
    return external, internal, outgoing, outgoing_bytes


def _boundary_input_sizes(
    group_id: int,
    internal: list[dict[int | tuple, tuple[int, int]]],
    features: list[SubgraphFeatures],
) -> dict[int | tuple, tuple[int, int]]:
    result = dict(internal[group_id])
    for tensor_id, size in features[group_id].external_inputs.items():
        result[("external", tensor_id)] = (-1, size)
    return result


def _update_cache(
    cache: OrderedDict[int, tuple[int, float]],
    cache_bytes: int,
    tensor_id: int,
    size: int,
    available_at: float,
    capacity: int,
) -> int:
    if size > capacity or tensor_id in cache:
        return cache_bytes
    while cache and cache_bytes + size > capacity:
        _, (old_size, _) = cache.popitem(last=False)
        cache_bytes -= old_size
    cache[tensor_id] = (size, available_at)
    return cache_bytes + size


def schedule_partition(
    analysis: GraphAnalysis,
    partition: Partition,
    ncores: int,
    problem: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    features, bounds = build_subgraph_features(analysis, partition)
    count = len(features)
    if count == 0:
        plan = {"node_to_subgraph": {}, "core_schedules": [[] for _ in range(ncores)]}
        validate_plan(analysis.graph, plan, ncores)
        return plan

    external_inputs, internal_inputs, output_tensors, output_copy_bytes = _input_maps(
        analysis, partition, features, bounds
    )
    predecessors = [set(item.predecessor_subgraphs) for item in features]
    successors = [set(item.successor_subgraphs) for item in features]
    boundary_sizes: dict[tuple[int, int], dict[int | tuple, int]] = bounds
    rank = [0] * count
    sync = (
        config["scene_a"]["task_cross_core_wait_cycles"]
        if problem == 1
        else config["scene_b"]["cross_core_copy_delay_cycles"]
    )
    for group_id in range(count - 1, -1, -1):
        children = []
        for child in successors[group_id]:
            size = sum(boundary_sizes.get((group_id, child), {}).values())
            copy_time = 2 * size / config["bandwidth"]
            children.append(rank[child] + sync + copy_time)
        rank[group_id] = features[group_id].estimated_compute + max(children, default=0)

    remaining = [len(items) for items in predecessors]
    ready = [(-rank[group_id], -features[group_id].estimated_compute, group_id)
             for group_id, degree in enumerate(remaining) if degree == 0]
    heapq.heapify(ready)
    core_schedules: list[list[int]] = [[] for _ in range(ncores)]
    core_task_end = [0.0] * ncores
    core_pipe_end: list[dict[str, float]] = [defaultdict(float) for _ in range(ncores)]
    assigned_core: dict[int, int] = {}
    estimated_finish: dict[int, float] = {}
    external_core_users: dict[int, set[int]] = defaultdict(set)
    tensor_core_users: dict[int, set[int]] = defaultdict(set)
    cache: OrderedDict[int, tuple[int, float]] = OrderedDict()
    cache_bytes = 0
    cache_capacity = config["problem_3"]["cache_capacity_bytes"]
    cache_bandwidth = config["problem_3"]["cache_bandwidth_bytes_per_cycle"]
    ddr_bandwidth = config["bandwidth"]

    while ready:
        _, _, group_id = heapq.heappop(ready)
        best: tuple[tuple[float, float, int], int, float, dict[str, float], set[int]] | None = None
        inputs = _boundary_input_sizes(group_id, internal_inputs, features)

        for core_id in range(ncores):
            group_work = dict(features[group_id].work_by_pipe)
            source_ready = 0.0
            ddr_read_cycles = 0.0
            cache_read_cycles = 0.0
            cache_misses: set[int] = set()

            if problem == 1:
                for parent in predecessors[group_id]:
                    wait = 0 if assigned_core[parent] == core_id else config[
                        "scene_a"]["task_cross_core_wait_cycles"]
                    source_ready = max(
                        source_ready,
                        estimated_finish[parent] + wait,
                    )
                ddr_read_cycles += sum(inputs[key][1] for key in inputs) / ddr_bandwidth
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
                score = (finish, float(sum(inputs[key][1] for key in inputs)), core_id)
            else:
                for parent in predecessors[group_id]:
                    edge_data = boundary_sizes.get((parent, group_id), {})
                    if assigned_core[parent] == core_id:
                        source_ready = max(source_ready, estimated_finish[parent])
                    else:
                        source_ready = max(
                            source_ready,
                            estimated_finish[parent] + sync
                            + sum(edge_data.values()) / ddr_bandwidth,
                        )

                cache_check = max(
                    source_ready,
                    core_pipe_end[core_id].get("PIPE_MTE2", 0.0),
                )
                for tensor_id, (producer_group, size) in inputs.items():
                    if producer_group >= 0:
                        if assigned_core[producer_group] == core_id:
                            continue
                        if isinstance(tensor_id, int) and core_id in tensor_core_users[tensor_id]:
                            continue
                    elif core_id in external_core_users[tensor_id[1]]:
                        continue

                    if isinstance(tensor_id, int):
                        logical_id = tensor_id
                    elif tensor_id[0] == "external":
                        logical_id = tensor_id[1]
                    else:
                        logical_id = None
                    cache_entry = cache.get(logical_id) if problem == 3 and logical_id is not None else None
                    if cache_entry is not None and cache_entry[1] <= cache_check:
                        cache_read_cycles += size / cache_bandwidth
                    else:
                        ddr_read_cycles += size / ddr_bandwidth
                        if problem == 3 and logical_id is not None:
                            cache_misses.add(logical_id)
                group_work["PIPE_MTE2"] = group_work.get("PIPE_MTE2", 0) + ddr_read_cycles + cache_read_cycles
                start = source_ready
                pipe_ends = {
                    pipe: max(core_pipe_end[core_id][pipe], start) + duration
                    for pipe, duration in group_work.items()
                }
                finish = max(
                    start + features[group_id].internal_critical_path,
                    max(pipe_ends.values(), default=start),
                )
                extra_bytes = sum(inputs[key][1] for key in inputs)
                score = (finish, float(extra_bytes), core_id)

            candidate = (score, core_id, start, pipe_ends, cache_misses)
            if best is None or score < best[0]:
                best = candidate

        if best is None:
            raise RuntimeError("no core candidate was available")
        _, core_id, start, pipe_ends, cache_misses = best
        assigned_core[group_id] = core_id
        estimated_finish[group_id] = pipe_ends.get("__task__", max(pipe_ends.values(), default=start))
        core_schedules[core_id].append(group_id)
        if problem == 1:
            core_task_end[core_id] = estimated_finish[group_id]
        else:
            for pipe, end in pipe_ends.items():
                core_pipe_end[core_id][pipe] = end

        for tensor_id in external_inputs[group_id]:
            external_core_users[tensor_id].add(core_id)
        for tensor_id in output_tensors[group_id]:
            tensor_core_users[tensor_id].add(core_id)
        for tensor_id, (producer_group, _) in inputs.items():
            if isinstance(tensor_id, int) and producer_group >= 0 and assigned_core[producer_group] != core_id:
                tensor_core_users[tensor_id].add(core_id)

        if problem == 3:
            cache_ready = pipe_ends.get("PIPE_MTE2", estimated_finish[group_id])
            for tensor_id in cache_misses:
                tensor_size = analysis.graph.tensors.get(tensor_id, {}).get("size", 0)
                cache_bytes = _update_cache(
                    cache, cache_bytes, tensor_id, tensor_size,
                    cache_ready, cache_capacity,
                )

        for child in successors[group_id]:
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(
                    ready,
                    (-rank[child], -features[child].estimated_compute, child),
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
    validate_plan(analysis.graph, plan, ncores)
    return plan