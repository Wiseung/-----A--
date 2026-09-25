from __future__ import annotations

import heapq
from dataclasses import dataclass

from .graph_analysis import GraphAnalysis
from .features import operation_features


@dataclass(slots=True)
class Partition:
    groups: list[list[int]]
    op_to_subgraph: dict[int, int]


def _topological_order_for_partition(
    analysis: GraphAnalysis, strategy: str
) -> list[int]:
    if strategy == "id":
        return list(analysis.topological_order)
    if strategy not in {"critical_path", "branch_locality", "release_bytes"}:
        raise ValueError(f"unknown topology strategy: {strategy}")

    release_bytes = {op_id: 0 for op_id in analysis.eligible_ops}
    for tensor_id, tensor in analysis.graph.tensors.items():
        consumer_ops = (
            analysis.graph.tensor_consumers.get(tensor_id, set())
            & analysis.eligible_ops
        )
        if not consumer_ops:
            continue
        last_consumer = max(
            analysis.topological_index[op_id] for op_id in consumer_ops
        )
        for op_id in consumer_ops:
            if analysis.topological_index[op_id] == last_consumer:
                release_bytes[op_id] += tensor["size"]

    indegree = {
        op_id: len(analysis.predecessors[op_id])
        for op_id in analysis.eligible_ops
    }
    ready: list[tuple[int, int, int, int]] = []

    def priority(op_id: int) -> tuple[int, int, int, int]:
        if strategy == "critical_path":
            return (
                -analysis.backward_path[op_id],
                -analysis.forward_path[op_id],
                0,
                op_id,
            )
        if strategy == "branch_locality":
            continuation = sum(
                len(analysis.predecessors[child]) == 1
                for child in analysis.successors[op_id]
            )
            return (
                -continuation,
                -analysis.backward_path[op_id],
                -analysis.forward_path[op_id],
                op_id,
            )
        return (
            -release_bytes[op_id],
            -analysis.backward_path[op_id],
            -analysis.forward_path[op_id],
            op_id,
        )

    for op_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, priority(op_id))
    order = []
    while ready:
        _, _, _, op_id = heapq.heappop(ready)
        order.append(op_id)
        for child in analysis.successors[op_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, priority(child))
    if len(order) != len(analysis.eligible_ops):
        raise ValueError("eligible operation graph contains a cycle")
    return order


def _cut_bytes_by_position(
    analysis: GraphAnalysis, order: list[int]
) -> list[int]:
    graph = analysis.graph
    count = len(order)
    position = {op_id: index for index, op_id in enumerate(order)}
    delta = [0] * (count + 1)
    for tensor_id, tensor in graph.tensors.items():
        if tensor["pos"] == "DDR":
            continue
        producer_positions = [
            position[op_id]
            for op_id in graph.tensor_producers.get(tensor_id, set())
            if op_id in analysis.eligible_ops
        ]
        consumer_positions = [
            position[op_id]
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
            input_users.setdefault(tensor_id, []).append(position[op_id])
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


def partition_contiguous(
    analysis: GraphAnalysis,
    group_count: int,
    problem: int,
    topology_strategy: str = "id",
) -> Partition:
    order = _topological_order_for_partition(analysis, topology_strategy)
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
        cuts = _cut_bytes_by_position(analysis, order)
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


_CAGG_WEIGHTS = {
    1: (0.35, 0.30, 0.15, 0.05, 0.55),
    2: (0.25, 0.45, 0.15, 0.10, 0.40),
    3: (0.20, 0.40, 0.15, 0.15, 0.35),
}


def _spread_candidates(candidates: list[int], count: int) -> list[int]:
    if not candidates or count <= 0:
        return []
    if len(candidates) <= count:
        return list(candidates)
    if count == 1:
        return [candidates[len(candidates) // 2]]
    return [
        candidates[index * (len(candidates) - 1) // (count - 1)]
        for index in range(count)
    ]


def _cagg_seed_order(analysis: GraphAnalysis, count: int) -> list[int]:
    graph = analysis.graph
    order = list(analysis.topological_order)
    features = operation_features(analysis)
    critical_length = max(analysis.forward_path.values(), default=0)
    critical_ops = [
        op_id for op_id in order
        if (
            analysis.forward_path[op_id]
            + analysis.backward_path[op_id]
            - max(1, graph.ops[op_id]["cycles"])
            == critical_length
        )
    ]
    critical_ops.sort(key=analysis.topological_index.__getitem__)
    work_ops = sorted(
        order,
        key=lambda op_id: (
            -features[op_id]["cycles"],
            -analysis.backward_path[op_id],
            analysis.topological_index[op_id],
            op_id,
        ),
    )
    shared_tensors = [
        tensor_id for tensor_id, tensor in graph.tensors.items()
        if tensor["pos"] == "DDR"
        and len(
            graph.tensor_consumers.get(tensor_id, set()) & analysis.eligible_ops
        ) > 1
    ]
    shared_tensors.sort(key=lambda tensor_id: (-graph.tensors[tensor_id]["size"], tensor_id))
    shared_ops = []
    for tensor_id in shared_tensors:
        consumers = graph.tensor_consumers.get(tensor_id, set()) & analysis.eligible_ops
        representative = min(
            consumers,
            key=lambda op_id: (
                -features[op_id]["cycles"],
                -analysis.backward_path[op_id],
                analysis.topological_index[op_id],
                op_id,
            ),
        )
        shared_ops.append(representative)

    sources = [
        _spread_candidates(critical_ops, count),
        work_ops,
        shared_ops,
        _spread_candidates(order, count),
    ]
    seeds: list[int] = []
    positions = [0] * len(sources)
    while len(seeds) < count:
        progressed = False
        for source_index, source in enumerate(sources):
            while positions[source_index] < len(source):
                candidate = source[positions[source_index]]
                positions[source_index] += 1
                if candidate not in seeds:
                    seeds.append(candidate)
                    progressed = True
                    break
            if len(seeds) == count:
                break
        if progressed:
            continue
        for candidate in order:
            if candidate not in seeds:
                seeds.append(candidate)
                break
    seeds.sort(key=analysis.topological_index.__getitem__)
    return seeds


def _cagg_tensor_cost(
    analysis: GraphAnalysis,
    assignment: dict[int, int],
    tensor_id: int,
) -> int:
    graph = analysis.graph
    tensor = graph.tensors[tensor_id]
    producer_groups = {
        assignment[op_id]
        for op_id in graph.tensor_producers.get(tensor_id, set())
        if op_id in analysis.eligible_ops and op_id in assignment
    }
    consumer_groups = {
        assignment[op_id]
        for op_id in graph.tensor_consumers.get(tensor_id, set())
        if op_id in analysis.eligible_ops and op_id in assignment
    }
    if tensor["pos"] == "DDR":
        return tensor["size"] * (len(producer_groups) + len(consumer_groups))
    return tensor["size"] * sum(
        source_group != target_group
        for source_group in producer_groups
        for target_group in consumer_groups
    )


def _cagg_hyperedge_delta(
    analysis: GraphAnalysis,
    assignment: dict[int, int],
    op_id: int,
    group_id: int,
) -> int:
    graph = analysis.graph
    incident_tensors = set(graph.op_inputs[op_id]) | set(graph.op_outputs[op_id])
    before = sum(
        _cagg_tensor_cost(analysis, assignment, tensor_id)
        for tensor_id in incident_tensors
    )
    assignment[op_id] = group_id
    after = sum(
        _cagg_tensor_cost(analysis, assignment, tensor_id)
        for tensor_id in incident_tensors
    )
    del assignment[op_id]
    return after - before


def _cagg_shared_input_cost(
    analysis: GraphAnalysis,
    assignment: dict[int, int],
    tensor_id: int,
) -> int:
    graph = analysis.graph
    tensor = graph.tensors[tensor_id]
    consumers = graph.tensor_consumers.get(tensor_id, set()) & analysis.eligible_ops
    if tensor["pos"] != "DDR" or len(consumers) <= 1:
        return 0
    consumer_groups = {
        assignment[op_id] for op_id in consumers if op_id in assignment
    }
    return (
        tensor["size"]
        * max(0, len(consumer_groups) - 1)
        * max(1, len(consumers) - 1)
    )


def _cagg_coverage_delta(
    analysis: GraphAnalysis,
    assignment: dict[int, int],
    op_id: int,
    group_id: int,
) -> int:
    graph = analysis.graph
    incident_tensors = (
        set(graph.op_inputs[op_id])
        | set(graph.op_outputs[op_id])
        | set(analysis.external_sources.get(op_id, set()))
    )
    before = sum(
        _cagg_shared_input_cost(analysis, assignment, tensor_id)
        for tensor_id in incident_tensors
    )
    assignment[op_id] = group_id
    after = sum(
        _cagg_shared_input_cost(analysis, assignment, tensor_id)
        for tensor_id in incident_tensors
    )
    del assignment[op_id]
    return after - before


def _cagg_load_cost(
    m_cycles: list[int],
    v_cycles: list[int],
    total_m_cycles: int,
    total_v_cycles: int,
) -> float:
    return sum(
        (cycles / max(total_cycles, 1)) ** 2
        for loads, total_cycles in (
            (m_cycles, total_m_cycles),
            (v_cycles, total_v_cycles),
        )
        for cycles in loads
    )


def _cagg_critical_delta(
    analysis: GraphAnalysis,
    assignment: dict[int, int],
    critical_ops: set[int],
    op_id: int,
    group_id: int,
) -> int:
    if op_id not in critical_ops:
        return 0
    neighbours = analysis.predecessors[op_id] | analysis.successors[op_id]
    return sum(
        neighbour in critical_ops
        and neighbour in assignment
        and assignment[neighbour] != group_id
        for neighbour in neighbours
    )


def _cagg_assignment_edges(
    analysis: GraphAnalysis,
    assignment: dict[int, int],
    op_id: int,
    group_id: int,
) -> list[tuple[int, int]]:
    edges = []
    for parent in analysis.predecessors[op_id]:
        if parent in assignment and assignment[parent] != group_id:
            edges.append((assignment[parent], group_id))
    for child in analysis.successors[op_id]:
        if child in assignment and assignment[child] != group_id:
            edges.append((group_id, assignment[child]))
    return edges


def _cagg_would_cycle(
    group_edges: list[set[int]],
    additional_edges: list[tuple[int, int]],
) -> bool:
    candidate_edges = [set(children) for children in group_edges]
    for source, target in additional_edges:
        candidate_edges[source].add(target)
    indegree = [0] * len(candidate_edges)
    for children in candidate_edges:
        for child in children:
            indegree[child] += 1
    ready = [group_id for group_id, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while ready:
        group_id = ready.pop()
        visited += 1
        for child in candidate_edges[group_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return visited != len(candidate_edges)


def _cagg_group_bounds(
    analysis: GraphAnalysis,
    seeds: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    seed_groups = {op_id: group_id for group_id, op_id in enumerate(seeds)}
    lower: dict[int, int] = {}
    for op_id in analysis.topological_order:
        if op_id in seed_groups:
            lower[op_id] = seed_groups[op_id]
        else:
            lower[op_id] = max(
                (lower[parent] for parent in analysis.predecessors[op_id]),
                default=0,
            )
    upper: dict[int, int] = {}
    for op_id in reversed(analysis.topological_order):
        if op_id in seed_groups:
            upper[op_id] = seed_groups[op_id]
        else:
            upper[op_id] = min(
                (upper[child] for child in analysis.successors[op_id]),
                default=len(seeds) - 1,
            )
    return lower, upper


def partition_cagg_lite(
    analysis: GraphAnalysis,
    group_count: int,
    problem: int,
    coverage_aware: bool = False,
) -> Partition:
    if problem not in _CAGG_WEIGHTS:
        raise ValueError(f"unknown problem: {problem}")
    order = list(analysis.topological_order)
    if not order:
        return Partition([], {})
    group_count = max(1, min(group_count, len(order)))
    if group_count == 1:
        groups = [order]
        return Partition(
            groups,
            {op_id: 0 for op_id in order},
        )

    graph = analysis.graph
    features = operation_features(analysis)
    seeds = _cagg_seed_order(analysis, group_count)
    assignment = {op_id: group_id for group_id, op_id in enumerate(seeds)}
    groups = [[op_id] for op_id in seeds]
    group_edges = [set() for _ in range(group_count)]
    lower_bounds, upper_bounds = _cagg_group_bounds(analysis, seeds)
    m_cycles = [
        features[op_id]["cycles"] if features[op_id]["pipe"] == "PIPE_M" else 0
        for op_id in seeds
    ]
    v_cycles = [
        features[op_id]["cycles"] if features[op_id]["pipe"] == "PIPE_V" else 0
        for op_id in seeds
    ]
    total_m_cycles = sum(
        item["cycles"] for item in features.values() if item["pipe"] == "PIPE_M"
    )
    total_v_cycles = sum(
        item["cycles"] for item in features.values() if item["pipe"] == "PIPE_V"
    )
    total_tensor_bytes = max(
        sum(tensor["size"] for tensor in graph.tensors.values()), 1
    )
    total_shared_input_bytes = max(
        sum(
            tensor["size"]
            for tensor_id, tensor in graph.tensors.items()
            if tensor["pos"] == "DDR"
            and len(
                graph.tensor_consumers.get(tensor_id, set()) & analysis.eligible_ops
            ) > 1
        ),
        1,
    )
    total_memory_bytes = max(
        sum(
            tensor["size"] for tensor in graph.tensors.values()
            if tensor["pos"] in {"L1", "UB"}
        ),
        1,
    )
    critical_ops = {
        op_id for op_id in order
        if (
            analysis.forward_path[op_id]
            + analysis.backward_path[op_id]
            - max(1, graph.ops[op_id]["cycles"])
            == max(analysis.forward_path.values(), default=0)
        )
    }
    alpha, beta, gamma, delta, eta = _CAGG_WEIGHTS[problem]
    remaining = [op_id for op_id in order if op_id not in assignment]

    while remaining:
        before_load = _cagg_load_cost(
            m_cycles, v_cycles, total_m_cycles, total_v_cycles
        )
        best_key: tuple[float, int, int, int] | None = None
        best_choice: tuple[int, int] | None = None
        for op_id in remaining[:1]:
            op_feature = features[op_id]
            memory_bytes = sum(
                graph.tensors[tensor_id]["size"]
                for tensor_id in set(graph.op_inputs[op_id]) | set(graph.op_outputs[op_id])
                if graph.tensors[tensor_id]["pos"] in {"L1", "UB"}
            )
            lower_group = max(
                [
                    lower_bounds[op_id],
                    *(
                        assignment[parent]
                        for parent in analysis.predecessors[op_id]
                        if parent in assignment
                    ),
                ]
            )
            upper_group = min(
                [
                    upper_bounds[op_id],
                    *(
                        assignment[child]
                        for child in analysis.successors[op_id]
                        if child in assignment
                    ),
                ]
            )
            for group_id in range(lower_group, upper_group + 1):
                assignment_edges = _cagg_assignment_edges(
                    analysis, assignment, op_id, group_id
                )
                if _cagg_would_cycle(group_edges, assignment_edges):
                    continue
                stranger_pred = sum(
                    assignment.get(parent) != group_id
                    for parent in analysis.predecessors[op_id]
                )
                candidate_m = list(m_cycles)
                candidate_v = list(v_cycles)
                if op_feature["pipe"] == "PIPE_M":
                    candidate_m[group_id] += op_feature["cycles"]
                elif op_feature["pipe"] == "PIPE_V":
                    candidate_v[group_id] += op_feature["cycles"]
                load_delta = _cagg_load_cost(
                    candidate_m, candidate_v, total_m_cycles, total_v_cycles
                ) - before_load
                hyperedge_delta = _cagg_hyperedge_delta(
                    analysis, assignment, op_id, group_id
                ) / total_tensor_bytes
                coverage_delta = (
                    _cagg_coverage_delta(analysis, assignment, op_id, group_id)
                    / total_shared_input_bytes
                    if coverage_aware else 0.0
                )
                critical_delta = _cagg_critical_delta(
                    analysis, assignment, critical_ops, op_id, group_id
                ) / max(len(critical_ops), 1)
                cost = (
                    alpha * load_delta
                    + beta * hyperedge_delta
                    + gamma * stranger_pred / max(len(analysis.predecessors[op_id]), 1)
                    + delta * memory_bytes / total_memory_bytes
                    + eta * critical_delta
                    + 0.35 * coverage_delta
                )
                key = (
                    cost,
                    analysis.topological_index[op_id],
                    group_id,
                    op_id,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_choice = (op_id, group_id)
        if best_choice is None:
            raise RuntimeError("CAGG-lite failed to assign an operation")
        op_id, group_id = best_choice
        assignment[op_id] = group_id
        groups[group_id].append(op_id)
        for source, target in _cagg_assignment_edges(
            analysis, assignment, op_id, group_id
        ):
            group_edges[source].add(target)
        if features[op_id]["pipe"] == "PIPE_M":
            m_cycles[group_id] += features[op_id]["cycles"]
        elif features[op_id]["pipe"] == "PIPE_V":
            v_cycles[group_id] += features[op_id]["cycles"]
        remaining.remove(op_id)

    return Partition(groups, assignment)


def partition_cagg_lite_coverage(
    analysis: GraphAnalysis,
    group_count: int,
    problem: int,
) -> Partition:
    return partition_cagg_lite(
        analysis, group_count, problem, coverage_aware=True
    )


def partition_with_strategy(
    analysis: GraphAnalysis,
    group_count: int,
    problem: int,
    partition_strategy: str = "contiguous",
    topology_strategy: str = "id",
) -> Partition:
    if partition_strategy == "contiguous":
        return partition_contiguous(
            analysis,
            group_count,
            problem,
            topology_strategy=topology_strategy,
        )
    if partition_strategy == "cagg_lite":
        return partition_cagg_lite(analysis, group_count, problem)
    if partition_strategy == "cagg_lite_coverage":
        return partition_cagg_lite_coverage(analysis, group_count, problem)
    if partition_strategy in {"chain_contiguous", "chain_partition"}:
        from .chain_partition import partition_chain_contiguous

        return partition_chain_contiguous(analysis, group_count, problem)
    raise ValueError(f"unknown partition strategy: {partition_strategy}")


def candidate_group_counts(num_ops: int, ncores: int, problem: int) -> list[int]:
    if not num_ops:
        return [0]
    if ncores == 1:
        return [1]
    multipliers = {1: (1, 2, 4), 2: (2, 4), 3: (2, 4, 6)}[problem]
    return list(dict.fromkeys(min(num_ops, ncores * value) for value in multipliers))