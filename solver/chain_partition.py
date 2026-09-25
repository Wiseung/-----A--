from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

from .graph_analysis import GraphAnalysis
from .partition import Partition


@dataclass(slots=True)
class ChainFeatures:
    chain_id: int
    members: list[int]
    m_cycles: int
    v_cycles: int
    topological_min_index: int


def build_chains(
    analysis: GraphAnalysis,
) -> tuple[list[ChainFeatures], dict[int, int], dict[tuple[int, int], int]]:
    """Build maximal single-successor/single-predecessor operation chains."""
    graph = analysis.graph
    assigned: set[int] = set()
    chains: list[ChainFeatures] = []
    chain_of_op: dict[int, int] = {}

    for start in analysis.topological_order:
        if start in assigned:
            continue
        members = [start]
        assigned.add(start)
        current = start
        while len(analysis.successors[current]) == 1:
            (child,) = tuple(analysis.successors[current])
            if len(analysis.predecessors[child]) != 1 or child in assigned:
                break
            members.append(child)
            assigned.add(child)
            current = child

        m_cycles = 0
        v_cycles = 0
        for op_id in members:
            cycles = max(1, graph.ops[op_id]["cycles"])
            if graph.ops[op_id]["pipe"] == "PIPE_M":
                m_cycles += cycles
            elif graph.ops[op_id]["pipe"] == "PIPE_V":
                v_cycles += cycles
        chain_id = len(chains)
        chains.append(ChainFeatures(
            chain_id=chain_id,
            members=members,
            m_cycles=m_cycles,
            v_cycles=v_cycles,
            topological_min_index=analysis.topological_index[members[0]],
        ))
        for op_id in members:
            chain_of_op[op_id] = chain_id

    edge_bytes_by_op: dict[tuple[int, int], int] = defaultdict(int)
    for tensor_id, tensor in graph.tensors.items():
        producers = graph.tensor_producers.get(tensor_id, set()) & analysis.eligible_ops
        consumers = graph.tensor_consumers.get(tensor_id, set()) & analysis.eligible_ops
        for producer in producers:
            for consumer in consumers:
                if producer != consumer:
                    edge_bytes_by_op[(producer, consumer)] += max(
                        0, int(tensor["size"])
                    )
    for source, target, size in graph.direct_op_edges:
        if source in analysis.eligible_ops and target in analysis.eligible_ops:
            edge_bytes_by_op[(source, target)] += max(0, int(size))

    edge_bytes: dict[tuple[int, int], int] = defaultdict(int)
    for (source, target), size in edge_bytes_by_op.items():
        source_chain = chain_of_op[source]
        target_chain = chain_of_op[target]
        if source_chain != target_chain:
            edge_bytes[(source_chain, target_chain)] += size
    return chains, chain_of_op, dict(edge_bytes)


def _chain_dag(
    chains: list[ChainFeatures],
    chain_of_op: dict[int, int],
    analysis: GraphAnalysis,
) -> tuple[list[set[int]], list[set[int]]]:
    successors = [set() for _ in chains]
    predecessors = [set() for _ in chains]
    for source in analysis.eligible_ops:
        source_chain = chain_of_op[source]
        for target in analysis.successors[source]:
            target_chain = chain_of_op[target]
            if source_chain != target_chain:
                successors[source_chain].add(target_chain)
                predecessors[target_chain].add(source_chain)
    return successors, predecessors


def _chain_topological_order(
    chains: list[ChainFeatures],
    successors: list[set[int]],
    predecessors: list[set[int]],
) -> list[int]:
    indegree = [len(items) for items in predecessors]
    ready = [
        (chains[chain_id].topological_min_index, chain_id)
        for chain_id, degree in enumerate(indegree)
        if degree == 0
    ]
    heapq.heapify(ready)
    order = []
    while ready:
        _, chain_id = heapq.heappop(ready)
        order.append(chain_id)
        for child in sorted(successors[chain_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(
                    ready,
                    (chains[child].topological_min_index, child),
                )
    if len(order) != len(chains):
        raise ValueError("chain dependency graph contains a cycle")
    return order


def _crossing_bytes(
    order: list[int],
    edge_bytes: dict[tuple[int, int], int],
) -> list[int]:
    positions = {chain_id: index for index, chain_id in enumerate(order)}
    delta = [0] * (len(order) + 1)
    for (source, target), size in edge_bytes.items():
        source_position = positions[source]
        target_position = positions[target]
        if source_position >= target_position:
            continue
        delta[source_position + 1] += size
        delta[target_position + 1] -= size
    crossing = [0] * (len(order) + 1)
    active = 0
    for boundary in range(1, len(order)):
        active += delta[boundary]
        crossing[boundary] = active
    return crossing


def _chain_intervals(
    chains: list[ChainFeatures],
    order: list[int],
    edge_bytes: dict[tuple[int, int], int],
    group_count: int,
    problem: int,
) -> list[list[int]]:
    if not order:
        return []
    group_count = max(1, min(group_count, len(order)))
    if group_count == 1:
        return [list(order)]

    m_load = [chains[chain_id].m_cycles for chain_id in order]
    v_load = [chains[chain_id].v_cycles for chain_id in order]
    prefix_m = [0]
    prefix_v = [0]
    for m_cycles, v_cycles in zip(m_load, v_load):
        prefix_m.append(prefix_m[-1] + m_cycles)
        prefix_v.append(prefix_v[-1] + v_cycles)

    def interval_load(start: int, end: int) -> int:
        return max(
            prefix_m[end + 1] - prefix_m[start],
            prefix_v[end + 1] - prefix_v[start],
        )

    crossing = _crossing_bytes(order, edge_bytes)
    max_crossing = max(crossing, default=0)
    cut_weight = {1: 0.30, 2: 0.12, 3: 0.05}[problem]
    cuts: list[int] = []
    previous = 0
    total_load = interval_load(0, len(order) - 1)
    for group_id in range(1, group_count):
        low = previous + 1
        high = len(order) - (group_count - group_id)
        target = total_load * group_id / group_count
        cut = min(
            range(low, high + 1),
            key=lambda boundary: (
                abs(interval_load(0, boundary - 1) - target)
                / max(total_load, 1)
                + cut_weight * crossing[boundary] / max(max_crossing, 1),
                boundary,
            ),
        )
        cuts.append(cut - 1)
        previous = cut

    intervals = []
    start = 0
    for end in [*cuts, len(order) - 1]:
        intervals.append(order[start:end + 1])
        start = end + 1
    return intervals


def partition_chain_contiguous(
    analysis: GraphAnalysis,
    group_count: int,
    problem: int,
) -> Partition:
    if problem not in {1, 2, 3}:
        raise ValueError(f"unknown problem: {problem}")
    chains, chain_of_op, edge_bytes = build_chains(analysis)
    successors, predecessors = _chain_dag(chains, chain_of_op, analysis)
    order = _chain_topological_order(chains, successors, predecessors)
    intervals = _chain_intervals(
        chains, order, edge_bytes, group_count, problem
    )
    groups = [
        [op_id for chain_id in interval for op_id in chains[chain_id].members]
        for interval in intervals
    ]
    op_to_subgraph = {
        op_id: group_id
        for group_id, members in enumerate(groups)
        for op_id in members
    }
    return Partition(groups, op_to_subgraph)