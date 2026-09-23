from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

from .graph_io import Graph


COPY_TYPES = {"COPY_IN", "COPY_OUT"}


@dataclass(slots=True)
class GraphAnalysis:
    graph: Graph
    eligible_ops: set[int]
    predecessors: dict[int, set[int]]
    successors: dict[int, set[int]]
    topological_order: list[int]
    topological_index: dict[int, int]
    topological_level: dict[int, int]
    forward_path: dict[int, int]
    backward_path: dict[int, int]
    external_inputs: dict[int, set[int]]
    external_sources: dict[int, set[int]]
    tensor_lifetimes: dict[int, tuple[int, int]]


def _topological_order(nodes: set[int], predecessors: dict[int, set[int]]) -> list[int]:
    successors = {node: set() for node in nodes}
    indegree = {node: len(predecessors[node]) for node in nodes}
    for node, parents in predecessors.items():
        for parent in parents:
            successors[parent].add(node)
    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for child in successors[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(nodes):
        raise ValueError("eligible operation graph contains a cycle")
    return order


def analyze_graph(graph: Graph) -> GraphAnalysis:
    all_op_ids = set(graph.ops)
    full_successors = {op_id: set() for op_id in all_op_ids}
    for source, target, _ in graph.direct_op_edges:
        full_successors[source].add(target)
    for tensor_id, producers in graph.tensor_producers.items():
        consumers = graph.tensor_consumers.get(tensor_id, set())
        for producer in producers:
            full_successors[producer].update(
                consumer for consumer in consumers if consumer != producer
            )
    full_predecessors = {op_id: set() for op_id in all_op_ids}
    for source, children in full_successors.items():
        for child in children:
            full_predecessors[child].add(source)

    eligible = {
        op_id for op_id, op in graph.ops.items() if op["op"] not in COPY_TYPES
    }
    predecessors = {op_id: set() for op_id in eligible}
    successors = {op_id: set() for op_id in eligible}

    full_order = _topological_order(all_op_ids, full_predecessors)
    nearest_eligible: dict[int, set[int]] = {}
    for op_id in full_order:
        nearest = set()
        for parent in full_predecessors[op_id]:
            if parent in eligible:
                nearest.add(parent)
            else:
                nearest.update(nearest_eligible[parent])
        if op_id in eligible:
            predecessors[op_id].update(nearest)
            nearest_eligible[op_id] = {op_id}
        else:
            nearest_eligible[op_id] = nearest

    for op_id, parents in predecessors.items():
        for parent in parents:
            successors[parent].add(op_id)

    order = _topological_order(eligible, predecessors)
    index = {op_id: i for i, op_id in enumerate(order)}
    level: dict[int, int] = {}
    forward: dict[int, int] = {}
    for op_id in order:
        cycles = max(1, graph.ops[op_id]["cycles"])
        level[op_id] = 1 + max((level[parent] for parent in predecessors[op_id]), default=-1)
        forward[op_id] = cycles + max(
            (forward[parent] for parent in predecessors[op_id]), default=0
        )
    backward: dict[int, int] = {}
    for op_id in reversed(order):
        cycles = max(1, graph.ops[op_id]["cycles"])
        backward[op_id] = cycles + max(
            (backward[child] for child in successors[op_id]), default=0
        )

    external_inputs: dict[int, set[int]] = defaultdict(set)
    external_sources: dict[int, set[int]] = defaultdict(set)
    for consumer in eligible:
        for tensor_id in graph.op_inputs[consumer]:
            producer_ops = graph.tensor_producers.get(tensor_id, set())
            copyins = [
                op_id for op_id in producer_ops
                if graph.ops[op_id]["op"] == "COPY_IN"
            ]
            if copyins:
                external_inputs[consumer].add(tensor_id)
                for copyin in copyins:
                    external_sources[consumer].update(
                        source_id for source_id in graph.op_inputs[copyin]
                        if graph.tensors[source_id]["pos"] == "DDR"
                    )
            elif graph.tensors[tensor_id]["pos"] == "DDR":
                external_inputs[consumer].add(tensor_id)
                external_sources[consumer].add(tensor_id)

    tensor_lifetimes = {}
    for tensor_id, tensor in graph.tensors.items():
        if tensor["pos"] == "DDR":
            continue
        producers = graph.tensor_producers.get(tensor_id, set()) & eligible
        consumers = graph.tensor_consumers.get(tensor_id, set()) & eligible
        positions = [index[op_id] for op_id in producers | consumers]
        if positions:
            tensor_lifetimes[tensor_id] = (min(positions), max(positions))

    return GraphAnalysis(
        graph=graph,
        eligible_ops=eligible,
        predecessors=predecessors,
        successors=successors,
        topological_order=order,
        topological_index=index,
        topological_level=level,
        forward_path=forward,
        backward_path=backward,
        external_inputs={op: set(ids) for op, ids in external_inputs.items()},
        external_sources={op: set(ids) for op, ids in external_sources.items()},
        tensor_lifetimes=tensor_lifetimes,
    )