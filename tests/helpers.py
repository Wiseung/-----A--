from __future__ import annotations

from typing import Any


def make_chain_graph(op_count: int, size: int = 16) -> dict[str, Any]:
    tensors = [
        {"id": 1, "pos": "DDR", "size": size},
        {"id": 101, "pos": "UB", "size": size},
        {"id": 2, "pos": "DDR", "size": 0},
    ]
    ops = [
        {"id": 10, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 1},
        {"id": 90, "op": "COPY_OUT", "pipe": "PIPE_MTE3", "cycles": 1},
    ]
    edges = [
        {"source": 1, "target": 10},
        {"source": 10, "target": 101},
    ]
    previous_tensor = 101
    for index in range(op_count):
        op_id = 20 + index
        tensor_id = 201 + index
        tensors.append({"id": tensor_id, "pos": "UB", "size": size if index + 1 < op_count else 0})
        ops.append({
            "id": op_id,
            "op": "ADD" if index % 2 == 0 else "MATMUL",
            "pipe": "PIPE_V" if index % 2 == 0 else "PIPE_M",
            "cycles": 4,
        })
        edges.extend([
            {"source": previous_tensor, "target": op_id},
            {"source": op_id, "target": tensor_id},
        ])
        previous_tensor = tensor_id
    edges.extend([
        {"source": previous_tensor, "target": 90},
        {"source": 90, "target": 2},
    ])
    return {"tensors": tensors, "ops": ops, "edges": edges}


def make_shared_input_graph(consumer_count: int, size: int = 60) -> dict[str, Any]:
    tensors = [
        {"id": 1, "pos": "DDR", "size": size},
        {"id": 101, "pos": "UB", "size": size},
    ]
    ops = [{"id": 10, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 1}]
    edges = [{"source": 1, "target": 10}, {"source": 10, "target": 101}]
    for index in range(consumer_count):
        op_id = 20 + index
        local_output = 201 + index
        ddr_output = 301 + index
        copy_out = 90 + index
        tensors.extend([
            {"id": local_output, "pos": "UB", "size": 0},
            {"id": ddr_output, "pos": "DDR", "size": 0},
        ])
        ops.extend([
            {"id": op_id, "op": "ADD", "pipe": "PIPE_V", "cycles": 1},
            {"id": copy_out, "op": "COPY_OUT", "pipe": "PIPE_MTE3", "cycles": 1},
        ])
        edges.extend([
            {"source": 101, "target": op_id},
            {"source": op_id, "target": local_output},
            {"source": local_output, "target": copy_out},
            {"source": copy_out, "target": ddr_output},
        ])
    return {"tensors": tensors, "ops": ops, "edges": edges}


def plan_for(assignments: list[int], ncores: int) -> dict[str, Any]:
    mapping = {str(20 + index): index for index in range(len(assignments))}
    schedules = [[] for _ in range(ncores)]
    for subgraph_id, core_id in enumerate(assignments):
        schedules[core_id].append(subgraph_id)
    return {"node_to_subgraph": mapping, "core_schedules": schedules}


def make_fifo_reuse_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    tensors = [
        {"id": 1, "pos": "DDR", "size": 8},
        {"id": 2, "pos": "UB", "size": 8},
        {"id": 3, "pos": "DDR", "size": 8},
        {"id": 4, "pos": "UB", "size": 8},
        {"id": 5, "pos": "DDR", "size": 8},
        {"id": 6, "pos": "UB", "size": 8},
    ]
    ops = [
        {"id": 10, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 1},
        {"id": 11, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 1},
        {"id": 12, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 1},
    ]
    edges = [
        {"source": 1, "target": 10}, {"source": 10, "target": 2},
        {"source": 3, "target": 11}, {"source": 11, "target": 4},
        {"source": 5, "target": 12}, {"source": 12, "target": 6},
    ]

    dataflow = [
        (20, {2}, 201, 0),
        (21, {201}, 202, 0),
        (22, {2, 202}, 203, 0),
        (23, {203}, 204, 0),
        (24, {4, 204}, 205, 0),
        (25, {205}, 206, 0),
        (26, {6, 206}, 207, 0),
        (27, {207}, 208, 0),
        (28, {2, 208}, 209, 0),
    ]
    for op_id, input_tensors, output_tensor, size in dataflow:
        tensors.append({"id": output_tensor, "pos": "UB", "size": size})
        ops.append({"id": op_id, "op": "ADD", "pipe": "PIPE_V", "cycles": 1})
        for input_tensor in input_tensors:
            edges.append({"source": input_tensor, "target": op_id})
        edges.append({"source": op_id, "target": output_tensor})

    tensors.append({"id": 300, "pos": "DDR", "size": 0})
    ops.append({"id": 99, "op": "COPY_OUT", "pipe": "PIPE_MTE3", "cycles": 1})
    edges.extend([
        {"source": 209, "target": 99},
        {"source": 99, "target": 300},
    ])

    assignments = [0, 1, 1, 2, 2, 3, 3, 4, 4]
    schedules = [[0], [1, 2], [3, 4], [5, 6], [7, 8]]
    return (
        {"tensors": tensors, "ops": ops, "edges": edges},
        {
            "node_to_subgraph": {
                str(20 + index): group for index, group in enumerate(range(9))
            },
            "core_schedules": schedules,
        },
    )