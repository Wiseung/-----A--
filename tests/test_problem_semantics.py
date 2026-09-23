from __future__ import annotations

import unittest
from pathlib import Path

from solver.graph_analysis import analyze_graph
from solver.graph_io import load_config, parse_graph
from solver.partition import Partition, _topological_order_for_partition, partition_contiguous
from solver.schedule_common import _estimated_partition_copy_bytes, schedule_partition
from tests.helpers import (
    make_chain_graph,
    make_fifo_reuse_graph,
    make_shared_input_graph,
    plan_for,
)

from multicore_cut_evaluate_problem_1 import evaluate_scene_a
from multicore_cut_evaluate_problem_2 import evaluate_scene_b
from multicore_cut_evaluate_problem_3 import evaluate_problem_3


class OfficialSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = load_config(cls.root / "2026_official")

    def eval_a(self, graph, plan):
        return evaluate_scene_a(
            graph, plan,
            bandwidth=self.config["bandwidth"],
            capacity=self.config["capacity"],
            cross_core_wait=self.config["scene_a"]["task_cross_core_wait_cycles"],
            same_core_wait=self.config["scene_a"]["task_same_core_wait_cycles"],
        )

    def eval_b(self, graph, plan):
        return evaluate_scene_b(
            graph, plan,
            bandwidth=self.config["bandwidth"],
            capacity=self.config["capacity"],
            cross_core_copy_delay=self.config["scene_b"]["cross_core_copy_delay_cycles"],
        )

    def eval_3(self, graph, plan, cache_capacity=None):
        cache = dict(self.config["problem_3"])
        if cache_capacity is not None:
            cache["cache_capacity_bytes"] = cache_capacity
        return evaluate_problem_3(
            graph, plan,
            bandwidth=self.config["bandwidth"],
            capacity=self.config["capacity"],
            cross_core_copy_delay=self.config["scene_b"]["cross_core_copy_delay_cycles"],
            **cache,
        )

    def test_problem_a_same_core_subgraphs_still_copy_via_ddr(self) -> None:
        graph = make_chain_graph(2)
        plan = plan_for([0, 0], 2)
        result_a = self.eval_a(graph, plan)
        result_b = self.eval_b(graph, plan)
        self.assertEqual(result_a["data_movement_bytes"]["added_copy_bytes"], 32)
        self.assertEqual(result_b["data_movement_bytes"]["added_copy_bytes"], 0)

    def test_problem_b_allows_core_zero_to_one_to_zero(self) -> None:
        graph = make_chain_graph(3)
        plan = plan_for([0, 1, 0], 2)
        result = self.eval_b(graph, plan)
        self.assertGreater(result["makespan"], 0)
        self.assertEqual(result["num_cores"], 2)

    def test_global_ddr_bandwidth_is_shared(self) -> None:
        graph = make_shared_input_graph(2, size=60)
        plan = plan_for([0, 1], 2)
        result = self.eval_b(graph, plan)
        copies = [
            op for core in result["per_core_timeline"] for op in core["ops"]
            if op["op"] == "COPY_IN"
        ]
        self.assertEqual(len(copies), 2)
        self.assertEqual({op["start"] for op in copies}, {0})
        self.assertEqual({op["duration"] for op in copies}, {2})

    def test_simultaneous_first_cache_reads_both_miss(self) -> None:
        graph = make_shared_input_graph(2, size=128)
        plan = plan_for([0, 1], 2)
        result = self.eval_3(graph, plan)
        stats = result["cache_stats"]
        self.assertEqual(stats["copy_in_misses"], 2)
        self.assertEqual(stats["copy_in_hits"], 0)

    def test_fifo_hit_does_not_refresh_eviction_order(self) -> None:
        graph, plan = make_fifo_reuse_graph()
        result = self.eval_3(graph, plan, cache_capacity=16)
        reads = [
            event for event in result["cache_events"]
            if event["event"] in {"hit", "miss"}
        ]
        reads.sort(key=lambda event: (event["time"], event["core_id"], event["op_id"]))
        tensor_a = 2
        a_reads = [event for event in reads if event["tensor_id"] == tensor_a]
        self.assertEqual([event["event"] for event in a_reads], ["miss", "hit", "miss"])
        self.assertTrue(any(
            event["event"] == "insert" and tensor_a in event["evicted_tensor_ids"]
            for event in result["cache_events"]
        ))

    def test_tensor_larger_than_cache_is_not_inserted(self) -> None:
        size = 1_048_577
        graph = make_shared_input_graph(1, size=size)
        plan = plan_for([0], 1)
        large_capacity = {"L1": 2_000_000, "UB": 2_000_000}
        result = evaluate_problem_3(
            graph, plan,
            bandwidth=self.config["bandwidth"],
            capacity=large_capacity,
            cross_core_copy_delay=self.config["scene_b"]["cross_core_copy_delay_cycles"],
            **self.config["problem_3"],
        )
        self.assertEqual(result["cache_stats"]["copy_in_misses"], 1)
        self.assertEqual(result["cache_final_entries"], [])

    def test_scene_b_fanout_copy_pairs_match_official(self) -> None:
        size = 128
        for target_core_count in (1, 2, 3):
            consumer_count = target_core_count
            tensors = [
                {"id": 1, "pos": "DDR", "size": size},
                {"id": 101, "pos": "UB", "size": size},
                {"id": 201, "pos": "UB", "size": size},
            ]
            ops = [
                {"id": 5, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 1},
                {"id": 10, "op": "ADD", "pipe": "PIPE_M", "cycles": 10},
            ]
            edges = [
                {"source": 1, "target": 5},
                {"source": 5, "target": 101},
                {"source": 101, "target": 10},
                {"source": 10, "target": 201},
            ]
            groups = [[10]]
            for index in range(consumer_count):
                op_id = 20 + index
                tensor_id = 300 + index
                ops.append({
                    "id": op_id, "op": "ADD", "pipe": "PIPE_V", "cycles": 10,
                })
                tensors.append({"id": tensor_id, "pos": "UB", "size": 0})
                edges.extend([
                    {"source": 201, "target": op_id},
                    {"source": op_id, "target": tensor_id},
                ])
                groups.append([op_id])
            if target_core_count == 3:
                tensors.append({"id": 400, "pos": "DDR", "size": size})
                ops.append({
                    "id": 90, "op": "COPY_OUT", "pipe": "PIPE_MTE3", "cycles": 1,
                })
                edges.extend([
                    {"source": 201, "target": 90},
                    {"source": 90, "target": 400},
                ])
            raw = {"tensors": tensors, "ops": ops, "edges": edges}
            graph = parse_graph(raw)
            analysis = analyze_graph(graph)
            op_to_subgraph = {
                op_id: group_id
                for group_id, members in enumerate(groups)
                for op_id in members
            }
            partition = Partition(groups, op_to_subgraph)
            owners = {group_id: group_id for group_id in range(len(groups))}
            plan = {
                "node_to_subgraph": {
                    str(op_id): group_id
                    for op_id, group_id in op_to_subgraph.items()
                },
                "core_schedules": [[group_id] for group_id in range(len(groups))],
            }

            estimated = _estimated_partition_copy_bytes(analysis, partition, owners)
            official = self.eval_b(raw, plan)

            self.assertEqual(estimated, 2 * target_core_count * size)
            self.assertEqual(
                official["data_movement_bytes"]["partition_added_copy_bytes"],
                estimated,
            )

    def test_scene_b_same_target_core_fanout_is_merged(self) -> None:
        size = 128
        raw = {
            "tensors": [
                {"id": 1, "pos": "DDR", "size": size},
                {"id": 101, "pos": "UB", "size": size},
                {"id": 201, "pos": "UB", "size": size},
                {"id": 300, "pos": "UB", "size": 0},
                {"id": 301, "pos": "UB", "size": 0},
            ],
            "ops": [
                {"id": 5, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 1},
                {"id": 10, "op": "ADD", "pipe": "PIPE_M", "cycles": 10},
                {"id": 20, "op": "ADD", "pipe": "PIPE_V", "cycles": 10},
                {"id": 21, "op": "ADD", "pipe": "PIPE_V", "cycles": 10},
            ],
            "edges": [
                {"source": 1, "target": 5}, {"source": 5, "target": 101},
                {"source": 101, "target": 10}, {"source": 10, "target": 201},
                {"source": 201, "target": 20}, {"source": 20, "target": 300},
                {"source": 201, "target": 21}, {"source": 21, "target": 301},
            ],
        }
        graph = parse_graph(raw)
        analysis = analyze_graph(graph)
        partition = Partition(
            groups=[[10], [20], [21]],
            op_to_subgraph={10: 0, 20: 1, 21: 2},
        )
        owners = {0: 0, 1: 1, 2: 1}
        plan = {
            "node_to_subgraph": {"10": 0, "20": 1, "21": 2},
            "core_schedules": [[0], [1, 2]],
        }

        estimated = _estimated_partition_copy_bytes(analysis, partition, owners)
        official = self.eval_b(raw, plan)

        self.assertEqual(estimated, 2 * size)
        self.assertEqual(
            official["data_movement_bytes"]["partition_added_copy_bytes"],
            estimated,
        )

    def test_scheduler_transfer_deltas_match_official(self) -> None:
        raw = make_shared_input_graph(4, size=60)
        graph = parse_graph(raw)
        analysis = analyze_graph(graph)
        groups = [[20], [21], [22], [23]]
        partition = Partition(
            groups=groups,
            op_to_subgraph={op_id: group_id
                            for group_id, members in enumerate(groups)
                            for op_id in members},
        )
        diagnostics = {}

        plan = schedule_partition(
            analysis, partition, 3, 2, self.config, diagnostics
        )
        official = self.eval_b(raw, plan)

        self.assertEqual(
            diagnostics["estimated_partition_added_copy_bytes"],
            official["data_movement_bytes"]["partition_added_copy_bytes"],
        )
        self.assertEqual(
            diagnostics["estimated_partition_added_copy_bytes_recomputed"],
            official["data_movement_bytes"]["partition_added_copy_bytes"],
        )

    def test_scheduler_waits_for_input_copyin_before_consumer(self) -> None:
        graph = parse_graph(make_shared_input_graph(1, size=120))
        analysis = analyze_graph(graph)
        partition = Partition(
            groups=[[20]],
            op_to_subgraph={20: 0},
        )
        diagnostics = {}

        schedule_partition(analysis, partition, 1, 2, self.config, diagnostics)

        chosen = diagnostics["groups"][0]
        self.assertEqual(chosen["group_start"], 2)
        self.assertEqual(chosen["group_finish"], 3)
        self.assertEqual(
            chosen["candidate_placements"][0]["pipe_ends"]["PIPE_MTE2"],
            2,
        )

    def test_scheduler_communication_delta_depends_on_candidate_core(self) -> None:
        graph = parse_graph(make_chain_graph(2, size=16))
        analysis = analyze_graph(graph)
        partition = Partition(
            groups=[[20], [21]],
            op_to_subgraph={20: 0, 21: 1},
        )
        diagnostics = {}

        schedule_partition(analysis, partition, 2, 2, self.config, diagnostics)

        child = next(item for item in diagnostics["groups"] if item["group_id"] == 1)
        candidates = {item["core_id"]: item for item in child["candidate_placements"]}
        self.assertEqual(candidates[0]["delta_cross_copy_bytes"], 0)
        self.assertEqual(candidates[1]["delta_cross_copy_bytes"], 2 * 16)
        self.assertNotEqual(candidates[0]["score"][1], candidates[1]["score"][1])

    def test_scheduler_models_source_write_and_target_read_pipes(self) -> None:
        raw = {
            "tensors": [
                {"id": 1, "pos": "DDR", "size": 0},
                {"id": 101, "pos": "UB", "size": 0},
                {"id": 201, "pos": "UB", "size": 60},
                {"id": 202, "pos": "UB", "size": 0},
            ],
            "ops": [
                {"id": 5, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 1},
                {"id": 10, "op": "MATMUL", "pipe": "PIPE_M", "cycles": 80},
                {"id": 20, "op": "ADD", "pipe": "PIPE_V", "cycles": 120},
                {"id": 30, "op": "ADD", "pipe": "PIPE_V", "cycles": 100},
            ],
            "edges": [
                {"source": 1, "target": 5}, {"source": 5, "target": 101},
                {"source": 101, "target": 10}, {"source": 10, "target": 201},
                {"source": 201, "target": 30},
                {"source": 30, "target": 202},
            ],
        }
        graph = parse_graph(raw)
        analysis = analyze_graph(graph)
        partition = Partition(
            groups=[[10], [20], [30]],
            op_to_subgraph={10: 0, 20: 1, 30: 2},
        )
        config = {
            **self.config,
            "scene_b": {**self.config["scene_b"], "cross_core_copy_delay_cycles": 0},
        }
        diagnostics = {}

        plan = schedule_partition(analysis, partition, 2, 2, config, diagnostics)

        consumer = next(item for item in diagnostics["groups"] if item["group_id"] == 2)
        candidates = {item["core_id"]: item for item in consumer["candidate_placements"]}
        self.assertEqual(candidates[0]["delta_cross_copy_bytes"], 0)
        self.assertEqual(candidates[1]["delta_cross_copy_bytes"], 120)
        self.assertEqual(consumer["chosen_core"], 1, msg=str(consumer))
        self.assertEqual(consumer["group_start"], 83)
        result = self.eval_b(raw, plan)
        self.assertEqual(
            result["data_movement_bytes"]["partition_added_copy_bytes"], 120
        )

    def test_subgraph_rank_uses_dag_order_not_group_id(self) -> None:
        graph = parse_graph(make_chain_graph(2, size=16))
        analysis = analyze_graph(graph)
        partition = Partition(
            groups=[[21], [20]],
            op_to_subgraph={21: 0, 20: 1},
        )
        diagnostics = {}

        schedule_partition(analysis, partition, 1, 2, self.config, diagnostics)

        group_ranks = {
            group["group_id"]: group["rank"] for group in diagnostics["groups"]
        }
        self.assertGreater(group_ranks[1], group_ranks[0])

    def test_alternate_topological_orders_are_legal_candidates(self) -> None:
        raw = {
            "tensors": [
                {"id": 1, "pos": "DDR", "size": 8},
                {"id": 2, "pos": "DDR", "size": 8},
                {"id": 101, "pos": "UB", "size": 8},
                {"id": 102, "pos": "UB", "size": 0},
                {"id": 201, "pos": "UB", "size": 8},
                {"id": 202, "pos": "UB", "size": 8},
                {"id": 203, "pos": "UB", "size": 0},
            ],
            "ops": [
                {"id": 20, "op": "ADD", "pipe": "PIPE_V", "cycles": 1},
                {"id": 21, "op": "ADD", "pipe": "PIPE_V", "cycles": 1},
                {"id": 30, "op": "ADD", "pipe": "PIPE_V", "cycles": 1},
                {"id": 31, "op": "ADD", "pipe": "PIPE_V", "cycles": 1},
                {"id": 32, "op": "ADD", "pipe": "PIPE_V", "cycles": 1},
            ],
            "edges": [
                {"source": 1, "target": 20}, {"source": 20, "target": 101},
                {"source": 101, "target": 21}, {"source": 21, "target": 102},
                {"source": 2, "target": 30}, {"source": 30, "target": 201},
                {"source": 201, "target": 31}, {"source": 31, "target": 202},
                {"source": 202, "target": 32}, {"source": 32, "target": 203},
            ],
        }
        analysis = analyze_graph(parse_graph(raw))
        id_order = _topological_order_for_partition(analysis, "id")
        critical_order = _topological_order_for_partition(analysis, "critical_path")
        release_order = _topological_order_for_partition(analysis, "release_bytes")

        self.assertEqual(id_order[0], 20)
        self.assertEqual(critical_order[0], 30)
        self.assertEqual(set(release_order), analysis.eligible_ops)
        critical_partition = partition_contiguous(
            analysis, 2, 2, topology_strategy="critical_path"
        )
        self.assertEqual(
            set(op for group in critical_partition.groups for op in group),
            analysis.eligible_ops,
        )

    def test_scene_b_direct_op_edge_copy_pair_matches_official(self) -> None:
        raw = {
            "tensors": [{"id": 300, "pos": "UB", "size": 0}],
            "ops": [
                {"id": 10, "op": "MATMUL", "pipe": "PIPE_M", "cycles": 10},
                {"id": 20, "op": "ADD", "pipe": "PIPE_V", "cycles": 10},
            ],
            "edges": [
                {"source": 10, "target": 20, "data_size": 32},
                {"source": 20, "target": 300},
            ],
        }
        graph = parse_graph(raw)
        analysis = analyze_graph(graph)
        partition = Partition(
            groups=[[10], [20]],
            op_to_subgraph={10: 0, 20: 1},
        )
        owners = {0: 0, 1: 1}
        plan = {
            "node_to_subgraph": {"10": 0, "20": 1},
            "core_schedules": [[0], [1]],
        }

        estimated = _estimated_partition_copy_bytes(analysis, partition, owners)
        official = self.eval_b(raw, plan)

        self.assertEqual(estimated, 64)
        self.assertEqual(
            official["data_movement_bytes"]["partition_added_copy_bytes"],
            estimated,
        )


if __name__ == "__main__":
    unittest.main()