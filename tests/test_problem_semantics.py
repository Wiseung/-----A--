from __future__ import annotations

import unittest
from pathlib import Path

from solver.graph_io import load_config
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


if __name__ == "__main__":
    unittest.main()