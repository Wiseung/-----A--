from __future__ import annotations

import unittest

from solver.cache_aware import generate_schedule as schedule_3
from solver.graph_analysis import analyze_graph
from solver.graph_io import load_config, parse_graph
from solver.legality import validate_plan
from solver.improve import generate_neighbor_candidates
from solver.partition import Partition, partition_contiguous
from solver.schedule_a import generate_schedule as schedule_1
from solver.schedule_b import generate_schedule as schedule_2
from solver.schedule_common import _cache_contains_at, schedule_partition
from tests.helpers import make_chain_graph, make_shared_input_graph


class SmallGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = load_config(cls.root / "2026_official")

    def test_chain_and_shared_fork_graphs_are_schedulable(self) -> None:
        for raw in (make_chain_graph(5), make_shared_input_graph(4)):
            graph = parse_graph(raw)
            analysis = analyze_graph(graph)
            for problem, generator in ((1, schedule_1), (2, schedule_2), (3, schedule_3)):
                for cores in range(1, 6):
                    count = min(len(analysis.eligible_ops), cores * 2)
                    partition = partition_contiguous(analysis, count, problem)
                    plan = generator(analysis, cores, self.config, len(partition.groups))
                    validate_plan(graph, plan, cores)

    def test_two_independent_ops_can_use_two_cores(self) -> None:
        graph = parse_graph(make_shared_input_graph(2))
        analysis = analyze_graph(graph)
        plan = schedule_1(analysis, 2, self.config, 2)
        owners = {
            core for core, schedule in enumerate(plan["core_schedules"])
            if schedule
        }
        self.assertEqual(owners, {0, 1})

    def test_problem_b_commits_critical_path_finish(self) -> None:
        raw = {
            "tensors": [
                {"id": 1, "pos": "DDR", "size": 0},
                {"id": 101, "pos": "UB", "size": 0},
                {"id": 201, "pos": "UB", "size": 0},
                {"id": 202, "pos": "UB", "size": 0},
                {"id": 203, "pos": "UB", "size": 0},
                {"id": 204, "pos": "UB", "size": 0},
                {"id": 205, "pos": "UB", "size": 0},
            ],
            "ops": [
                {"id": 10, "op": "MATMUL", "pipe": "PIPE_M", "cycles": 80},
                {"id": 11, "op": "ADD", "pipe": "PIPE_V", "cycles": 80},
                {"id": 20, "op": "ADD", "pipe": "PIPE_V", "cycles": 150},
                {"id": 30, "op": "ADD", "pipe": "PIPE_V", "cycles": 79},
                {"id": 40, "op": "ADD", "pipe": "PIPE_V", "cycles": 1},
            ],
            "edges": [
                {"source": 1, "target": 10},
                {"source": 1, "target": 20},
                {"source": 1, "target": 30},
                {"source": 10, "target": 201},
                {"source": 201, "target": 11},
                {"source": 11, "target": 202},
                {"source": 202, "target": 40},
                {"source": 20, "target": 203},
                {"source": 30, "target": 204},
                {"source": 40, "target": 205},
            ],
        }
        graph = parse_graph(raw)
        analysis = analyze_graph(graph)
        partition = Partition(
            groups=[[10, 11], [20], [30], [40]],
            op_to_subgraph={10: 0, 11: 0, 20: 1, 30: 2, 40: 3},
        )
        config = {
            **self.config,
            "scene_b": {**self.config["scene_b"], "cross_core_copy_delay_cycles": 0},
        }

        plan = schedule_partition(analysis, partition, 2, 2, config)

        self.assertIn(3, plan["core_schedules"][0])
        self.assertNotIn(3, plan["core_schedules"][1])

    def test_future_cache_insert_does_not_evict_past_query_state(self) -> None:
        events = [(0.0, 0, 1, 8), (100.0, 1, 2, 8)]

        self.assertTrue(_cache_contains_at(events, 50.0, 8, 1))
        self.assertFalse(_cache_contains_at(events, 100.0, 8, 1))
        self.assertTrue(_cache_contains_at(events, 100.0, 8, 2))

    def test_migration_preserves_other_groups_relative_order(self) -> None:
        graph = parse_graph(make_shared_input_graph(4))
        analysis = analyze_graph(graph)
        plan = {
            "node_to_subgraph": {str(20 + index): index for index in range(4)},
            "core_schedules": [[0, 2], [1, 3]],
        }

        candidate, family = next(
            generate_neighbor_candidates(analysis, plan, problem=2, limit=1)
        )

        self.assertEqual(family, "move")
        self.assertEqual(candidate["core_schedules"][0], [2])
        self.assertEqual(
            [group for group in candidate["core_schedules"][1] if group != 0],
            [1, 3],
        )

    def test_small_neighbor_budget_covers_merge_and_split(self) -> None:
        graph = parse_graph(make_chain_graph(4))
        analysis = analyze_graph(graph)
        merge_plan = {
            "node_to_subgraph": {
                str(op_id): group_id
                for group_id, op_id in enumerate(analysis.topological_order)
            },
            "core_schedules": [[0, 1, 2, 3], []],
        }
        merge_candidates = list(generate_neighbor_candidates(
            analysis, merge_plan, problem=1, limit=2
        ))
        self.assertEqual([family for _, family in merge_candidates], ["move", "merge"])

        split_plan = {
            "node_to_subgraph": {str(op_id): 0 for op_id in analysis.topological_order},
            "core_schedules": [[0], []],
        }
        split_candidates = list(generate_neighbor_candidates(
            analysis, split_plan, problem=2, limit=2
        ))
        self.assertEqual([family for _, family in split_candidates], ["move", "split"])


if __name__ == "__main__":
    unittest.main()