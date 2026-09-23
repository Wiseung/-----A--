from __future__ import annotations

import unittest

from solver.graph_io import parse_graph
from solver.legality import validate_plan
from tests.helpers import make_chain_graph, plan_for


class LegalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = parse_graph(make_chain_graph(3))

    def test_accepts_empty_core(self) -> None:
        plan = plan_for([0, 0, 0], 2)
        self.assertEqual(validate_plan(self.graph, plan)["num_cores"], 2)

    def test_rejects_missing_compute_op(self) -> None:
        plan = plan_for([0, 0], 2)
        with self.assertRaises(RuntimeError):
            validate_plan(self.graph, plan)

    def test_rejects_same_core_reverse_order(self) -> None:
        plan = {
            "node_to_subgraph": {"20": 0, "21": 1, "22": 2},
            "core_schedules": [[2, 1, 0], []],
        }
        with self.assertRaises(RuntimeError):
            validate_plan(self.graph, plan)

    def test_rejects_noncontiguous_merge_cycle(self) -> None:
        plan = {
            "node_to_subgraph": {"20": 0, "21": 1, "22": 0},
            "core_schedules": [[0, 1], []],
        }
        with self.assertRaises(RuntimeError):
            validate_plan(self.graph, plan)

    def test_rejects_non_plan_top_level_field(self) -> None:
        plan = plan_for([0, 0, 0], 2)
        plan["metadata"] = {}
        with self.assertRaises(RuntimeError):
            validate_plan(self.graph, plan)


if __name__ == "__main__":
    unittest.main()