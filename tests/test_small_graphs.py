from __future__ import annotations

import unittest

from solver.cache_aware import generate_schedule as schedule_3
from solver.graph_analysis import analyze_graph
from solver.graph_io import load_config, parse_graph
from solver.legality import validate_plan
from solver.partition import partition_contiguous
from solver.schedule_a import generate_schedule as schedule_1
from solver.schedule_b import generate_schedule as schedule_2
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


if __name__ == "__main__":
    unittest.main()