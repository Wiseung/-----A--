from __future__ import annotations

from typing import Any

from .graph_analysis import GraphAnalysis
from .partition import candidate_group_counts, partition_with_strategy
from .schedule_common import schedule_partition


def generate_schedule(
    analysis: GraphAnalysis,
    ncores: int,
    config: dict[str, Any],
    group_count: int,
    diagnostics: dict[str, Any] | None = None,
    topology_strategy: str = "id",
    partition_strategy: str = "contiguous",
    placement_scoring: str = "baseline",
    cache_ordering: str = "fifo",
) -> dict[str, Any]:
    partition = partition_with_strategy(
        analysis,
        group_count,
        problem=1,
        partition_strategy=partition_strategy,
        topology_strategy=topology_strategy,
    )
    return schedule_partition(
        analysis, partition, ncores, problem=1, config=config,
        diagnostics=diagnostics,
        placement_scoring=placement_scoring,
        cache_ordering=cache_ordering,
    )


def candidate_counts(analysis: GraphAnalysis, ncores: int) -> list[int]:
    return candidate_group_counts(len(analysis.eligible_ops), ncores, problem=1)