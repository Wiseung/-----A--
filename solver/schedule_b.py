from __future__ import annotations

from typing import Any

from .graph_analysis import GraphAnalysis
from .partition import candidate_group_counts, partition_contiguous
from .schedule_common import schedule_partition


def generate_schedule(
    analysis: GraphAnalysis,
    ncores: int,
    config: dict[str, Any],
    group_count: int,
    diagnostics: dict[str, Any] | None = None,
    topology_strategy: str = "id",
) -> dict[str, Any]:
    partition = partition_contiguous(
        analysis, group_count, problem=2,
        topology_strategy=topology_strategy,
    )
    return schedule_partition(
        analysis, partition, ncores, problem=2, config=config,
        diagnostics=diagnostics,
    )


def candidate_counts(analysis: GraphAnalysis, ncores: int) -> list[int]:
    return candidate_group_counts(len(analysis.eligible_ops), ncores, problem=2)