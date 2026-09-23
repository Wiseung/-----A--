from __future__ import annotations

from typing import Any

from .graph_io import Graph, OFFICIAL_CODE

import sys

if str(OFFICIAL_CODE) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE))

from stub_multicore_cut_and_schedule import derive_multicore_plan  # noqa: E402


def validate_plan(graph: Graph, plan: dict[str, Any], ncores: int | None = None) -> dict[str, Any]:
    view = derive_multicore_plan(graph.raw, plan)
    if ncores is not None and view["num_cores"] != ncores:
        raise ValueError(f"plan has {view['num_cores']} core schedules; expected {ncores}")
    return view