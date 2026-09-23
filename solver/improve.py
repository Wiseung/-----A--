from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .graph_analysis import GraphAnalysis
from .legality import validate_plan


def _owners(plan: dict[str, Any]) -> dict[int, int]:
    return {
        subgraph_id: core_id
        for core_id, schedule in enumerate(plan["core_schedules"])
        for subgraph_id in schedule
    }


def _plan_with_owners(
    mapping: dict[str, int], owners: dict[int, int], ncores: int
) -> dict[str, Any]:
    schedules = [[] for _ in range(ncores)]
    for subgraph_id, core_id in owners.items():
        schedules[core_id].append(subgraph_id)
    for schedule in schedules:
        schedule.sort()
    return {"node_to_subgraph": mapping, "core_schedules": schedules}


def generate_neighbors(
    analysis: GraphAnalysis,
    plan: dict[str, Any],
    problem: int,
    limit: int = 32,
) -> Iterator[dict[str, Any]]:
    ncores = len(plan["core_schedules"])
    owners = _owners(plan)
    group_ids = sorted(owners)
    seen = {json.dumps(plan, sort_keys=True)}
    yielded = 0

    for subgraph_id in group_ids:
        source_core = owners[subgraph_id]
        for target_core in range(ncores):
            if target_core == source_core:
                continue
            moved = dict(owners)
            moved[subgraph_id] = target_core
            candidate = _plan_with_owners(dict(plan["node_to_subgraph"]), moved, ncores)
            key = json.dumps(candidate, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            try:
                validate_plan(analysis.graph, candidate, ncores)
            except (ValueError, RuntimeError):
                continue
            yield candidate
            yielded += 1
            if yielded >= limit:
                return

    if problem == 1:
        for left, right in zip(group_ids, group_ids[1:]):
            mapping = {}
            for op_id, old_group in plan["node_to_subgraph"].items():
                if old_group == right:
                    new_group = left
                elif old_group > right:
                    new_group = old_group - 1
                else:
                    new_group = old_group
                mapping[op_id] = new_group
            for target_core in dict.fromkeys((owners[left], owners[right])):
                merged_owners = {}
                for group_id, core_id in owners.items():
                    if group_id == right:
                        merged_owners[left] = target_core
                    elif group_id > right:
                        merged_owners[group_id - 1] = core_id
                    elif group_id == left:
                        merged_owners[left] = target_core
                    else:
                        merged_owners[group_id] = core_id
                candidate = _plan_with_owners(mapping, merged_owners, ncores)
                key = json.dumps(candidate, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    validate_plan(analysis.graph, candidate, ncores)
                except (ValueError, RuntimeError):
                    continue
                yield candidate
                yielded += 1
                if yielded >= limit:
                    return


def plan_signature(plan: dict[str, Any]) -> str:
    return json.dumps(plan, sort_keys=True, separators=(",", ":"))