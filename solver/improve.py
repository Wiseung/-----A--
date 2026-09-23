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


def _move_group(plan: dict[str, Any], subgraph_id: int, target_core: int) -> dict[str, Any]:
    schedules = [list(schedule) for schedule in plan["core_schedules"]]
    source_core = next(
        core_id for core_id, schedule in enumerate(schedules)
        if subgraph_id in schedule
    )
    schedules[source_core].remove(subgraph_id)
    schedules[target_core].append(subgraph_id)
    return {"node_to_subgraph": dict(plan["node_to_subgraph"]), "core_schedules": schedules}


def _move_neighbors(
    plan: dict[str, Any], owners: dict[int, int]
) -> Iterator[dict[str, Any]]:
    ncores = len(plan["core_schedules"])
    for subgraph_id in sorted(owners):
        source_core = owners[subgraph_id]
        for target_core in range(ncores):
            if target_core != source_core:
                yield _move_group(plan, subgraph_id, target_core)


def _swap_neighbors(
    plan: dict[str, Any], owners: dict[int, int]
) -> Iterator[dict[str, Any]]:
    groups = sorted(owners)
    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            left_core, right_core = owners[left], owners[right]
            if left_core == right_core:
                continue
            schedules = [list(schedule) for schedule in plan["core_schedules"]]
            left_index = schedules[left_core].index(left)
            right_index = schedules[right_core].index(right)
            schedules[left_core][left_index] = right
            schedules[right_core][right_index] = left
            yield {
                "node_to_subgraph": dict(plan["node_to_subgraph"]),
                "core_schedules": schedules,
            }


def _reinsert_neighbors(plan: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for core_id, original in enumerate(plan["core_schedules"]):
        for source_index, subgraph_id in enumerate(original):
            for target_index in range(len(original)):
                if target_index == source_index:
                    continue
                schedule = list(original)
                schedule.pop(source_index)
                schedule.insert(target_index, subgraph_id)
                schedules = [list(items) for items in plan["core_schedules"]]
                schedules[core_id] = schedule
                yield {
                    "node_to_subgraph": dict(plan["node_to_subgraph"]),
                    "core_schedules": schedules,
                }


def _split_neighbors(
    analysis: GraphAnalysis, plan: dict[str, Any], owners: dict[int, int]
) -> Iterator[dict[str, Any]]:
    old_groups = sorted(owners)
    mapping = {int(op_id): int(group_id)
               for op_id, group_id in plan["node_to_subgraph"].items()}
    for split_group in old_groups:
        members = sorted(
            (op_id for op_id, group_id in mapping.items() if group_id == split_group),
            key=analysis.topological_index.__getitem__,
        )
        if len(members) < 2:
            continue
        midpoint = len(members) // 2
        left_members = set(members[:midpoint])
        old_to_new: dict[int, list[int]] = {}
        new_id = 0
        for old_group in old_groups:
            if old_group == split_group:
                old_to_new[old_group] = [new_id, new_id + 1]
                new_id += 2
            else:
                old_to_new[old_group] = [new_id]
                new_id += 1
        new_mapping = {}
        for op_id, old_group in mapping.items():
            if old_group == split_group:
                new_group = old_to_new[old_group][0 if op_id in left_members else 1]
            else:
                new_group = old_to_new[old_group][0]
            new_mapping[str(op_id)] = new_group
        schedules = []
        for schedule in plan["core_schedules"]:
            new_schedule = []
            for old_group in schedule:
                new_schedule.extend(old_to_new[old_group])
            schedules.append(new_schedule)
        yield {"node_to_subgraph": new_mapping, "core_schedules": schedules}


def _merge_neighbors(
    plan: dict[str, Any], owners: dict[int, int]
) -> Iterator[dict[str, Any]]:
    groups = sorted(owners)
    for left, right in zip(groups, groups[1:]):
        for target_core in dict.fromkeys((owners[left], owners[right])):
            mapping = {}
            for op_id, old_group in plan["node_to_subgraph"].items():
                if old_group == right:
                    new_group = left
                elif old_group > right:
                    new_group = old_group - 1
                else:
                    new_group = old_group
                mapping[op_id] = new_group
            schedules = [[] for _ in plan["core_schedules"]]
            merged_inserted = False
            for core_id, schedule in enumerate(plan["core_schedules"]):
                for group_id in schedule:
                    if group_id in (left, right):
                        if core_id == target_core and not merged_inserted:
                            schedules[core_id].append(left)
                            merged_inserted = True
                        continue
                    schedules[core_id].append(
                        group_id if group_id < right else group_id - 1
                    )
            yield {"node_to_subgraph": mapping, "core_schedules": schedules}


def generate_neighbor_candidates(
    analysis: GraphAnalysis,
    plan: dict[str, Any],
    problem: int,
    limit: int = 32,
) -> Iterator[tuple[dict[str, Any], str]]:
    if limit <= 0:
        return
    owners = _owners(plan)
    families = [("move", iter(_move_neighbors(plan, owners)))]
    if problem == 1:
        families.append(("merge", iter(_merge_neighbors(plan, owners))))
    families.extend([
        ("swap", iter(_swap_neighbors(plan, owners))),
        ("reinsert", iter(_reinsert_neighbors(plan))),
        ("split", iter(_split_neighbors(analysis, plan, owners))),
    ])
    seen = {plan_signature(plan)}
    yielded = 0
    while families and yielded < limit:
        next_families = []
        for family, candidates in families:
            while True:
                candidate = next(candidates, None)
                if candidate is None:
                    break
                signature = plan_signature(candidate)
                if signature in seen:
                    continue
                seen.add(signature)
                try:
                    validate_plan(analysis.graph, candidate, len(plan["core_schedules"]))
                except (ValueError, RuntimeError):
                    continue
                next_families.append((family, candidates))
                yielded += 1
                yield candidate, family
                if yielded >= limit:
                    return
                break
        families = next_families


def generate_neighbors(
    analysis: GraphAnalysis,
    plan: dict[str, Any],
    problem: int,
    limit: int = 32,
) -> Iterator[dict[str, Any]]:
    for candidate, _ in generate_neighbor_candidates(analysis, plan, problem, limit):
        yield candidate


def plan_signature(plan: dict[str, Any]) -> str:
    return json.dumps(plan, sort_keys=True, separators=(",", ":"))