from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .features import build_subgraph_features
from .graph_analysis import GraphAnalysis
from .legality import validate_plan
from .partition import Partition
from .schedule_common import _schedule_live_range_metrics


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


def _partition_for_plan(plan: dict[str, Any]) -> Partition:
    mapping = {
        int(op_id): int(group_id)
        for op_id, group_id in plan["node_to_subgraph"].items()
    }
    group_count = max(mapping.values(), default=-1) + 1
    groups = [[] for _ in range(group_count)]
    for op_id, group_id in mapping.items():
        groups[group_id].append(op_id)
    return Partition(groups=groups, op_to_subgraph=mapping)


def _residence_key(
    analysis: GraphAnalysis,
    partition: Partition,
    plan: dict[str, Any],
    capacity: dict[str, int],
) -> tuple[float, float, float, str]:
    metrics = _schedule_live_range_metrics(
        analysis, partition, plan, problem=2, capacity=capacity
    )
    l1_values = metrics.get("l1_residence_bytes_est", {}).values()
    ub_values = metrics.get("ub_residence_bytes_est", {}).values()
    max_l1 = max((float(value) for value in l1_values), default=0.0)
    max_ub = max((float(value) for value in ub_values), default=0.0)
    return (
        max_l1,
        max_ub,
        max_l1 + max_ub,
        plan_signature(plan),
    )


def _residence_reinsert_neighbors(
    analysis: GraphAnalysis,
    plan: dict[str, Any],
    capacity: dict[str, int],
    inspect_limit: int = 64,
) -> Iterator[dict[str, Any]]:
    partition = _partition_for_plan(plan)
    scored: list[tuple[tuple[float, float, float, str], dict[str, Any]]] = []
    for index, candidate in enumerate(_reinsert_neighbors(plan)):
        if index >= inspect_limit:
            break
        try:
            validate_plan(
                analysis.graph,
                candidate,
                len(plan["core_schedules"]),
            )
        except (ValueError, RuntimeError):
            continue
        scored.append((
            _residence_key(analysis, partition, candidate, capacity),
            candidate,
        ))
    for _, candidate in sorted(scored, key=lambda item: item[0]):
        yield candidate


def _shared_input_skew_neighbors(
    analysis: GraphAnalysis,
    plan: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    partition = _partition_for_plan(plan)
    features, _ = build_subgraph_features(analysis, partition)
    owners = _owners(plan)
    positions = {
        group_id: (core_id, index)
        for core_id, schedule in enumerate(plan["core_schedules"])
        for index, group_id in enumerate(schedule)
    }
    groups_by_tensor: dict[int, set[int]] = {}
    for group_id, feature in enumerate(features):
        for tensor_id in feature.shared_ddr_inputs:
            groups_by_tensor.setdefault(tensor_id, set()).add(group_id)

    seen: set[str] = set()
    for tensor_id, groups in sorted(groups_by_tensor.items()):
        groups_by_core: dict[int, list[int]] = {}
        for group_id in groups:
            core_id = owners[group_id]
            groups_by_core.setdefault(core_id, []).append(group_id)
        if len(groups_by_core) < 2:
            continue
        ordered_cores = sorted(
            groups_by_core,
            key=lambda core_id: min(positions[group_id] for group_id in groups_by_core[core_id]),
        )
        for target_core in ordered_cores[1:]:
            target_group = min(
                groups_by_core[target_core],
                key=lambda group_id: positions[group_id],
            )
            target_index = plan["core_schedules"][target_core].index(target_group)
            blockers = [
                group_id
                for group_id in plan["core_schedules"][target_core][target_index + 1:]
                if features[group_id].external_inputs
            ]
            for blocker in blockers[:2]:
                schedule = list(plan["core_schedules"][target_core])
                schedule.remove(target_group)
                blocker_index = schedule.index(blocker)
                schedule.insert(blocker_index + 1, target_group)
                schedules = [list(items) for items in plan["core_schedules"]]
                schedules[target_core] = schedule
                candidate = {
                    "node_to_subgraph": dict(plan["node_to_subgraph"]),
                    "core_schedules": schedules,
                }
                signature = plan_signature(candidate)
                if signature in seen or signature == plan_signature(plan):
                    continue
                seen.add(signature)
                yield candidate


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
    residence_ordering: bool = False,
    capacity: dict[str, int] | None = None,
    shared_input_skew: bool = False,
) -> Iterator[tuple[dict[str, Any], str]]:
    if limit <= 0:
        return
    owners = _owners(plan)
    if shared_input_skew and problem == 3:
        families = [(
            "shared_input_skew",
            iter(_shared_input_skew_neighbors(analysis, plan)),
        )]
    elif residence_ordering and problem == 2 and capacity:
        families = [(
            "residence_reinsert",
            iter(_residence_reinsert_neighbors(analysis, plan, capacity)),
        )]
    else:
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