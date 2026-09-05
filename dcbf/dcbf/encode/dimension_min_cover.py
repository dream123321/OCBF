from __future__ import annotations

from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
from pathlib import Path
import socket
import time

import numpy as np

from .find_min_cover_set import find_min_cover_set
from .compact_indices import IndexBucket, count_selected_indices, find_multi_cover_set_compact
from ..memory_guard import MIB, current_guard, work_memory, require_memory, stage_progress


def normalize_dimension_min_cover_workers(value) -> int:
    if isinstance(value, bool):
        raise ValueError("dimension_min_cover_workers must be an integer")
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("dimension_min_cover_workers must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("dimension_min_cover_workers must be an integer")
    if workers < -1:
        raise ValueError("dimension_min_cover_workers must be -1, 0, or a positive integer")
    return workers


def build_dimension_tasks(nested_classes, body, elements, prefix=(), nested_targets=None):
    tasks = []
    for element_index, dimension_classes in enumerate(nested_classes or []):
        element = elements[element_index] if element_index < len(elements) else f"type-{element_index}"
        for dimension, classes in enumerate(dimension_classes or []):
            populated = [bucket if isinstance(bucket, IndexBucket) else list(bucket) for bucket in (classes or []) if bucket]
            if not populated:
                continue
            targets = None
            if nested_targets is not None:
                raw_targets = nested_targets[element_index][dimension]
                targets = [int(target) for bucket, target in zip(classes, raw_targets) if bucket]
            tasks.append(
                {
                    "key": tuple(prefix) + (str(body), str(element), int(dimension)),
                    "classes": populated,
                    "targets": targets,
                }
            )
    return tasks


def merge_dimension_tasks(tasks):
    merged = {}
    order = []
    for task in tasks:
        key = tuple(task["key"])
        if key not in merged:
            merged[key] = {"key": key, "classes": [], "targets": None}
            order.append(key)
        merged[key]["classes"].extend(task.get("classes") or [])
        targets = task.get("targets")
        if targets is not None:
            if merged[key]["targets"] is None:
                merged[key]["targets"] = []
            merged[key]["targets"].extend(targets)
    return [merged[key] for key in order]


def find_multi_cover_set(classes, targets):
    if not classes:
        return []
    if len(classes) != len(targets):
        raise ValueError("Internal error: classes and targets length mismatch")

    remaining = [max(0, int(target)) for target in targets]
    structure_to_classes = defaultdict(list)
    for class_index, bucket in enumerate(classes):
        counts = defaultdict(int)
        for raw_index in bucket:
            counts[int(raw_index)] += 1
        for structure_index, count in counts.items():
            structure_to_classes[structure_index].append((class_index, count))

    selected = []
    available = set(structure_to_classes)
    while available and any(value > 0 for value in remaining):
        best_index = None
        best_score = 0
        for structure_index in available:
            score = 0
            for class_index, count in structure_to_classes[structure_index]:
                if remaining[class_index] > 0:
                    score += min(count, remaining[class_index])
            if score > best_score or (
                score == best_score and best_index is not None and structure_index < best_index
            ):
                best_index = structure_index
                best_score = score
        if best_index is None or best_score <= 0:
            break

        selected.append(best_index)
        available.remove(best_index)
        for class_index, count in structure_to_classes[best_index]:
            if remaining[class_index] > 0:
                remaining[class_index] = max(0, remaining[class_index] - count)
    return selected


def _solve_dimension_task(task):
    started = time.perf_counter()
    targets = task.get("targets")
    classes = task["classes"]
    if targets is None:
        selected = find_min_cover_set(classes)
    else:
        selected = find_multi_cover_set_compact(classes, targets)
    return {
        "key": tuple(task["key"]),
        "selected": sorted(set(int(index) for index in selected)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _parse_positive_int(value):
    try:
        parsed = int(str(value).split("(", 1)[0])
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _lsf_local_allocation(environ, hostname):
    hostfile = environ.get("LSB_DJOB_HOSTFILE")
    if hostfile and Path(hostfile).is_file():
        short_name = hostname.split(".", 1)[0]
        count = 0
        for line in Path(hostfile).read_text(encoding="utf-8", errors="ignore").splitlines():
            listed = line.strip()
            if listed and listed.split(".", 1)[0] == short_name:
                count += 1
        if count > 0:
            return count
    return _parse_positive_int(environ.get("LSB_DJOB_NUMPROC"))


def detect_local_cpu_limit(environ=None, affinity_count=None, hostname=None):
    environ = os.environ if environ is None else environ
    hostname = socket.gethostname() if hostname is None else hostname
    if affinity_count is None:
        if hasattr(os, "sched_getaffinity"):
            affinity_count = len(os.sched_getaffinity(0))
        else:
            affinity_count = os.cpu_count() or 1
    affinity_count = max(1, int(affinity_count))

    scheduler_limit = None
    scheduler = None
    if any(key in environ for key in ("SLURM_JOB_ID", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE")):
        scheduler = "slurm"
        scheduler_limit = _parse_positive_int(
            environ.get("SLURM_CPUS_ON_NODE") or environ.get("SLURM_JOB_CPUS_PER_NODE")
        )
    elif any(key in environ for key in ("LSB_JOBID", "LSB_DJOB_HOSTFILE", "LSB_DJOB_NUMPROC")):
        scheduler = "lsf"
        scheduler_limit = _lsf_local_allocation(environ, hostname)

    available = affinity_count
    if scheduler_limit is not None:
        available = min(available, scheduler_limit)
    return max(1, available), scheduler


def _cancel_pending(pending):
    for future in pending:
        future.cancel()


def stop_pool(executor, children):
    executor.shutdown(wait=False, cancel_futures=True)
    for child in children:
        if child.is_alive():
            child.terminate()
    deadline = time.monotonic() + 5
    for child in children:
        child.join(timeout=max(0, deadline - time.monotonic()))
        if child.is_alive():
            child.kill()
            child.join(timeout=1)


def prune_dimension_union(tasks, selected):
    started = time.perf_counter()
    selected = sorted(set(int(index) for index in selected))
    if not selected:
        return [], {
            "enabled": True,
            "state_count": 0,
            "input_selected": 0,
            "output_selected": 0,
            "removed": 0,
            "elapsed_seconds": 0.0,
        }

    selected_array = np.asarray(selected, dtype=np.int64)
    target_by_state = []
    contribution_by_structure = defaultdict(dict)
    task_keys_by_structure = defaultdict(set)

    for task_index, task in enumerate(tasks):
        classes = task.get("classes") or []
        raw_targets = task.get("targets")
        if raw_targets is not None and len(classes) != len(raw_targets):
            raise ValueError(
                f"Internal error: classes and targets length mismatch for {tuple(task['key'])}"
            )
        for class_index, bucket in enumerate(classes):
            target = 1 if raw_targets is None else max(0, int(raw_targets[class_index]))
            if target <= 0 or not bucket:
                continue
            state_index = len(target_by_state)
            target_by_state.append(target)
            counts = count_selected_indices(bucket, selected_array)
            for selected_position in np.flatnonzero(counts):
                structure_index = selected[int(selected_position)]
                count = int(counts[selected_position])
                contribution = 1 if raw_targets is None else count
                contribution_by_structure[structure_index][state_index] = contribution
                task_keys_by_structure[structure_index].add(task_index)

    coverage = [0] * len(target_by_state)
    for contributions in contribution_by_structure.values():
        for state_index, count in contributions.items():
            coverage[state_index] += count

    unsatisfied = [
        state_index
        for state_index, target in enumerate(target_by_state)
        if coverage[state_index] < target
    ]
    if unsatisfied:
        raise RuntimeError(
            "Per-dimension union does not satisfy all cover targets before global pruning: "
            f"{len(unsatisfied)} state(s) are under-covered"
        )

    # Remove narrow, low-contribution structures first so cross-dimension structures
    # remain available to replace specialists selected independently per dimension.
    removal_order = sorted(
        selected,
        key=lambda structure_index: (
            len(task_keys_by_structure.get(structure_index, ())),
            len(contribution_by_structure.get(structure_index, {})),
            sum(contribution_by_structure.get(structure_index, {}).values()),
            -structure_index,
        ),
    )
    kept = set(selected)
    removed = []
    for structure_index in removal_order:
        contributions = contribution_by_structure.get(structure_index, {})
        if all(
            coverage[state_index] - count >= target_by_state[state_index]
            for state_index, count in contributions.items()
        ):
            kept.remove(structure_index)
            removed.append(structure_index)
            for state_index, count in contributions.items():
                coverage[state_index] -= count

    output = sorted(kept)
    return output, {
        "enabled": True,
        "state_count": len(target_by_state),
        "input_selected": len(selected),
        "output_selected": len(output),
        "removed": len(removed),
        "removed_indices": removed,
        "elapsed_seconds": time.perf_counter() - started,
    }


def solve_dimension_tasks(tasks, requested_workers):
    requested_workers = normalize_dimension_min_cover_workers(requested_workers)
    tasks = [task for task in tasks if task.get("classes")]
    started = time.perf_counter()
    if not tasks:
        return [], {
            "mode": "per_dimension",
            "requested_workers": requested_workers,
            "effective_workers": 0,
            "task_count": 0,
            "sum_selected": 0,
            "union_selected": 0,
            "overlap_removed": 0,
            "global_prune": {
                "enabled": True,
                "state_count": 0,
                "input_selected": 0,
                "output_selected": 0,
                "removed": 0,
                "elapsed_seconds": 0.0,
            },
            "dimension_solve_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "scheduler": None,
        }, {}
    if requested_workers == 0:
        raise ValueError("solve_dimension_tasks requires a non-zero worker setting")

    available_cpus, scheduler = detect_local_cpu_limit()
    worker_limit = available_cpus if requested_workers == -1 else requested_workers
    effective_workers = max(1, min(len(tasks), worker_limit, available_cpus))
    guarded = current_guard() is not None
    if guarded:
        largest = max(sum(len(bucket) for bucket in task['classes']) for task in tasks)
        per_task_bytes = 64 * MIB + largest * 96
        require_memory(per_task_bytes)
        effective_workers = min(effective_workers, max(1, work_memory() // per_task_bytes))
        stage_progress('dimension_min_cover', workers=effective_workers)
    results = []

    if effective_workers == 1:
        for task in tasks:
            results.append(_solve_dimension_task(task))
    else:
        context = multiprocessing.get_context("fork")
        max_in_flight = effective_workers if guarded else effective_workers * 2
        task_iterator = iter(tasks)
        pending = {}
        existing_children = {p.pid for p in multiprocessing.active_children()}
        executor = ProcessPoolExecutor(max_workers=effective_workers, mp_context=context)
        children = []
        try:
            while len(pending) < max_in_flight:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    break
                pending[executor.submit(_solve_dimension_task, task)] = tuple(task["key"])
            children = [p for p in multiprocessing.active_children() if p.pid not in existing_children]

            while pending:
                completed, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                if guarded:
                    require_memory(0)
                for future in completed:
                    key = pending.pop(future)
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        _cancel_pending(pending)
                        raise RuntimeError(f"Dimension min-cover failed for {key}: {exc}") from exc
                    try:
                        task = next(task_iterator)
                    except StopIteration:
                        continue
                    pending[executor.submit(_solve_dimension_task, task)] = tuple(task["key"])
        except BaseException:
            _cancel_pending(pending)
            children = [p for p in multiprocessing.active_children() if p.pid not in existing_children]
            stop_pool(executor, children)
            raise
        else:
            executor.shutdown(wait=True)

    dimension_solve_seconds = time.perf_counter() - started
    results.sort(key=lambda item: item["key"])
    selected_by_task = {tuple(result["key"]): result["selected"] for result in results}
    union_before_prune = sorted({index for result in results for index in result["selected"]})
    sum_selected = sum(len(result["selected"]) for result in results)
    union, prune_stats = prune_dimension_union(tasks, union_before_prune)
    stats = {
        "mode": "per_dimension",
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "task_count": len(tasks),
        "sum_selected": sum_selected,
        "union_selected_before_prune": len(union_before_prune),
        "union_selected": len(union),
        "overlap_removed": sum_selected - len(union_before_prune),
        "global_prune": prune_stats,
        "dimension_solve_seconds": dimension_solve_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "scheduler": scheduler,
    }
    return union, stats, selected_by_task


def format_dimension_min_cover_stats(stats):
    message = (
        "Dimension min-cover completed: "
        f"tasks={stats['task_count']} workers={stats['effective_workers']} "
        f"sum_selected={stats['sum_selected']} "
        f"union_selected={stats.get('union_selected_before_prune', stats['union_selected'])} "
        f"elapsed={stats['elapsed_seconds']:.3f}s"
    )
    prune_stats = stats.get("global_prune") or {}
    message += (
        " global_prune="
        f"{prune_stats.get('input_selected', 0)}->{prune_stats.get('output_selected', 0)} "
        f"prune_elapsed={prune_stats.get('elapsed_seconds', 0.0):.3f}s"
    )
    return message
