from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from ase.io import iread

from .file_conversion import write_normalized_extxyz


MANIFEST_VERSION = 1
MANIFEST_NAME = "scf_filter_sources.json"


def collection_failure_reason(exc):
    if isinstance(exc, (FileNotFoundError, EOFError)):
        return "missing_efs"
    message = str(exc).lower()
    missing = any(
        token in message
        for token in ("missing", "not found", "no such file", "does not exist", "empty")
    )
    efs_source = any(
        token in message
        for token in (
            "energy",
            "force",
            "stress",
            "efs",
            "vasprun",
            "logout",
            "running_scf",
            "output",
        )
    )
    return "missing_efs" if missing and efs_source else "collection_failed"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def manifest_path_for_output(out_name):
    return Path(out_name).resolve().with_name(MANIFEST_NAME)


def _relative_task(root, task):
    root = Path(root).resolve()
    task = Path(task).resolve()
    try:
        return task.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"SCF source task is outside the collection root: {task}") from exc


def _normalize_excluded(root, records):
    normalized = []
    seen = set()
    for record in records:
        source_task = _relative_task(root, record["task"])
        if source_task in seen:
            continue
        seen.add(source_task)
        item = {
            "source_task": source_task,
            "reason": str(record.get("reason") or "collection_failed"),
        }
        if record.get("detail"):
            item["detail"] = str(record["detail"])
        if record.get("max_force") is not None:
            item["max_force"] = float(record["max_force"])
        normalized.append(item)
    return normalized


def write_scf_filter_sources(
    current,
    out_name,
    force_threshold,
    selected_records,
    excluded_records,
):
    current = Path(current).resolve()
    output = Path(out_name).resolve()
    frames = []
    for frame_index, record in enumerate(selected_records):
        frames.append(
            {
                "frame_index": frame_index,
                "source_task": _relative_task(current, record["task"]),
                "max_force": float(record["max_force"]),
                "selection_reason": str(record["selection_reason"]),
            }
        )

    output_sha256 = _sha256(output) if output.is_file() and output.stat().st_size else None
    for frame in frames:
        frame["scf_filter_sha256"] = output_sha256
    selected_tasks = {item["source_task"] for item in frames}
    excluded = [
        item
        for item in _normalize_excluded(current, excluded_records)
        if item["source_task"] not in selected_tasks
    ]
    payload = {
        "version": MANIFEST_VERSION,
        "scf_filter": str(output),
        "scf_filter_sha256": output_sha256,
        "frame_count": len(frames),
        "force_threshold": float(force_threshold),
        "frames": frames,
        "excluded": excluded,
    }
    path = manifest_path_for_output(output)
    _atomic_json(path, payload)
    return payload


def load_scf_filter_sources(out_name, required=True):
    output = Path(out_name).resolve()
    path = manifest_path_for_output(output)
    if not path.exists():
        if required:
            raise RuntimeError(f"SCF filter source manifest is missing: {path}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != MANIFEST_VERSION:
        raise RuntimeError(f"Unsupported SCF filter source manifest: {path}")

    expected_sha256 = payload.get("scf_filter_sha256")
    if expected_sha256 is None:
        if output.exists() and output.stat().st_size:
            raise RuntimeError(f"SCF filter source manifest expects an empty output: {output}")
    else:
        if not output.is_file() or _sha256(output) != expected_sha256:
            raise RuntimeError(f"SCF filter source manifest does not match {output}")
    frames = payload.get("frames") or []
    if int(payload.get("frame_count", -1)) != len(frames):
        raise RuntimeError(f"SCF filter source manifest frame count is invalid: {path}")
    actual_frame_count = sum(1 for _ in iread(str(output), index=":")) if expected_sha256 else 0
    if actual_frame_count != len(frames):
        raise RuntimeError(
            f"SCF filter source manifest does not match frame count in {output}: "
            f"manifest={len(frames)} xyz={actual_frame_count}"
        )
    for expected_index, frame in enumerate(frames):
        if int(frame.get("frame_index", -1)) != expected_index:
            raise RuntimeError(f"SCF filter source manifest frame order is invalid: {path}")
        if frame.get("scf_filter_sha256") != expected_sha256:
            raise RuntimeError(f"SCF filter source manifest frame hash is invalid: {path}")
    return payload


def finalize_scf_collection(
    current,
    out_name,
    ori_out_name,
    force_threshold,
    ok_count,
    collected_tasks,
    no_success_paths,
    excluded_records,
):
    current = Path(current).resolve()
    output = Path(out_name).resolve()
    original = Path(ori_out_name).resolve()
    data = []
    if original.is_file() and original.stat().st_size:
        data = list(iread(str(original), index=":"))
    if len(data) != len(collected_tasks):
        raise RuntimeError(
            "SCF source tracking mismatch: "
            f"parsed_frames={len(data)} source_tasks={len(collected_tasks)}"
        )

    max_forces = [float(np.linalg.norm(atom.get_forces(), axis=1).max()) for atom in data]
    selected_indices = [
        index for index, max_force in enumerate(max_forces) if max_force < force_threshold
    ]
    force_count = len(selected_indices)
    fallback_force = "None"
    selection_reason = "force_threshold_pass"
    if not selected_indices and data:
        selected_indices = [int(np.argmin(max_forces))]
        fallback_force = max_forces[selected_indices[0]]
        selection_reason = "minimum_force_fallback"

    selected_set = set(selected_indices)
    selected_records = []
    for output_index, source_index in enumerate(selected_indices):
        write_normalized_extxyz(
            str(output),
            data[source_index],
            append=(selection_reason != "minimum_force_fallback" or output_index > 0),
        )
        selected_records.append(
            {
                "task": collected_tasks[source_index],
                "max_force": max_forces[source_index],
                "selection_reason": selection_reason,
            }
        )

    all_excluded = list(excluded_records)
    for index, task in enumerate(collected_tasks):
        if index in selected_set:
            continue
        all_excluded.append(
            {
                "task": task,
                "reason": "force_threshold_excluded",
                "max_force": max_forces[index],
                "detail": f"max_force={max_forces[index]} threshold={float(force_threshold)}",
            }
        )

    write_scf_filter_sources(
        current,
        output,
        force_threshold,
        selected_records,
        all_excluded,
    )
    return (
        int(ok_count),
        len(data),
        list(no_success_paths),
        force_count,
        fallback_force,
    )
