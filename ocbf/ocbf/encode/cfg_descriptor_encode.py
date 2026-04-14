from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import shutil
import subprocess


def _count_cfg_blocks(cfg_path):
    count = 0
    with open(cfg_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line == "BEGIN_CFG\n":
                count += 1
    return count


def _build_chunk_ranges(total, parts):
    parts = max(1, min(int(parts), int(total)))
    base, remainder = divmod(total, parts)
    ranges = []
    start = 0
    for index in range(parts):
        stop = start + base + (1 if index < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def _split_cfg_by_blocks(cfg_path, output_dir, prefix, parts):
    total_blocks = _count_cfg_blocks(cfg_path)
    if total_blocks <= 1 or parts <= 1:
        return [Path(cfg_path)]

    ranges = _build_chunk_ranges(total_blocks, parts)
    part_cfg_paths = [Path(output_dir) / f"{prefix}.part_{index:04d}.cfg" for index in range(len(ranges))]
    handles = [open(path, "w", encoding="utf-8") for path in part_cfg_paths]
    try:
        current_part = 0
        block_index = -1
        target_stop = ranges[current_part][1]
        in_block = False
        with open(cfg_path, "r", encoding="utf-8") as source:
            for line in source:
                if line == "BEGIN_CFG\n":
                    block_index += 1
                    while current_part < len(ranges) - 1 and block_index >= target_stop:
                        current_part += 1
                        target_stop = ranges[current_part][1]
                    in_block = True
                if in_block:
                    handles[current_part].write(line)
                if in_block and line.startswith("END_CFG"):
                    in_block = False
    finally:
        for handle in handles:
            handle.close()
    return part_cfg_paths


def _build_calc_descriptors_command(sus2_mlp_exe, mtp_path, cfg_path, out_path, train_env=None):
    argv = [str(sus2_mlp_exe), "calc-descriptors", str(mtp_path), str(cfg_path), str(out_path)]
    train_env = str(train_env or "").strip()
    if not train_env:
        return argv, False
    quoted = " ".join(shlex.quote(item) for item in argv)
    shell_command = f"{train_env}\n{quoted}"
    return ["bash", "-lc", shell_command], True


def _run_calc_descriptors(sus2_mlp_exe, mtp_path, cfg_path, out_path, train_env=None):
    command, shell_mode = _build_calc_descriptors_command(
        sus2_mlp_exe,
        mtp_path,
        cfg_path,
        out_path,
        train_env=train_env,
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        rendered = command if shell_mode else " ".join(command)
        raise RuntimeError(
            f"calc-descriptors failed (exit={completed.returncode}): {rendered}\n"
            f"{completed.stderr[-2000:]}"
        )


def encode_cfg_parallel(cfg_path, out_path, sus2_mlp_exe, mtp_path, encoding_cores=1, train_env=None):
    cfg_path = Path(cfg_path)
    out_path = Path(out_path)
    encoding_cores = max(1, int(encoding_cores))

    total_blocks = _count_cfg_blocks(cfg_path)
    if total_blocks == 0:
        out_path.write_text("", encoding="utf-8")
        return 0

    worker_count = min(encoding_cores, total_blocks)
    if worker_count == 1:
        _run_calc_descriptors(sus2_mlp_exe, mtp_path, cfg_path, out_path, train_env=train_env)
        return 1

    temp_dir = out_path.parent / f".{out_path.stem}.parts"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        part_cfg_paths = _split_cfg_by_blocks(cfg_path, temp_dir, out_path.stem, worker_count)
        part_out_paths = [path.with_suffix(".out") for path in part_cfg_paths]
        with ThreadPoolExecutor(max_workers=len(part_cfg_paths)) as executor:
            futures = [
                executor.submit(
                    _run_calc_descriptors,
                    sus2_mlp_exe,
                    mtp_path,
                    part_cfg,
                    part_out,
                    train_env,
                )
                for part_cfg, part_out in zip(part_cfg_paths, part_out_paths)
            ]
            for future in futures:
                future.result()

        with open(out_path, "w", encoding="utf-8") as merged:
            for part_out in part_out_paths:
                with open(part_out, "r", encoding="utf-8") as source:
                    shutil.copyfileobj(source, merged)
        return len(part_cfg_paths)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
