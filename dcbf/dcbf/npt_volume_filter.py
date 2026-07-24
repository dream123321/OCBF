from __future__ import annotations

from bisect import bisect_right
import json
import math
from pathlib import Path
import warnings

from ase.io import read


DEFAULT_NPT_MAX_CELL_VOLUME_FILTER_FACTOR = 1.5
MIN_NPT_MAX_CELL_VOLUME_FILTER_FACTOR = 1.1
NPT_VOLUME_FILTER_REPORT = "npt_cell_volume_filter.json"
_ENSEMBLE_INFO_KEY = "_dcbf_md_ensemble"
_NPT_SEED_VOLUME_INFO_KEY = "_dcbf_npt_seed_volume"


def normalize_npt_max_cell_volume_filter_factor(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            "npt_max_cell_volume_filter_factor must be null or a finite number"
        )
    try:
        factor = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "npt_max_cell_volume_filter_factor must be null or a finite number"
        ) from exc
    if not math.isfinite(factor):
        raise ValueError(
            "npt_max_cell_volume_filter_factor must be null or a finite number"
        )
    if factor < MIN_NPT_MAX_CELL_VOLUME_FILTER_FACTOR:
        warnings.warn(
            f"npt_max_cell_volume_filter_factor={value} is below "
            f"{MIN_NPT_MAX_CELL_VOLUME_FILTER_FACTOR}; using default "
            f"{DEFAULT_NPT_MAX_CELL_VOLUME_FILTER_FACTOR}",
            RuntimeWarning,
        )
        return DEFAULT_NPT_MAX_CELL_VOLUME_FILTER_FACTOR
    return factor


def _finite_positive_volume(atoms, description):
    try:
        volume = float(atoms.get_volume())
    except Exception as exc:
        raise RuntimeError(f"Cannot read cell volume for {description}") from exc
    if not math.isfinite(volume) or volume <= 0.0:
        raise RuntimeError(
            f"Invalid cell volume for {description}: {volume!r}"
        )
    return volume


def _volume_within_factor(current_volume, seed_volume, factor):
    volume_factor = current_volume / seed_volume
    return volume_factor <= factor or math.isclose(
        volume_factor,
        factor,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )


def npt_seed_volume(case_dir):
    case_path = Path(case_dir)
    structure_name = case_path.parent.parent.name
    seed_path = case_path / f"{structure_name}.vasp"
    if not seed_path.is_file():
        raise RuntimeError(
            "NPT cell-volume filter cannot find the corresponding seed VASP: "
            f"{seed_path}"
        )
    try:
        seed_atoms = read(str(seed_path), index=0)
    except Exception as exc:
        raise RuntimeError(
            f"NPT cell-volume filter cannot read seed VASP: {seed_path}"
        ) from exc
    return _finite_positive_volume(seed_atoms, f"seed VASP {seed_path}")


def filter_selected_indices(
    atoms,
    case_dirs,
    case_structure_counts,
    selected_indices,
    factor,
):
    selected = [int(index) for index in selected_indices]
    if factor is None or not selected:
        return selected, {
            "original_selected_count": len(selected),
            "kept_count": len(selected),
            "removed_count": 0,
        }

    counts = [int(count) for count in case_structure_counts]
    if len(case_dirs) != len(counts):
        raise RuntimeError(
            "NPT cell-volume filter cannot map candidates: "
            "case directory and structure-count lengths differ"
        )
    if any(count < 0 for count in counts):
        raise RuntimeError(
            "NPT cell-volume filter cannot map candidates: "
            "negative case structure count"
        )

    cumulative_ends = []
    total = 0
    for count in counts:
        total += count
        cumulative_ends.append(total)
    if total != len(atoms):
        raise RuntimeError(
            "NPT cell-volume filter cannot map candidates: "
            f"case counts total {total}, but merged trajectory contains {len(atoms)} structures"
        )

    seed_volumes = {}
    kept = []
    for index in selected:
        if index < 0 or index >= total:
            raise RuntimeError(
                f"NPT cell-volume filter candidate index is out of range: {index}"
            )
        case_index = bisect_right(cumulative_ends, index)
        case_dir = Path(case_dirs[case_index])
        if case_dir.parent.name.lower() != "npt":
            kept.append(index)
            continue

        if case_index not in seed_volumes:
            seed_volumes[case_index] = npt_seed_volume(case_dir)
        current_volume = _finite_positive_volume(
            atoms[index],
            f"NPT candidate index {index} from {case_dir}",
        )
        if _volume_within_factor(
            current_volume,
            seed_volumes[case_index],
            factor,
        ):
            kept.append(index)

    return kept, {
        "original_selected_count": len(selected),
        "kept_count": len(kept),
        "removed_count": len(selected) - len(kept),
    }


def annotate_das_candidates(structures, case_dir):
    case_path = Path(case_dir)
    ensemble = case_path.parent.name.lower()
    if ensemble not in {"npt", "nvt"}:
        raise RuntimeError(
            "NPT cell-volume filter cannot identify MD ensemble from case path: "
            f"{case_path}"
        )
    seed_volume = npt_seed_volume(case_path) if ensemble == "npt" else None
    for atoms in structures:
        atoms.info[_ENSEMBLE_INFO_KEY] = ensemble
        if seed_volume is not None:
            atoms.info[_NPT_SEED_VOLUME_INFO_KEY] = seed_volume


def filter_annotated_das_atoms(atoms_list, factor):
    original_count = len(atoms_list)
    if factor is None:
        return list(atoms_list), {
            "original_selected_count": original_count,
            "kept_count": original_count,
            "removed_count": 0,
        }

    kept = []
    for index, atoms in enumerate(atoms_list):
        ensemble = atoms.info.pop(_ENSEMBLE_INFO_KEY, None)
        seed_volume = atoms.info.pop(_NPT_SEED_VOLUME_INFO_KEY, None)
        if ensemble not in {"npt", "nvt"}:
            raise RuntimeError(
                "NPT cell-volume filter cannot identify the source ensemble for "
                f"DAS candidate {index}; rerun this generation from MD selection"
            )
        if ensemble == "npt":
            if seed_volume is None:
                raise RuntimeError(
                    "NPT cell-volume filter is missing seed-volume metadata for "
                    f"DAS candidate {index}; rerun this generation from MD selection"
                )
            seed_volume = float(seed_volume)
            if not math.isfinite(seed_volume) or seed_volume <= 0.0:
                raise RuntimeError(
                    f"Invalid NPT seed volume for DAS candidate {index}: {seed_volume!r}"
                )
            current_volume = _finite_positive_volume(
                atoms,
                f"DAS NPT candidate {index}",
            )
            if not _volume_within_factor(current_volume, seed_volume, factor):
                continue
        kept.append(atoms)

    return kept, {
        "original_selected_count": original_count,
        "kept_count": len(kept),
        "removed_count": original_count - len(kept),
    }


def write_npt_volume_filter_report(workspace, factor, stats):
    report = {
        "enabled": factor is not None,
        "factor": factor,
        "original_selected_count": int(stats["original_selected_count"]),
        "kept_count": int(stats["kept_count"]),
        "removed_count": int(stats["removed_count"]),
    }
    report_path = Path(workspace) / NPT_VOLUME_FILTER_REPORT
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def load_npt_volume_filter_report(workspace):
    report_path = Path(workspace) / NPT_VOLUME_FILTER_REPORT
    if not report_path.is_file():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read NPT cell-volume filter report: {report_path}"
        ) from exc
