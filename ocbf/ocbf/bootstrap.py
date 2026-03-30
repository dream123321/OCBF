from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil

from ase.data import atomic_numbers, chemical_symbols
from ase.io import iread, write
import yaml

from .das.file_conversion import cfg2xyz
from .mtp import normalize_mtp_type
from .runtime_config import load_json_config, normalize_scheduler_config, save_runtime_config


class WorkspaceBootstrapper:
    LEGACY_INIT_FILES = ("sub_loop.py", "old_parameter.yaml", "bsub_script.py", "l2k2.mtp", "l2k3.mtp", "original.cfg", "original.mtp")
    SUBMISSION_BACKENDS = {
        "bsub": {
            "task_submission_method": "bsub<bsub.lsf",
            "start_calc_command": "bsub<",
        },
        "sbatch": {
            "task_submission_method": "sbatch bsub.lsf",
            "start_calc_command": "sbatch",
        },
    }
    DEFAULT_PARAMETER_VALUES = {
        "mlp_nums": 3,
        "size": "(1, 1, 1)",
        "sort_ele": True,
        "nvt_lattice_scaling_factor": [1],
        "das_ambiguity": True,
        "af_default": 0.01,
        "af_limit": 0.2,
        "af_failed": 0.5,
        "over_fitting_factor": 1.1,
        "af_adaptive": None,
        "threshold_low": 0.08,
        "threshold_high": 0.3,
        "select_stru_num": None,
        "end": 1,
        "num_elements": 6,
        "sample": {
            "n": 5,
            "cluster_threshold_init": 0.5,
            "k": 2,
            "clustering_by_ambiguity": True,
        },
        "mlp_encode_model": True,
        "encoding_cores": 2,
        "iw_method": "Freedman_Diaconis",
        "iw": 0.01,
        "iw_scale": 1.0,
        "body_list": ["two", "three"],
        "mtp_type": "l2k2",
        "selection_budget_schedule": [20, 15, 10],
        "coverage_threshold_schedule": [99.5, 99.9, 99.95],
        "coverage_rate_method": "mean",
        "coverage_calculation_mode": "per_configuration",
        "report_per_configuration_details": True,
        "dft": {
            "calc_dir_num": 5,
            "force_threshold": 20,
        },
    }

    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config_dir = self.config_path.parent
        self.config = load_json_config(self.config_path)

    @staticmethod
    def _coerce_bool(value, default=True):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)

    @staticmethod
    def resolve_path(base_dir, raw_path):
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (base_dir / path).resolve()

    def prepare_workspace(self):
        dataset = self.config["dataset"]
        run_dir = self.resolve_path(self.config_dir, self.config.get("run_dir", "."))
        init_source_dir = self.resolve_path(self.config_dir, dataset.get("init_source_dir", "init"))
        init_dir = run_dir / "init"
        init_dir.mkdir(parents=True, exist_ok=True)
        if init_source_dir.resolve() != init_dir.resolve():
            shutil.copytree(init_source_dir, init_dir, dirs_exist_ok=True)

        for legacy_name in self.LEGACY_INIT_FILES:
            legacy_path = init_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

        xyz_input = self.resolve_path(self.config_dir, dataset["xyz_input"])
        elements = self.infer_elements_from_xyz(xyz_input)
        mtp_type = self.write_parameter_yaml(init_dir, elements, xyz_input)
        runtime_config = dict(self.config)
        runtime_config["scheduler"] = self.normalize_scheduler_keys(dict(self.config["scheduler"]))
        save_runtime_config(run_dir, runtime_config)
        return run_dir, xyz_input, elements, mtp_type

    @staticmethod
    def infer_elements_from_xyz(xyz_input):
        element_set = set()
        for atoms in iread(str(xyz_input)):
            element_set.update(atoms.get_chemical_symbols())
        ordered_atomic_numbers = sorted(atomic_numbers[element] for element in element_set)
        return [chemical_symbols[number] for number in ordered_atomic_numbers]

    def write_parameter_yaml(self, init_dir, elements, xyz_input):
        raw_parameter = dict(self.config["parameter"])
        legacy_keys = [key for key in ("ele_model", "bw_method", "bw", "bw_coff", "stru_num", "coverage_rate_threshold") if key in raw_parameter]
        if legacy_keys:
            raise ValueError(
                "Legacy parameter keys are no longer supported: "
                f"{legacy_keys}. Use sort_ele / selection_budget_schedule / coverage_threshold_schedule / iw_method / iw / iw_scale instead."
            )
        parameter = self.apply_parameter_defaults(raw_parameter)
        parameter.pop("init_threshold", None)
        parameter.pop("threshold_coff", None)
        parameter["ele"] = elements
        parameter["sort_ele"] = self._coerce_bool(parameter.get("sort_ele", True), default=True)
        parameter["encoding_cores"] = int(parameter.get("encoding_cores", 2))
        parameter["iw"] = float(parameter.get("iw", 0.01))
        parameter["iw_scale"] = float(parameter.get("iw_scale", 1.0))
        parameter["dataset_xyz_input"] = str(xyz_input)
        parameter["mtp_type"] = normalize_mtp_type(parameter.get("mtp_type", "l2k2"))
        mtp_type = parameter["mtp_type"]
        with open(init_dir / "parameter.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(parameter, handle, default_flow_style=False, sort_keys=False)
        return mtp_type

    @classmethod
    def normalize_parameter_keys(cls, parameter):
        normalized = dict(parameter)
        if "ele_model" in normalized and "sort_ele" not in normalized:
            normalized["sort_ele"] = int(normalized["ele_model"]) == 1
        if "bw_method" in normalized and "iw_method" not in normalized:
            normalized["iw_method"] = normalized["bw_method"]
        if "bw" in normalized and "iw" not in normalized:
            normalized["iw"] = normalized["bw"]
        if "bw_coff" in normalized and "iw_scale" not in normalized:
            normalized["iw_scale"] = normalized["bw_coff"]
        if "stru_num" in normalized and "selection_budget_schedule" not in normalized:
            normalized["selection_budget_schedule"] = normalized["stru_num"]
        if "coverage_rate_threshold" in normalized and "coverage_threshold_schedule" not in normalized:
            normalized["coverage_threshold_schedule"] = normalized["coverage_rate_threshold"]
        return normalized

    @classmethod
    def apply_parameter_defaults(cls, parameter):
        normalized = cls.normalize_parameter_keys(dict(parameter))
        for key, default_value in cls.DEFAULT_PARAMETER_VALUES.items():
            if isinstance(default_value, dict):
                merged = copy.deepcopy(default_value)
                merged.update(normalized.get(key) or {})
                normalized[key] = merged
            elif key not in normalized:
                normalized[key] = copy.deepcopy(default_value)
        return normalized

    @classmethod
    def normalize_scheduler_keys(cls, scheduler):
        return normalize_scheduler_config(scheduler)

    @staticmethod
    def export_final_xyz(run_dir, config):
        workflow = config["workflow"]
        if not workflow.get("output_xyz", True):
            return

        main_list = [item for item in os.listdir(run_dir) if item.startswith("main_")]
        if not main_list:
            return
        main_list = sorted(main_list, key=lambda item: int(item.replace("main_", "")))
        last_main = main_list[-1]

        gen_dir = Path(run_dir) / last_main
        gen_list = [item for item in os.listdir(gen_dir) if item.startswith("gen_")]
        gen_list = sorted(gen_list, key=lambda item: int(item.replace("gen_", "")))
        last_gen = gen_list[-1]

        output_name = workflow.get("output_xyz_name", "all_sample_data.xyz")
        output_path = Path(run_dir) / output_name
        if output_path.exists():
            output_path.unlink()

        parameter_path = Path(run_dir) / "init" / "parameter.yaml"
        with open(parameter_path, "r", encoding="utf-8") as handle:
            yaml_data = yaml.safe_load(handle)
        ele = yaml_data["ele"]
        sort_ele = WorkspaceBootstrapper._coerce_bool(yaml_data.get("sort_ele", True), default=True)

        train_cfg = Path(run_dir) / last_main / last_gen / "train_mlp" / "train.cfg"
        cfg2xyz(ele, sort_ele, str(train_cfg), str(output_path))
        label = config["dataset"].get("all_label")
        if label:
            atoms = []
            for atom in list(iread(output_path)):
                atom.info["label"] = label
                atoms.append(atom)
            write(output_path, atoms, format="extxyz")
