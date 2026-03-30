from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ase.io import iread, write
import yaml
from tqdm import tqdm

from .das.calc_ensemble_ambiguity import ambiguity_extract, check_lmp_error, get_force_ambiguity
from .das.das_update_ambiguity import (
    af_limit_record,
    af_limit_update,
    das_update_ambiguity,
    record_yaml,
)
from .das.logger import setup_logger
from .das.main_calc import check_and_modify_calc_dir, main_calc
from .das.mkdir import mkdir_vasp
from .das.other import end_yaml, mkdir, remove, touch
from .das.sample_xyz import sample_main
from .das.scf_lmp_data import scf_lammps_data
from .das.train_mlp import pre_train_mlp, start_train
from .das.work_dir import (
    bsub_dir,
    check_filter_xyz_0,
    check_finish,
    check_scf,
    deepest_dir,
    delete_dump,
    scf_dir,
    submit_lammps_task,
    work_deepest_dir,
)
from .encode.mlp_encode_sample_flow import main_sample_flow
from .bootstrap import WorkspaceBootstrapper
from .runtime_config import build_scheduler_spec, load_runtime_config


class GenerationRunner:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.logger = setup_logger()
        self.parameter = self._load_parameter_yaml()
        self.runtime_config = load_runtime_config(self.workspace)
        self.scheduler = build_scheduler_spec(self.runtime_config["scheduler"])
        self.scf2xyz = self._resolve_scf_handler()

    def _load_parameter_yaml(self):
        with open(self.workspace / "parameter.yaml", "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return WorkspaceBootstrapper.apply_parameter_defaults(data)

    def _resolve_scf_handler(self):
        scf_cal_engine = self.scheduler.scf_cal_engine
        if scf_cal_engine == "abacus":
            from .das.abacus_main_xyz import abacus_main_xyz as scf2xyz
        elif scf_cal_engine == "cp2k":
            from .das.cp2k_main_xyz import cp2k_main_xyz as scf2xyz
        elif scf_cal_engine == "qe":
            from .das.qe_main_xyz import qe_main_xyz as scf2xyz
        elif scf_cal_engine == "vasp":
            from .das.vasp_main_xyz import vasp_main_xyz as scf2xyz
        else:
            raise ValueError(f"{scf_cal_engine} does not exist")
        return scf2xyz

    @property
    def generation_index(self):
        return int(self.workspace.name.replace("gen_", ""))

    def _load_end_state(self):
        try:
            return end_yaml(str(self.workspace / "end.yaml"))
        except Exception:
            return None, None, None, None, None

    def _validate_training_outputs(self):
        gen_num = self.generation_index
        train_dirs = deepest_dir(str(self.workspace), "train_mlp")
        if gen_num == 0 and len(train_dirs) == 1:
            return

        for train_dir in train_dirs:
            log_file = os.path.join(train_dir, "logout")
            with open(log_file, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            for line in lines:
                if "Killed" in line:
                    self.logger.info(f"Scaling_mlp training error, please check! : {log_file}")
                    touch(str(self.workspace), "__error__")
                    raise RuntimeError(f"Scaling_mlp training error: {log_file}")
            for line in lines[-25:]:
                if "nan" in line:
                    self.logger.info(f"Scaling_mlp training error, please check! : {log_file}")
                    touch(str(self.workspace), "__error__")
                    raise RuntimeError(f"Scaling_mlp training error: {log_file}")

    def _train_models(self):
        ele = self.parameter["ele"]
        ele_model = 1 if self.parameter["sort_ele"] else 2
        mlp_nums = 1 if self.parameter["mlp_encode_model"] else self.parameter["mlp_nums"]
        if not self.parameter["mlp_encode_model"] and mlp_nums < 3:
            raise ValueError("During the use of DAS, mlp_nums should be greater than or equal to 3")

        label = pre_train_mlp(
            str(self.workspace),
            mlp_nums,
            ele,
            ele_model,
            self.logger,
            self.scheduler,
        )
        if label:
            start_train(str(self.workspace), self.scheduler.task_submission_method, mlp_nums, self.logger)
        self._validate_training_outputs()

    def _prepare_md(self):
        mkdir_vasp(
            str(self.workspace),
            self.parameter["mlp_MD"],
            self.parameter["ele"],
            tuple(eval(self.parameter["size"])),
            1 if self.parameter["mlp_encode_model"] else self.parameter["mlp_nums"],
            self.parameter["sort_ele"],
            self.parameter["nvt_lattice_scaling_factor"],
            self.parameter["mlp_encode_model"],
            self.scheduler,
        )
        dirs_1 = work_deepest_dir(str(self.workspace))
        dirs_2 = bsub_dir(str(self.workspace))
        submit_lammps_task(str(self.workspace), self.logger, self.scheduler.task_submission_method)
        check_finish(dirs_1, self.logger, "All MD calculations have been completed")
        return dirs_1, dirs_2

    def _select_by_ambiguity(self, dirs_1, dirs_2, end_state):
        end_threshold_low, end_threshold_high, end_n, end_cluster_threshold_init, end_k = end_state
        threshold_low = self.parameter["threshold_low"]
        threshold_high = self.parameter["threshold_high"]
        sample = self.parameter["sample"]
        n = sample["n"]
        cluster_threshold_init = sample["cluster_threshold_init"]
        k = sample["k"]

        last_gen = "gen_" + str(self.generation_index - 1)
        last_gen_path = self.workspace.parent / last_gen
        mtp_path = self.workspace / "mtp"
        last_scf_filter_xyz = last_gen_path / "scf_lammps_data" / "scf_filter.xyz"

        if end_threshold_low != threshold_low or end_threshold_high != threshold_high:
            for directory in dirs_2:
                os.chdir(directory)
                for file_name in glob.glob("*filter*"):
                    remove(file_name)

            af_adaptive = None
            if self.parameter["das_ambiguity"]:
                if self.generation_index == 0:
                    xyz = None
                    model_fns = None
                else:
                    xyz = str(last_scf_filter_xyz)
                    model_fns = glob.glob(os.path.join(mtp_path, "current*"))
                yaml_file = os.path.join(self.workspace, "parameter.yaml")
                new_af_limit = af_limit_update(str(self.workspace), yaml_file)
                af_adaptive, label = das_update_ambiguity(
                    ele=self.parameter["ele"],
                    sort_ele=self.parameter["sort_ele"],
                    af_default=self.parameter["af_default"],
                    af_limit=new_af_limit,
                    af_failed=self.parameter["af_failed"],
                    over_fitting_factor=self.parameter["over_fitting_factor"],
                    logger=self.logger,
                ).run(xyz, model_fns, self.generation_index)
                af_limit_record(str(self.workspace), label)
                threshold_low = float(af_adaptive)
                threshold_high = self.parameter["af_failed"]

            select_stru_num = 0
            for directory_name in tqdm(dirs_1):
                directory = os.path.join(self.workspace, directory_name)
                total_stru = get_force_ambiguity(directory)
                num, structures, interval, hist = ambiguity_extract(
                    directory,
                    "force.0.dump",
                    "af.out",
                    threshold_low,
                    threshold_high,
                    self.parameter["ele"],
                    self.parameter["sort_ele"],
                    self.parameter["end"],
                    self.parameter["num_elements"],
                )
                path_parts = os.path.normpath(directory).split(os.sep)
                final_path = os.sep.join(path_parts[-3:])
                filter_path = os.path.join(os.sep.join(path_parts[:-2]), "filter.xyz")
                write(filter_path, structures, format="extxyz", append=True)
                self.logger.info(
                    f"{final_path}: According to the ambiguity in the {round(threshold_low, 3)}-{threshold_high} range , "
                    f"{num} structures are selected from {total_stru} structures. Interval:{interval} Statistical number:{hist}"
                )
                select_stru_num += num
                error, message = check_lmp_error(directory)
                if error:
                    self.logger.warning(message)

            yaml_file = os.path.join(self.workspace, "parameter.yaml")
            adaptive_value = float(af_adaptive) if af_adaptive is not None else None
            record_yaml(yaml_file, adaptive_value, int(select_stru_num))
        else:
            self.logger.info("end_threshold equals threshold: skip to select the structure by ambiguity")

        if (
            end_threshold_low == threshold_low
            and end_threshold_high == threshold_high
            and end_n == n
            and end_cluster_threshold_init == cluster_threshold_init
            and end_k == k
        ):
            self.logger.info("(threshold, n, cluster_threshold_init, k) parameters are equal: skip to select the structure by MBTR+Brich")
            return

        for directory in dirs_2:
            os.chdir(directory)
            if os.path.getsize("filter.xyz") != 0:
                num = len(list(iread("filter.xyz")))
                if n * k <= num:
                    select, total = sample_main(
                        os.getcwd(),
                        n=n,
                        threshold_init=cluster_threshold_init,
                        k=k,
                        clustering_by_ambiguity=sample["clustering_by_ambiguity"],
                    )
                    name = os.path.basename(directory)
                    self.logger.info(f"{name}: selected {select} structures from {total} structures in data by MBTR+Brich.")
                else:
                    shutil.copy("filter.xyz", f"{num}_sample_filter.xyz")

    def _select_by_encoding(self, dirs_1):
        sample_xyz_list = glob.glob(os.path.join(self.workspace, "work", "*_sample_filter.xyz"))
        if len(sample_xyz_list) == 0:
            main_sample_flow(
                str(self.workspace),
                dirs_1,
                self.parameter["iw"],
                self.parameter["iw_method"],
                self.parameter.get("iw_scale", 1.0),
                self.parameter["body_list"],
                self.parameter["ele"],
                self.parameter["sort_ele"],
                self.parameter["mtp_type"],
                self.parameter["selection_budget_schedule"],
                self.parameter["coverage_threshold_schedule"],
                self.parameter["coverage_rate_method"],
                self.logger,
                self.parameter.get("coverage_calculation_mode", "per_configuration"),
                self.parameter.get("report_per_configuration_details", False),
            )
        elif len(sample_xyz_list) == 1:
            self.logger.info(f"*_sample_filter.xyz already exists.({sample_xyz_list[0]})")
        else:
            raise ValueError("Multiple *_sample_filter.xyz, Please delete!")

    def _persist_end_state(self):
        shutil.copy(self.workspace / "parameter.yaml", self.workspace / "end.yaml")

    def _submit_scf_jobs(self, total_atom_list):
        scf_lammps_data_path = self.workspace / "scf_lammps_data"
        scf_path = scf_lammps_data_path / "scf"
        mkdir(str(scf_lammps_data_path))
        mkdir(str(scf_path))
        os.chdir(scf_path)
        calc_dir_num = check_and_modify_calc_dir(str(self.workspace), total_atom_list, self.parameter["dft"]["calc_dir_num"])
        num = main_calc(total_atom_list, calc_dir_num, str(self.workspace), self.scheduler)
        if not check_scf(str(self.workspace)):
            subprocess.run([sys.executable, "start_calc.py"], check=True)
        self.logger.info(f"The {num} structures are divided into {calc_dir_num} dft calculation tasks to be submitted")

    def _collect_scf_results(self, force_threshold):
        current = self.workspace / "scf_lammps_data" / "scf" / "filter"
        out_name = self.workspace / "scf_lammps_data" / "scf_filter.xyz"
        ori_out_name = self.workspace / "scf_lammps_data" / "ori_scf_filter.xyz"
        remove(str(out_name))
        return self.scf2xyz(str(current), str(out_name), str(ori_out_name), force_threshold)

    def _run_scf_stage_without_encoding(self, end_state):
        end_threshold_low, end_threshold_high, end_n, end_cluster_threshold_init, end_k = end_state
        sample = self.parameter["sample"]
        if not check_filter_xyz_0(str(self.workspace)):
            self.logger.info("The active learning loop ends")
            touch(str(self.workspace), "__end__")
            return

        if (
            end_threshold_low == self.parameter["threshold_low"]
            and end_threshold_high == self.parameter["threshold_high"]
            and end_n == sample["n"]
            and end_cluster_threshold_init == sample["cluster_threshold_init"]
            and end_k == sample["k"]
            and self.parameter["dft"]["calc_dir_num"]
        ):
            self.logger.info("(calc_dir_num, threshold, n, cluster_threshold_init, k) parameters are equal: skip scf calculations")
        else:
            total_atom_list = scf_lammps_data("no", str(self.workspace))
            scf_path = self.workspace / "scf_lammps_data" / "scf"
            os.chdir(scf_path)
            for item in [name for name in os.listdir() if name != "total_sample_filter.xyz"]:
                remove(item)
            self._submit_scf_jobs(total_atom_list)

        self.logger.info("In the process of checking whether the SCF calculation is complete......")
        check_finish(scf_dir(str(self.workspace)), self.logger, "All scf calculations have been completed")

        ok_count, len_count, no_success_path, force_count, _ = self._collect_scf_results(self.parameter["dft"]["force_threshold"])
        with open(self.workspace / "no_success_path.json", "w", encoding="utf-8") as handle:
            json.dump(no_success_path, handle)
        gen = self.workspace.name
        self.logger.info(
            f"Active learning continues: {gen} | {self.scheduler.scf_cal_engine}_completed_number:{ok_count} | "
            f"Successful_collection_structure/scf_convergent_number:{len_count} | "
            f"force_threshold_number({self.parameter['dft']['force_threshold']}):{force_count}"
        )

    def _run_scf_stage_with_encoding(self):
        sample_xyz_list = glob.glob(os.path.join(self.workspace, "work", "*_sample_filter.xyz"))
        if len(sample_xyz_list) == 0:
            self.logger.info("The active learning loop ends")
            touch(str(self.workspace), "__end__")
            return

        sample_xyz = sample_xyz_list[0]
        if os.path.getsize(sample_xyz) == 0:
            self.logger.info("No structures were selected for SCF. The active learning loop ends")
            touch(str(self.workspace), "__end__")
            return

        total_atom_list = list(iread(sample_xyz))
        scf_lammps_data_path = self.workspace / "scf_lammps_data"
        scf_path = self.workspace / "scf_lammps_data" / "scf"
        mkdir(str(scf_lammps_data_path))
        mkdir(str(scf_path))
        os.chdir(scf_path)
        for item in [name for name in os.listdir() if name != "total_sample_filter.xyz"]:
            remove(item)
        self._submit_scf_jobs(total_atom_list)

        self.logger.info("In the process of checking whether the SCF calculation is complete......")
        check_finish(scf_dir(str(self.workspace)), self.logger, "All scf calculations have been completed")

        ok_count, len_count, no_success_path, force_count, force_of_force_count_0 = self._collect_scf_results(
            self.parameter["dft"]["force_threshold"]
        )
        with open(self.workspace / "no_success_path.json", "w", encoding="utf-8") as handle:
            json.dump(no_success_path, handle)

        gen = self.workspace.name
        if force_count == 0:
            self.logger.info(
                f"Active learning continues: {gen} | {self.scheduler.scf_cal_engine}_completed_number:{ok_count} | "
                f"Successful_collection_structure/scf_convergent_number:{len_count} | "
                f"force_threshold_number({self.parameter['dft']['force_threshold']}):{force_count} | "
                f"minimum_max_force:{force_of_force_count_0}"
            )
        else:
            self.logger.info(
                f"Active learning continues: {gen} | {self.scheduler.scf_cal_engine}_completed_number:{ok_count} | "
                f"Successful_collection_structure/scf_convergent_number:{len_count} | "
                f"force_threshold_number({self.parameter['dft']['force_threshold']}):{force_count}"
            )

    def run(self):
        end_state = self._load_end_state()
        self._train_models()
        dirs_1, dirs_2 = self._prepare_md()
        if self.parameter["mlp_encode_model"]:
            self._select_by_encoding(dirs_1)
            self._persist_end_state()
            self._run_scf_stage_with_encoding()
        else:
            self._select_by_ambiguity(dirs_1, dirs_2, end_state)
            self._persist_end_state()
            self._run_scf_stage_without_encoding(end_state)
        self._persist_end_state()
        delete_dump(dirs_1)
        touch(str(self.workspace), "__ok__")
