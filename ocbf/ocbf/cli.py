import argparse

from .das.gen_while_loop import check_run_position, gen_while_loop, mkdir

from .bootstrap import WorkspaceBootstrapper
from .generation import GenerationRunner
from .reduce import OCBFReducer
from .selection.benchmark import SelectionBenchmark
from .selection.calibrate import SelectionCalibrator


class OCBFApplication:
    def run_from_config(self, config_path, prepare_only=False):
        bootstrapper = WorkspaceBootstrapper(config_path)
        run_dir, _, _, _ = bootstrapper.prepare_workspace()
        if prepare_only:
            return run_dir

        workflow = bootstrapper.config["workflow"]
        parameter = WorkspaceBootstrapper.apply_parameter_defaults(dict(bootstrapper.config.get("parameter", {})))
        main_loop_npt = workflow.get("main_loop_npt")
        main_loop_nvt = workflow.get("main_loop_nvt")
        selected_loop = self._select_loop_reference(main_loop_npt, main_loop_nvt)

        start_position, gen_num, main_num = check_run_position(str(run_dir), selected_loop)
        mode = workflow.get("mode", "full-automatic")
        sleep_time = workflow.get("sleep_time", 10)
        max_gen = workflow.get("max_gen", 10)

        if mode == "semi-automatic":
            for index in range(main_num, len(selected_loop)):
                npt = main_loop_npt[index] if main_loop_npt is not None and index < len(main_loop_npt) else None
                nvt = main_loop_nvt[index] if main_loop_nvt is not None and index < len(main_loop_nvt) else None
                gen_while_loop(str(run_dir), npt, nvt, start_position, gen_num, sleep_time, max_gen)
        elif mode == "full-automatic":
            for index in range(main_num, len(selected_loop)):
                npt = main_loop_npt[index] if main_loop_npt is not None and index < len(main_loop_npt) else None
                nvt = main_loop_nvt[index] if main_loop_nvt is not None and index < len(main_loop_nvt) else None
                start_position, gen_num, new_main_num = check_run_position(str(run_dir), selected_loop)
                gen_while_loop(str(run_dir), npt, nvt, start_position, gen_num, sleep_time, max_gen)
                if new_main_num < len(selected_loop) - 1:
                    next_path = run_dir / ("main_" + str(new_main_num + 1)) / "gen_0"
                    mkdir(str(next_path))
        else:
            raise ValueError("workflow.mode must be 'semi-automatic' or 'full-automatic'")

        WorkspaceBootstrapper.export_final_xyz(str(run_dir), bootstrapper.config)
        return run_dir

    @staticmethod
    def _select_loop_reference(main_loop_npt, main_loop_nvt):
        if main_loop_npt is not None:
            return main_loop_npt
        if main_loop_nvt is not None:
            return main_loop_nvt
        raise ValueError("workflow.main_loop_npt and workflow.main_loop_nvt cannot both be null")

    @staticmethod
    def run_generation(workspace):
        runner = GenerationRunner(workspace)
        runner.run()
        return runner.workspace

    @staticmethod
    def benchmark_selection(output_path):
        benchmark = SelectionBenchmark()
        output_path, results = benchmark.write_report(output_path)
        return output_path, results

    @staticmethod
    def calibrate_selection(output_path, cases):
        calibrator = SelectionCalibrator()
        output_path, results = calibrator.write_report(output_path, cases=cases)
        return output_path, results

    @staticmethod
    def reduce_from_config(config_path):
        reducer = OCBFReducer(config_path)
        return reducer.run()


def build_parser():
    parser = argparse.ArgumentParser(prog="ocbf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="prepare and launch an OCBF workflow from one JSON file")
    run_parser.add_argument("config", help="path to the JSON config file")
    run_parser.add_argument("--prepare-only", action="store_true", help="only materialize the workspace files")

    generation_parser = subparsers.add_parser("run-generation", help="run one generated workspace iteration")
    generation_parser.add_argument("--workspace", default=".", help="generation workspace path")

    reduce_parser = subparsers.add_parser("reduce", help="reduce redundant database structures from one JSON file")
    reduce_parser.add_argument("config", help="path to the JSON config file")

    benchmark_parser = subparsers.add_parser("benchmark-selection", help="profile the structure-selection hot path")
    benchmark_parser.add_argument(
        "--output",
        default="benchmark_outputs/selection_profile.json",
        help="where to write the JSON benchmark report",
    )

    calibrate_parser = subparsers.add_parser("calibrate-selection", help="verify exact equality with the original min_cover")
    calibrate_parser.add_argument(
        "--output",
        default="benchmark_outputs/selection_calibration.json",
        help="where to write the JSON calibration report",
    )
    calibrate_parser.add_argument(
        "--cases",
        type=int,
        default=120,
        help="number of randomized calibration cases",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    app = OCBFApplication()
    if args.command == "run":
        app.run_from_config(args.config, prepare_only=args.prepare_only)
        return 0
    if args.command == "run-generation":
        app.run_generation(args.workspace)
        return 0
    if args.command == "benchmark-selection":
        app.benchmark_selection(args.output)
        return 0
    if args.command == "calibrate-selection":
        app.calibrate_selection(args.output, args.cases)
        return 0
    if args.command == "reduce":
        app.reduce_from_config(args.config)
        return 0
    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
