import os
import concurrent.futures
import shlex
from .file_conversion import dump2cfg, merge_cfg_out
import subprocess


def main_dump2cfg(path, cfg_name):
    input_path = os.path.join(path, 'force.0.dump')
    output_path = os.path.join(path, cfg_name)
    length = dump2cfg(input_path, output_path)
    return length


def _build_calc_descriptors_shell(sus2_mlp_exe, mtp_path, md_cfg, md_out, train_env=None):
    quoted = " ".join(
        shlex.quote(item)
        for item in [str(sus2_mlp_exe), 'calc-descriptors', str(mtp_path), str(md_cfg), str(md_out)]
    )
    train_env = str(train_env or '').strip()
    if train_env:
        return f"{train_env}\n{quoted}"
    return quoted


def mul_encode(pwd, mtp_path, dirs, cfg_name, out_name, sus2_mlp_exe, train_env=None):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        cfg_names = [cfg_name for _ in dirs]
        results = list(executor.map(main_dump2cfg, dirs, cfg_names))

    commands = []
    for path in dirs:
        md_cfg = os.path.join(path, cfg_name)
        md_out = os.path.join(path, out_name)
        commands.append(_build_calc_descriptors_shell(sus2_mlp_exe, mtp_path, md_cfg, md_out, train_env=train_env))

    processes = [subprocess.Popen(['bash', '-lc', cmd]) for cmd in commands]

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"calc-descriptors failed during mul_encode with exit code {process.returncode}")

    merge_cfg_out(pwd, dirs, cfg_name, out_name)
    return results


if __name__ == '__main__':
    pwd = os.getcwd()
    dirs = ''
    mul_encode(pwd, mtp_path, dirs, cfg_name, out_name)
