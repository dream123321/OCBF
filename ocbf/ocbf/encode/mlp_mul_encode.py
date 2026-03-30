import os
import concurrent.futures
from functools import partial
from .file_conversion import dump2cfg, merge_cfg_out
import subprocess

def main_dump2cfg(path, cfg_name):
    input = os.path.join(path,'force.0.dump')
    output = os.path.join(path, cfg_name)
    length = dump2cfg(input,output)
    return length

def mul_encode(pwd, mtp_path, dirs, cfg_name, out_name, sus2_mlp_exe):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #results = list(executor.map(main_dump2cfg, dirs))
        cfg_names = [cfg_name for _ in dirs]
        results = list(executor.map(main_dump2cfg,dirs,cfg_names))
    commands = []
    for a in dirs:
        md_cfg = os.path.join(a, cfg_name)
        md_out = os.path.join(a, out_name)
        commands.append(f'{sus2_mlp_exe} calc-descriptors {mtp_path} {md_cfg} {md_out}')
    # 创建进程列表
    processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]

    # 等待所有进程完成
    for process in processes:
        process.wait()

    merge_cfg_out(pwd, dirs, cfg_name, out_name)
    return results

if __name__ == '__main__':
    pwd = os.getcwd()
    dirs = ''
    mul_encode(pwd, mtp_path,dirs,cfg_name, out_name)