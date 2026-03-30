import os
import glob
import sys
import shutil
from pathlib import Path
import yaml
from .file_conversion import xyz2cfg, merge_cfg
from .work_dir import deepest_dir,check_finish
from ..mtp import write_mtp_template

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def check(file):
    if not os.path.exists(file):
        sys.exit(f'{file} train.cfg no exists')

def train_mlp_lsf(current_num,out,COMMAND,pwd, scheduler):
    str = f"""#!/bin/bash
{scheduler.bsub_script_train_sus_job_name}{current_num}
"""
    str = str + scheduler.bsub_script_train_sus

    content = f"""
start_time=$(date +%s.%N)
touch __start__
$COMMAND_std > logout 2>&1
end_time=$(date +%s.%N)
touch __ok__
runtime=$(echo "$end_time - $start_time" | bc)
echo "total_runtime：$runtime s" >> time.txt"""
    COMMAND_std = 'COMMAND_std="' + COMMAND + f'../../mtp/current_{current_num}.mtp"'
    total = str + COMMAND_std+content
    with open(out,'w') as file:
        file.write(total)

def pre_train_mlp(pwd, mlp_nums, ele, ele_model,logger, scheduler):
    mtp_path = os.path.join(pwd,'mtp')
    mkdir(mtp_path)
    os.chdir(mtp_path)
    names = glob.glob('current*')

    if len(names) != mlp_nums:
        train_mlp_path = os.path.join(pwd,'train_mlp')
        mkdir(train_mlp_path)

        gen_num = os.path.basename(pwd).replace('gen_','')
        main_dir = os.path.dirname(os.path.dirname(pwd))
        main_num = os.path.basename(os.path.dirname(pwd)).replace('main_','')

        if int(gen_num) == 0 and int(main_num) != 0:
            last_main_num = 'main_' + str((int(main_num) - 1))
            last_main_path = os.path.join(main_dir, last_main_num)
            last_main_gen_num = sorted([int(a.replace('gen_','')) for a in os.listdir(last_main_path) if not a.endswith('.pkl')])[-1]
            last_main_mtp_path = os.path.join(last_main_path, 'gen_'+str(last_main_gen_num),'mtp')
            last_main_train_path = os.path.join(last_main_path, 'gen_'+str(last_main_gen_num),'train_mlp','train.cfg')
            shutil.copy(last_main_train_path, train_mlp_path)
            for a in os.listdir(last_main_mtp_path):
                shutil.copy(os.path.join(last_main_mtp_path,a),mtp_path)
            logger.info(f'main_{main_num} | gen_0: start')
            return False
        elif int(gen_num) == 0 and int(main_num) == 0:
            parameter_yaml = os.path.join(main_dir, 'init', 'parameter.yaml')
            with open(parameter_yaml, 'r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle)
            xyz2cfg(ele, ele_model, data['dataset_xyz_input'], os.path.join(train_mlp_path, 'train.cfg'))
            for i in range(mlp_nums):
                sub_train_mlp_path = os.path.join(train_mlp_path, str(i))
                mkdir(sub_train_mlp_path)
                os.chdir(sub_train_mlp_path)
                write_mtp_template(Path('hyx.mtp'), data['mtp_type'], len(ele))
                train_mlp_lsf(i, 'bsub.lsf',scheduler.original_COMMAND,pwd, scheduler)
                #os.system('dos2unix bsub.lsf')
            logger.info(f'main_0 | gen_0: start')
            return True
        else:
            for i in range(mlp_nums):
                sub_train_mlp_path = os.path.join(train_mlp_path, str(i))
                mkdir(sub_train_mlp_path)
                os.chdir(sub_train_mlp_path)
                train_mlp_lsf(i, 'bsub.lsf',scheduler.subsequent_COMMAND,pwd, scheduler)
                #os.system('dos2unix bsub.lsf')
            last_gen = 'gen_'+str((int(gen_num)-1))
            last_gen_path = os.path.join(os.path.dirname(pwd),last_gen)
            last_train_cfg = os.path.join(last_gen_path,'train_mlp','train.cfg')
            last_mtp_path = os.path.join(last_gen_path,'mtp')
            last_scf_filter_xyz = os.path.join(last_gen_path,'scf_lammps_data','scf_filter.xyz')
            os.chdir(last_mtp_path)
            for a in glob.glob('current*'):
                current_num = a.split('_')[1].split('.')[0]
                shutil.copy(a,os.path.join(train_mlp_path,str(current_num),'hyx.mtp'))
            check(last_train_cfg)

            scf_filter_cfg = os.path.join(train_mlp_path,'scf_filter.cfg')

            xyz2cfg(ele, ele_model, last_scf_filter_xyz, scf_filter_cfg)
            merge_cfg(last_train_cfg, scf_filter_cfg, os.path.join(train_mlp_path,'train.cfg'))
            logger.info(f'main_{main_num} | gen_{gen_num}: start')
            return True
    else:
        dirs = deepest_dir(pwd, 'train_mlp')
        if len(dirs) != 1:
            check_finish(dirs, logger, 'All training_mlp tasks have been completed')
        else:
            logger.info(f'mtp already exists')
        return False

def start_train(pwd,task_submission_method, mlp_nums, logger):
    dirs = deepest_dir(pwd,'train_mlp')
    for a in dirs:
        if not os.path.exists(os.path.join(a,'__start__')):
            os.chdir(a)
            os.system(f"{task_submission_method}")
            logger.info(f'mlp_{os.path.basename(a)} has been submitted')
        else:
            logger.info(f'mlp_{os.path.basename(a)} is already in training')
    check_finish(dirs, logger, 'All training_mlp tasks have been completed')

    mtp_path = os.path.join(pwd, 'mtp')
    mkdir(mtp_path)
    os.chdir(mtp_path)
    names = glob.glob('current*')
    if len(names) != mlp_nums:
        sys.exit(f'Please check the training process! Successful training={len(names)}')
        logger.error(f'Please check the training process! Successful training={len(names)}')
if __name__=='__main__':
    path =''
    train_mlp(path, 3, 'taiying')
