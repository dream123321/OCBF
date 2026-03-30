import os
import time

'''获取目录的最大深度'''
def get_directory_depth(directory):
    max_depth = 0
    for root, _, _ in os.walk(directory):
        depth = root[len(directory):].count(os.sep)  # 计算当前目录相对于基础目录的层级
        max_depth = max(max_depth, depth)
    return max_depth

'''获取npy和tpye的路径'''
def search_directories_in_directory(directory,depth):
    npy_path = []
    for root, dirs, files in os.walk(directory):
        temp_depth = root[len(directory):].count(os.sep)
        if temp_depth == depth:
            npy_path.append(root)
    return npy_path

def work_deepest_dir(pwd):
    work = os.path.join(pwd,'work')
    depth = get_directory_depth(work)
    calc_list = search_directories_in_directory(work, depth)
    return calc_list

def deepest_dir(pwd,dir):
    work = os.path.join(pwd,dir)
    depth = get_directory_depth(work)
    calc_list = search_directories_in_directory(work, depth)
    return calc_list

def scf_dir(pwd):
    work = os.path.join(pwd,'scf_lammps_data','scf','filter')
    dir_path = [os.path.join(work,a) for a in os.listdir(work) if a != 'time.txt']
    temp_list = []
    for sub_dir in dir_path:
        temp_list = [os.path.join(sub_dir,a) for a in os.listdir(sub_dir)] + temp_list
    return temp_list

def check_finish(dirs,logger,log):
    while True:
        count = 0
        for a in dirs:
            ok_file = os.path.join(a, '__ok__')
            if os.path.exists(ok_file):
                count += 1
        if count == len(dirs):
            logger.info(log)
            break
        time.sleep(10)

#与check_finish区别，不用while循环
def check_scf(pwd):
    count = 0
    dirs = scf_dir(pwd)
    for a in dirs:
        ok_file = os.path.join(a, '__ok__')
        if os.path.exists(ok_file):
            count += 1
    if count == len(dirs):
        return True
    else:
        return False

#每个结构用bsub提交，所以叫bsub_dir，即每个结构的主目录
def bsub_dir(pwd):
    work = os.path.join(pwd,'work')
    dirs = [os.path.join(work,f) for f in os.listdir(work) if os.path.isdir(os.path.join(work,f))]
    return dirs

def submit_lammps_task(pwd,logger,method):
    for dir in bsub_dir(pwd):
        os.chdir(dir)
        count_1 = 0
        name = os.path.basename(dir)
        for a in work_deepest_dir(pwd):
            ok_file = os.path.join(a, '__ok__')
            if not os.path.exists(ok_file):
                count_1 += 1
        if count_1 != 0:
            os.system(f'{method}')
            logger.info(f'{name}: Task is submitted')

#检测分歧阈值筛选出来的每个结构的xyz,是否数量为0,为0说明收敛
def check_filter_xyz_0(pwd):
    count = 0
    dirs_2 = bsub_dir(pwd)
    for dir in dirs_2:
        os.chdir(dir)
        if os.path.getsize('filter.xyz') == 0:
            count = count + 1
    if count == len(dirs_2):
        return False
    else:
        return True

def delete_dump(dirs):
    for dir in dirs:
        file = os.path.join(dir,'force.0.dump')
        if os.path.exists(file):
            os.remove(file)