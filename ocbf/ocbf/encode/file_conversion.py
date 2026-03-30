from ase.io import iread
from ase.data import atomic_numbers,chemical_symbols
import os
from tqdm import tqdm

def _ordered_elements(elements, sort_ele):
    if sort_ele:
        atomic_order = sorted(atomic_numbers[item] for item in elements)
        return [chemical_symbols[number] for number in atomic_order]
    return list(elements)


def xyz2cfg(elements, sort_ele, input_path, output_path):
    fin = iread(input_path)
    map_dic = {}
    ordered_elements = _ordered_elements(elements, sort_ele)
    for index, element in enumerate(ordered_elements):
        map_dic.update({element: index})

    b=list(fin)
    ff = open(output_path,"w")
    for i in range(len(b)):
        #print(i)
        atoms=b[i]
        ele=atoms.get_chemical_symbols()
        nat=len(ele)
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        force= atoms.get_forces()
        en=atoms.get_potential_energy()
        virial=atoms.info['virial']
        ff.write("""BEGIN_CFG\n""")
        ff.write(""" Size\n""")
        ff.write("""  {:6} \n""".format(nat))
        ff.write(""" Supercell \n""")
        ff.write("""{:15.10f} {:15.10f} {:15.10f}\n""".format(cell[0, 0], cell[0, 1], cell[0, 2]))
        ff.write("""{:15.10f} {:15.10f} {:15.10f}\n""".format(cell[1, 0], cell[1, 1], cell[1, 2]))
        ff.write("""{:15.10f} {:15.10f} {:15.10f}\n""".format(cell[2, 0], cell[2, 1], cell[2, 2]))
        ff.write("""AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n""")
        for i in range(nat):
            ff.write(
                """ {:6} {:6} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n""".format(i + 1, map_dic[ele[i]], pos[i, 0], pos[i, 1], pos[i, 2],
                                                                                             force[i,0], force[i,1], force[i,2]))
        ff.write("""Energy \n""")
        ff.write(f"""\t{en} \n""")
        ff.write("""PlusStress:  xx          yy          zz          yz          xz          xy \n""")
        ff.write(f"\t{virial[0,0]}  \t{virial[1,1]}  \t{virial[2,2]}  \t{virial[1,2]}  \t{virial[0,2]}  \t{virial[0,1]} \n")
        ff.write("""END_CFG \n""")
    ff.close()

'''不收集丢原子的结构'''
def dump2cfg(input,out):
    fin= iread(input)
    data = list(fin)

    std_num = data[0].get_global_number_of_atoms()
    index = 0
    for temp in data:
        if temp.get_global_number_of_atoms() == std_num:
            index = index + 1
    b = data[:index]

    map_dic={}
    ele = list(set(b[0].get_chemical_symbols()))
    # if sort_ele:
    #     temp = sorted([atomic_numbers[i] for i in ele])
    #     ele = [chemical_symbols[a] for a in temp]
    #     print(temp, ele)
    # else:
    #     ele=ele
    for i in range(len(ele)):
        map_dic.update({ele[i]:chemical_symbols.index(ele[i])-1})
    #print(map_dic)
    ff = open(out,"a")
    for i in range(len(b)):
        #print(i)
        atoms=b[i]
        ele=atoms.get_chemical_symbols()
        nat=len(ele)
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        #force= atoms.get_forces()
        #en=atoms.get_potential_energy()
        #virial=atoms.info['virial']
        ff.write("""BEGIN_CFG\n""")
        ff.write(""" Size\n""")
        ff.write("""  {:6} \n""".format(nat))
        ff.write(""" Supercell \n""")
        ff.write("""{:15.10f} {:15.10f} {:15.10f}\n""".format(cell[0, 0], cell[0, 1], cell[0, 2]))
        ff.write("""{:15.10f} {:15.10f} {:15.10f}\n""".format(cell[1, 0], cell[1, 1], cell[1, 2]))
        ff.write("""{:15.10f} {:15.10f} {:15.10f}\n""".format(cell[2, 0], cell[2, 1], cell[2, 2]))
        ff.write("""AtomData:  id type       cartes_x      cartes_y      cartes_z     \n""")
        # for i in range(nat):
        #     ff.write(
        #         """ {:6} {:6} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n""".format(i + 1, map_dic[ele[i]], pos[i, 0], pos[i, 1], pos[i, 2],
        #                                                                                      force[i,0], force[i,1], force[i,2]))
        for i in range(nat):
            ff.write(
                """ {:6} {:6} {:12.6f} {:12.6f} {:12.6f} \n""".format(i + 1, map_dic[ele[i]], pos[i, 0], pos[i, 1], pos[i, 2],
                                                                                             ))
        #ff.write("""Energy \n""")
        #ff.write(f"""\t{en} \n""")
        #ff.write("""PlusStress:  xx          yy          zz          yz          xz          xy \n""")
        #ff.write(f"\t{virial[0,0]}  \t{virial[1,1]}  \t{virial[2,2]}  \t{virial[1,2]}  \t{virial[0,2]}  \t{virial[0,1]} \n")
        ff.write("""END_CFG \n""")
    ff.close()
    return len(b)

def merge_cfg(file_path_1,file_path_2,output_file_path):
    with open(file_path_1, 'r') as file1:
        content_1 = file1.read()
    with open(file_path_2, 'r') as file2:
        content_2 = file2.read()
    merged_content = content_1 + '\n' + content_2
    with open(output_file_path, 'w') as output_file:
        output_file.write(merged_content)

def cfg2xyz(elements, sort_ele, cfg_file_path, xyz_file_path):
    map_dic = {}
    ordered_elements = _ordered_elements(elements, sort_ele)
    for index, element in enumerate(ordered_elements):
        map_dic.update({str(index): element})
    # cfgs=sys.argv[1]

    cfgs = cfg_file_path
    with open(cfgs) as f:
        lines = f.readlines()
        cfgcnt = 0
        for line in lines:
            if line == ' Size\n':
                cfgcnt += 1

        cntr = 1
        for i in range(len(lines)):
            if lines[i] != 'BEGIN_CFG\n':
                continue
            #        else:
            #            print("reading cfg#"+str(cntr+1))
            size = int(lines[i + 2].split()[0])
            lat = lines[i + 4].split() + lines[i + 5].split() + lines[i + 6].split()
            for k in range(len(lat)):
                lat[k] = float(lat[k])

            nruter = []
            for j in range(size):
                tmp = []
                words = lines[i + 8 + j].split()
                #print(words)
                tmp.append(map_dic[str(words[1])])
                tmp.append(float(words[2]))
                tmp.append(float(words[3]))
                tmp.append(float(words[4]))
                nruter.append(tmp)

            with open(xyz_file_path, 'a') as ff:
                ff.write(str(size) + '\n')
                ff.write(
                    """Lattice="{: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} " Properties=species:S:1:pos:R:3 pbc="T T T" \n""".format(
                        *lat))
                for k in range(size):
                    #print(*nruter[k])
                    ff.write('{} {: 12.8f} {: 12.8f} {: 12.8f}  \n'.format(*nruter[k]))
            #print(cntr)
            cntr += 1

def remove(file):
    if os.path.exists(file):
        os.remove(file)

def merge_cfg_out(pwd,merge_file_dirs,cfg_name,out_name):
    md_cfg = os.path.join(pwd, 'work', cfg_name)
    md_out = os.path.join(pwd, 'work', out_name)
    # remove(md_cfg)
    # remove(md_out)
    with open(md_cfg, 'w') as outfile:
        for path in merge_file_dirs:
            single_md_cfg = os.path.join(path, cfg_name)
            with open(single_md_cfg, 'r') as infile:
                outfile.write(infile.read() + '\n')  # 添加换行符以分隔文件内
    with open(md_out, 'w') as outfile:
        for path in merge_file_dirs:
            single_md_out = os.path.join(path, out_name)
            with open(single_md_out, 'r') as infile:
                outfile.write(infile.read() + '\n')  # 添加换行符以分隔文件内
    # for path in merge_file_dirs:
    #     single_md_out = os.path.join(path, 'md.out')
    #     single_md_cfg = os.path.join(path, 'md.cfg')
    #     os.remove(single_md_out)
    #     os.remove(single_md_cfg)

if __name__ == '__main__':
    sort_ele = True
    input = 'force.0.dump'
    out = 'md.cfg'
    #dump2cfg(input, out)
    ele = ['Al','As','Ga']
    cfg_file_path, xyz_file_path = 'md.cfg','md.xyz'
    cfg2xyz(ele, sort_ele, cfg_file_path, xyz_file_path)
