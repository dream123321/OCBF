from ase.io import iread
from ase.data import atomic_numbers,chemical_symbols
import numpy as np


def xyz2cfg(ele,ele_model,input,out):
    fin=iread(input)
    map_dic={}
    if ele_model == 1:
        temp = sorted([atomic_numbers[i] for i in ele])
        ele = [chemical_symbols[a] for a in temp]
    elif ele_model ==2:
        ele=ele
    for i in range(len(ele)):
        map_dic.update({ele[i]:i})

    b=list(fin)
    ff = open(out,"w")
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

def sort_xyz2cfg(input,out):
    fin = iread(input)
    b = list(fin)
    map_dic = {}
    ele_set = set()
    for i in b:
        ele_set = ele_set | set(i.get_chemical_symbols())
    ele_ = list(ele_set)
    order = True
    if (order):
        ele_order = {}
        for i in ele_:
            ele_order.update({i: atomic_numbers[i]})
        ele_ = sorted(ele_, key=lambda x: ele_order[x])
    for i in range(len(ele_)):
        map_dic.update({ele_[i]: i})
    ff = open(out, "w")
    for i in range(len(b)):
        # print(i)
        atoms = b[i]
        ele = atoms.get_chemical_symbols()
        nat = len(ele)
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        force = atoms.get_forces()
        en = atoms.get_potential_energy()
        virial = atoms.info['virial']
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
                """ {:6} {:6} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n""".format(i + 1, map_dic[ele[i]],
                                                                                                pos[i, 0], pos[i, 1],
                                                                                                pos[i, 2],
                                                                                                force[i, 0],
                                                                                                force[i, 1],
                                                                                                force[i, 2]))
        ff.write("""Energy \n""")
        ff.write(f"""\t{en} \n""")
        ff.write("""PlusStress:  xx          yy          zz          yz          xz          xy \n""")
        ff.write(
            f"\t{virial[0, 0]}  \t{virial[1, 1]}  \t{virial[2, 2]}  \t{virial[1, 2]}  \t{virial[0, 2]}  \t{virial[0, 1]} \n")
        ff.write("""END_CFG \n""")
    ff.close()
    return ele_

def dump2cfg(ele_model,input,out):
    fin= iread(input)
    b = list(fin)
    map_dic={}
    ele = list(set(b[0].get_chemical_symbols()))
    if ele_model == 1:
        temp = sorted([atomic_numbers[i] for i in ele])
        ele = [chemical_symbols[a] for a in temp]
    elif ele_model ==2:
        ele=ele
    for i in range(len(ele)):
        map_dic.update({ele[i]:i})

    ff = open(out,"w")
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

def merge_cfg(file_path_1,file_path_2,output_file_path):
    with open(file_path_1, 'r') as file1:
        content_1 = file1.read()
    with open(file_path_2, 'r') as file2:
        content_2 = file2.read()
    merged_content = content_1 + '\n' + content_2
    with open(output_file_path, 'w') as output_file:
        output_file.write(merged_content)

def cfg2xyz(ele,ele_model,cfgs,out):
    map_dic = {}
    if ele_model == 1:
        ele_ = [atomic_numbers[i] for i in ele]
        ele_ = sorted(ele_)
        ele = [chemical_symbols[a] for a in ele_]
        for i in range(len(ele)):
            map_dic.update({i: ele[i]})
    elif ele_model == 2:
        for i in range(len(ele)):
            map_dic.update({i: ele[i]})
    print(map_dic)
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
            energy = float(lines[i + 9 + size].split()[0])
            lat = lines[i + 4].split() + lines[i + 5].split() + lines[i + 6].split()
            for k in range(len(lat)):
                lat[k] = float(lat[k])
            stress = lines[i + 11 + size].split()
            for l in range(len(stress)):
                stress[l] = float(stress[l])
            _stress = [stress[0], stress[5], stress[4], stress[5], stress[1], stress[3], stress[4], stress[3],
                       stress[2]]

            nruter = []
            for j in range(size):
                tmp = []
                words = lines[i + 8 + j].split()
                tmp.append(map_dic[int(words[1])])
                tmp.append(float(words[2]))
                tmp.append(float(words[3]))
                tmp.append(float(words[4]))
                tmp.append(float(words[5]))
                tmp.append(float(words[6]))
                tmp.append(float(words[7]))
                nruter.append(tmp)

            with open(out, 'a') as ff:
                ff.write(str(size) + '\n')
                ff.write(
                    """Lattice="{: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} " Properties=species:S:1:pos:R:3:forces:R:3 energy={: 12.8f} virial= "{: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f}" pbc="T T T" \n""".format(
                        *lat, energy, *_stress))
                for k in range(size):
                    ff.write('{} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f}\n'.format(*nruter[k]))
            #print(cntr)
            cntr += 1

