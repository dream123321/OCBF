# OCBF Installation

This file only keeps the installation workflow.

## 1. Recommended: One-Button Deployment Package

```bash
tar -zxvf ocbf_one-button_deployment.tar.gz
cd ocbf_one-button_deployment
bash install.sh
source activate.sh
bash verify.sh

cd source/OCBF_stability/example/sample

Then modify the submission queue, dft_env and dft_command
ocbf run ocbf.init_dataset.vasp.test.json 
```

## 2. Kill A Managed Background Run

```bash
ocbf kill .
```

Or specify the run directory/config path:

```bash
ocbf kill ocbf.init_dataset.vasp.test.json 
```

## 3. Manual Build From Source

### 3.1 Install Python dependencies

```bash
cd ocbf
python -m pip install -r requirement.txt
python setup.py install
```

### 3.2 Build SUS2-MLIP

```bash
install [SUS2-MLIP](https://github.com/hu-yanxiao/SUS2-MLIP)
```

Expected outputs:
- `bin/mlp-sus2`
- `lib/lib_mlip_interface.a`

### 2.3 Build pymlip

```bash
tar -zxvf pysus2mlip.tar.gz
cd pysus2mlip
```

Edit `setup.py` so that:
- `mlip_include_dir` points to:
  - `<SUS2-MLIP_latest>/src/common`
  - `<SUS2-MLIP_latest>/src`
  - `<SUS2-MLIP_latest>/dev_src`
- `extra_objects` points to:
  - `<SUS2-MLIP_latest>/lib/lib_mlip_interface.a`

Then install:

```bash
CC=gcc CXX=g++ python -m pip install .
```

### 3.4 Build LAMMPS Interface

```bash
SUS2-MLIP models can be used in [LAMMPS](https://github.com/lammps/lammps) simulation via the sus2-interface

tar -zxzf sus2-interface-20260410.tar.gz
cd /path/to/lammps/src
cp -r /path/to/sus2-interface-20260410/ML-SUS2 ./ML-SUS2
make no-user-mlip || true
make no-ML-SUS2 || true
make yes-ML-SUS2
make mpi -j 16
```

Expected output:
- `lmp_mpi`

