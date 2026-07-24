#!/bin/bash

module purge
source /work/phy-huangj/hj_mlp/dcbf_one-button_deployment/activate.sh
dcbf run dcbf.init_dataset.vasp.qiming.json
