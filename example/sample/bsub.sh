#!/bin/bash

module purge
source /data/phy-huangj/app/temp/dcbf_lowdisk_20260902/dcbf_one-button_deployment/activate.sh
dcbf run dcbf.init_dataset.vasp.qiming.json
