#!/bin/bash
#SBATCH --partition=cms-desy
#SBATCH --time=12:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --nodes=1
#SBATCH --constraint="P100"
export LD_PRELOAD=""
source /etc/profile.d/modules.sh


if ! command -v nvidia-smi &> /dev/null
then
    module load maxwell gcc/9.3
else
    module load maxwell gcc/9.3 cuda/11.1
    CUDA=111
fi

TORCH=1.9.0
cd ~/fgsim
source .tox/py38/bin/activate
python3 -m fgsim $@
