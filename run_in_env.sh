#!/bin/bash
set -e
export LD_PRELOAD=""
source /etc/profile.d/modules.sh

ulimit -n `ulimit -H -n`

if ! command -v nvidia-smi &> /dev/null
then
    module load maxwell gcc/9.3
else
    module load maxwell gcc/9.3 cuda/11.1
fi

cd ~/fgsim
source v/venv1.10.1+cu111/bin/activate
python3 -m fgsim $@
echo "Command python3 -m fgsim $@ finished."
