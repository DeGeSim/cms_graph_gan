#!/usr/bin/bash
export LD_PRELOAD=""
source /etc/profile.d/modules.sh

ulimit -n `ulimit -H -n`

if ! command -v nvidia-smi &> /dev/null
then
    module load maxwell gcc/9.3
else
    module load maxwell gcc/9.3 cuda/11.1 &> /dev/null
fi

pushd ~/fgsim
source bashFunctionCollection.sh
popd

RAMPATH="/dev/shm/mscham/venv1.10.1+cu111"

if [ ! -d $RAMPATH ]; then
  echo "Installing  $RAMPATH"
  mkdir -p $RAMPATH
  tar -x -f ~/beegfs/venvs/venv1.10.1+cu111.tar --directory $RAMPATH
else
  echo "Installation $RAMPATH already exists. Just activating."
fi
source $RAMPATH/bin/activate

# save ram env
# tar -c -f ~/beegfs/venvs/venv1.10.1+cu111.tar --directory=/dev/shm/mscham/venv1.10.1+cu111 .
