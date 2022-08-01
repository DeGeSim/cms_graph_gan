#!/usr/bin/bash
ulimit -n `ulimit -H -n`

RAMPATH="/dev/shm/$USER/fgsim"

if [ ! -d $RAMPATH ]; then
  echo "Installing  $RAMPATH"
  mkdir -p $RAMPATH
  tar -x -f ~/beegfs/conda/fgsim.tar --directory ${RAMPATH}
fi
if alias ls &>/dev/null ; then
  unalias ls
fi

export CONDA_DIR=~/beegfs/conda/miniconda
# Init conda and mamba by hand
eval "$(${CONDA_DIR}/bin/conda shell.bash hook 2> /dev/null)"
source ${CONDA_DIR}/etc/profile.d/mamba.sh
mamba activate $RAMPATH

# save ram env
# tar -c -f ~/beegfs/venvs/fgsim.tar --directory=${RAMPATH} .
