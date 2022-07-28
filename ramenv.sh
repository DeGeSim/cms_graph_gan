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
source $RAMPATH/bin/activate

# save ram env
# tar -c -f ~/beegfs/venvs/fgsim.tar --directory=${RAMPATH} .
