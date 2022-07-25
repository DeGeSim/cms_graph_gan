#!/usr/bin/bash
ulimit -n `ulimit -H -n`

RAMPATH="/dev/shm/$USER/fgsim"

if [ ! -d $RAMPATH ]; then
  echo "Installing  $RAMPATH"
  mkdir -p $RAMPATH
  tar -x -f ~/beegfs/venvs/fgsim.tar --directory ${RAMPATH}
fi
unalias ls || true
source $RAMPATH/bin/activate

# save ram env
# tar -c -f ~/beegfs/venvs/fgsim.tar --directory=${RAMPATH} .
