#!/usr/bin/bash
ulimit -n `ulimit -H -n`

RAMPATH=/dev/shm/$USER/fgsim
TIMESTAMP_FILE=$RAMPATH/tsfile
LOCKFILE=/dev/shm/$USER-ramenv.lock
TARBALL=~/fgsim/env.tar

# # Get the file descript and lock the file
# LOCKPATH=/dev/shm/${USER}/ramenv.lock
# exec 4> ${LOCKPATH} || exit 1
# flock 4 || exit 1
# trap "rm -f ${LOCKPATH}" EXIT

(
flock -n 9 || exit 1
# Remove the old env from RAM if a newer one has been written to the tarball
if [ -d $RAMPATH ] && [ $TARBALL -nt $TIMESTAMP_FILE ]; then
    echo "Tarball $TARBALL is newer then the env in $RAMPATH"
    if ! [ -f $TARBALL ] || ! [ -s $TARBALL ] || ! [ -r $TARBALL ] ; then
        echo "Something is wrong with $TARBALL. Not making any changes"
        return
    fi
    echo "Removing the env in $RAMPATH"
    rm -rf $RAMPATH
fi
# Install/Update the env in RAM
if [ ! -d $RAMPATH ]; then
    echo "Installing env in $RAMPATH from Tarball $TARBALL"
    mkdir -p $RAMPATH
    tar -x -f $TARBALL --directory ${RAMPATH}
    touch $TIMESTAMP_FILE
fi
) 9>$LOCKFILE

# remove aliases that break the conda setup script
if alias ls &>/dev/null ; then
    unalias ls
fi

# Init conda and mamba by hand
export CONDA_DIR=~/beegfs/conda/miniconda
eval "$(${CONDA_DIR}/bin/conda shell.bash hook 2> /dev/null)"
source ${CONDA_DIR}/etc/profile.d/mamba.sh
mamba activate fgsim
# save ram env
alias ramenv_save="tar -c -f ${TARBALL} --directory=${RAMPATH} ."
