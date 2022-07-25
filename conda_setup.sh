#!/bin/bash
set -e
module purge
conda init zsh
conda install mamba
mamba init
set -x
mamba install -c conda-forge conda-pack
mamba create --name fgsim python=3.9

mamba activate fgsim
mamba install pytorch=1.11 cudatoolkit=11.3 -c pytorch
mamba install numpy
python -c 'import torch; assert torch.cuda.is_available()'
mamba install glib
mamba install pyg -c pyg
# Assert that torch scatter works
python -c 'import torch_scatter'


#Project dependencies
mamba install omegaconf typeguard tqdm uproot awkward tensorboard tblib pytorch-lightning scikit-learn multiprocessing-logging  icecream prettytable pretty_errors
mamba install -c comet_ml comet_ml
# dev tools
mamba install black isort flake8 mypy pytest pre-commit ipykernel

# jetnet requirements
mamba install coffea h5py wurlitzer
pip install jetnet torchtyping

export TARBALL_BASE_PATH=~/beegfs/conda/fgsim_base.tar
export TARBALL_PATH=~/beegfs/conda/fgsim.tar
export RAMDIR=/dev/shm/$USER/fgsim

mamba pack -n fgsim -o ${TARBALL_BASE_PATH}


mkdir -p ${RAMDIR}
tar -x -f ${TARBALL_BASE_PATH} --directory ${RAMDIR}

# activate the enviroment in ram
unalias -a
source ${RAMDIR}/bin/activate

# remove torch-spline-conv that causes an error
# python -c  'from torch_spline_conv import spline_basis, spline_weighting'
pip uninstall torch-spline-conv


pushd ~/fgsim
pip install -e .
popd

[[ -d  ~/queueflow ]] || git clone git@github.com:DeGeSim/queueflow.git ~/queueflow
pushd ~/queueflow
pip install -e .
popd
# save the manipulated tarball
tar -c -f ${TARBALL_PATH} --directory=${RAMDIR} .
