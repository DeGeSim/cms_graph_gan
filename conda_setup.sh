#!/bin/bash
set -ex

export CONDA_DIR=~/beegfs/conda/miniconda
export RAMDIR=/dev/shm/$USER/fgsim
export TARBALL_PATH=~/beegfs/conda/fgsim.tar
export TARBALL_BASE_PATH=~/beegfs/conda/fgsim_base.tar
mkdir -p /dev/shm/$USER

# module purge
# # conda init zsh
# conda install mamba --yes
# # mamba init
# # mamba install -c conda-forge conda-pack
#
# mamba create --yes --prefix /dev/shm/$USER/fgsim python=3.9

# Init conda and mamba by hand
set +x
eval "$(${CONDA_DIR}/bin/conda shell.bash hook 2> /dev/null)"
source ${CONDA_DIR}/etc/profile.d/mamba.sh
set -x

mamba activate /dev/shm/mscham/fgsim
mamba install --yes  -c pytorch  pytorch=1.11 cudatoolkit=11.3
mamba install --yes numpy
python -c 'import torch; assert torch.cuda.is_available()'
# mamba install glib
mamba install --yes pyg -c pyg
# Assert that torch scatter works
python -c 'import torch_scatter'


#Project dependencies
mamba install --yes omegaconf typeguard tqdm uproot awkward tensorboard tblib pytorch-lightning scikit-learn multiprocessing-logging  icecream prettytable pretty_errors
mamba install --yes -c comet_ml comet_ml
# dev tools
mamba install --yes black isort flake8 mypy pytest pre-commit ipykernel

# jetnet requirements
mamba install --yes coffea h5py wurlitzer
pip install jetnet torchtyping


# mamba pack -n fgsim -o ${TARBALL_BASE_PATH}

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
