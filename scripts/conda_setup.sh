#!/bin/bash
set -ex

export CONDA_DIR=~/beegfs/conda/miniconda
export RAMDIR=/dev/shm/${USER}/fgsim
export TARBALL=~/fgsim/env.tar
mkdir -p /dev/shm/${USER}

if [[ ! -f  ${CONDA_DIR}/bin/activate ]]; then
    mkdir -p ${CONDA_DIR}
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | sh -b -s -p ${CONDA_DIR}
fi


source ${CONDA_DIR}/bin/activate

conda config --set auto_activate_base false
conda config --append channels conda-forge
# Add the user directory in ram to the list of env locations
conda config --append envs_dirs /dev/shm/${USER}/

# remove modules
if ! command -v module &> /dev/null
then
    module purge
fi

if ! command -v mamba &> /dev/null
then
    # Init conda by hand
    set +x
    eval "$(${CONDA_DIR}/bin/conda shell.bash hook 2> /dev/null)"
    set -x
    conda install mamba --yes
fi

# Init conda and mamba by hand
set +x
eval "$(${CONDA_DIR}/bin/conda shell.bash hook 2> /dev/null)"
source ${CONDA_DIR}/etc/profile.d/mamba.sh
set -x

# Create the env
mamba create --yes --prefix /dev/shm/${USER}/fgsim python=3.9

mamba activate ${RAMDIR}
conda config --add channels pytorch
conda config --add channels pyg

mamba install --yes  pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
python -c 'import torch; assert torch.cuda.is_available()'

mamba install --yes omegaconf typeguard tqdm uproot awkward tensorboard tblib pytorch-lightning scikit-learn multiprocessing-logging   prettytable pretty_errors torchinfo seaborn sqlitedict wandbray-tune hyperopt

# crypto vs openssl bug
# https://stackoverflow.com/questions/74981558/error-updating-python3-pip-attributeerror-module-lib-has-no-attribute-openss
mamba install cryptography==38.0.4

# dev tools
mamba install --yes black isort flake8 mypy pytest pre-commit ipykernel jupyter notebook openai icecream

## PyG
mamba install --yes pyg -c pyg
# Assert that torch scatter works
python -c 'import torch_scatter'
# remove torch-spline-conv that causes an error
# python -c  'from torch_spline_conv import spline_basis, spline_weighting'
pip uninstall torch-spline-conv

# jetnet requirements
mamba install --yes coffea h5py wurlitzer

pip install jetnet

# activate the enviroment in ram
unalias -a
source ${RAMDIR}/bin/activate

pip install -e ~/fgsim

[[ -d  ~/queueflow ]] || git clone git@github.com:DeGeSim/queueflow.git ~/queueflow
pip install -e ~/queueflow
# save the manipulated tarball
tar -c -f ${TARBALL} --directory=${RAMDIR} .
