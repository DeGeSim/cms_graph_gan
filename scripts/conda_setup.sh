#!/bin/env bash
set -ex

export CONDA_DIR=~/dust/miniconda
export RAMDIR=/dev/shm/${USER}/fgsim
export TARBALL=~/fgsim/env.tar
export PYTHON_VERSION=3.10
export CUDA_VERSION=11.8
export TORCH_VERSION=2.0.\*

## Conda setup & configuration
# remove modules existing
if ! command -v module &> /dev/null
then
    module purge
fi


if [[ ! -f  ${CONDA_DIR}/bin/activate ]]; then
    mkdir -p ${CONDA_DIR}
    echo "Installing Conda"
    tmpfile="$(mktemp).sh"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  -O ${tmpfile}
    bash ${tmpfile} -b -s -p ${CONDA_DIR}
    rm ${tmpfile}
fi

set +x
echo "Activating Conda"
source ${CONDA_DIR}/bin/activate
set -x


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
echo "Activating Mamba"
eval "$(${CONDA_DIR}/bin/conda shell.bash hook 2> /dev/null)"
source "${CONDA_DIR}/etc/profile.d/mamba.sh"
set -x

conda config --set auto_activate_base false
conda deactivate
# Add the user directory in ram to the list of env locations
conda config --append envs_dirs /dev/shm/${USER}/


## Create the env
mkdir -p /dev/shm/${USER}
mamba create --yes --prefix /dev/shm/${USER}/fgsim python=${PYTHON_VERSION}

mamba activate ${RAMDIR}

conda config --append channels conda-forge
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels pyg


mamba install --yes pytorch=${TORCH_VERSION}=py${PYTHON_VERSION}_cuda${CUDA_VERSION}\*  pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia
python -c 'import torch; assert torch.cuda.is_available()'
# pin the torch and cuda versions
conda list "^(pytorch|pytorch-cuda)$" | tail -n+4 | awk '{ print $1 "==" $2 }' > $RAMDIR/conda-meta/pinned


mamba install --yes omegaconf typeguard tqdm tensorboard tblib pytorch-lightning multiprocessing-logging prettytable pretty_errors rich torchinfo wandb
# Data Science stack
mamba install --yes scipy pandas scikit-learn seaborn matplotlib
# IO
mamba install --yes sqlitedict h5py uproot awkward

## Optional
# Hyperparemter Tuning
mamba install --yes  ray-default ray-tune ray-air hyperopt -c conda-forge
# dev tools
mamba install --yes ruff black mypy pytest pre-commit ipykernel jupyter notebook

## PyG
mamba install --yes pytorch-scatter pytorch-sparse pytorch-cluster pyg -c pyg
# Assert that torch scatter works
python -c 'import torch_scatter'


# jetnet requirements
mamba install --yes coffea numexpr cython py-cpuinfo cmake lit wurlitzer certifi

# mamba install -c nvidia libcublas cuda-cupti cuda-nvrtc cuda-runtime  libcufft libcurand libcusolver libcusparse
# found in conda
# cudnn-cu11 nccl-cu11 nvtx-cu11
pip install --no-deps tables wasserstein energyflow
pip install --no-deps jetnet

## Workarounds
# # crypto vs openssl bug
# # https://stackoverflow.com/questions/74981558/error-updating-python3-pip-attributeerror-module-lib-has-no-attribute-openss
# mamba install cryptography==38.0.4
# # remove torch-spline-conv that causes an error
# # python -c  'from torch_spline_conv import spline_basis, spline_weighting'
# pip uninstall torch-spline-conv



# activate the enviroment in ram
# source ${RAMDIR}/bin/activate

for REPO in fgsim queueflow; do
    if ! [[ -d  ~/${REPO} ]]; then
        git clone git@github.com:DeGeSim/${REPO}.git ~/${REPO}
    fi
    pip install --no-deps -e ~/${REPO}
done

# save the manipulated tarball
# tar -c -f ${TARBALL} --directory=${RAMDIR} .
