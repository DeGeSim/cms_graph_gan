#!/bin/bash
set -e
if [[ ! -z "$VIRTUAL_ENV" ]]; then
    deactivate
fi
yes | module clear
set -x

# Install the utilities in isolated eviroments
pip install -U pipx
packages=(black isort #formating \
    flake8 mypy #linting\
    pytest #testing \
    pre-commit #precommit hooks
)
for package in  ${packages[@]}; do
    command -v $package &> /dev/null || pipx install $package
done
command -v pflake8 &> /dev/null || pipx inject --include-apps flake8 pyproject-flake8


#provide cuda
module load maxwell gcc/9.3 cuda/11.1
export CUDA=cu111 TORCH=1.10.1
VENV=venv${TORCH}+${CUDA}

# compile multicore
export MAKEFLAGS="-j20"


# setup and load the virtual enviroment
# Here the script assumes that python>=3.8 has been provided
RAMPATH="/dev/shm/$USER/"
if [ ! -d $RAMPATH ]; then
  mkdir -p $RAMPATH
fi
# go into the ram path
pushd $RAMPATH
# create the venv
python -m venv $VENV
# activate it
source $VENV/bin/activate
# proceed th installation the base directory
# the pip packages will be installed into the $RAMPATH
popd


pip install --upgrade pip
pip install wheel
# pytorch_geometric neeeds some patched versions of torch* and cannot be installed in the setup
pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
if [[ ${TORCH} == 1.10.1  ]]; then
    pip install torchvision==0.11.2+${CUDA}
fi

# Provide the package as editable, so that we can do "from fgsim import ..."
pip install -e .[dev]

# install queueflow
[-d  ~/queueflow] || git clone git@github.com:DeGeSim/queueflow.git ~/queueflow
pushd ~/queueflow
pip install -e .
popd

# save the enviroment from ram to disk
tar -c -f ~/beegfs/venvs/$VENV.tar --directory=$RAMPATH/$VENV .
