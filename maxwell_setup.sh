#!/bin/bash
deactivate
yes | module clear
set -ex

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


# setup and load the virtual enviroment
[[ -d v ]] || mkdir v
pushd v
# Here the script assumes that python>=3.8 has been provided
python -m venv $VENV
source $VENV/bin/activate
popd


pip install wheel
# pytorch_geometric neeeds some patched versions of torch* and cannot be installed in the setup
pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

pip install torch-geometric

# Provide the package as editable, so that we can do "from fgsim import ..."
pip install -e .[dev]
# rope #refactoring \
# jedi

# install queueflow
git clone git@github.com:DeGeSim/queueflow.git ~/queueflow
cd ~/queueflow
pip install -e .
