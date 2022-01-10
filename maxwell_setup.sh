#!/bin/bash
deactivate
yes | module clear

set -ex

#provide cuda
module load maxwell gcc/9.3 cuda/11.3

# Here the script assumes that python>=3.8 has been provided
# Install the utilities in isolated eviroments
pip install -U pipx
pipx install black isort #formating \
    rope #refactoring \
    pyproject-flake8 mypy #linting\
    coverage pytest jedi #testing \
    pre-commit #precommit hooks



# setup and load the virtual enviroment
virtualenv venv
source venv/bin/activate

# pytorch_geometric neeeds some patched versions of torch* and cannot be installed in the setup
export CUDA=cu113 TORCH=1.10.1
pip3 install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

pip install torch-geometric

# Provide the package as editable, so that we can do "from fgsim import ..."
pip install -e .[dev]

# install queueflow
git clone git@github.com:DeGeSim/queueflow.git ~/queueflow
cd ~/queueflow
pip install -e .
