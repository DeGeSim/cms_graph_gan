#!/bin/bash
deactivate
yes | module clear
module load maxwell gcc/9.3 cuda/11.1
CUDA=111
TORCH=1.9.0
virtualenv pyenv
source pyenv/bin/activate


pip3 install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html


pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

pip install torch-geometric

pip install .
pip install .[test]
pip install .[code]
