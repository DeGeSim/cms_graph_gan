#!/bin/bash

source scripts/ramenv.sh
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
python ./scripts/run.py $@
exit $!
