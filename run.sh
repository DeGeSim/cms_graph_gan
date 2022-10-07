#!/bin/bash

source ramenv.sh
python ./fgsim/utils/split_run.py $@
exit $!
