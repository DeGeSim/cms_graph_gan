#!/bin/bash

source scripts/ramenv.sh
python ./scripts/run.py $@
exit $!
