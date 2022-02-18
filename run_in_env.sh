#!/bin/bash
set -e
export LD_PRELOAD=""
source /etc/profile.d/modules.sh

ulimit -n `ulimit -H -n`

if ! command -v nvidia-smi &> /dev/null
then
    module load maxwell gcc/9.3
else
    module load maxwell gcc/9.3 cuda/11.1
fi

cd ~/fgsim
source bashFunctionCollection.sh

source venv1.10.1+cu111/bin/activate
logandrun python3 -m fgsim $@ &
export COMMANDPID=$!
trap "echo 'run_in_env.sh got SIGTERM' && kill $COMMANDPID " SIGINT SIGTERM
wait $COMMANDPID
echo "Command python3 -m fgsim $@ finished."
