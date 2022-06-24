#!/bin/bash
set -e
export LD_PRELOAD=""
source /etc/profile.d/modules.sh

ulimit -n `ulimit -H -n`

if ! command -v nvidia-smi &> /dev/null
then
    module load maxwell gcc/9.3
else
    module load maxwell gcc/9.3 cuda/11.1 &> /dev/null
fi

cd ~/fgsim
source bashFunctionCollection.sh

# load env from ram, if it is alreay loaded, else take the local one
RAMPATH="/dev/shm/$USER/venv1.10.1+cu111"
if [ -d $RAMPATH ]; then
    source ramenv.sh
else
    source venv1.10.1+cu111/bin/activate
fi

logandrun python3 -m fgsim $@ &
export COMMANDPID=$!
trap "echo 'run_in_env.sh got SIGTERM' && kill $COMMANDPID " SIGINT SIGTERM
wait $COMMANDPID
echo "Command python3 -m fgsim $@ finished."
