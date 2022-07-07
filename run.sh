#!/bin/bash
if [[ $1 == 'local' ]]; then
    REMOTE=false
    elif [[ $1 == 'remote' ]]; then
    REMOTE=true
else
    echo "First argument must be local or remote."
    exit 1
fi

shift
cd ~/fgsim
source bashFunctionCollection.sh
source ramenv.sh
# Let python resolve the cli arguments
set -x
tmpfile=`mktemp`
ARGS=(`echo $@ |tr " " "\n" `) # convert string to array
python ./fgsim/utils/cli.py ${ARGS[@]} > $tmpfile
IFS=$'\n'
readarray -t lines <$tmpfile
rm $tmpfile


function elementf {
    RESTCMD=(`echo $RESTCMD |tr " " "\n" `) # convert string to array
    if [[ $CMD == setup ]]; then
        ./run_in_env.sh $RESTCMD
    else
        if [[ "${TAG_OR_HASH_ARG}" == "--tag" ]]; then
            export HASH=$( python3 -m fgsim --tag ${TAG_OR_HASH} gethash 2>/dev/null )
        else
            export HASH=${TAG_OR_HASH}
        fi
        if [[ $REMOTE == 'true' ]]; then
            sbatch \
            --partition=allgpu \
            --time=24:00:00 \
            --nodes=1 \
            --constraint="P100|V100|A100" \
            --output=wd/slurm-$CMD-${HASH}-%j.out \
            --job-name=${HASH} run.sh local ${RESTCMD[@]}
        else

            logandrun python3 -m fgsim ${RESTCMD[@]} &
            export COMMANDPID=$!
            trap "echo 'run.sh got SIGTERM' && kill $COMMANDPID " SIGINT SIGTERM
            wait $COMMANDPID
        fi
    fi
}

#construct the correct job with it
for line in ${lines[@]}; do
    # split the lines into an array
    IFS=' ' read -r -a linesplit <<< $line
    export CMD=${linesplit[0]}
    export TAG_OR_HASH=${linesplit[1]}
    export TAG_OR_HASH_ARG=${linesplit[2]}
    export RESTCMD=${linesplit[@]:2}
    if [[ $REMOTE == 'true' || $CMD != 'train' ]]; then
        elementf &
    else
        elementf
    fi
done
wait

exit
