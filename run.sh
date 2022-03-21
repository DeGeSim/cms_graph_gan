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

# Let python resolve the cli arguments
IFS=$'\n'
readarray -t lines < <(python ./fgsim/utils/cli.py $@)

function elementf {
    if [[ $CMD == setup ]]; then
        ./run_in_env.sh $RESTCMD
    else
        if [[ "${TAG_OR_HASH_ARG}" == "--tag" ]]; then
            export HASH=$( source venv1.10.1+cu111/bin/activate; python3 -m fgsim --tag ${TAG_OR_HASH} setup 2>/dev/null )
        else
            export HASH=${TAG_OR_HASH}
        fi
        if [[ $REMOTE == 'true' ]]; then
            sbatch \
            --partition=allgpu,cms-desy \
            --time=78:00:00 \
            --nodes=1 \
            --constraint="P100" \
            --output=wd/slurm-$CMD-${HASH}-%j.out \
            --job-name=${HASH} run_in_env.sh --hash $HASH $CMD
        else
            ./run_in_env.sh --hash $HASH $CMD &
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
