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

#construct the correct job with it
for line in ${lines[@]}; do
    IFS=' ' read -r -a linesplit <<< $line
    CMD=${linesplit[0]}
    TAG=${linesplit[1]}
    RESTCMD=${linesplit[@]:2}
    # echo "Line: $line"
    # echo "... CMD $CMD tag $TAG Rest $RESTCMD"
    if [[ $REMOTE == 'true' ]]; then
        sbatch \
        --partition=allgpu,cms-desy \
        --time=78:00:00 \
        --mail-type=ALL \
        --nodes=1 \
        --constraint="P100" \
        --output=wd/slurm-$CMD-$TAG-%j.out \
        --job-name=$CMD-$TAG-%j run_in_env.sh $RESTCMD
    else
        ./run_in_env.sh $RESTCMD
    fi
done

exit
