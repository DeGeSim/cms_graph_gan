#!/bin/bash
set -e
SWEEPID=$( cat fgsim/wd/$1/sweepid )

for ((i = 1; i <= 25; i++)); do
    sbatch << EOF
#!/bin/bash
#SBATCH --partition=maxgpu,allgpu
#SBATCH --job-name=wandbagent
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --constraint='A100|V100'
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/mscham/slurm/rayworker-%j.log
source ${HOME}/fgsim/scripts/ramenv.sh
wandb agent --count 50 $SWEEPID
EOF
done
