#!/bin/bash
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO PRODUCTION!
#SBATCH --partition=cms-desy
#SBATCH --job-name=rayhead
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --constraint='P100|V100|A100'
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/mscham/slurm/rayhead-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --exclusive

# Load modules or your own conda environment here
source ~/fgsim/ramenv.sh

# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# # if we detect a space character in the head node IP, we'll
# # convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<< "$ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        ip=${ADDR[1]}
    else
        ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun \
    --task-prolog="${HOME}/fgsim/ramenv.sh" \
    --nodes=1 \
    --ntasks=1 \
    -w "$node_1" \
    ray start --head --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --block &

for ((i = 1; i <= 35; i++)); do
    sbatch << EOF
#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --job-name=rayworker
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --constraint='P100|V100|A100'
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/mscham/slurm/rayworker-%j.log
source ${HOME}/fgsim/ramenv.sh
ray start --address "$ip_head" --redis-password="$redis_password" --block
EOF
done
while [[ $(squeue -u ${USER} -n rayworker -t R | wc -l) -lt 2 ]] ; do
    sleep 1
done
sleep 10
# ===== Call your code below =====
python ~/fgsim/raytune/tune.py
