#!/bin/bash
#SBATCH --account=def-hansenhe
#SBATCH --gres=gpu:v100l:4              # Number of GPU(s) per node
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=10              # Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem=100G                      # Memory per node (adjust as needed)
#SBATCH --time=05-00:00                 # Time limit (days-hours:minutes)
#SBATCH --output=pretraining_LLM_48_mlm_0.15_scratch_continueT_1.out  

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/ENV
source $SLURM_TMPDIR/ENV/bin/activate
pip3 install --no-index --upgrade pip
pip3 install --no-index -r requirements_pep.txt

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"


srun python3 LLM/train_ddp.py --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS --scratch
