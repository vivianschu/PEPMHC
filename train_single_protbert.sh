#!/bin/bash
#SBATCH --account=def-hansenhe
#SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=10              # Number of CPUs per task (adjust as needed)
#SBATCH --mem=20G                      # Memory per node (adjust as needed)
#SBATCH --time=02-10:00                 # Time limit (days-hours:minutes)
#SBATCH --output=single_protbert.out       # Output file, Need to be frequently updated

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/ENV
source $SLURM_TMPDIR/ENV/bin/activate
pip3 install --no-index --upgrade pip
pip3 install --no-index -r requirements_prot_bert.txt

python3 /home/patrick3/scratch/MHCAttnNet_ft/src/train_single_protbert.py 
