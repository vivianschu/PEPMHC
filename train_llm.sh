#!/bin/bash
#SBATCH --account=def-hansenhe
#SBATCH --gres=gpu:a100:4              # Number of GPU(s) per node
#SBATCH --cpus-per-task=10              # Number of CPUs per task (adjust as needed)
#SBATCH --mem=30G                      # Memory per node (adjust as needed)
#SBATCH --time=03-00:00                 # Time limit (days-hours:minutes)
#SBATCH --output=pretraining_LLM_48_mlm_0.15.out       # Output file

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/ENV
source $SLURM_TMPDIR/ENV/bin/activate
pip3 install --no-index --upgrade pip
pip3 install --no-index -r requirements_pep.txt

python3 /scratch/ssd004/scratch/vchu/PEPMHC/LLM/train.py
