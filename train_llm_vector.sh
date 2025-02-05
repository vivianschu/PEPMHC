#!/bin/bash
#SBATCH --job-name=test_llm
#SBATCH --gres=gpu:4
#SBATCH --partition=t4v1,t4v2,a40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12               # Number of CPUs per task (adjust as needed)
#SBATCH --mem=100G                       # Memory per node (adjust as needed)
#SBATCH --time=12:00:00                  # Time limit (days-hours:minutes)
#SBATCH --output=pretraining_LLM_48_mlm_0.15_%j.out       # Output file

module load cuda-11.8
echo $(nvcc --version)
# module load anaconda
# conda activate neo

python3 /scratch/ssd004/scratch/vchu/PEPMHC/LLM/train.py