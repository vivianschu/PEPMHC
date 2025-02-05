#!/bin/bash
#SBATCH --job-name=test_protbert
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v1,t4v2,a40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12               # Number of CPUs per task (adjust as needed)
#SBATCH --mem=100G                       # Memory per node (adjust as needed)
#SBATCH --time=12:00:00                  # Time limit (days-hours:minutes)
#SBATCH --output=prot_bert_48_350_mlm_0.25_44000_new_split_%j.out       # Output file

module load cuda-11.8
echo $(nvcc --version)

python3 /scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_ft/src/train_protbert_ft_fast.py --pep-max-len 48 --new-split-flag 