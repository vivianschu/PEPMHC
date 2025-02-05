import os
import math

import torch
import torchtext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_num = 5 # save model after n epochs
epochs = 25
batch_size = 32
model_name = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/models/classI/check.pt"
pep_n = 1
mhc_n = 3
pep_vectors_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/1-gram-vectors.txt" # set based on pep_n
mhc_vectors_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/3-gram-vectors.txt" # set based on mhc_n
cache_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/"
base_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/classI/"
train_file = "train.csv"
val_file = "val.csv"
test_file = "test.csv"

new_base_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/classI/new_mhc_split/"


EMBED_DIM = 100 # change only after re-training the vectors in the new space
PEPTIDE_LENGTH = 45 # set based on pep_n
MHC_AMINO_ACID_LENGTH = 120 # set based on mhc_n
BiLSTM_HIDDEN_SIZE = 64
BiLSTM_PEPTIDE_NUM_LAYERS = 3
BiLSTM_MHC_NUM_LAYERS = 3
LINEAR0_OUT = 2048
LINEAR1_OUT = 64
LINEAR2_OUT = 2
CONTEXT_DIM = 16
MAXPOOL_FLAG = True

KERNEL_SIZE = 3
STRIDE_SIZE = 1


USE_MHC_SPLIT = True
