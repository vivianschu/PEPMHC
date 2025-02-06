import os
import math
import torch
import torchtext

# Select device: CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_num = 1                # Save model checkpoint every n epochs
epochs = 5                  # Total number of training epochs
batch_size = 64             # Batch size during training/inference

# Store/load model checkpoint
model_name = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/models/classI/check.pt"

# N-gram parameters for peptide and MHC sequences
pep_n = 1                   # Peptides split into 1-grams
mhc_n = 3                   # MHC split into 3-grams
# Paths to pretrained embedding vectors for n-grams
pep_vectors_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/1-gram-vectors.txt" # set based on pep_n
mhc_vectors_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/3-gram-vectors.txt" # set based on mhc_n

# Paths for caching and storing dataset files
cache_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/"
base_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/classI/"

# Train, val, test CSVs
train_file = "train.csv"
val_file = "val.csv"
test_file = "test.csv"

# Alternative split path for MHC
new_base_path = "/scratch/ssd004/scratch/vchu/PEPMHC/MHCAttnNet_dataset/classI/new_mhc_split/"

EMBED_DIM = 100                     # change only after re-training the vectors in the new space
PEPTIDE_LENGTH = 45                 # set based on pep_n
MHC_AMINO_ACID_LENGTH = 120         # set based on mhc_n

BiLSTM_HIDDEN_SIZE = 64             # Hidden dim size for LSTM
BiLSTM_PEPTIDE_NUM_LAYERS = 3       # Number of stacked LSTM layers for peptides
BiLSTM_MHC_NUM_LAYERS = 3           # Number of stacked LSTM layers for MHC

LINEAR0_OUT = 2048          # Output size of first linear layer
LINEAR1_OUT = 64            # Output size of second linear layer
LINEAR2_OUT = 2             # Final output size (e.g., 2 for binary class)

# Attention and pooling params
CONTEXT_DIM = 16            # Dim of context vector in attention mech
MAXPOOL_FLAG = True         # Apply max pooling?

KERNEL_SIZE = 3
STRIDE_SIZE = 1

# Switch between standard splitting or new MHC-based dataset split
USE_MHC_SPLIT = True
