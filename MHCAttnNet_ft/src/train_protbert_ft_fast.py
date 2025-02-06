import os
import time
import sys
import argparse
from rich.progress import track
from data_loader import get_dataset, IEDB_raw
import config

# Metrics and stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
    precision_recall_fscore_support
)
from scipy.stats import pearsonr, spearmanr

# PyTorch and utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter

# Hugging Face Transformers
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    pipeline,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification
)

# Custom data loader and distributed training
import data_loader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import parsing
import argparse
from pynvml import *
import logging

# Confirm running cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parse commmand-line arguments
parser = parsing.create_parser()
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Set manual seed for reproducibility
torch.manual_seed(3)

# Create directory if doesn't exist
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

# Fetch device and number of epochs
epochs = config.epochs

# Configure logging for INFO-level messages
logging.basicConfig(level=logging.INFO)
base="/scratch/ssd004/scratch/vchu/PEPMHC"
last_ckpt="0"

def print_gpu_utilization():
    '''
    Print current GPU memory usage
    '''
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    '''
    Summarize stats from Trainer result
    (time, samples/sec, GPU usage)
    '''
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def init_process_group(local_rank):
    '''
    Initialize distributed process group (NCCL backend) for multi-GPU setup
    '''
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

def compute_metrics(pred):
    '''
    Custom metric function for eval
    '''
    # Extract labels and predictions
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(labels, pred.predictions.argmax(-1))

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, preds)
    prc_auc = average_precision_score(labels, preds)
    # Pearson and spearman correlations
    pcc, p = pearsonr(labels, preds)
    srcc, p = spearmanr(labels, preds)
    # Confusion matrix for sensitivity/PPV
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = float(tp)/(tp+fn)
    PPV = float(tp)/(tp+fp)

    return {
        'Accuracy': accuracy,
        'precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC_AUC': roc_auc,
        'PRC_AUC': prc_auc,
        'PCC': pcc,
        'Sensitivity': sensitivity,
        'PPV': PPV,
        'SRCC': srcc
    }

def train():
    '''
    Main train function
    '''
    print("Starting training...")
    # --- Data Loading ---
    # Create train and test dataset using custom IEDB_PEP_MHC Dataset class
    train_dataset = data_loader.IEDB_PEP_MHC(
        split="train",
        pep_max_len=args.pep_max_len,
        new_split_flag=args.new_split_flag
    )
    test_dataset = data_loader.IEDB_PEP_MHC(
        split="test",
        pep_max_len=args.pep_max_len,
        new_split_flag=args.new_split_flag
    )

    print(len(train_dataset), len(test_dataset))

    train_batch_size = 16
    test_batch_size = 16
    
    # --- Model Checkpoint Paths ---
    # Different base paths dependent on if args.scratch is True
    if args.scratch: 
        base_path= f'{base}/output/pep_scratch/lr_e_5/mlm_0.15/max_len_{args.pep_max_len}/500_epochs/'
    else:
        base_path= f'{base}/output/pep/lr_e_5/max_len_{args.pep_max_len}/'
    
    # --- Current Model ---
    # If using a pretrained model, else try loading a different cehckpoint
    if args.pretrained:           
        model_name1 = 'Rostlab/prot_bert_bfd'
    else: 
        checkpoint_path = base_path + f'checkpoint-{last_ckpt}'
        if os.path.exists(checkpoint_path):
            model_name1 = checkpoint_path
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Falling back to 'Rostlab/prot_bert_bfd'.")
            model_name1 = 'Rostlab/prot_bert_bfd'
    
    # Second model uses Rostlab/prot_bert_bfd
    model_name2 = 'Rostlab/prot_bert_bfd'
    # Load two backbone models from HuggingFace 
    model1 = AutoModel.from_pretrained(model_name1)
    model2 = AutoModel.from_pretrained(model_name2)

    # --- Combined Model ---
    class CombinedModel(nn.Module):
        def __init__(self, model1, model2):
            super().__init__()
            self.model1 = model1
            self.model2 = model2
            # Final classifier that takes in [CLS] embeddings from both models
            self.classifier = nn.Linear(model1.config.hidden_size + model2.config.hidden_size, 2)

                
        def forward(
            self, 
            pep_input_ids, pep_token_type_ids, pep_attention_mask,
            mhc_input_ids, mhc_token_type_ids, mhc_attention_mask, 
            labels=None
        ):

            outputs1 = self.model1(
                input_ids=pep_input_ids.squeeze(dim=1),
                token_type_ids=pep_token_type_ids.squeeze(dim=1),
                attention_mask=pep_attention_mask.squeeze(dim=1)
            )
            outputs2 = self.model2(
                input_ids=mhc_input_ids.squeeze(dim=1),
                token_type_ids=mhc_token_type_ids.squeeze(dim=1),
                attention_mask=mhc_attention_mask.squeeze(dim=1)
            )
            
            # Extract [CLS] token embedding from each model
            embeddings1 = outputs1.last_hidden_state[:, 0, :]
            embeddings2 = outputs2.last_hidden_state[:, 0, :]
            
            # Concatenate embeddings and pass through linear layer
            combined_embeddings = torch.cat((embeddings1, embeddings2), dim=-1)
            logits = self.classifier(combined_embeddings)
            
            # If labels are provided, compute cross-entropy loss
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                return (loss, logits)
            else:
                return logits
    
    # Instantiate combined model which merges the two separate Transformer models
    combined_model = CombinedModel(model1, model2)
    
    # --- Custom Trainer ---
    class CustomTrainer(Trainer):
        def __init__(self, *args,  train_dataloader=None, eval_dataloader=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_function = nn.CrossEntropyLoss()
            self.train_dataloader = train_dataloader
            self.eval_dataloader = eval_dataloader
        
        # Override compute_loss to handle (loss, logits) outputs from CombinedModel
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            labels = inputs.pop("labels").long()
            loss = outputs[0]   # CombinedModel returns (loss, logits)
            
            return (loss, outputs) if return_outputs else loss

    # --- Training Arguments ---
    save_path = f"{base}/output/pepmhc/mlm_0.15_max_len_{args.pep_max_len}_mhc_350/scratch_180000_step/new_split/"
    make_dir(save_path)

    training_args = TrainingArguments(
            output_dir=save_path,               # Directory for checkpoints/predictions
            overwrite_output_dir=True,          # Overwrite the output dir if exists
            num_train_epochs=epochs,                # Number of epochs to train
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=test_batch_size,
            max_steps=int(5000),                 # Total training steps (can override epochs if reached first)
            learning_rate=1e-05,                # Learning rate
            warmup_steps=int(1e3),              # Steps to warm up LR
            weight_decay=0.01,                  # Weight decay for regularizaiton
            logging_dir='./logs',               # Log directory
            logging_steps=int(1e3),             # Log every n steps
            do_train=True,                      # Perform training
            do_eval=True,                       # Perform evaluation
            eval_steps = int(5e3),              # Evaluation every n steps
            evaluation_strategy="steps",        # Evaluate every few steps (not just every epoch)
            gradient_accumulation_steps=1,      # Total number of steps before back propagation
            save_steps=int(1e3),                # Save checkpoint every n steps
            save_total_limit=2,                 # Keep only last n checkpoints
            bf16=True,
            # fp16=True,                          # Use mixed precision
            # fp16_opt_level="02",              # Mixed precision mode
            run_name="Siamese",                 # Experiment name
            metric_for_best_model="loss",       # Metric to select the best model
            # local_rank=int(os.environ["LOCAL_RANK"]),
            seed=3                              # Seed for experiment reproducibility
    )

    # --- Trainer Instance ---
    trainer = CustomTrainer(
        model=combined_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Train the model ---
    result = trainer.train()
    # Print training summary (time, samples/sec, GPU usage)
    print_summary(result)

if __name__ == "__main__":
    train()