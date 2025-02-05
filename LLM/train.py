from tqdm import tqdm
from transformers import BertTokenizerFast
import load_sentence_pair
import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, pipeline
import torch
import os
# create a python generator to dynamically load the data

from transformers import DataCollatorForLanguageModeling
import logging
import parsing

logging.basicConfig(level=logging.INFO)

tokenizer_name = 'Rostlab/prot_bert_bfd'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors='pt')

from transformers import TrainingArguments, AutoModelForMaskedLM, Trainer

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()



max_len = 48
save_path = "/scratch/ssd004/scratch/vchu/PEPMHC/output/pep/lr_e_5/mlm_0.15/max_len_{}/500_epochs/".format(max_len)

make_dir(save_path)

training_args = TrainingArguments(
    output_dir=save_path,                  # Output directory for model checkpoints and predictions
    overwrite_output_dir=True,             # Overwrite the output directory if it already exists
    do_train=True,                         # Perform training
    do_eval=False,                         # Perform evaluation
    # evaluation_strategy="steps",         # Evaluate the model at each logging step
    per_device_train_batch_size=256,       # Batch size for training
    # per_device_eval_batch_size=1,        # Batch size for evaluation
    gradient_accumulation_steps=4,         # Update model weights every n steps
    learning_rate=1e-5,                    # Learning rate
    weight_decay=0.01,                     # Weight decay for regularization
    adam_epsilon=1e-8,                     # Epsilon for the Adam optimizer
    max_grad_norm=1.0,                     # Gradient clipping
    # max_steps=5,
    num_train_epochs=1000,                 # Number of training epochs
    warmup_steps=0,                        # Number of warm-up steps
    logging_dir="./logs",                  # Directory for storing logs
    logging_steps=2000,                    # Log every n steps
    save_steps=12000,                      # Save the model every n steps
    save_total_limit=2,                    # Maximum number of model checkpoints to keep
    fp16=True ,                             # Use mixed precision training (if supported by GPU)
    # load_best_model_at_end=True,         # Load the best model at the end of training
    metric_for_best_model="loss",          # Metric to select the best model
    # greater_is_better=False,             # Set to "False" if lower metric values are better
    # Additional arguments for distributed training
    # local_rank=int(os.environ["LOCAL_RANK"]),
    seed=3

)


model = AutoModelForMaskedLM.from_pretrained('Rostlab/prot_bert_bfd')



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=load_sentence_pair.IEDB_pep(tokenizer_name='Rostlab/prot_bert_bfd', max_length=max_len),
    data_collator=data_collator,
)

result = trainer.train(resume_from_checkpoint = False)

print_summary(result)
