from tqdm import tqdm
from transformers import BertTokenizerFast
import load_sentence_pair
import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel, pipeline
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





import torch.distributed as dist
import torch.utils.data.distributed

import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')


parser.add_argument('--scratch', action='store_true', help='')



print("Starting...")

args = parser.parse_args()

ngpus_per_node = torch.cuda.device_count()

""" This next line is the key to getting DistributedDataParallel working on SLURM:
    SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
    current process inside a node and is also 0 or 1 in this example."""

local_rank = int(os.environ.get("SLURM_LOCALID")) 
rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

current_device = local_rank

torch.cuda.set_device(current_device)

""" this block initializes a process group and initiate communications
    between all processes running on all nodes """

print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
#init the process group
dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
print("process group ready!")

print('From Rank: {}, ==> Making model..'.format(rank))






max_len = 48
if args.scratch ==True:
    save_path = "/home/patrick3/scratch/output_directory/pep_scratch/lr_e_5/mlm_0.15/max_len_{}/500_epochs/".format(max_len)
else:

    save_path = "/home/patrick3/scratch/output_directory/pep/lr_e_5/mlm_0.15/max_len_{}/500_epochs/".format(max_len)

make_dir(save_path)

training_args = TrainingArguments(
    output_dir=save_path,         # Output directory for model checkpoints and predictions
    overwrite_output_dir=True,             # Overwrite the output directory if it already exists
    do_train=True,                         # Perform training
    do_eval=False,                          # Perform evaluation
    # evaluation_strategy="steps",           # Evaluate the model at each logging step
    per_device_train_batch_size=256,         # Batch size for training
    # per_device_eval_batch_size=1,          # Batch size for evaluation
    gradient_accumulation_steps=4,        # Update model weights every n steps  # 16 if single gpu, 8 if 2 gpus, and 4 if 4 gpus
    learning_rate=1e-5,                    # Learning rate
    weight_decay=0.01,                     # Weight decay for regularization
    adam_epsilon=1e-8,                     # Epsilon for the Adam optimizer
    max_grad_norm=1.0,                     # Gradient clipping
    # max_steps=5,
    num_train_epochs=1000,                    # Number of training epochs
    warmup_steps=0,                        # Number of warm-up steps
    logging_dir="./logs",                    # Directory for storing logs
    logging_steps=2000,                     # Log every n steps
    save_steps=12000,                       # Save the model every n steps
    save_total_limit=2,                    # Maximum number of model checkpoints to keep
    fp16=True,                             # Use mixed precision training (if supported by GPU)
    # load_best_model_at_end=True,           # Load the best model at the end of training
    metric_for_best_model="loss",          # Metric to select the best model
    # greater_is_better=False,               # Set to "False" if lower metric values are better
    # Additional arguments for distributed training
    # local_rank=int(os.environ["LOCAL_RANK"]),
    seed=3,

)

model = AutoModelForMaskedLM.from_pretrained('./Rostlab/prot_bert_bfd')

if args.scratch ==True:
# Model randomly initialized (starting from scratch)
    config = AutoConfig.for_model('bert')
    # Update config if you'd like
    # config.update({"param": value})
    model = AutoModelForMaskedLM.from_config(config)


model.cuda()
net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])

print('From Rank: {}, ==> Preparing data..'.format(rank))    



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=load_sentence_pair.IEDB_pep(tokenizer_name='./Rostlab/prot_bert_bfd', max_length=max_len),
    data_collator=data_collator,

)


result = trainer.train(resume_from_checkpoint = True)

