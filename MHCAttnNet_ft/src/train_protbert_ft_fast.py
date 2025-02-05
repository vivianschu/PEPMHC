import os
import time
import sys
import argparse
from rich.progress import track
from data_loader import get_dataset, IEDB_raw
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import data_loader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import parsing
import argparse
from pynvml import *
import logging

parser = parsing.create_parser()
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# if __name__ == "__main__":
torch.manual_seed(3)  # for reproducibility

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


# device = config.device
# epochs = config.epochs

logging.basicConfig(level=logging.INFO)

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()



def init_process_group(local_rank):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(labels, pred.predictions.argmax(-1))


    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, preds)
    prc_auc = average_precision_score(labels, preds)
    pcc, p = pearsonr(labels, preds)
    srcc, p = spearmanr(labels, preds)
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

   

    train_dataset = data_loader.IEDB_PEP_MHC(split="train", pep_max_len=args.pep_max_len, new_split_flag=args.new_split_flag)
    test_dataset = data_loader.IEDB_PEP_MHC(split="test",   pep_max_len=args.pep_max_len, new_split_flag=args.new_split_flag)

    print(len(train_dataset), len(test_dataset))

    train_batch_size = 16
    test_batch_size = 16



    # base_path= '/home/patrick3/projects/def-wanglab-ab/patrick3/output_directory/pep/lr_0.001/'
    # base_path= '/home/patrick3/scratch/output_directory/pep/lr_e_5/max_len_{}/'.format(args.pep_max_len)

    if args.scratch: 
        base_path= '/scratch/ssd004/scratch/vchu/PEPMHC/output/pep_scratch/lr_e_5/mlm_0.15/max_len_{}/500_epochs/'.format(args.pep_max_len)
    else:
        # base_path= '/home/patrick3/scratch/output_directory/pep/lr_e_5/mlm_0.15/max_len_{}/500_epochs/'.format(args.pep_max_len)
        base_path= '/scratch/ssd004/scratch/vchu/PEPMHC/output/pep/lr_e_5/max_len_{}/'.format(args.pep_max_len)
    
    
    if args.pretrained:           
        model_name1 = 'Rostlab/prot_bert_bfd'
    else: 
        checkpoint_path = base_path + 'checkpoint-{}'.format(44000)
        if os.path.exists(checkpoint_path):
            model_name1 = checkpoint_path
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Falling back to 'Rostlab/prot_bert_bfd'.")
            model_name1 = 'Rostlab/prot_bert_bfd'

    model_name2 = 'Rostlab/prot_bert_bfd'

    model1 = AutoModel.from_pretrained(model_name1)
    model2 = AutoModel.from_pretrained(model_name2)



    class CombinedModel(nn.Module):
        def __init__(self, model1, model2):
            super().__init__()
            self.model1 = model1
            self.model2 = model2
            self.classifier = nn.Linear(model1.config.hidden_size + model2.config.hidden_size, 2)

                
        def forward(self, pep_input_ids, pep_token_type_ids, pep_attention_mask, mhc_input_ids, mhc_token_type_ids, mhc_attention_mask, labels=None):

            outputs1 = self.model1(input_ids=pep_input_ids.squeeze(dim=1), token_type_ids=pep_token_type_ids.squeeze(dim=1), attention_mask=pep_attention_mask.squeeze(dim=1))
            outputs2 = self.model2(input_ids=mhc_input_ids.squeeze(dim=1), token_type_ids=mhc_token_type_ids.squeeze(dim=1), attention_mask=mhc_attention_mask.squeeze(dim=1))


            embeddings1 = outputs1.last_hidden_state[:, 0, :]
            embeddings2 = outputs2.last_hidden_state[:, 0, :]

    
            combined_embeddings = torch.cat((embeddings1, embeddings2), dim=-1)
            logits = self.classifier(combined_embeddings)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))

                return (loss, logits)

            else:
                return logits


    

    combined_model = CombinedModel(model1, model2)

    class CustomTrainer(Trainer):
        def __init__(self, *args,  train_dataloader=None, eval_dataloader=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_function = nn.CrossEntropyLoss()
            self.train_dataloader = train_dataloader
            self.eval_dataloader = eval_dataloader

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            labels = inputs.pop("labels").long()
     
            loss = outputs[0]    
            # loss = self.loss_function(outputs[1].view(-1, 2), labels.view(-1))

            return (loss, outputs) if return_outputs else loss

 

    save_path = "/scratch/ssd004/scratch/vchu/PEPMHC/output/pep_mhc/mlm_0.15_max_len_{}_mhc_350/scratch_180000_step/new_split/".format(args.pep_max_len)
    make_dir(save_path)

    training_args = TrainingArguments(
            output_dir=save_path,         # Output directory for model checkpoints and predictions
            overwrite_output_dir=True,             # Overwrite the output directory if it already exists
            num_train_epochs=10,                # total number of training epochs
            per_device_train_batch_size=train_batch_size,      # batch size per device during training
            per_device_eval_batch_size=test_batch_size,       # batch size for evaluation
            max_steps=int(3e5),
            learning_rate=1e-05,                    # learning rate
            warmup_steps=int(3e3),              # number of warmup steps for learning rate scheduler
            weight_decay=0.01,                  # strength of weight decay
            logging_dir='./logs',               # directory for storing logs
            logging_steps=int(3e3),             # How often to print logs
            do_train=True,                      # Perform training
            do_eval=True,                       # Perform evaluation
            eval_steps = int(3e4),
            evaluation_strategy="steps",        # evalute after eachh epoch
            gradient_accumulation_steps=1,      # total number of steps before back propagation
            save_steps=int(3e4),                # Save the model every n steps
            save_total_limit=2,                 # Maximum number of model checkpoints to keep
            fp16=True,                          # Use mixed precision
            # fp16_opt_level="02",              # mixed precision mode
            run_name="Two Model",               # experiment name
            metric_for_best_model="loss",       # Metric to select the best model
            # local_rank=int(os.environ["LOCAL_RANK"]),
            seed=3                              # Seed for experiment reproducibility 3x3
            
    )



    trainer = CustomTrainer(
        model=combined_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )


    # Step 7: Fine-tune the custom model using the Trainer.train() method

    result = trainer.train()

    print_summary(result)


    # Load model from checkpoint
    
    # checkpoint = './checkpoint'  # Replace with your checkpoint directory
    # model = BertForSequenceClassification.from_pretrained(checkpoint)
    # trainer.model = model


if __name__ == "__main__":
 
    train()







# def model_init():
#     # local_rank = torch.distributed.get_rank()
#     # torch.cuda.set_device(local_rank)
#     model= AutoModelForSequenceClassification.from_pretrained(model_name)
#     return  model
#     # return  nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])



# tokenizer_list= ['Rostlab/prot_bert_bfd', 'Rostlab/prot_bert']
# pep_model_name = '/home/patrick3/projects/def-wanglab-ab/patrick3/output_directory/pep/checkpoint-4500/'
# mhc_model_name = 'Rostlab/prot_bert_bfd'



# for tokenizer_name in tokenizer_list:

#     if config.USE_MHC_SPLIT==True:
#         from data_loader_new_split import IEDB_raw
#         train_dataset = IEDB_raw(split="train", tokenizer_name=tokenizer_name, max_length=512)
#         test_dataset  = IEDB_raw(split="test",  tokenizer_name=tokenizer_name, max_length=512)
#     else:
#         train_dataset = IEDB_raw(split="train", tokenizer_name=tokenizer_name, max_length=512)
#         test_dataset  = IEDB_raw(split="test",  tokenizer_name=tokenizer_name, max_length=512)



#     training_args = TrainingArguments(
#         output_dir='./results/'+model_name,      # output directory
#         num_train_epochs=5,                     # total number of training epochs
#         per_device_train_batch_size=4,          # batch size per device during training
#         per_device_eval_batch_size=10,          # batch size for evaluation
#         warmup_steps=1000,                      # number of warmup steps for learning rate scheduler
#         learning_rate=2e-05,                    # learning rate
#         weight_decay=0.01,                      # strength of weight decay
#         logging_dir='./logs',                   # directory for storing logs
#         logging_steps=2000,                    # How often to print logs
#         do_train=True,                          # Perform training
#         do_eval=True,                           # Perform evaluation
#         evaluation_strategy="epoch",            # evalute after eachh epoch
#         gradient_accumulation_steps=64,         # total number of steps before back propagation
#         fp16=True,                              # Use mixed precision
#         fp16_opt_level="02",                    # mixed precision mode
#         run_name=model_name.strip('Rostlab/'),  # experiment name
#         seed=3,                                  # Seed for experiment reproducibility 3x3
#         # Additional arguments for distributed training
#         local_rank=int(os.environ["LOCAL_RANK"]),
#     )

#     trainer = Trainer(
#         model_init=model_init,                  # the instantiated 🤗 Transformers model to be trained
#         args=training_args,                     # training arguments, defined above
#         train_dataset=train_dataset,            # training dataset
#         eval_dataset=test_dataset,              # evaluation dataset
#         compute_metrics = compute_metrics,      # evaluation metrics
#     )

#     trainer.train()

#     torch.cuda.empty_cache()