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
import ray
from ray import tune
from ray import air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.util.sgd.torch import TorchTrainer, TrainingOperator
from ray.util.sgd.torch.resnet_creator import resnet_creator
# if __name__ == "__main__":
torch.manual_seed(3)  # for reproducibility

# device = config.device
# epochs = config.epochs


def print_gpu_memory_usage():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check if your system has an NVIDIA GPU and if it's properly installed.")
        return

    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = reserved_memory - allocated_memory

    print(f"Total GPU memory: {total_memory / (1024 ** 2):.2f} MB")
    print(f"Reserved GPU memory: {reserved_memory / (1024 ** 2):.2f} MB")
    print(f"Allocated GPU memory: {allocated_memory / (1024 ** 2):.2f} MB")
    print(f"Free GPU memory: {free_memory / (1024 ** 2):.2f} MB")


def init_process_group(local_rank):
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

def set_environment_variables(local_rank):
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["LOCAL_RANK"] = str(local_rank)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

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
            'Recall':   recall,
            'F1':   f1,
            'ROC_AUC': roc_auc,
            'PRC_AUC': prc_auc,
            'PCC': pcc,
            'Sensitivity': sensitivity,
            'PPV': PPV,
            'SRCC': srcc
            }




def train():


        
    class CombinedModel(nn.Module):
        def __init__(self, model1, model2):
            super().__init__()
            self.model1 = model1
            self.model2 = model2
            self.classifier = nn.Linear(model1.config.hidden_size + model2.config.hidden_size, 2)

        def forward(self, pep_input_ids, pep_token_type_ids, pep_attention_mask, mhc_input_ids, mhc_token_type_ids, mhc_attention_mask, labels=None):
            outputs1 = self.model1(input_ids=pep_input_ids.squeeze(), token_type_ids=pep_token_type_ids.squeeze(), attention_mask=pep_attention_mask.squeeze())
            outputs2 = self.model2(input_ids=mhc_input_ids.squeeze(), token_type_ids=mhc_token_type_ids.squeeze(), attention_mask=mhc_attention_mask.squeeze())

            embeddings1 = outputs1.last_hidden_state[:, 0, :]
            embeddings2 = outputs2.last_hidden_state[:, 0, :]

            combined_embeddings = torch.cat((embeddings1, embeddings2), dim=-1)
            logits = self.classifier(combined_embeddings)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                return (logits, loss)
            else:
                return logits


    combined_model = CombinedModel(model1, model2)


    class CustomTrainingOperator(TrainingOperator):
        def setup(self, config):
            # Load datasets
            train_dataset = data_loader.IEDB_PEP_MHC(split="train")
            test_dataset = data_loader.IEDB_PEP_MHC(split="test")

            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

            # Load pre-trained models
            base_path = '/home/patrick3/projects/def-wanglab-ab/patrick3/output_directory/pep/lr_0.001/'
            model_name1 = base_path + 'checkpoint-22500'
            model_name2 = 'Rostlab/prot_bert_bfd'
            model1 = AutoModel.from_pretrained(model_name1)
            model2 = AutoModel.from_pretrained(model_name2)

            # Create combined model
            combined_model = CombinedModel(model1, model2)

            # Set up optimizer and loss function
            optimizer = torch.optim.SGD(combined_model.parameters(), lr=config["lr"])
            loss_function = nn.CrossEntropyLoss()

            self.model, self.optimizer, self.criterion = self.register(models=combined_model, optimizers=optimizer, criterion=loss_function)
            self.register_data(train_loader=train_loader, validation_loader=test_loader)

def train(config, num_workers=1, use_gpu=False, num_epochs=5):
    trainer = TorchTrainer(
        training_operator_cls=CustomTrainingOperator,
        config=config,
        num_workers=num_workers,
        use_gpu=use_gpu,
        backend="nccl" if use_gpu else "gloo",
    )

    for i in range(num_epochs):
        stats = trainer.train()
        print(f"Epoch {i + 1}: {stats}")
        val_stats = trainer.validate()
        print(f"Validation: {val_stats}")

    trainer.shutdown()


if __name__ == "__main__":
    # Set up hyperparameter search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64])
    }

    # Configure Ray Tune
    num_samples = 10
    scheduler = ASHAScheduler(metric="val_loss", mode="min")
    reporter = CLIReporter(metric_columns=["val_loss", "val_accuracy", "training_iteration"])


    # Start Ray Tune
    tune.run(
        tune.with_parameters(train, num_epochs=5),
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )








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







    #s
