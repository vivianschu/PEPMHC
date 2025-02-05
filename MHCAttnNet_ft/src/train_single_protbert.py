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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification


if __name__ == "__main__":
    torch.manual_seed(3)  # for reproducibility

    device = config.device
    epochs = config.epochs

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




    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     model = torch.nn.DataParallel(model)
        return model

    model_list = ['Rostlab/prot_bert_bfd']

    for model_name in model_list:

        if config.USE_MHC_SPLIT==True:
            from data_loader_new_split import IEDB_raw
            train_dataset = IEDB_raw(split="train", tokenizer_name=model_name, max_length=512)
            test_dataset  = IEDB_raw(split="test",  tokenizer_name=model_name, max_length=512)
        else:
            train_dataset = IEDB_raw(split="train", tokenizer_name=model_name, max_length=512)
            test_dataset  = IEDB_raw(split="test",  tokenizer_name=model_name, max_length=512)


        print(len(train_dataset), len(test_dataset))

        training_args = TrainingArguments(
            output_dir='./results/'+model_name,     # output directory
            num_train_epochs=10,                    # total number of training epochs
            # max_steps=int(5),
            per_device_train_batch_size=4,          # batch size per device during training
            per_device_eval_batch_size=4,           # batch size for evaluation
            warmup_steps=1000,                      # number of warmup steps for learning rate scheduler
            learning_rate=1e-05,                    # learning rate
            weight_decay=0.01,                      # strength of weight decay
            logging_dir='./logs',                   # directory for storing logs
            logging_steps=20000,                    # How often to print logs
            do_train=True,                          # Perform training
            do_eval=True,                           # Perform evaluation
            evaluation_strategy="epoch",            # evalute after eachh epoch
            gradient_accumulation_steps=32,         # total number of steps before back propagation
            fp16=True,                              # Use mixed precision
            fp16_opt_level="02",                    # mixed precision mode
            run_name=model_name.strip('Rostlab/'),  # experiment name
            # local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            seed=3                                  # Seed for experiment reproducibility 3x3
        )

        trainer = Trainer(
            model_init=model_init,                  # the instantiated 🤗 Transformers model to be trained
            args=training_args,                     # training arguments, defined above
            train_dataset=train_dataset,            # training dataset
            eval_dataset=test_dataset,              # evaluation dataset
            compute_metrics = compute_metrics,      # evaluation metrics
        )

        trainer.train()

        torch.cuda.empty_cache()







    #s
