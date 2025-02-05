import math
import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchtext
from torchtext.data import BucketIterator, Field, interleave_keys, RawField
from torchtext.data.dataset import TabularDataset
from torchtext.data.pipeline import Pipeline
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers import BertModel, BertTokenizer
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, pipeline
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import requests
from tqdm import tqdm
import numpy as np
import re
from rich.progress import track

import config


def tokenize_pep(seq):
    return list(map("".join, zip(*[iter(seq)]*config.pep_n)))

def tokenize_mhc(seq):
    return list(map("".join, zip(*[iter(seq)]*config.mhc_n)))




class IEDB_raw(Dataset):

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd', max_length=512):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasetFolderPath = config.new_base_path
        self.trainFilePath = os.path.join(self.datasetFolderPath, config.train_file)
        self.testFilePath = os.path.join(self.datasetFolderPath,  config.test_file)


        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.split = split
        if split=="train":
          self.peps, self.mhcs, self.labels = self.load_dataset(self.trainFilePath)
        else:
          self.peps, self.mhcs, self.labels = self.load_dataset(self.testFilePath)

        self.max_length = max_length


    def load_dataset(self,path):
        df = pd.read_csv(path,names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)

        pep = list(df['peptide'])
        mhc = list(df['mhc_amino_acid'])
        label = list(df['bind'])

        assert len(pep) == len(label)

        return pep, mhc, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pep = " ".join("".join(self.peps[idx].split()))
        pep = re.sub(r"[UZOB]", "X", pep)

        mhc = " ".join("".join(self.mhcs[idx].split()))
        mhc = re.sub(r"[UZOB]", "X", mhc)

        seq_ids = self.tokenizer(pep ,mhc,truncation=True, padding='max_length', max_length=self.max_length)
        # print(seq_ids)
        # exit(0)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample









#
