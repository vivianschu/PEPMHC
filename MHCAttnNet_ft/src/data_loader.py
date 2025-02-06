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
import requests
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from rich.progress import track
from sentence_transformers import SentenceTransformer, InputExample, losses
import config


def tokenize_pep(seq):
    return list(map("".join, zip(*[iter(seq)]*config.pep_n)))

def tokenize_mhc(seq):
    return list(map("".join, zip(*[iter(seq)]*config.mhc_n)))

class IEDB_PEP_MHC(Dataset):

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd', pep_max_len=48, new_split_flag=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if new_split_flag==True:

            self.datasetFolderPath = config.new_base_path
            self.trainFilePath = os.path.join(self.datasetFolderPath, config.train_file)
            self.testFilePath = os.path.join(self.datasetFolderPath,  config.test_file)
        
        else:
            self.datasetFolderPath = config.base_path
            self.trainFilePath = os.path.join(self.datasetFolderPath, config.train_file)
            self.valFilePath = os.path.join(self.datasetFolderPath,   config.val_file)
            self.testFilePath = os.path.join(self.datasetFolderPath,  config.test_file)


        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.split = split
        self.new_split_flag = new_split_flag
        if split=="train":
          self.peps, self.mhcs, self.labels = self.load_dataset(self.trainFilePath)
        else:
          self.peps, self.mhcs, self.labels = self.load_dataset(self.testFilePath)

        self.pep_max_length = pep_max_len # max 43
        self.mhc_max_length = 350 # max 347 in class I and 232 in class II



    def load_dataset(self,path):
        df = pd.read_csv(path,names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)


        pep = list(df['peptide'])
        mhc = list(df['mhc_amino_acid'])
        label = list(df['bind'])

        if self.split=="train" and self.new_split_flag==False:
            df_val = pd.read_csv(self.valFilePath, names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)

            pep_val = list(df_val['peptide'])
            mhc_val = list(df_val['mhc_amino_acid'])
            label_val = list(df_val['bind'])

            pep += pep_val
            mhc += mhc_val
            label += label_val


        assert len(pep) == len(label) and len(mhc) == len(label)

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

        pep_seq_ids = self.tokenizer(pep, truncation=True, padding='max_length', max_length=self.pep_max_length, return_tensors='pt')
        mhc_seq_ids = self.tokenizer(mhc, truncation=True, padding='max_length', max_length=self.mhc_max_length, return_tensors='pt')


        sample = {'pep_input_ids': pep_seq_ids['input_ids'], 'pep_token_type_ids': pep_seq_ids['token_type_ids'], 'pep_attention_mask': pep_seq_ids['attention_mask'],
                    'mhc_input_ids': mhc_seq_ids['input_ids'], 'mhc_token_type_ids': mhc_seq_ids['token_type_ids'], 'mhc_attention_mask': mhc_seq_ids['attention_mask'],  
                    'labels': torch.tensor(self.labels[idx],  dtype=torch.long)            
                  }


        return sample
    
    

    

class IEDB_raw(Dataset):

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd', max_length=1024):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasetFolderPath = config.base_path
        self.trainFilePath = os.path.join(self.datasetFolderPath, config.train_file)
        self.valFilePath = os.path.join(self.datasetFolderPath,   config.val_file)
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

        if self.split=="train":
            df = pd.read_csv(self.valFilePath, names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)

            pep = pep + list(df['peptide'])
            mhc = mhc + list(df['mhc_amino_acid'])
            label = label + list(df['bind'])


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



class IEDB_feat1(Dataset): # output [1024]

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasetFolderPath = config.base_path
        self.trainFilePath = os.path.join(self.datasetFolderPath, config.train_file)
        self.valFilePath = os.path.join(self.datasetFolderPath,   config.val_file)
        self.testFilePath = os.path.join(self.datasetFolderPath,  config.test_file)
        self.device = config.device

        self.tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
        self.model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        if split=="train":
          self.peps, self.mhcs, self.labels = self.load_dataset(self.trainFilePath)
        elif split=="val":
          self.peps, self.mhcs, self.labels = self.load_dataset(self.valFilePath)
        else:
          self.peps, self.mhcs, self.labels = self.load_dataset(self.testFilePath)


        print(split)

    def load_dataset(self,path):
        df = pd.read_csv(path,names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)

        pep_list = list(df['peptide'])
        mhc_list = list(df['mhc_amino_acid'])
        label = list(df['bind'])


        # assert len(pep) == len(label)

        # self.peps = [' '.join(peps) for peps in self.peps]

        with torch.no_grad():
            # self.peps = " ".join("".join(self.peps[idx].split()))
            pep_data = []
            for peps in tqdm(pep_list):
                peps = [" ".join(re.sub(r"[UZOB]", "X", peps))]
                ids = self.tokenizer.batch_encode_plus(peps,  max_length=45 ,add_special_tokens=True, pad_to_max_length=True)
                input_ids = torch.tensor(ids['input_ids']).to(self.device)
                attention_mask = torch.tensor(ids['attention_mask']).to(self.device)


                with torch.no_grad():
                    embedding = self.model(input_ids=input_ids,attention_mask=attention_mask)[0]
                    embedding = embedding.cpu().numpy()
                    pep_data.append(embedding.tolist())

            pep_data = torch.tensor(np.array(pep_data))

            mhc_data = []
            for mhcs in tqdm(mhc_list):
                mhcs = [" ".join(re.sub(r"[UZOB]", "X", mhcs))]
                ids = self.tokenizer.batch_encode_plus(mhcs,  max_length=350 ,add_special_tokens=True, pad_to_max_length=True)
                input_ids = torch.tensor(ids['input_ids']).to(self.device)
                attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

                with torch.no_grad():
                    embedding = self.model(input_ids=input_ids,attention_mask=attention_mask)[0]
                    embedding = embedding.cpu().numpy()
                    mhc_data.append(embedding.tolist())

            mhc_data = torch.tensor(np.array(mhc_data))


        return pep_data, mhc_data, label

class IEDB_feat2(Dataset): # output [seq, 1024]

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasetFolderPath = config.base_path
        self.trainFilePath = os.path.join(self.datasetFolderPath, config.train_file)
        self.valFilePath = os.path.join(self.datasetFolderPath,   config.val_file)
        self.testFilePath = os.path.join(self.datasetFolderPath,  config.test_file)
        self.device = config.device

        self.tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
        self.model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        if split=="train":
          self.peps, self.mhcs, self.labels = self.load_dataset(self.trainFilePath)
        elif split=="val":
          self.peps, self.mhcs, self.labels = self.load_dataset(self.valFilePath)
        else:
          self.peps, self.mhcs, self.labels = self.load_dataset(self.testFilePath)


        print(split)

    def load_dataset(self,path):
        df = pd.read_csv(path,names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)

        pep_list = list(df['peptide'])
        mhc_list = list(df['mhc_amino_acid'])
        label = list(df['bind'])

        return pep_list, mhc_list, label

    # def __len__(self):
    #     return len(self.labels)

    # def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
    # def __iter__(self):
    #     for peps, mhcs, labels in zip(self.peps, self.mhcs, self.labels):
    #
    #         peps = [" ".join(re.sub(r"[UZOB]", "X", peps))]
    #         ids = self.tokenizer.batch_encode_plus(peps,  max_length=45, add_special_tokens=True, pad_to_max_length=True)
    #         input_ids = torch.tensor(ids['input_ids']).to(self.device)
    #         attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
    #
    #         with torch.no_grad():
    #             embedding = self.model(input_ids=input_ids,attention_mask=attention_mask)[0]
    #             embedding = embedding.cpu().numpy()
    #
    #         pep_data = torch.tensor(np.array(embedding))
    #
    #         mhcs = [" ".join(re.sub(r"[UZOB]", "X", mhcs))]
    #         ids = self.tokenizer.batch_encode_plus(mhcs,  max_length=350, add_special_tokens=True, pad_to_max_length=True)
    #         input_ids = torch.tensor(ids['input_ids']).to(self.device)
    #         attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
    #
    #         with torch.no_grad():
    #             embedding = self.model(input_ids=input_ids,attention_mask=attention_mask)[0]
    #             embedding = embedding.cpu().numpy()
    #
    #
    #         mhc_data = torch.tensor(np.array(embedding))
    #
    #         label = torch.tensor(labels)
    #
    #
    #         yield pep_data, mhc_data, label

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):


        peps = [" ".join(re.sub(r"[UZOB]", "X", self.peps[idx]))]
        ids = self.tokenizer.batch_encode_plus(peps,  max_length=45, add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        # token_type=  torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1 ]).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            embedding = embedding.cpu().numpy()

        pep_data = torch.tensor(np.array(embedding))
        print(pep_data)
        exit(0)
        mhcs = [" ".join(re.sub(r"[UZOB]", "X", self.mhcs[idx]))]
        ids = self.tokenizer.batch_encode_plus(mhcs,  max_length=350, add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_ids=input_ids,attention_mask=attention_mask)[0]
            embedding = embedding.cpu().numpy()


        mhc_data = torch.tensor(np.array(embedding))

        label = torch.tensor(self.labels[idx])


        return pep_data, mhc_data, label




class IEDB_features(Dataset):

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd', max_length=1024):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasetFolderPath = config.base_path
        self.trainFilePath = os.path.join(self.datasetFolderPath, config.train_file)
        self.valFilePath = os.path.join(self.datasetFolderPath,   config.val_file)
        self.testFilePath = os.path.join(self.datasetFolderPath,  config.test_file)


        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

        if split=="train":
          self.seqs, self.labels = self.load_dataset(self.trainFilePath)
        elif split=="val":
          self.seqs, self.labels = self.load_dataset(self.valFilePath)
        else:
          self.seqs, self.labels = self.load_dataset(self.testFilePath)

        self.max_length = max_length


    def load_dataset(self,path):
        df = pd.read_csv(path,names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)


        pep = list(df['peptide'])
        mhc = list(df['mhc_amino_acid'])
        label = list(df['bind'])

        cat = list(df['peptide']+df['mhc_amino_acid'])

        assert len(pep) == len(label)

        return cat, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample



class IEDB_feat_pmhc(Dataset): # output [seq, 1024]

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasetFolderPath = config.base_path
        self.trainFilePath = os.path.join(self.datasetFolderPath, config.train_file)
        self.valFilePath = os.path.join(self.datasetFolderPath,   config.val_file)
        self.testFilePath = os.path.join(self.datasetFolderPath,  config.test_file)
        self.device = config.device

        self.tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
        self.model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        if split=="train":
          self.peps, self.mhcs, self.labels = self.load_dataset(self.trainFilePath)
        elif split=="val":
          self.peps, self.mhcs, self.labels = self.load_dataset(self.valFilePath)
        else:
          self.peps, self.mhcs, self.labels = self.load_dataset(self.testFilePath)


        print(split)

    def load_dataset(self,path):
        df = pd.read_csv(path,names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)

        pmhc_list   = list(df['peptide'] + df['mhc_amino_acid'])
        label       = list(df['bind'])

        return  pmhc_list, label

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):


        peps = [" ".join(re.sub(r"[UZOB]", "X", self.peps[idx]))]
        ids = self.tokenizer.batch_encode_plus(peps,  max_length=45, add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        # token_type=  torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1 ]).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            embedding = embedding.cpu().numpy()

        pep_data = torch.tensor(np.array(embedding))
        print(pep_data)
        exit(0)
        mhcs = [" ".join(re.sub(r"[UZOB]", "X", self.mhcs[idx]))]
        ids = self.tokenizer.batch_encode_plus(mhcs,  max_length=350, add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_ids=input_ids,attention_mask=attention_mask)[0]
            embedding = embedding.cpu().numpy()


        mhc_data = torch.tensor(np.array(embedding))

        label = torch.tensor(self.labels[idx])


        return pep_data, mhc_data, label

def train_examples(model_name):

    datasetFolderPath = config.base_path
    trainFilePath = os.path.join(datasetFolderPath, config.train_file)
    valFilePath   = os.path.join(datasetFolderPath,   config.val_file)



    df_train = pd.read_csv(trainFilePath,names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)
    df_val   = pd.read_csv(valFilePath,names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],  skiprows=1)



    pep_list      = list(df_train['peptide']) + list(df_val['peptide'])
    mhc_list      = list(df_train['mhc_amino_acid']) + list(df_val['mhc_amino_acid'])
    label_list    = list(df_train['bind']) + list(df_val['bind'])

    idx_list  = list(range(len(label_list)))

    # sequences_Example = 'A B C', 'D E F'
    # print(sequences_Example)



    if model_name == 'Rostlab/prot_bert':
        # print([" ".join(re.sub(r"[UZOB]", "X", pep_list[9]))])
        # exit(0)

        pep_list = [" ".join(re.sub(r"[UZOB]", "X", sequence)) for sequence in pep_list]
        mhc_list = [" ".join(re.sub(r"[UZOB]", "X", sequence)) for sequence in mhc_list]
        samples = [InputExample(texts=[pep_list[idx], mhc_list[idx]], label=label_list[idx]) for idx in idx_list]
        # print('pep', pep_list[1], 'mhc', mhc_list[1], 'label', label_list[1])
        # exit(0)
        # tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False )
        # tokenized = tokenizer(pep_list[0], mhc_list[0], padding=True, truncation=True, return_tensors="pt", max_length=512)
        # print(tokenized)
        # exit(0)
    else:
        samples = [InputExample(texts=[pep_list[idx], mhc_list[idx]], label=label_list[idx]) for idx in idx_list]

    return samples


def test_examples(model_name):

    datasetFolderPath = config.base_path
    testFilePath = os.path.join(datasetFolderPath, config.test_file)
    # valFilePath   = os.path.join(datasetFolderPath,   config.val_file)


    df_test = pd.read_csv(testFilePath,names=['peptide','mhc_amino_acid','bind', 'mhc_allele'],skiprows=1)


    pep_list      = list(df_test['peptide'])
    mhc_list      = list(df_test['mhc_amino_acid'])
    label_list    = list(df_test['bind'])

    idx_list  = list(range(len(label_list)))
    #
    if model_name == 'Rostlab/prot_bert':
        pep_list = [" ".join(re.sub(r"[UZOB]", "X", sequence)) for sequence in pep_list]
        mhc_list = [" ".join(re.sub(r"[UZOB]", "X", sequence)) for sequence in mhc_list]
        samples = [InputExample(texts=[pep_list[idx], mhc_list[idx]], label=label_list[idx]) for idx in idx_list]
    else:
        samples = [InputExample(texts=[pep_list[idx], mhc_list[idx]], label=label_list[idx]) for idx in idx_list]



    return samples



class IEDB(TabularDataset):

    def __init__(self, path, format, fields, skip_header=True, **kwargs):
        super(IEDB, self).__init__(path, format, fields, skip_header, **kwargs)

        # keep a raw copy of the sentence for debugging
        RAW_TEXT_FIELD = RawField()
        for ex in self.examples:
            raw_peptide, raw_mhc_amino_acid = ex.peptide[:], ex.mhc_amino_acid[:]
            setattr(ex, "raw_peptide", raw_peptide)
            setattr(ex, "raw_mhc_amino_acid", raw_mhc_amino_acid)
        self.fields["raw_peptide"] = RAW_TEXT_FIELD
        self.fields["raw_mhc_amino_acid"] = RAW_TEXT_FIELD

    @staticmethod
    def sort_key(ex):
        return interleave_keys(len(ex.peptide), len(ex.mhc_amino_acid))

    @classmethod
    def splits(cls, peptide_field, mhc_amino_acid_field, label_field, path=config.base_path, train=config.train_file, validation=config.val_file, test=config.test_file):
        return super(IEDB, cls).splits(
            path=path,
            train=train,
            validation=validation,
            test=test,
            format="csv",
            fields=[
                ("peptide", peptide_field),
                ("mhc_amino_acid", mhc_amino_acid_field),
                ("bind", label_field)
            ],
            skip_header=True
        )

    @classmethod
    def iters(cls, batch_size=64, device=0, shuffle=True, pep_vectors_path=config.pep_vectors_path, mhc_vectors_path=config.mhc_vectors_path, cache_path=config.cache_path):
        cls.PEPTIDE = Field(sequential=True, tokenize=tokenize_pep, batch_first=True, fix_length=config.PEPTIDE_LENGTH)
        cls.MHC_AMINO_ACID = Field(sequential=True, tokenize=tokenize_mhc, batch_first=True, fix_length=config.MHC_AMINO_ACID_LENGTH)
        cls.LABEL = Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)

        train, val, test = cls.splits(cls.PEPTIDE, cls.MHC_AMINO_ACID, cls.LABEL)

        pep_vec = torchtext.vocab.Vectors(pep_vectors_path, cache=cache_path) #[46,100]  1-gram
        mhc_vec = torchtext.vocab.Vectors(mhc_vectors_path, cache=cache_path) #[9418,100] 3-gram


        cls.PEPTIDE.build_vocab(train, val, vectors=pep_vec)
        cls.MHC_AMINO_ACID.build_vocab(train, val, vectors=mhc_vec)

        # print(cls.PEPTIDE.vocab.freqs, cls.PEPTIDE.vocab.stoi, cls.PEPTIDE.vocab.itos)
        # print(cls.MHC_AMINO_ACID.vocab.freqs, cls.MHC_AMINO_ACID.vocab.stoi, cls.MHC_AMINO_ACID.vocab.itos)
        # exit(0)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, shuffle=shuffle, repeat=False, device=device)



def get_dataset(device):
    train_loader, val_loader, test_loader = IEDB.iters(batch_size=config.batch_size, device=device, shuffle=True)
    obj = next(iter(train_loader))
    # print(obj)

    '''
    The following embedding dimension, eg., 23, 365, depends and sorted based on the frequency of sequences
    in the train.csv so that the numbers of dict should be less than the total number of combinations in the dictionary.
    The embedding represents the corresponding vectors in n-gram files of the sequence appearing the train.csv.
    For example in 1-gram, it is from unnamed (0), padding (1), L (2), V (3), A (4)....X (22). To be noticed,
    the vectors in unamned and padding are all zeros!
    '''

    print("Peptide embedding dimension [23, 100]", IEDB.PEPTIDE.vocab.vectors.size())

    peptide_embedding_dim = IEDB.PEPTIDE.vocab.vectors.size()
    peptide_embedding = nn.Embedding(peptide_embedding_dim[0], peptide_embedding_dim[1])
    peptide_embedding.weight = nn.Parameter(IEDB.PEPTIDE.vocab.vectors)
    peptide_embedding.weight.required_grad = True

    print("MHC Amino Acid embedding dimension [365, 100]", IEDB.MHC_AMINO_ACID.vocab.vectors.size())

    mhc_amino_acid_embedding_dim = IEDB.MHC_AMINO_ACID.vocab.vectors.size()
    mhc_amino_acid_embedding = nn.Embedding(mhc_amino_acid_embedding_dim[0], mhc_amino_acid_embedding_dim[1])
    mhc_amino_acid_embedding.weight = nn.Parameter(IEDB.MHC_AMINO_ACID.vocab.vectors)
    mhc_amino_acid_embedding.weight.required_grad = True

    return IEDB, train_loader, val_loader, test_loader, peptide_embedding, mhc_amino_acid_embedding

if __name__ == "__main__":
    device = config.device

    dataset_cls, train_loader, val_loader, test_loader, peptide_embedding, mhc_embedding = get_dataset(device)
    #
    # print(next(iter(train_loader)))
