import numpy as np
import re
import torch
import pandas as pd
import itertools
from torch.utils.data import Dataset, IterableDataset
from transformers import BertModel, BertTokenizer
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, pipeline


parent_path ='/scratch/ssd004/scratch/vchu/PEPMHC/LLM/'

# with open(parent_path + 'uniprot_sprot.fasta') as f:
#     lines = f.readlines()

# proteins = []
# protein_seq = ""

# for line in lines:
#     if line.startswith(">"):
#         if protein_seq:
#             proteins.append(protein_seq)
#             protein_seq = ""
#     else:
#         protein_seq += line.strip()

# if protein_seq:
#     proteins.append(protein_seq)


# np.save(parent_path + 'peptide', set(proteins))

# exit(0)

# mhc_1 = np.loadtxt(parent_path + 'MHC_I/train.csv', delimiter=',')
# print(len(mhc_1))


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


class IEDB_pep(Dataset):

    def __init__(self, tokenizer_name='Rostlab/prot_bert_bfd', max_length=48):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

        self.pep = self.load_dataset()


    def load_dataset(self):


        train_path = parent_path + 'data/MHC_I/train.csv'
        test_path  = parent_path + 'data/MHC_I/test.csv'

        train = pd.read_csv(train_path, names=['peptide','mhc_amino_acid','bind', 'mhc_allele'], skiprows=1)
        test = pd.read_csv(test_path,   names=['peptide','mhc_amino_acid','bind', 'mhc_allele'], skiprows=1)

        pep = list(train['peptide'])        + list(test['peptide'])

        # Load entire peptide sequences from IEDB   
        ideb_pep = list(pd.read_csv(parent_path + 'data/iedb_pep.csv', names=['peptide'], skiprows=1).squeeze("columns"))

        pep = pep + ideb_pep

        pep = unique(pep)
        # length =len(pep)
        # min_length = min(len(s) for s in pep) 
        # max_length = max(len(s) for s in pep)

        # print(length, min_length, max_length)
        # exit(0)

        return pep

    def __len__(self):
        return len(self.pep)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pep = " ".join("".join(self.pep[idx].split()))
        pep = re.sub(r"[UZOB]", "X", pep)

        seq_ids = self.tokenizer(pep, truncation=True, padding='max_length', max_length=self.max_length)
        # print(seq_ids)
        # exit(0)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        # sample['labels'] = torch.tensor(self.labels[idx])

        return sample
    

    
class IEDB_raw(Dataset):

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd', max_length=512):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

        self.pair = self.load_dataset()


    def load_dataset(self):


        train_path = parent_path + 'data/MHC_I/train.csv'
        test_path  = parent_path + 'data/MHC_I/test.csv'

        train = pd.read_csv(train_path, names=['peptide','mhc_amino_acid','bind', 'mhc_allele'], skiprows=1)
        test = pd.read_csv(test_path,   names=['peptide','mhc_amino_acid','bind', 'mhc_allele'], skiprows=1)

        pep = list(train['peptide'])    
        mhc = list(train['mhc_amino_acid'])

            

        pep = unique(pep)
        mhc = unique(mhc)


        pep = pep[:int(len(pep)*0.2)]

        pair_list = list(itertools.product(pep, mhc)) 

        square_list = [list(pair) for pair in pair_list] 


        return square_list

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pep = " ".join("".join(self.pair[idx][0].split()))
        pep = re.sub(r"[UZOB]", "X", pep)

        mhc = " ".join("".join(self.pair[idx][1].split()))
        mhc = re.sub(r"[UZOB]", "X", mhc)

        seq_ids = self.tokenizer(pep, mhc, truncation=True, padding='max_length', max_length=self.max_length)
        # print(seq_ids)
        # exit(0)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        # sample['labels'] = torch.tensor(self.labels[idx])

        return sample

