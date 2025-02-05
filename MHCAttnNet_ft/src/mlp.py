

import os
import math
import sys
import time
from tqdm import tqdm

import config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext
from model import Attention

def pooling2d(input_size, kernel_size, stride_size):
    return math.floor((input_size-kernel_size)/stride_size+1)



class MLP1(nn.Module):

    def __init__(self,
            dropout=0.2,
            activation="relu",
            classifier_dropout=0.1
            ):

        super(MLP1, self).__init__()
        self.hidden_size = config.BiLSTM_HIDDEN_SIZE


        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.peptide_linear = nn.Linear(1024, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(1024, config.LINEAR1_OUT)
        self.hidden_linear = nn.Linear(config.LINEAR1_OUT*2, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT, config.LINEAR2_OUT)




    def forward(self, pep, mhc):

        pep_out = self.dropout(self.relu(self.peptide_linear(pep)))
        mhc_out = self.dropout(self.relu(self.mhc_linear(mhc)))

        # pep_out = pep_emb.mean(axis=1)
        # mhc_out = mhc_emb.mean(axis=1)

        conc = torch.cat((pep_out, mhc_out), dim=1)


        out = self.relu(self.out_linear(self.hidden_linear(conc)))

        return out



class MLP(nn.Module):

    def __init__(self,
            dropout=0.2,
            activation="relu",
            # activation="mish",
            classifier_dropout=0.1
            ):

        super(MLP, self).__init__()
        self.hidden_size = config.BiLSTM_HIDDEN_SIZE


        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.mish = nn.Mish()

        self.dropout = nn.Dropout(p=0.1)


        self.peptide_linear0 = nn.Linear(config.PEPTIDE_LENGTH*100, config.LINEAR0_OUT)
        self.mhc_linear0 = nn.Linear(config.MHC_AMINO_ACID_LENGTH*100, config.LINEAR0_OUT)

        self.peptide_linear = nn.Linear(1024, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(1024, config.LINEAR1_OUT)
        self.hidden_linear = nn.Linear(config.LINEAR1_OUT*2, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT, config.LINEAR2_OUT)


        self.max_pool  = nn.MaxPool2d(20, stride=2)
        self.avg_pool  = nn.AvgPool2d(20, stride=2)



        self.pool_out = nn.Linear(pooling2d(45, 20 , 2)*
                                  pooling2d(1024, 20, 2), config.LINEAR1_OUT)


        self.mhc_out = nn.Linear(pooling2d(350, 20 , 2)*
                                  pooling2d(1024, 20, 2), config.LINEAR1_OUT)





    def forward(self, pep, mhc):


        # exit(0)
        # if config.MAXPOOL_FLAG == True:
        #     pep_emb = self.avg_pool(pep_emb)
        #     mhc_emb = self.avg_pool(mhc_emb)
        #
        #     pep_emb = self.dropout(self.mish(nn.Flatten(start_dim=1)(pep_emb)))
        #     mhc_emb = self.dropout(self.mish(nn.Flatten(start_dim=1)(mhc_emb)))
        #
        #     pep_emb = self.pool_out(pep_emb)
        #     mhc_emb = self.mhc_out(mhc_emb)
        #
        #
        #     pep_out = self.dropout(self.mish(pep_emb))
        #     mhc_out = self.dropout(self.mish(mhc_emb))
        # else:

        pep_emb = pep
        mhc_emb = mhc

        pep_out = pep_emb.mean(axis=1)
        mhc_out = mhc_emb.mean(axis=1)



        # pep_emb =self.avg_pool(pep_emb)
        # mhc_emb =self.avg_pool(mhc_emb)
        #
        # pep_emb = self.dropout(self.mish(nn.Flatten(start_dim=1)(pep_emb)))
        # mhc_emb = self.dropout(self.mish(nn.Flatten(start_dim=1)(mhc_emb)))
        #
        # pep_out = self.pool_out(pep_emb)
        # mhc_out = self.mhc_out(mhc_emb)


        #
            # pep_out = self.peptide_attn(pep_emb)
            # mhc_out = self.mhc_attn(mhc_emb)

            # pep_out = self.dropout(nn.Flatten(start_dim=1)(pep_emb))
            # mhc_out = self.dropout(nn.Flatten(start_dim=1)(mhc_emb))

            # pep_out = self.dropout(self.relu(self.peptide_linear0(pep_out)))
            # mhc_out = self.dropout(self.relu(self.mhc_linear0(mhc_out)))

        pep_out = self.dropout(self.mish(self.peptide_linear(pep_out)))
        mhc_out = self.dropout(self.mish(self.mhc_linear(mhc_out)))

        conc = torch.cat((pep_out, mhc_out), dim=1)

        out = self.out_linear(self.mish(self.hidden_linear(conc)))


        return out


























#







import os
import math
import sys
import time
from tqdm import tqdm

import config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext
from model import Attention

def pooling2d(input_size, kernel_size, stride_size):
    return math.floor((input_size-kernel_size)/stride_size+1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])



class Transformer(nn.Module):

    def __init__(self,
            peptide_embedding,
            mhc_embedding,
            nhead=1,
            dim_feedforward=2048,
            num_layers=1,
            dropout=0.2,
            activation="relu",
            # activation="mish",
            classifier_dropout=0.1
            ):

        super(Transformer, self).__init__()
        self.hidden_size = config.BiLSTM_HIDDEN_SIZE

        self.peptide_embedding = peptide_embedding
        self.mhc_embedding = mhc_embedding

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.mish = nn.Mish()

        self.dropout = nn.Dropout(p=0.1)


        self.peptide_linear0 = nn.Linear(config.PEPTIDE_LENGTH*100, config.LINEAR0_OUT)
        self.mhc_linear0 = nn.Linear(config.MHC_AMINO_ACID_LENGTH*100, config.LINEAR0_OUT)

        self.peptide_linear = nn.Linear(100, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(100, config.LINEAR1_OUT)
        self.hidden_linear = nn.Linear(config.LINEAR1_OUT*2, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT, config.LINEAR2_OUT)


        self.max_pool  = nn.MaxPool2d(config.KERNEL_SIZE, stride=config.STRIDE_SIZE)
        self.avg_pool  = nn.AvgPool2d(config.KERNEL_SIZE, stride=config.STRIDE_SIZE)



        self.pool_out = nn.Linear(pooling2d(config.PEPTIDE_LENGTH,config.KERNEL_SIZE,config.STRIDE_SIZE)*
                                  pooling2d(config.EMBED_DIM,config.KERNEL_SIZE,config.STRIDE_SIZE), config.LINEAR1_OUT)

        self.mhc_out  = nn.Linear(pooling2d(config.MHC_AMINO_ACID_LENGTH,config.KERNEL_SIZE,config.STRIDE_SIZE)*
                                  pooling2d(config.EMBED_DIM,config.KERNEL_SIZE,config.STRIDE_SIZE), config.LINEAR1_OUT)

        self.peptide_attn = Attention(config.EMBED_DIM, config.PEPTIDE_LENGTH,config.LINEAR1_OUT)
        self.mhc_attn = Attention(config.EMBED_DIM, config.MHC_AMINO_ACID_LENGTH, config.LINEAR1_OUT)


        d_model = config.EMBED_DIM
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.pep_pos_encoder = PositionalEncoding(
            dim_model=d_model,
            dropout_p=dropout,
            max_len=config.PEPTIDE_LENGTH,
        )

        self.mhc_pos_encoder = PositionalEncoding(
            dim_model=d_model,
            dropout_p=dropout,
            max_len=config.MHC_AMINO_ACID_LENGTH,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.pep_transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.mhc_transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.d_model = d_model

    def forward(self, pep, mhc):

        pep_emb = self.peptide_embedding(pep)  * math.sqrt(self.d_model)
        mhc_emb = self.mhc_embedding(mhc) * math.sqrt(self.d_model)


        pep_emb = self.pep_pos_encoder(pep_emb)
        pep_emb = self.pep_transformer_encoder(pep_emb)

        mhc_emb = self.mhc_pos_encoder(mhc_emb)
        mhc_emb = self.mhc_transformer_encoder(mhc_emb)

        if config.MAXPOOL_FLAG == True:
            pep_emb = self.avg_pool(pep_emb)
            mhc_emb = self.avg_pool(mhc_emb)

            pep_emb = self.dropout(self.mish(nn.Flatten(start_dim=1)(pep_emb)))
            mhc_emb = self.dropout(self.mish(nn.Flatten(start_dim=1)(mhc_emb)))

            pep_emb = self.pool_out(pep_emb)
            mhc_emb = self.mhc_out(mhc_emb)


            pep_out = self.dropout(self.mish(pep_emb))
            mhc_out = self.dropout(self.mish(mhc_emb))
        else:

            pep_out = pep_emb.mean(axis=2)
            mhc_out = mhc_emb.mean(axis=2)

            # pep_out = self.peptide_attn(pep_emb)
            # mhc_out = self.mhc_attn(mhc_emb)

            # pep_out = self.dropout(nn.Flatten(start_dim=1)(pep_emb))
            # mhc_out = self.dropout(nn.Flatten(start_dim=1)(mhc_emb))

            # pep_out = self.dropout(self.relu(self.peptide_linear0(pep_out)))
            # mhc_out = self.dropout(self.relu(self.mhc_linear0(mhc_out)))

            pep_out = self.dropout(self.mish(self.peptide_linear(pep_out)))
            mhc_out = self.dropout(self.mish(self.mhc_linear(mhc_out)))



        conc = torch.cat((pep_out, mhc_out), dim=1)

        out = self.out_linear(self.mish(self.hidden_linear(conc)))


        return out


























#
















#
