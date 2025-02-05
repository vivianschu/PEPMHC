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
            nhead=10,
            dim_feedforward=512,
            num_layers=6,
            dropout=0.2,
            activation="relu",
            classifier_dropout=0.1
            ):

        super(Transformer, self).__init__()
        self.hidden_size = config.BiLSTM_HIDDEN_SIZE

        self.peptide_embedding = peptide_embedding
        self.mhc_embedding = mhc_embedding

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.peptide_linear = nn.Linear(45*100, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(120*100, config.LINEAR1_OUT)
        self.hidden_linear = nn.Linear(config.LINEAR1_OUT*2, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT, config.LINEAR2_OUT)



        vocab_size, d_model = 500, 100
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.pep_pos_encoder = PositionalEncoding(
            dim_model=d_model,
            dropout_p=dropout,
            max_len=vocab_size,
        )

        self.mhc_pos_encoder = PositionalEncoding(
            dim_model=d_model,
            dropout_p=dropout,
            max_len=vocab_size,
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

        pep_out = self.dropout(nn.Flatten(start_dim=1)(pep_emb))
        mhc_out = self.dropout(nn.Flatten(start_dim=1)(mhc_emb))

        pep_out = self.dropout(self.relu(self.peptide_linear(pep_out)))
        mhc_out = self.dropout(self.relu(self.mhc_linear(mhc_out)))

        # pep_out = pep_emb.mean(axis=1)
        # mhc_out = mhc_emb.mean(axis=1)


        conc = torch.cat((pep_out, mhc_out), dim=1)

        out = self.relu(self.out_linear(self.hidden_linear(conc)))


        return out


























#
