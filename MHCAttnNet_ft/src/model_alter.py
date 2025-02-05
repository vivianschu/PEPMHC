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


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, context_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.context_dim = context_dim
        self.tanh = nn.Tanh()

        weight = torch.zeros(feature_dim, context_dim)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.b = nn.Parameter(torch.zeros(step_dim, context_dim))

        u = torch.zeros(context_dim, 1)
        nn.init.kaiming_uniform_(u)
        self.context_vector = nn.Parameter(u)

    def forward(self, x):
        eij = torch.matmul(x, self.weight)
        # eij = [batch_size, seq_len, context_dim]
        eij = self.tanh(torch.add(eij, self.b))
        # eij = [batch_size, seq_len, context_dim]
        v = torch.exp(torch.matmul(eij, self.context_vector))  # dot product
        # v = [batch_size, seq_len, 1]
        v = v / (torch.sum(v, dim=1, keepdim=True))
        # v = [batch_size, seq_len, 1]
        weighted_input = x * v
        # weighted_input = [batch_size, seq_len, 2*hidden_dim]             -> 2 : bidirectional
        s = torch.sum(weighted_input, dim=1)
        # s = [batch_size, 2*hidden_dim]                                   -> 2 : bidirectional
        return s

class AttnNet(nn.Module):

    def __init__(self, peptide_embedding, mhc_embedding):
        super(AttnNet, self).__init__()
        self.hidden_size = config.BiLSTM_HIDDEN_SIZE
        self.peptide_num_layers = config.BiLSTM_PEPTIDE_NUM_LAYERS
        self.mhc_num_layers = config.BiLSTM_MHC_NUM_LAYERS

        self.peptide_embedding = peptide_embedding
        self.mhc_embedding = mhc_embedding
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(100, self.hidden_size, num_layers=self.peptide_num_layers, batch_first=True, bidirectional=True)


        self.attn = Attention(2*self.hidden_size, 165, config.CONTEXT_DIM)


        self.peptide_linear = nn.Linear(2*self.hidden_size, config.LINEAR1_OUT)
        self.mhc_linear = nn.Linear(2*self.hidden_size, config.LINEAR1_OUT)
        self.hidden_linear = nn.Linear(config.LINEAR1_OUT, config.LINEAR1_OUT)
        self.out_linear = nn.Linear(config.LINEAR1_OUT, config.LINEAR2_OUT)

    def forward(self, peptide, mhc):
        pep_emb = self.peptide_embedding(peptide)
        mhc_emb = self.mhc_embedding(mhc)

        concat_emb = torch.cat((pep_emb, mhc_emb), dim=1)
        lstm_output, (pep_last_hidden_state, pep_last_cell_state) = self.lstm(concat_emb)

        attn_linear_inp = self.attn(lstm_output)
        linear_out = self.relu(self.peptide_linear(attn_linear_inp))
        out = self.relu(self.out_linear(self.hidden_linear(linear_out)))

        # out = [batch_size, LINEAR2_OUT]

        return out












#
