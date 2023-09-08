import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math



class AggAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(AggAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class Higorder(nn.Module):
    def __init__(self, hid_dim, length, feature_drop):
        super(Higorder,self).__init__()

        self.dropout = nn.Dropout(p=feature_drop)
        self.non_linear = nn.Tanh()
        self.intra_agg = nn.ModuleList([Intra_agg(hid_dim,length[i]) for i in range(len(length))])#for i in rang(length[0])
        self.inter_agg = AggAttention(hid_dim)

    def forward(self, features, ADJ):
        h = [self.intra_agg[i](features, ADJ[i]) for i in range(len(ADJ))]
        return self.inter_agg(torch.stack(h, dim=1))

class Intra_agg(nn.Module):
    def __init__(self, hid_dim, length, bias=True):
        super(Intra_agg, self).__init__()

        self.weight = [Parameter(torch.FloatTensor(hid_dim,hid_dim)) for i in range(length)]
        if bias:
            self.bias = [Parameter(torch.FloatTensor(hid_dim)) for i in range(length)]
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(length)
        self.intra_attention = AggAttention(in_size=hid_dim)

    def reset_parameters(self, length):
        for i in range(length):
            stdv = 1. / math.sqrt(self.weight[i].size(1))
            self.weight[i].data.uniform_(-stdv, stdv)
            if self.bias[i] is not None:
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        h_all=[]
        for i in range(len(adj)):
            output = torch.spmm(adj[i], torch.spmm(inputs, self.weight[i]))
            if self.bias[i] is not None:
                h_all.append(F.elu(output + self.bias[i]))
            else:
                h_all.append(F.elu(output))
        return self.intra_attention(torch.stack(h_all, dim=1))
