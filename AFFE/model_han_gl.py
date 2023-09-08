import torch
import torch.nn as nn
from NS_AGG import NetSchema,GCN
from HO_AGG import Higorder
import torch.nn.functional as F
from model import HAN
import numpy as np


class HAN_GL(nn.Module):
    def __init__(self, input_dim, hid_dim, sample_rate, nei_num, feature_drop, attn_drop, length, outsize):
        super(HAN_GL,self).__init__()

        self.dropout = nn.Dropout(p=feature_drop)
        self.non_linear = nn.ReLU()
        self.feat_mapping = nn.ModuleList([nn.Linear(m, hid_dim, bias=True) for m in input_dim])
        self.network_schema = NetSchema(hid_dim, sample_rate, nei_num, attn_drop)
        self.Higher_order = Higorder(hid_dim, length, feature_drop)
        self.predict2 = nn.Linear(3*hid_dim, hid_dim)

        self.predict = nn.Linear(hid_dim, outsize)



    def forward(self, features, NS, ADJ):
        h = [self.dropout(self.feat_mapping[i](features[i])) for i in range(len(features))]
        h2 = self.dropout(self.network_schema(h, NS))
        h3 = self.dropout(self.Higher_order(h[0], ADJ))
        h4 = torch.cat([h[0], h2, h3], dim=1)
        h4 = self.dropout(self.predict2(h4))
        return self.predict(h4), h4




