#the model of a PNA to predict the dsp

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential, LeakyReLU, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, SAGEConv, global_mean_pool
from torch_geometric.utils import degree


class SAGE_dsp(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels):
        super(SAGE_dsp, self).__init__()
        torch.manual_seed(12345)

        self.prelin = Linear(in_channels,hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.lin_hh1 = Linear(hidden_channels, hidden_channels)
        self.lin_hh2 = Linear(hidden_channels, hidden_channels)
        self.lin_oo = Linear(out_channels, out_channels)
        self.lin_oh = Linear(out_channels, hidden_channels)
        self.lin_h1 = Linear(hidden_channels, 1)

        self.leakyrelu1 = LeakyReLU()
        self.leakyrelu2 = LeakyReLU()
        self.norm_o = BatchNorm1d(out_channels)
        self.norm_h = BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, batch):
        
        x = self.prelin(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.lin_hh1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.leakyrelu1(x)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.lin_hh2(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.leakyrelu1(x)
   
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.lin_oo(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.leakyrelu1(x)

        x = global_mean_pool(x, batch) 
        x = self.lin_oh(x)
        x = self.norm_h(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.leakyrelu1(x)

        x = self.lin_h1(x)
        x = x.relu()
        
        return x