#the model of a PNA to predict the dsp

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential, LeakyReLU, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, global_mean_pool, SAGEConv
from torch_geometric.utils import degree

from torch_geometric.nn.conv import MessagePassing

class LastLayer(MessagePassing):
    def __init__(self):
        super(LastLayer, self).__init__(aggr='add')  
    def forward(self, x, edge_index):
        log_score = self.propagate(edge_index, x=x)
        #print(torch.exp(log_score))
        log_score = log_score + torch.log(x + 1e-6)
        return torch.exp(log_score)
    def message(self, x_j):
        return torch.log(x_j+1e-6)


class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            #print("Entered")
            w=module.weight.data
            w=w.clamp(0)
            module.weight.data=w
            #print("done")
            
class LINEAR_lut(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels):
        super(LINEAR_lut, self).__init__()
        torch.manual_seed(12345)

        self.prelin = Linear(in_channels,hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.lin_hh1 = Linear(hidden_channels, hidden_channels)
        self.lin_hh2 = Linear(hidden_channels, hidden_channels)
        self.lin_oo = Linear(out_channels, out_channels)
        self.lin_oo2 = Linear(out_channels, out_channels)
        self.lin_oh = Linear(out_channels, hidden_channels)
        self.lin_h1 = Linear(hidden_channels, 1)

 
        self.leakyrelu = LeakyReLU()
        self.norm_o = BatchNorm1d(out_channels)
        self.norm_h = BatchNorm1d(hidden_channels)
     
        self.linear_prop = LastLayer()
        self.lin_64_1 = Linear(256, 128)
        self.lin32_1 = Linear(128, 1)
        self.lin_xvar_64 = Linear(1, 64)

    def forward(self, x, edge_index, batch):
        z = torch.clone(x[:,:10])
        x_var = torch.clone(x[:,10]).reshape(-1,1)
        
        z = self.prelin(z)
        
        #z = z.relu()
        z = self.conv1(z, edge_index)
        
        z = z.relu()
        z = self.lin_hh1(z)
        #z = F.dropout(z, p=0.1, training=self.training)
        #z = self.leakyrelu(z)
        
        z = self.conv2(z, edge_index)
        z = z.relu()
        z = self.lin_hh2(z)
        #z = F.dropout(z, p=0.1, training=self.training)
        #z = self.leakyrelu(z)
   
        z = self.conv3(z, edge_index)
        #z = z.relu()
        z2 = self.lin_oo2(z)
        z = self.lin_oo(z)
        
        #z = F.dropout(z, p=0.1, training=self.training)
        z = z.relu()
        z_front = z[:,:256]
        z_behind = z[:,256:]

        z2 = z2.relu()
        z2_front = z2[:,:256]
        z2_behind = z2[:,256:]
        
        #x_var = torch.square(self.lin_xvar_64(x_var))
        x_combine = z_front * x_var + z_behind

        x_combine = self.linear_prop(x_combine,edge_index)

        x_combine = global_mean_pool(x_combine, batch) 
   
        x_combine = self.lin_64_1(x_combine)
        x_combine = 7000 - x_combine.relu()
        x_combine = self.lin32_1(x_combine)

        x_linear = z2_front * x_var + z2_behind
        x_linear = self.linear_prop(x_linear, edge_index)
        x_linear = global_mean_pool(x_linear, batch)
        x_linear = self.lin_linear(x_linear)
        x_out = x_combine + x_linear
        
        return x_out