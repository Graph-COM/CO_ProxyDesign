#the model of a graph sage to predict the relative error

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, global_mean_pool
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

class PNA_concave(torch.nn.Module):
    def __init__(self):
        super().__init__()
        deg_file = torch.load(' path to deg').to(torch.device("cuda:7"))# add the path to the deg file
        #self.node_emb = Embedding(11, 75)
        self.pre_lin = Linear(1, 80)
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
     
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=80, out_channels=80,
                           aggregators=aggregators, scalers=scalers, deg=deg_file,
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(80))
        self.linear_propagation = LastLayer()
        self.mlp_1 = Sequential(Linear(40, 20))
        self.mlp_2 = Sequential(Linear(20, 1))
        self.mlp_lin = Sequential(Linear(40, 1))

    def forward(self, x, edge_index, batch):
        
        #import pdb; pdb.set_trace()
        operation = torch.clone(x[:,0]).reshape(-1,1)
        assignment = torch.clone(x[:,1]).reshape(-1,1)
        operation = self.pre_lin(operation)
        #import pdb; pdb.set_trace()
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            operation = F.relu(batch_norm(conv(operation, edge_index)))
        operation_front = operation[:,:40]
        operation_behind = operation[:,40:]
        x_combine = operation_front * assignment + operation_behind
        x_combine = self.linear_propagation(x_combine, edge_index)

        x_combine = global_mean_pool(x_combine, batch)

        x_lin = self.mlp_lin(x_combine)
        x_concave = self.mlp_1(x_combine)
        x_concave = 20 - x_concave.relu()
        x_concave = self.mlp_2(x_concave)
        x_out = x_concave + x_lin
        
        
        return x_out
