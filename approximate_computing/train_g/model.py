#the model of a graph sage to predict the relative error

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.utils import degree

state_dict_path = '' # path to deg file
device = torch.device("cuda:3")
state_dict_path_model = '' # path to deg file
device_model = torch.device("cuda:3")

class PNA_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        deg_file = torch.load(state_dict_path_model).to(device_model)
        #self.node_emb = Embedding(11, 75)
        self.pre_lin = Linear(2, 75)
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
     
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg_file,
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, edge_index, batch):
        
        #import pdb; pdb.set_trace()
        
        x = self.pre_lin(x)
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        return self.mlp(x)


class PNA_model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        deg_file = torch.load(state_dict_path).to(device)
        #self.node_emb = Embedding(11, 75)
        self.pre_lin = Linear(2, 75)
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
     
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg_file,
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(76, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, alpha, edge_index, batch):
        
        #import pdb; pdb.set_trace()
        fixed_feature = torch.clone(x[:,0]).reshape(-1, 1)
        x = self.pre_lin(x)
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        to_attach = alpha[batch].float().reshape(-1,1)
        x = torch.cat([x, to_attach], -1)
      
        x = self.mlp(x)
        random_noise = torch.empty_like(x).uniform_(1e-10, 1 - 1e-10)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        x = (x + random_noise) / 1.0
        x = x.sigmoid()

        return x, fixed_feature

class PNA_model2test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        deg_file = torch.load(state_dict_path).to(device)
        self.pre_lin = Linear(2, 75)
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
     
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg_file,
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(76, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, alpha, edge_index, batch):
        
        #import pdb; pdb.set_trace()
        fixed_feature = torch.clone(x[:,0]).reshape(-1, 1)
        x = self.pre_lin(x)
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        
        to_attach = alpha[batch].float().reshape(-1,1)
        x = torch.cat([x, to_attach], -1)
        x = self.mlp(x)
        x = x.sigmoid()
        return x, fixed_feature
