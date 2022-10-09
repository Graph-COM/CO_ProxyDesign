#the model of a graph sage to predict the relative error
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import BatchNorm, PNAConv, global_mean_pool
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

class PNA_aff(torch.nn.Module):
    def __init__(self, gpu_num, save_path):
        super().__init__()
        self.gpu_num = gpu_num
        self.save_path = save_path
        deg_file = torch.load(self.save_path+'/deg.pt').to(torch.device("cuda:"+str(self.gpu_num)))
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
        self.mlp = Sequential(Linear(40, 1))

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

        return self.mlp(x_combine)
