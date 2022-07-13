import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool

import sys
sys.path.append("..")
from model import Linear_Mnist, ResBlock
from torch_geometric.nn.conv import MessagePassing
gpu_num = '3'

class Penalty(MessagePassing):
    def __init__(self):
        super(Penalty, self).__init__(aggr='add')  
        self.device = torch.device("cuda:"+str(gpu_num) if torch.cuda.is_available() else "cpu")
    def forward(self, x, edge_index, edge_attr):
        log_score = self.propagate(edge_index, x=x, edge_attr = edge_attr)
        #print(torch.exp(log_score))
        return torch.exp(log_score)*16*200*3
    def message(self, edge_attr):
        one = torch.ones(edge_attr.shape[0]).to(self.device).reshape(-1,1)
        #print(one.shape[0])
        #print(torch.log(one-edge_attr))
        return torch.log(one-edge_attr+1e-6)

device = torch.device("cuda:"+str(gpu_num) if torch.cuda.is_available() else "cpu")
model = Linear_Mnist(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
state_dict = torch.load(' ', map_location = torch.device("cpu"))
model.load_state_dict(state_dict)
model.to(device)
model.eval()

class ErdosLoss(torch.nn.Module):
    def __init__(self):
        super(ErdosLoss,self).__init__()
        self.penalty = Penalty()
    def forward(self, x, edge_index, edge_feature, batch):
        """the erdos loss function
        """
        
        x = x.unsqueeze(1) # x: B C H W
        _, loss_1 = model(x,edge_index,edge_feature, batch)
        loss_2 = self.penalty(x,edge_index, edge_feature)
        erdos_loss1 = torch.mean(loss_1)
        erdos_loss2 = torch.mean(loss_2)

        # calculate loss function 3 (for matching)
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        index_ = edge_index.T
        H = torch.zeros(num_edges,num_nodes).long().to(device)
        a = torch.ones(num_edges,num_nodes).long().to(device)
        H.scatter_(1,index_.long(),a)
        H = H.float()
        H_ = torch.mm(H,H.T)
        H_ = torch.triu(H_, diagonal = 1)
        p = edge_feature.T # 1 * 24b
        pt = p.T # 24b * 1
        loss_3 = torch.mm(p,H_.float())
        loss_3 = torch.mm(loss_3, pt)

        erdos_loss3 = loss_3 / (torch.max(batch).item() + 1)
        return erdos_loss1 + erdos_loss2 + 200 * erdos_loss3