import torch
from torch.nn import Linear,LeakyReLU,BatchNorm1d,Sigmoid
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool


from typing import Union, Tuple, Any, Callable
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn



class SAGEConv_edge(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv_edge, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        self.reset_parameters()
        
        self.lin_e = Linear(1, in_channels[0])
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, size: Size = None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        
        # propagate_type: (x: OptPairTensor)
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            #assert x[0].size(-1) == edge_attr.size(-1)
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out
    def message(self, x_j: Tensor, edge_attr: Tensor):
        #return x_j
        edge_feature = self.lin_e(edge_attr)
        return F.relu(x_j + edge_feature)
    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor):
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SAGE_edge(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels):
        super(SAGE_edge, self).__init__()
        torch.manual_seed(12345)

        self.prelin = Linear(in_channels,hidden_channels)
        #self.prelin_e = Linear(in_channels,hidden_channels)
        self.conv1 = SAGEConv_edge(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv_edge(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv_edge(hidden_channels, out_channels)

        self.lin_hh1 = Linear(hidden_channels, hidden_channels)
        self.lin_hh2 = Linear(hidden_channels, hidden_channels)
        self.lin_oo = Linear(out_channels, out_channels)
        self.lin_oh = Linear(out_channels, hidden_channels)
        self.lin_h1 = Linear(hidden_channels, 1)

        self.leakyrelu = LeakyReLU()
        self.norm_h = BatchNorm1d(hidden_channels)
        self.norm_o = BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.prelin(x)
        x = x.relu()
        #edge_attr = self.prelin_e(edge_attr)
        #edge_attr = edge_attr.relu()

        x = self.conv1(x, edge_index, edge_attr)
        #x = self.norm_h(x)
        x = x.relu()
        x = self.lin_hh1(x)
        x = self.leakyrelu(x)
        #x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        #x = self.norm_h(x)
        x = x.relu()
        x = self.lin_hh2(x)
        x = self.leakyrelu(x)
        #x = F.dropout(x, p=0.1, training=self.training)
   
        x = self.conv3(x, edge_index, edge_attr)
        #x = self.norm_o(x)
        x = x.relu()
        x = self.lin_oo(x)
        x = self.leakyrelu(x)
        #x = F.dropout(x, p=0.1, training=self.training)

        x = global_mean_pool(x, batch) 
        x = self.lin_oh(x)
        x = self.norm_h(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x = self.leakyrelu(x)

        x = self.lin_h1(x)
        x = x.relu()
        return x

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, ResBlock):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 64, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 64, 2, stride=2)        
        self.fc = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #import pdb; pdb.set_trace()
        out = F.avg_pool2d(out, 4)
        #import pdb; pdb.set_trace()
        out = out.view(out.size(0), -1)
        sage_input = self.fc(out)
        res_out = self.fc2(sage_input)
        return res_out, sage_input


class Model_Mnist(torch.nn.Module):
    def __init__(self, Res_Block, in_channels, hidden_channels,out_channels):
        super(Model_Mnist, self).__init__()
        torch.manual_seed(12345)
        self.ResNet = ResNet(Res_Block)
        self.SAGE_edge = SAGE_edge(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        res_out, sage_input = self.ResNet(x)
        predict = self.SAGE_edge(sage_input, edge_index, edge_attr, batch)
        return res_out, predict