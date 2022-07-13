import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool

import sys
sys.path.append("..")
from model import SAGE_lut, SAGE_dsp

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model_dsp = SAGE_dsp(in_channels = 11, hidden_channels = 32, out_channels = 128)
model_dsp.load_state_dict(torch.load(' '),torch.device("cpu"))
model_dsp.to(device)
model_dsp.eval()

model_lut = SAGE_lut(in_channels = 11, hidden_channels = 32, out_channels = 128)
model_lut.load_state_dict(torch.load(' '),torch.device("cpu"))
model_lut.to(device)
model_lut.eval()

class ErdosLoss(torch.nn.Module):
    def __init__(self):
        super(ErdosLoss,self).__init__()

    def forward(self, x, alpha, fixed_feature, edge_index, batch):
        """the erdos loss function
        """
        x = torch.cat([fixed_feature, x],1)
        predicted_dsp = model_dsp(x, edge_index, batch)
        predicted_lut = model_lut(x, edge_index, batch)
       
        loss = predicted_lut + alpha * predicted_dsp
        
        erdos_loss = torch.mean(loss)

        return erdos_loss
