import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool

import sys
sys.path.append("..")
from model import PNA_concave

device = torch.device("cuda:2")
model = PNA_concave()
state_dict = torch.load((' path to the model'),map_location = torch.device("cpu"))# add pth to the model
model.load_state_dict(state_dict)
model.to(device)
model.eval()

class ErdosLoss(torch.nn.Module):
    def __init__(self):
        super(ErdosLoss,self).__init__()

    def forward(self, x, fixed_feature, alpha, edge_index, batch):
        """the erdos loss function
        """
        input_feature = torch.cat([fixed_feature, x],1)
        relative_error = model(input_feature, edge_index, batch) # it's 100 times the relative error
        x = x.reshape(-1, 15)
        x = torch.sum(x, dim = 1)
        #import pdb; pdb.set_trace()
        loss = relative_error + alpha * (15 - x)
        erdos_loss = torch.mean(loss)

        return erdos_loss
        #return torch.mean(predicted_lut)
