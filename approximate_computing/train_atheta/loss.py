import torch

import sys
sys.path.append("..")


class ErdosLoss(torch.nn.Module):
    def __init__(self, gpu_num, model):
        super(ErdosLoss,self).__init__()
        self.gpu_num = gpu_num
        self.proxy = model
        self.device = torch.device("cuda:"+str(gpu_num))

    def forward(self, x, fixed_feature, alpha, edge_index, batch):
        """the erdos loss function
        """
        input_feature = torch.cat([fixed_feature, x],1)
        relative_error = self.proxy(input_feature, edge_index, batch) # it's 100 times the relative error
        x = x.reshape(-1, 15)
        x = torch.sum(x, dim = 1)
        loss = relative_error - alpha * x
        erdos_loss = torch.mean(loss)

        return erdos_loss
