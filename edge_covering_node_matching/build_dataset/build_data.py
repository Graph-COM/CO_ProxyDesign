import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pathlib import Path
import yaml
import re
import random

# the ground_truth function f()
def f(z1, z2, x):
    # z1: node feature in one side, E * 1
    # z2: node feature in the other side, E * 1
    # x: edge feature, optimization variable, E * 1
    g1 = (z1 + z2) / 3 + (z1 * z2) / 100
    g2 = (z1 / (z2+1) + z2 / (z1+1)) * 5
    ans = np.mean(g1 * x + g2)
    #ans = g1 * x + g2
    return ans

# generate a bernoulli 0 or 1 with the given probability
# the dataset class to generate the dataset
class Synthetic_Mnist_Dataset(InMemoryDataset):
    def __init__(self, config:dict):
        self.config = config
        self.data_path = Path(config['target_path'])
        self.splits = config['splits']
        super(Synthetic_Mnist_Dataset, self).__init__(root=self.data_path)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['synthetic_mnist.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def get_idx_split(self, split_type = 'Random'):
        data_idx = np.arange(100000)
        splits = self.splits
        train_num = int(float(splits['train'])*100000)
        valid_num = int(float(splits['valid'])*100000)
        if (split_type==None):
            train_idx = data_idx[:train_num]
            valid_idx = data_idx[train_num:train_num+valid_num]
            test_idx = data_idx[train_num+valid_num:]
        elif (split_type=='Random'):    
            shuffle_idx = np.random.shuffle(data_idx)
            train_idx = data_idx[:train_num]
            valid_idx = data_idx[train_num:train_num+valid_num]
            test_idx = data_idx[train_num+valid_num:]
        else:
            print("something went wrong in spliting the index?")        
        return {'train':torch.tensor(train_idx,dtype = torch.long),'valid':torch.tensor(valid_idx,dtype = torch.long),'test':torch.tensor(test_idx,dtype = torch.long)}

    def process(self):
        # create our synthetic dataset 2
        data_list = []
        max_y = 0
        # get the data of mnist
        mnist_list = torch.load('../mnist/processed/training.pt')

        for i in range(100000):

            if (i%1000==0):
                print(str(i)+"instances has been processed")
            # generate the edge index
            edge_direc1 = torch.tensor([[0,0,1,1,2,2,3,4,4,5,5,6,6, 7, 8,8, 9, 9, 10,10,11,12,13,14],
                                        [1,4,2,5,3,6,7,5,8,6,9,7,10,11,9,12,10,13,11,14,15,13,14,15]])
            edge_direc2_index = [1,0]                                      
            edge_direc2 = edge_direc1[edge_direc2_index]
            edge_index = torch.cat((edge_direc1, edge_direc2),1)
            # generate the node feature
            node_feature_index_decade = np.random.randint(1,60000,16)
            node_feature_index_unit = np.random.randint(1,60000,16)
            node_feature_decade = mnist_list[0][node_feature_index_decade]
            node_feature_unit = mnist_list[0][node_feature_index_unit]
            node_feature = torch.cat([node_feature_decade, node_feature_unit], -1)
            node_feature_label_decade = mnist_list[1][node_feature_index_decade]
            node_feature_label_unit = mnist_list[1][node_feature_index_unit]
            node_feature_label = (10 * node_feature_label_decade + node_feature_label_unit).reshape(-1,1)
            node_feature_label_numpy = node_feature_label.numpy()
            # generate the edge feature
            edge_feature_direc1 = np.random.randint(0,2,24)
            edge_feature_direc1_reshape = edge_feature_direc1.reshape(-1,1)
            edge_feature_direc1_tensor = torch.from_numpy(edge_feature_direc1_reshape)
            edge_feature = torch.cat((edge_feature_direc1_tensor,edge_feature_direc1_tensor),0)
            # generate the lifted node feature with the dimension of |E|
            lift_index1 = edge_direc1[0,:]
            node_feature_label_lifted1 = node_feature_label_numpy[lift_index1]
            lift_index2 = edge_direc1[1,:]
            node_feature_label_lifted2 = node_feature_label_numpy[lift_index2]
            # generate the label y
            y = f(node_feature_label_lifted1, node_feature_label_lifted2, edge_feature_direc1)
            # form the dataset
            if y>max_y:
                max_y = y
            node_feature = torch.tensor(node_feature).float()
            edge_feature = torch.tensor(edge_feature).float()
            
            data = Data(x = node_feature, y = y, edge_index = edge_index, edge_attr = edge_feature, node_feature_label = node_feature_label)
            #import pdb; pdb.set_trace()
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("max y:"+str(max_y))





if __name__ == '__main__':
    configs = Path('./configs')
    for cfg in configs.iterdir():
        if str(cfg).startswith("configs/config"):
            cfg_dict = yaml.safe_load(cfg.open('r'))
            dataset = Synthetic_Mnist_Dataset(cfg_dict['data'])
    
    # this is to test the content in dataset:
    '''
    for i in range(10):
        print(dataset[i]['y'])
    '''
