import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pathlib import Path
import yaml
import re
import os
from torch_geometric.data import DataLoader
import argparse
import shutil
import sys
sys.path.append("..")
from build_dataset.build_data import Synthetic_Mnist_Dataset
from tensorboardX import SummaryWriter
from train_g.model import Model_Mnist2test, ResBlock, Model_Mnist

import pickle

def f(z1, z2, x):
    # z1: node feature in one side, E * 1
    # z2: node feature in the other side, E * 1
    # x: edge feature, optimization variable, E * 1
    g1 = (z1 + z2) / 3 + (z1 * z2) / 100
    g2 = (z1 / (z2+1) + z2 / (z1+1)) * 5
    ans = np.mean(g1 * x + g2)
    return ans


def test_an_instance(model, edge_index, node_feature, node_feature_label, predictor):
    node_feature_numpy = node_feature
    node_feature = torch.tensor(node_feature.reshape(-1,28, 56)).float()
    node_feature = node_feature.unsqueeze(1)

    edge_feature_direc1 = np.random.randint(0,2,24)
    edge_feature_direc1_reshape = edge_feature_direc1.reshape(-1,1)
    edge_feature_direc1_tensor = torch.from_numpy(edge_feature_direc1_reshape)
    edge_feature = torch.cat((edge_feature_direc1_tensor,edge_feature_direc1_tensor),0)
    edge_feature = torch.tensor(edge_feature).float()   

    _, model_out = model(node_feature, edge_index, edge_feature)
    out_1 = model_out[:24,0].detach().numpy()
    #print(out_1)
    #set the output according to the output of the model
    out_index = np.argsort(out_1)
    #print(out_index)
    i = 24
    punishment = 300
    edge_feature_model = np.zeros(24).astype(np.int64)
    while(punishment >= 200):
        i = i-1
        edge_feature_model[out_index[i]] = 1
        punishment = (   (1-edge_feature_model[0])*(1-edge_feature_model[1])  
                           + (1-edge_feature_model[2])*(1-edge_feature_model[3])*(1-edge_feature_model[0])
                           + (1-edge_feature_model[4])*(1-edge_feature_model[5])*(1-edge_feature_model[2])
                           + (1-edge_feature_model[6])*(1-edge_feature_model[4])   
                           + (1-edge_feature_model[7])*(1-edge_feature_model[8])*(1-edge_feature_model[1]) 
                           + (1-edge_feature_model[9])*(1-edge_feature_model[10])*(1-edge_feature_model[3])*(1-edge_feature_model[7]) 
                           + (1-edge_feature_model[11])*(1-edge_feature_model[12])*(1-edge_feature_model[5])*(1-edge_feature_model[9])
                           + (1-edge_feature_model[13])*(1-edge_feature_model[6])*(1-edge_feature_model[11])
                           + (1-edge_feature_model[15])*(1-edge_feature_model[8])*(1-edge_feature_model[14])
                           + (1-edge_feature_model[16])*(1-edge_feature_model[17])*(1-edge_feature_model[10])*(1-edge_feature_model[14])
                           + (1-edge_feature_model[18])*(1-edge_feature_model[19])*(1-edge_feature_model[12])*(1-edge_feature_model[16])
                           + (1-edge_feature_model[20])*(1-edge_feature_model[13])*(1-edge_feature_model[18])
                           + (1-edge_feature_model[21])*(1-edge_feature_model[15])
                           + (1-edge_feature_model[22])*(1-edge_feature_model[17])*(1-edge_feature_model[21])
                           + (1-edge_feature_model[23])*(1-edge_feature_model[19])*(1-edge_feature_model[22])
                           + (1-edge_feature_model[20])*(1-edge_feature_model[23])
                        ) * 200  
        #print(punishment)
    #print(edge_feature_model)  
    # generate the f prediction of the model's decision
    lift_index1 = edge_index[0,:24]
    lift_index2 = edge_index[1,:24]
    node_feature_label_lifted1 = node_feature_label[lift_index1]
    node_feature_label_lifted2 = node_feature_label[lift_index2]
    y = f(node_feature_label_lifted1, node_feature_label_lifted2, edge_feature_model)
    batch = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    feature_model = torch.tensor(edge_feature_model)
    edgef_feed = torch.cat([feature_model, feature_model],0).reshape(-1,1).float()
    #import pdb; pdb.set_trace()
    _, model_out_rounded = predictor(node_feature, edge_index, edgef_feed, batch)
    #print(y)
    return y, model_out_rounded
        



def main():
    #load the model
    model = Model_Mnist2test(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
    model.load_state_dict(torch.load(' ')) # path to the model

    model.eval()

    predictor = Model_Mnist(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
    predictor.load_state_dict(torch.load(' ')) # path to the model

    predictor.eval()


    # initialize the edge index
    edge_direc1 = torch.tensor([[0,0,1,1,2,2,3,4,4,5,5,6,6, 7, 8,8, 9, 9, 10,10,11,12,13,14],
                                [1,4,2,5,3,6,7,5,8,6,9,7,10,11,9,12,10,13,11,14,15,13,14,15]])
    edge_direc2_index = [1,0]                                      
    edge_direc2 = edge_direc1[edge_direc2_index]
    edge_index = torch.cat((edge_direc1, edge_direc2),1)
    batch = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    # load the dataset and test
    test_folder = './testset_covering/'
    avg_accuracy = 0
    avg_score = 0
    gt_avg_score = 0 
    avg_model_rounded = 0
    for index in range(1,501):
        test_path = test_folder+str(index)+'.pkl'
        test_file = open(test_path, 'rb')
        test_instance = pickle.load(test_file)
        node_feature = test_instance['node_feature']
        node_feature_label = test_instance['node_feature_label']
        score = test_instance['score']
        score_by_model, model_out_rounded = test_an_instance(model, edge_index, node_feature, node_feature_label, predictor)
        print("instance "+str(index)+"this_score: "+str(score_by_model))
        print("instance "+str(index)+"gt_score: "+str(score))
        print("instance "+str(index)+"model out rounded "+str(model_out_rounded))
        avg_score = avg_score + score_by_model / 500
        gt_avg_score = gt_avg_score + score / 500
        avg_model_rounded = avg_model_rounded + model_out_rounded / 500

    print("the overall average score is:"+str(avg_score))
    print("the overall ground truth average score is:"+str(gt_avg_score))
    print("the overall model out rounded is:"+str(avg_model_rounded))


if __name__ == '__main__':
    main()