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
#from train_g_linear.model import Linear_Mnist, Linear_Mnist2, ResBlock
from train_g_concave.model import Linear_Mnist, Linear_Mnist2, ResBlock
import pickle

from tqdm import tqdm

H = torch.zeros(24,16).long()
a = torch.ones(24,16).long()
index_ = [[0,1],
          [0,4],
          [1,2],
          [1,5],
          [2,3],
          [2,6],
          [3,7],
          [4,5],
          [4,8],
          [5,6],
          [5,9],
          [6,7],
          [6,10],
          [7,11],
          [8,9],
          [8,12],
          [9,10],
          [9,13],
          [10,11],
          [10,14],
          [11,15],
          [12,13],
          [13,14],
          [14,15]]
H.scatter_(1,torch.LongTensor(index_),a)
H_ = torch.mm(H,H.T)
H_ = torch.triu(H_, diagonal = 1)


def f(z1, z2, x):
    # z1: node feature in one side, E * 1
    # z2: node feature in the other side, E * 1
    # x: edge feature, optimization variable, E * 1
    g1 = (z1 + z2) / 3 + (z1 * z2) / 100
    g2 = (z1 / (z2+1) + z2 / (z1+1)) * 5
    ans = np.mean(g1 * x + g2)
    return ans


def test_an_instance(predictor, model, edge_index, node_feature, node_feature_label, penalty_coefficient):
    node_feature_numpy = node_feature
    node_feature = torch.tensor(node_feature.reshape(-1,28,56)).float()
    node_feature = node_feature.unsqueeze(1)
    #import pdb; pdb.set_trace()
    edge_feature_direc1 = np.random.randint(0,2,24)
    edge_feature_direc1_reshape = edge_feature_direc1.reshape(-1,1)
    edge_feature_direc1_tensor = torch.from_numpy(edge_feature_direc1_reshape)
    edge_feature = torch.cat((edge_feature_direc1_tensor,edge_feature_direc1_tensor),0)
    edge_feature = torch.tensor(edge_feature).float()   
    
    _, model_out = model(node_feature, edge_index, edge_feature)
    out_1 = model_out[:24,0].detach().numpy()
    edge_feature_model = np.zeros(24)
    edge_feature_model = out_1

    # set each digit to either 0 or 1
    for i in range(24):
        edge_feature_tmp_0 = edge_feature_model.copy()
        edge_feature_tmp_0[i] = 0.0
        edge_feature_tmp_1 = edge_feature_model.copy()
        edge_feature_tmp_1[i] = 1.0
        punishment_0 = (   (1-edge_feature_tmp_0[0])*(1-edge_feature_tmp_0[1])  
                           + (1-edge_feature_tmp_0[2])*(1-edge_feature_tmp_0[3])*(1-edge_feature_tmp_0[0])
                           + (1-edge_feature_tmp_0[4])*(1-edge_feature_tmp_0[5])*(1-edge_feature_tmp_0[2])
                           + (1-edge_feature_tmp_0[6])*(1-edge_feature_tmp_0[4])   
                           + (1-edge_feature_tmp_0[7])*(1-edge_feature_tmp_0[8])*(1-edge_feature_tmp_0[1]) 
                           + (1-edge_feature_tmp_0[9])*(1-edge_feature_tmp_0[10])*(1-edge_feature_tmp_0[3])*(1-edge_feature_tmp_0[7]) 
                           + (1-edge_feature_tmp_0[11])*(1-edge_feature_tmp_0[12])*(1-edge_feature_tmp_0[5])*(1-edge_feature_tmp_0[9])
                           + (1-edge_feature_tmp_0[13])*(1-edge_feature_tmp_0[6])*(1-edge_feature_tmp_0[11])
                           + (1-edge_feature_tmp_0[15])*(1-edge_feature_tmp_0[8])*(1-edge_feature_tmp_0[14])
                           + (1-edge_feature_tmp_0[16])*(1-edge_feature_tmp_0[17])*(1-edge_feature_tmp_0[10])*(1-edge_feature_tmp_0[14])
                           + (1-edge_feature_tmp_0[18])*(1-edge_feature_tmp_0[19])*(1-edge_feature_tmp_0[12])*(1-edge_feature_tmp_0[16])
                           + (1-edge_feature_tmp_0[20])*(1-edge_feature_tmp_0[13])*(1-edge_feature_tmp_0[18])
                           + (1-edge_feature_tmp_0[21])*(1-edge_feature_tmp_0[15])
                           + (1-edge_feature_tmp_0[22])*(1-edge_feature_tmp_0[17])*(1-edge_feature_tmp_0[21])
                           + (1-edge_feature_tmp_0[23])*(1-edge_feature_tmp_0[19])*(1-edge_feature_tmp_0[22])
                           + (1-edge_feature_tmp_0[20])*(1-edge_feature_tmp_0[23])
                        ) * 200 * 3 
        
        penalty_0 = torch.mm(torch.tensor(edge_feature_tmp_0).reshape(1,-1),H_.float())
        penalty_0 = torch.mm(penalty_0,torch.tensor(edge_feature_tmp_0).reshape(-1,1))
        punishment_0 = 2 * punishment_0 + penalty_0 * penalty_coefficient

        punishment_1 = (   (1-edge_feature_tmp_1[0])*(1-edge_feature_tmp_1[1])  
                           + (1-edge_feature_tmp_1[2])*(1-edge_feature_tmp_1[3])*(1-edge_feature_tmp_1[0])
                           + (1-edge_feature_tmp_1[4])*(1-edge_feature_tmp_1[5])*(1-edge_feature_tmp_1[2])
                           + (1-edge_feature_tmp_1[6])*(1-edge_feature_tmp_1[4])   
                           + (1-edge_feature_tmp_1[7])*(1-edge_feature_tmp_1[8])*(1-edge_feature_tmp_1[1]) 
                           + (1-edge_feature_tmp_1[9])*(1-edge_feature_tmp_1[10])*(1-edge_feature_tmp_1[3])*(1-edge_feature_tmp_1[7]) 
                           + (1-edge_feature_tmp_1[11])*(1-edge_feature_tmp_1[12])*(1-edge_feature_tmp_1[5])*(1-edge_feature_tmp_1[9])
                           + (1-edge_feature_tmp_1[13])*(1-edge_feature_tmp_1[6])*(1-edge_feature_tmp_1[11])
                           + (1-edge_feature_tmp_1[15])*(1-edge_feature_tmp_1[8])*(1-edge_feature_tmp_1[14])
                           + (1-edge_feature_tmp_1[16])*(1-edge_feature_tmp_1[17])*(1-edge_feature_tmp_1[10])*(1-edge_feature_tmp_1[14])
                           + (1-edge_feature_tmp_1[18])*(1-edge_feature_tmp_1[19])*(1-edge_feature_tmp_1[12])*(1-edge_feature_tmp_1[16])
                           + (1-edge_feature_tmp_1[20])*(1-edge_feature_tmp_1[13])*(1-edge_feature_tmp_1[18])
                           + (1-edge_feature_tmp_1[21])*(1-edge_feature_tmp_1[15])
                           + (1-edge_feature_tmp_1[22])*(1-edge_feature_tmp_1[17])*(1-edge_feature_tmp_1[21])
                           + (1-edge_feature_tmp_1[23])*(1-edge_feature_tmp_1[19])*(1-edge_feature_tmp_1[22])
                           + (1-edge_feature_tmp_1[20])*(1-edge_feature_tmp_1[23])
                        ) * 200 * 3
        penalty_1 = torch.mm(torch.tensor(edge_feature_tmp_1).reshape(1,-1),H_.float())
        penalty_1 = torch.mm(penalty_1,torch.tensor(edge_feature_tmp_1).reshape(-1,1))
        punishment_1 = 2 * punishment_1 + penalty_1 * penalty_coefficient
        # generate the f prediction of the model's decision
        
        batch = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        edgef_feed0 = torch.tensor(edge_feature_tmp_0.reshape(-1,1))
        edgef_feed0 = torch.cat([edgef_feed0,edgef_feed0],0)
        edgef_feed1 = torch.tensor(edge_feature_tmp_1.reshape(-1,1))
        edgef_feed1 = torch.cat([edgef_feed1,edgef_feed1],0)
        _, y_0 = predictor(node_feature, edge_index, edgef_feed0, batch)
        _, y_1 = predictor(node_feature, edge_index, edgef_feed1, batch)
       
        
        # compare and set
        #print(punishment_0)
        #print(punishment_1)
        if (y_0 + punishment_0) <= (y_1 + punishment_1):
            
            edge_feature_model = edge_feature_tmp_0
        else: 
            edge_feature_model = edge_feature_tmp_1
        
    lift_index1 = edge_index[0,:24]
    lift_index2 = edge_index[1,:24]
    node_feature_label_lifted1 = node_feature_label[lift_index1]
    node_feature_label_lifted2 = node_feature_label[lift_index2]
    y = f(node_feature_label_lifted1, node_feature_label_lifted2, edge_feature_model)
    edge_num = sum(edge_feature_model)
    
    edge_f = torch.tensor(edge_feature_model)
    feed_f = torch.cat([edge_f, edge_f], 0).reshape(-1,1).float()
    _, model_out_rounded = predictor(node_feature, edge_index, feed_f, batch)
    return y, edge_feature_model, edge_num, model_out_rounded


def whether_match(edge_index, edge_feature):
    # return whether it's a perfect match
    node_flag = np.zeros(16)
    for i in range(24):
        if edge_feature[i]== 1:
            node_flag[edge_index[0,i]] = 1
            node_flag[edge_index[1,i]] = 1
    if sum(node_flag)==16:
        return 1
    else:
        return 0


def main():
    #load the model
    model = Linear_Mnist2(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
    model_state_dict = torch.load(' ', map_location = torch.device("cpu"))

    model.load_state_dict(model_state_dict)
    model.eval()

    # load the predictor
    predictor = Linear_Mnist(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
    predictor_state_dict =  torch.load(' ', map_location = torch.device("cpu"))
    predictor.load_state_dict(predictor_state_dict)
    
    predictor.eval()

    # initialize the edge index
    edge_direc1 = torch.tensor([[0,0,1,1,2,2,3,4,4,5,5,6,6, 7, 8,8, 9, 9, 10,10,11,12,13,14],
                                [1,4,2,5,3,6,7,5,8,6,9,7,10,11,9,12,10,13,11,14,15,13,14,15]])
    edge_direc2_index = [1,0]                                      
    edge_direc2 = edge_direc1[edge_direc2_index]
    edge_index = torch.cat((edge_direc1, edge_direc2),1)
    batch = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    # load the dataset and test
    test_folder = './testset_matching/'

    avg_accuracy = 0
    avg_score = 0
    gt_avg_score = 0
    avg_edge = 0
    avg_model_rounded = 0

    
    num_match = 0
    linear_list = []
    gt_list = []
    max_list = []
    for index in tqdm(range(1,501)):
        test_path = test_folder+str(index)+'.pkl'
        test_file = open(test_path, 'rb')
        test_instance = pickle.load(test_file)
        node_feature = test_instance['node_feature']
        node_feature_label = test_instance['node_feature_label']
        score = test_instance['score']
        max_score = test_instance['max_score']
        penalty_coeff = 300
        
        this_best = 10000
        big_match_flag = 0
        this_model_rounded = 0
        score_by_model, edge_feature_final, edge_num, model_out_rounded = test_an_instance(predictor,model, edge_index, node_feature, node_feature_label, penalty_coeff)
        while edge_num != 8:
            if edge_num > 8:
                penalty_coeff = penalty_coeff * 1.8
                score_by_model, edge_feature_final, edge_num, model_out_rounded = test_an_instance(predictor,model, edge_index, node_feature, node_feature_label, penalty_coeff)
            elif edge_num < 8:
                penalty_coeff = penalty_coeff * 0.8
                score_by_model, edge_feature_final, edge_num, model_out_rounded = test_an_instance(predictor,model, edge_index, node_feature, node_feature_label, penalty_coeff)
        match_flag = whether_match(edge_index, edge_feature_final)
        if match_flag ==1:
            big_match_flag = 1
            if score_by_model < this_best:
                this_best = score_by_model
                this_model_rounded = model_out_rounded

        if big_match_flag == 1:
            linear_list.append(this_best)
            gt_list.append(score)
            max_list.append(max_score)
            num_match = num_match + 1 

        #this_accuracy = 1.0 - (abs(score_by_model-score) / score )
        #print("instance "+str(index)+"this_accuracy: "+str(this_accuracy))
        #avg_accuracy = avg_accuracy + this_accuracy / 499
        print("instance "+str(index)+"whether match: "+str(match_flag))
        print("instance "+str(index)+"penalty_coefficient: "+str(penalty_coeff))
        print("instance "+str(index)+"num_edge: "+str(edge_num))
        print("instance "+str(index)+"this_score: "+str(this_best))
        print("instance "+str(index)+"gt_score: "+str(score))
        print("instance "+str(index)+"max_score: "+str(max_score))
        print("instance "+str(index)+"model out rounded: "+str(model_out_rounded))
       
        avg_score = avg_score + score_by_model / 500
        gt_avg_score = gt_avg_score + score / 500
        avg_edge = avg_edge + edge_num / 500
        avg_model_rounded = avg_model_rounded + this_model_rounded / 500

    print("total  number of matching: "+str(num_match))
    print("the overall average score is:"+str(np.mean(linear_list)))
    print("the overall average edge is:"+str(avg_edge))
    print("the overall ground truth average score is:"+str(np.mean(gt_list)))
    print("the overall max score is:"+str(np.mean(max_list)))
    print("the overall model out rounded is:"+str(avg_model_rounded))



if __name__ == '__main__':
    main()