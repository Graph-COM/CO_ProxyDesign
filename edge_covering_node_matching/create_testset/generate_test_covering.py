import numpy as np
from pathlib import Path
import os
from numba import jit
from numba.typed import List
from itertools import product
import pickle
import torch

@jit(nopython=True)
def f(z1, z2, x):
    # z1: node feature in one side, E * 1
    # z2: node feature in the other side, E * 1
    # x: edge feature, optimization variable, E * 1
    g1 = (z1 + z2) / 3 + (z1 * z2) / 100
    g2 = (z1 / (z2+1) + z2 / (z1+1)) * 5
    ans = np.mean(g1 * x + g2)
    return ans


mnist_list = torch.load('../mnist/processed/training.pt')
mnist_feature_list = mnist_list[0].numpy()
mnist_label_list = mnist_list[1].numpy()
#@jit(nopython=True)
def find_min_score(probs):
    edge_direc1 = np.array([[0,0,1,1,2,2,3,4,4,5,5,6,6, 7, 8,8, 9, 9, 10,10,11,12,13,14],
                            [1,4,2,5,3,6,7,5,8,6,9,7,10,11,9,12,10,13,11,14,15,13,14,15]])
    lift_index1 = edge_direc1[0,:]
    lift_index2 = edge_direc1[1,:]#
    #generate the node feature from mnist
    node_feature_index_decade = np.random.randint(1,60000,16)
    node_feature_index_unit = np.random.randint(1,60000,16)
    node_feature_decade = mnist_feature_list[node_feature_index_decade]
    node_feature_unit = mnist_feature_list[node_feature_index_unit]
    node_feature = np.concatenate([node_feature_decade, node_feature_unit], -1)
    # get the node feature
    node_feature_label_decade = mnist_label_list[node_feature_index_decade]
    node_feature_label_unit = mnist_label_list[node_feature_index_unit]
    node_feature_label = (10 * node_feature_label_decade + node_feature_label_unit)
    # initialize min score,target node feature, target edge feature
    min_score = 10000
    target_edge_feature = np.zeros(24).astype(np.int64)
    #to count
    #flag = 0
    for i in probs:
        #flag = flag + 1
        #if (flag % 10000 ==0):
            #print("tested flag"+str(flag)+"!")
        if sum(i)<8 or sum(i) > 16:
            continue
        else:
            # generate the edge feature
            edge_feature = np.array(i)
            # generate the lifted node feature with the dimension of |E|
            node_feature_label_lifted1 = node_feature_label[lift_index1]
            node_feature_label_lifted2 = node_feature_label[lift_index2]
            # generate the label y
            y = f(node_feature_label_lifted1, node_feature_label_lifted2, edge_feature)
            # generate the punishment
            punishment = (   (1-edge_feature[0])*(1-edge_feature[1])  
                           + (1-edge_feature[2])*(1-edge_feature[3])*(1-edge_feature[0])
                           + (1-edge_feature[4])*(1-edge_feature[5])*(1-edge_feature[2])
                           + (1-edge_feature[6])*(1-edge_feature[4])   
                           + (1-edge_feature[7])*(1-edge_feature[8])*(1-edge_feature[1]) 
                           + (1-edge_feature[9])*(1-edge_feature[10])*(1-edge_feature[3])*(1-edge_feature[7]) 
                           + (1-edge_feature[11])*(1-edge_feature[12])*(1-edge_feature[5])*(1-edge_feature[9])
                           + (1-edge_feature[13])*(1-edge_feature[6])*(1-edge_feature[11])
                           + (1-edge_feature[15])*(1-edge_feature[8])*(1-edge_feature[14])
                           + (1-edge_feature[16])*(1-edge_feature[17])*(1-edge_feature[10])*(1-edge_feature[14])
                           + (1-edge_feature[18])*(1-edge_feature[19])*(1-edge_feature[12])*(1-edge_feature[16])
                           + (1-edge_feature[20])*(1-edge_feature[13])*(1-edge_feature[18])
                           + (1-edge_feature[21])*(1-edge_feature[15])
                           + (1-edge_feature[22])*(1-edge_feature[17])*(1-edge_feature[21])
                           + (1-edge_feature[23])*(1-edge_feature[19])*(1-edge_feature[22])
                           + (1-edge_feature[20])*(1-edge_feature[23])
                        ) * 200
            score = y + punishment
            if score < min_score:
                min_score = score
                target_edge_feature = edge_feature
                
    return node_feature, node_feature_label, target_edge_feature, min_score

def make_an_instance(save_path,index):
    probs = product(range(2), repeat = 24)
    typed_probs = List()
    [typed_probs.append(x) for x in probs]
    
    node_feature, node_feature_label, edge_feature, score = find_min_score(typed_probs)
    #print(node_feature)
    #print(node_feature_label)
    #print(edge_feature)
    #print(score)
    #import pdb; pdb.set_trace()
    list_to_save = {'node_feature':node_feature, 'node_feature_label': node_feature_label, 'edge_feature': edge_feature, 'score': score}
    save_file = open(save_path+str(index)+'.pkl', 'wb')
    pickle.dump(list_to_save, save_file)
    save_file.close()

#@jit(nopython=True)
def main():
    
    save_path = './testset_covering/'
    for index in range(1,501):
        print("generating the "+str(index)+" instance!")
        make_an_instance(save_path,index)
 
if __name__ == '__main__':
    main()