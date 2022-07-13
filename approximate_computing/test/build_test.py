# generate application 2 dataset
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pathlib import Path
import yaml
import re

from numba import jit
from numba.typed import List
from itertools import product
import pickle
import time

@jit(nopython=True)
def sample_edge_index():
    '''randomly sample the edge index for the circuit'''
    #the initial input: whether has 14 different inputs or 16
    num_input = 14 + np.random.randint(0,2)*2
    # shuffle the 0 (14/16 nodes), 1 (8 nodes), 2(4 nodes), 3(2 nodes) index to generate edge_index
    layer0_index = np.arange(num_input)
    np.random.shuffle(layer0_index)
    if num_input == 14:
        layer0_index = np.append(layer0_index, layer0_index[:2])
    layer1_index = np.arange(num_input, num_input + 8)
    np.random.shuffle(layer1_index)
    layer2_index = np.arange(num_input + 8, num_input + 12)
    np.random.shuffle(layer2_index)
    layer3_index = np.arange(num_input + 12, num_input + 14)
    # generate the edge index (upper row)
    up_row_index = np.concatenate((layer0_index, layer1_index, layer2_index, layer3_index)).reshape(1,-1)
    #generate the lower row by reshape trick [14 14 15 15 ,..., 27 27] or [16 16 17 17 ,..., 29 29] 
    tmp_index = np.arange(num_input, num_input + 15).reshape(-1,1)
    low_row_index = np.concatenate((tmp_index, tmp_index), 1)
    low_row_index = low_row_index.reshape(1, -1)
    edge_index = np.concatenate((up_row_index, low_row_index), 0)    
    return edge_index, num_input


def cal_node(l, r, op, assign, err_chance, err_margin):
    '''calculate the result in a single node
       l: left input n*1
       r: right input n*1
       op: 0 add, 1 multiply
       err_chance: percentage that the assignment would cause an error is set to 1
       err_margin: how much error the assignment occur, also percentage
    '''
    switcher = {
        0 : l + r,
        1 : l * r,
    }
    chance = float(np.random.randint(0, 10000) / 100)
    res = switcher.get(op)
    if assign == 1 and chance <= err_chance:
        add_or_minus = np.random.randint(0,2)
        if add_or_minus:
            res *= (1 + float(err_margin / 100))
        else:
            res *= (1 - float(err_margin / 100))
    return res


def cal_result(initial_input, operation, assignment, edge_index, err_chance, err_margin):
    '''given an initial input, operation, assignment, edge_index, err_chance and err_margin, calculate the output
    n is the sampling of random initial inputs
    initial input: 14/16 digits randomly 0.1-10          n*14 / n* 16
    operation: 15 digit 0/1: 0: sum; 1: multiply         1*15
    assignment: 15 digit 0/1 0: no error; 1: error       1*15
    edge_index: 2*30 edge index                          2*30
    err_chance: percentage that the assignment would cause an error is set to 1
    err_margin: how much error the assignment occur, also percentage
    '''
    # calculate the number of sampling n and do some allocation
    num_sampling = initial_input.shape[0]
    num_initial = initial_input.shape[1]
    result_list = np.zeros((num_sampling, 15))
    result_list = np.concatenate((initial_input, result_list), 1)
    up_row_edge = edge_index[0,:]
    #calculate the result
    for i in range(15):
        op = operation[i]
        assign = assignment[i]
        l = result_list[:,up_row_edge[2*i]]
        r = result_list[:,up_row_edge[2*i+1]]
        result_list[:,num_initial + i] = cal_node(l, r, op, assign, err_chance, err_margin)
    return result_list[:,num_initial + 14].reshape(-1,1)
    #return result_list


def find_min_score(result_noerr, num_initials, edge_index, operation, initial_input, typed_probs, threshold_1, threshold_2, threshold_3):
    min_score_1 = 100
    node_assignment_1 = np.zeros(15)
    min_score_2 = 100
    node_assignment_2 = np.zeros(15)
    min_score_3 = 100
    node_assignment_3 = np.zeros(15)
    # each initial_input needs 500 sampling to calculate the average error
    num_iteration = 100
    index = 0
    for assignment in typed_probs:
        index = index + 1
        if index % 8000 ==0:
            print("testing index:"+str(index))
        if sum(assignment)!=threshold_1 and sum(assignment)!=threshold_2 and sum(assignment)!=threshold_3:
            continue
        elif sum(assignment)==threshold_1:
            assignment = np.array(assignment)
            result_list = np.zeros((num_initials, num_iteration))
            for index_iteration in range(num_iteration):
                result = cal_result(initial_input, operation, assignment, edge_index, 100, 10).reshape(-1)
                result_list[:,index_iteration] = result
            error_ = abs(result_list - result_noerr)
            relative_error = error_ / result_noerr
            avg_relative_error = np.mean(relative_error)
            if avg_relative_error < min_score_1:
                min_score_1 = avg_relative_error
                node_assignment_1 = assignment
        elif sum(assignment)==threshold_2:
            assignment = np.array(assignment)
            result_list = np.zeros((num_initials, num_iteration))
            for index_iteration in range(num_iteration):
                result = cal_result(initial_input, operation, assignment, edge_index, 100, 10).reshape(-1)
                result_list[:,index_iteration] = result
            error_ = abs(result_list - result_noerr)
            relative_error = error_ / result_noerr
            avg_relative_error = np.mean(relative_error)
            if avg_relative_error < min_score_2:
                min_score_2 = avg_relative_error
                node_assignment_2 = assignment   
        elif sum(assignment)==threshold_3:
            assignment = np.array(assignment)
            result_list = np.zeros((num_initials, num_iteration))
            for index_iteration in range(num_iteration):
                result = cal_result(initial_input, operation, assignment, edge_index, 100, 10).reshape(-1)
                result_list[:,index_iteration] = result
            error_ = abs(result_list - result_noerr)
            relative_error = error_ / result_noerr
            avg_relative_error = np.mean(relative_error)
            if avg_relative_error < min_score_3:
                min_score_3 = avg_relative_error
                node_assignment_3 = assignment    
    return node_assignment_1, min_score_1, node_assignment_2, min_score_2, node_assignment_3, min_score_3
                
                

def make_an_instance(save_path,index):
    #generate the initial metadata for one structure: edge_index, operation, initial_input
    edge_index, num_input = sample_edge_index()
    assignment_noerr = np.zeros(15)
    num_initials = 500
    initial_input = np.random.randint(1,100,(num_initials, num_input))/10
    operation = np.random.randint(0,2,15)

    #get the result with no error
    result_noerr = cal_result(initial_input, operation, assignment_noerr, edge_index, 100, 10)
    
    #iterate over all of the possible choices on the assignment and find out the best one
    probs = product(range(2), repeat = 15)
    typed_probs = List()
    [typed_probs.append(x) for x in probs]
    node_feature_3, score_3 , node_feature_5, score_5, node_feature_8, score_8 = find_min_score(result_noerr, num_initials, edge_index, operation, initial_input, typed_probs, 3, 5, 8)

    
    #it's currently double-direction with no edge feature, might be changed later, 
    # and get rid of the initial input (16 digits)
    graph_edge_index = torch.tensor(edge_index[:,16:])
    graph_edge_index = graph_edge_index - num_input
    reverse_edge_index = graph_edge_index[[1,0]]
    #construct the graph
    double_edge_index = torch.cat([graph_edge_index, reverse_edge_index], 1)
    list_to_save = {'operation': operation, 'edge_index': double_edge_index,'node_feature_3':node_feature_3, 'score_3': score_3, 'node_feature_5':node_feature_5, 'score_5': score_5, 'node_feature_8':node_feature_8, 'score_8': score_8}
    
    save_file = open(save_path+str(index)+'.pkl', 'wb')
    pickle.dump(list_to_save, save_file)
    save_file.close()

def main():
    start_time = time.time()
    save_path = '/scratch1/wang5272/erdos/application_2/test/testset_time/'
    for index in range(1,500):
        print("generating the "+str(index)+" instance!")
        make_an_instance(save_path,index)
        this_time = time.time()
        print("used time: "+str(this_time - start_time))
    end_time = time.time()
    print("used time: "+str(end_time - start_time))
    
if __name__ == '__main__':
    main()

        

