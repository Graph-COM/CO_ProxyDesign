# generate application 2 dataset
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pathlib import Path
import yaml
from tqdm import tqdm

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
     
class Application_2_Dataset(InMemoryDataset):
    def __init__(self, config:dict):
        self.config = config
        self.data_path = Path(config['data_dir'])
        self.splits = config['splits']
        super(Application_2_Dataset, self).__init__(root=self.data_path)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['data.pt']
    def download(self):
        # Download to `self.raw_dir`.
        pass
    def get_idx_split(self, split_type = 'Random'):
        data_idx = np.arange(100000)
        splits = self.splits
        train_num = int(float(splits['train'])*100000)
        test_num = int(float(splits['val'])*100000)
        if (split_type==None):
            train_idx = data_idx[:train_num]
            test_idx = data_idx[train_num:]
        elif (split_type=='Random'):    
            shuffle_idx = np.random.shuffle(data_idx)
            train_idx = data_idx[:train_num]
            test_idx = data_idx[train_num:]
        else:
            print("something went wrong?")        
        return {'train':torch.tensor(train_idx,dtype = torch.long), 'val':torch.tensor(test_idx,dtype = torch.long)}
    def process(self):
        data_list = []
        for index_structure in tqdm(range(1,501)):
            # randomly sample the graph structure (edge_index)
            edge_index, num_input = sample_edge_index()
            # randomly sample the operation in each node  0:sum, 1:multiply
            operation = np.random.randint(0,2,15)
            #first calculate assignment all 0 (which means all correct)
            assignment_noerr = np.zeros(15)
            num_initials = 2000
            initial_input = np.random.randint(1,100,(num_initials, num_input))/10
            result_noerr = cal_result(initial_input, operation, assignment_noerr, edge_index, 100, 10)
            for index_assignment in range(1,201):
                # in each structure, sample 200 different assignments
                assignment = np.random.randint(0,2,15)
                # each initial_input needs 1000 sampling to calculate the average error
                num_iteration = 500
                result_list = np.zeros((num_initials, num_iteration))
                for index_iteration in range(num_iteration):
                    result = cal_result(initial_input, operation, assignment, edge_index, 100, 10).reshape(-1)
                    result_list[:,index_iteration] = result
                error_ = abs(result_list - result_noerr)
                relative_error = error_ / result_noerr
                avg_relative_error = np.mean(relative_error)
                #construct the node feature, the first digit is operation, the second digit is assignment
                node_feature = np.concatenate((operation.reshape(-1,1), assignment.reshape(-1,1)), 1)       
                node_feature = torch.Tensor(node_feature)
                #it's currently double-direction with no edge feature, might be changed later, 
                # and get rid of the initial input (16 digits)
                graph_edge_index = torch.tensor(edge_index[:,16:])
                graph_edge_index = graph_edge_index - num_input
                reverse_edge_index = graph_edge_index[[1,0]]
                #construct the graph
                double_edge_index = torch.cat([graph_edge_index, reverse_edge_index], 1)
                data = Data(x = node_feature, y = avg_relative_error, edge_index = double_edge_index)
                data_list.append(data)
                if index_assignment % 50 == 0:
                    print("making the "+str(index_assignment)+ "-th assignment of structure "+str(index_structure))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

if __name__ == '__main__':
    import os
    configs = Path('./configs')
    for cfg in configs.iterdir():
        if str(cfg).startswith("configs/config"):
            cfg_dict = yaml.safe_load(cfg.open('r'))
            dataset = Application_2_Dataset(cfg_dict['data'])