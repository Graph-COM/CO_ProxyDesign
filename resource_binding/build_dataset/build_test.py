# generate application 1 dataset
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pickle
from pathlib import Path
import yaml
import re
import itertools
class Application_1_Test(InMemoryDataset):
    def __init__(self, config:dict):
        self.config = config
        self.data_path = Path(config['data_dir'])
        super(Application_1_Test, self).__init__(root=self.data_path)
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
    def get_idx_split(self):
        data_idx = np.arange(310)
        test_idx = data_idx
        return {'test':torch.tensor(test_idx,dtype = torch.long)}
    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        #case_index = [1, 2, 3, 4, 11, 12, 13, 14, 15]
        case_index = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,28,29,30,31,32,33,34,35,36,37,38,39,40]
        #case_index = [12]

        for k in range(len(case_index)):
            name='case_'+str(case_index[k])+'/case_'+str(case_index[k])+'_' 
            edge=pd.read_csv('./case/'+name+'edge.csv')
            source_list = edge['source'].values.tolist() # 'in14','in5','in2'
            target_list = edge['target'].values.tolist() # '14','5','2'
            source_list_num = []
            target_list_num = []
            for i in range(len(source_list)):
                source_list_num = source_list_num + re.findall(r"\d+\.?\d*",source_list[i])
                target_list_num = target_list_num + re.findall(r"\d+\.?\d*",target_list[i])
        
            tmp = source_list_num
            source_list_num = source_list_num + target_list_num
            target_list_num = target_list_num + tmp
            
            both_list = source_list_num + target_list_num
            both_array = np.array(both_list).reshape(2,-1).astype(float)-1
            edge_index = torch.from_numpy(both_array).long()
            graph_label_dsp = pd.read_csv('./case/'+name+'target_dsp.csv')
            graph_label_lut = pd.read_csv('./case/'+name+'target_lut.csv')
 
            num_nodes = torch.max(edge_index)
            nodes_list = np.arange(0,num_nodes+1)
            full_connect_edge = torch.tensor(list(itertools.permutations(nodes_list, 2))).T
            #print(full_connect_edge)

            save_path = './test_list/'
            save_file = open(save_path+str(case_index[k])+'.pkl', 'rb')
            list_tuple = pickle.load(save_file)
            instance_list_test = list_tuple['test_list']
            save_file.close()
            
            for i in instance_list_test:
                ith_label_dsp = np.array(graph_label_dsp)[i][0].astype(int)
                ith_label_lut = np.array(graph_label_lut)[i][0].astype(int)
                predict_dsp = torch.tensor(ith_label_dsp,dtype = torch.float).reshape(-1,1)
                predict_lut = torch.tensor(ith_label_lut,dtype = torch.float).reshape(-1,1)
                raw_content = pd.read_csv('./case/case_'+str(case_index[k])+'/case'+str(case_index[k])+'_'+str(i)+'.csv')
                node_feature = np.array(raw_content)
                node_feature = node_feature[...,1:].astype(int)
                x = torch.tensor(node_feature, dtype = torch.float)
                data = Data(x = x, edge_index = edge_index, predict_lut = predict_lut, predict_dsp = predict_dsp, case_index = case_index[k], instance_index = i)
                data_list.append(data)
     
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(len(data_list))


if __name__ == '__main__':
    import os
    configs = Path('./configs')
    for cfg in configs.iterdir():
        if str(cfg).startswith("configs/config"):
            cfg_dict = yaml.safe_load(cfg.open('r'))
            dataset = Application_1_Test(cfg_dict['test'])
    
    