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

class Application_1_Inference(InMemoryDataset):
    def __init__(self, config:dict):
        self.config = config
        self.data_path = Path(config['data_dir'])
        super(Application_1_Inference, self).__init__(root=self.data_path)
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
    # 8172 is all the cases, 9 is the threshold 0, 1, 2, 3, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    def get_idx_split(self):
        data_idx = np.arange(20) 
        test_idx = data_idx
        return {'inference':torch.tensor(test_idx,dtype = torch.long)}
    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        #case_index = [1, 2, 3, 4, 11, 12, 13, 14, 15]
        case_index = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23]

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
 
            length_list = len(graph_label_dsp)
            random_index = np.random.randint(0,length_list)
            raw_content = pd.read_csv('./case/case_'+str(case_index[k])+'/case'+str(case_index[k])+'_'+str(random_index)+'.csv')
            node_feature = np.array(raw_content)
            node_feature = node_feature[...,1:].astype(int)
            x = torch.tensor(node_feature, dtype = torch.float)
            data = Data(x = x, edge_index = edge_index, case_index = case_index[k], instance_index = random_index)
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
            dataset = Application_1_Inference(cfg_dict['inference'])
    
    