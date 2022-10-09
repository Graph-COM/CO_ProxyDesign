import torch
from pathlib import Path
import yaml
import os
from torch_geometric.data import DataLoader
import argparse
import shutil


import sys
sys.path.append("..")
from build_dataset.build_data import Application_2_Dataset
from tensorboardX import SummaryWriter
from model_aff import PNA_aff
from torch_geometric.utils import degree
from tqdm import tqdm

def train(model, train_loader,criterion, optimizer,device):
    model.train()
    
    for data in train_loader:
        data.to(device)
        data.y = 100*data.y.reshape(-1,1).float()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(model, test_loader,device,criterion):
    model.eval()
    fault_mse = 0
    loss = 0
    mean_relative_error = 0
    with torch.no_grad():
        for data in test_loader:
            data.y = 100*data.y.reshape(-1,1) 
            data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            # calculate the mean relative error
            relative_error = torch.mean(abs(out - data.y)/(data.y+1e-6))
            #print(relative_error)
            mean_relative_error = mean_relative_error + relative_error * (torch.max(data.batch) + 1)
            fault_mse = fault_mse + loss*(torch.max(data.batch)  +1 )
    return fault_mse / len(test_loader.dataset) , mean_relative_error / len(test_loader.dataset)

def main():
    
    parser = argparse.ArgumentParser(description='this is the arg parser for application dataset 1')
    parser.add_argument('--save_path', dest = 'save_path',default = './train_files/aff/new_train')
    parser.add_argument('--gpu', dest = 'gpu',default = '7')

    args = parser.parse_args()
    



    cfg = Path("../build_dataset/configs/config.yaml")
    cfg_dict = yaml.safe_load(cfg.open('r'))
    dataset = Application_2_Dataset(cfg_dict['data'])
    data_splits = dataset.get_idx_split()
    train_dataset = dataset[data_splits['train']]
    
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
    max_y = 0
    for data in train_loader:
        y = 100*data.y.reshape(-1,1).float()
        if y > max_y:
            max_y = y
    print('max_y'+str(max_y))

if __name__ == '__main__':
    main()



