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
from tqdm import tqdm

import sys
sys.path.append("..")
from build_dataset.build_data import Synthetic_Mnist_Dataset
from tensorboardX import SummaryWriter
from model import Model_Mnist, ResBlock

def train(model, train_loader,criterion, optimizer,device, schedular):
    model.train()

    for data in train_loader:
        data.to(device)
        x_input = data.x
        x_input = x_input.unsqueeze(1) # x: B C H W
        node_label = data.node_feature_label.float()
        predict_label = data.y.reshape(-1,1).float()

        res_out, predict = model(x_input, data.edge_index, data.edge_attr, data.batch)
        
        loss_resnet = criterion(res_out, node_label)
        loss_predict = criterion(predict, predict_label)
        loss = loss_resnet + loss_predict
        loss.backward()
        optimizer.step()
        schedular.step()
        optimizer.zero_grad()

def test(model, test_loader,device,criterion):
    model.eval()
    fault_mse_predict = 0
    fault_mse_res = 0
    res_loss = 0
    predict_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data.to(device)
            x_input = data.x
            x_input = x_input.unsqueeze(1) 
            node_label = data.node_feature_label.float()
            predict_label = data.y.reshape(-1,1).float()
            res_out, predict = model(x_input, data.edge_index, data.edge_attr, data.batch)
            res_loss = criterion(res_out, node_label)
            predict_loss = criterion(predict, predict_label)
            fault_mse_predict = fault_mse_predict + predict_loss*(torch.max(data.batch)  +1 )
            fault_mse_res = fault_mse_res + res_loss*(torch.max(data.batch)+1)
    return fault_mse_predict / len(test_loader.dataset) , fault_mse_res / len(test_loader.dataset)

def main():
    
    parser = argparse.ArgumentParser(description='this is the arg parser for synthetic dataset 2 with mnist')
    parser.add_argument('--save_path', dest = 'save_path',default = 'train_files/new_train')
    parser.add_argument('--gpu', dest = 'gpu',default = '7')

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # save the model and config for this training
    old_model_path = r'./model.py'
    new_model_path = os.path.join(args.save_path,'model.py')
    shutil.copyfile(old_model_path,new_model_path)

    old_config_path = r'../build_dataset/configs/config.yaml'
    new_config_path = os.path.join(args.save_path,'config.yaml')
    shutil.copyfile(old_config_path,new_config_path)


    cfg = Path("../build_dataset/configs/config.yaml")
    cfg_dict = yaml.safe_load(cfg.open('r'))
    dataset = Synthetic_Mnist_Dataset(cfg_dict['data'])
    data_splits = dataset.get_idx_split()
    train_dataset = dataset[data_splits['train']]
    valid_dataset = dataset[data_splits['valid']]
    test_dataset = dataset[data_splits['test']]
   

    train_loader = DataLoader(train_dataset, batch_size = 160, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 160, shuffle = False)
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    

    model = Model_Mnist(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    schedular=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.999)
    
    criterion = torch.nn.MSELoss()
    model.to(device)

    tensor_path = os.path.join(args.save_path,'tensor_log')

    writer = SummaryWriter(log_dir = tensor_path)
    best_mse_test = 10000
    best_mse_train = 10000
    for epoch in range(1,202):
 
        train(model,train_loader, criterion, optimizer,device, schedular)
        train_mse_predict, train_mse_res = test(model, train_loader,device,criterion)
        test_mse_predict, test_mse_res  = test(model, test_loader,device,criterion)
        if (test_mse_predict<best_mse_test):
            best_mse_test = test_mse_predict
            best_test_path = os.path.join(args.save_path,'best_test_model.pth')
            torch.save(model.state_dict(), best_test_path)
        if (train_mse_predict<best_mse_train):
            best_mse_train = train_mse_predict
            best_train_path = os.path.join(args.save_path,'best_train_model.pth')
            torch.save(model.state_dict(), best_train_path)
        if (epoch%40==1):
            PATH = os.path.join(args.save_path,'epoch'+str(epoch)+'.pth')
            torch.save(model.state_dict(), PATH)
        writer.add_scalar('scalar/train_mseloss_predict',train_mse_predict,epoch)
        writer.add_scalar('scalar/test_mseloss_predict',test_mse_predict,epoch)
        writer.add_scalar('scalar/train_mseloss_res',train_mse_res,epoch)
        writer.add_scalar('scalar/test_mseloss_res',train_mse_res,epoch)
        print(f'Epoch: {epoch:03d}, Train MSE Pre: {train_mse_predict:.4f}, Test MSE Pre: {test_mse_predict:.4f},Train MSE Res: {train_mse_res:.4f}, Test MSE Res: {test_mse_res:.4f} ')
    writer.close()


if __name__ == '__main__':
    main()



