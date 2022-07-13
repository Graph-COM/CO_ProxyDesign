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
from build_dataset.build_data import Application_2_Dataset
from tensorboardX import SummaryWriter
from model import PNA_linear2
from torch_geometric.utils import degree
from loss import ErdosLoss
import time

def train(model, train_loader,criterion, optimizer,device):
    model.train()
    
    for data in train_loader:
        data.to(device)
        # alpha is (0.01, 10)
        alpha = torch.randint(1,31,(torch.max(data.batch)+1, 1))/ 100
        alpha = alpha.to(device)
        out, fixed_feature = model(data.x, alpha, data.edge_index, data.batch)
        loss = criterion(out, fixed_feature, alpha,  data.edge_index, data.batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(model, test_loader,device,criterion):
    model.eval()
    fault_mse = 0
    loss = 0
    with torch.no_grad():
        for data in test_loader:
            data.to(device)
            #alpha is (0.01, 10)
            alpha = torch.randint(1,31,(torch.max(data.batch)+1, 1)) /100
            alpha = alpha.to(device)
            out, fixed_feature = model(data.x, alpha, data.edge_index, data.batch)
            loss = criterion(out, fixed_feature, alpha,  data.edge_index, data.batch)
            fault_mse = fault_mse + loss*(torch.max(data.batch)  +1 )
    return fault_mse / len(test_loader.dataset) 

def main():
    
    parser = argparse.ArgumentParser(description='this is the arg parser for application dataset 2')
    parser.add_argument('--save_path', dest = 'save_path',default = './train_files/new_train')
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

    old_train_path = r'./train_g_linear.py'
    new_train_path = os.path.join(args.save_path,'train_g_linear.py')
    shutil.copyfile(old_train_path,new_train_path)



    cfg = Path("../build_dataset/configs/config.yaml")
    cfg_dict = yaml.safe_load(cfg.open('r'))
    dataset = Application_2_Dataset(cfg_dict['data'])
    data_splits = dataset.get_idx_split()
    train_dataset = dataset[data_splits['train']]
    test_dataset = dataset[data_splits['test']]
   

    train_loader = DataLoader(train_dataset, batch_size = 2048, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 2048, shuffle = False)
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    torch.save(deg, args.save_path+'/deg.pt')
   
    model = PNA_linear2()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    #criterion = torch.nn.MSELoss() 
    #criterion = torch.nn.L1Loss()
    criterion = ErdosLoss()
    model.to(device)

    tensor_path = os.path.join(args.save_path,'tensor_log')

    writer = SummaryWriter(log_dir = tensor_path)
    best_loss_test = 10000
    best_loss_train = 10000
    start_time = time.time()
    for epoch in range(1,201):
        train(model,train_loader, criterion, optimizer,device)
        
        train_loss = test(model, train_loader,device,criterion)
        test_loss  = test(model, test_loader,device,criterion)
        if (test_loss<best_loss_test):
            best_loss_test = test_loss
            best_test_path = os.path.join(args.save_path,'best_test_model.pth')
            torch.save(model.state_dict(), best_test_path)
        if (train_loss<best_loss_train):
            best_loss_train = train_loss
            best_train_path = os.path.join(args.save_path,'best_train_model.pth')
            torch.save(model.state_dict(), best_train_path)
        if (epoch%1==0):
            this_time = time.time()
            print("time used: "+str(this_time - start_time))
            PATH = os.path.join(args.save_path,'epoch'+str(epoch)+'.pth')
            torch.save(model.state_dict(), PATH)
        writer.add_scalar('scalar/train_loss',train_loss,epoch)
        writer.add_scalar('scalar/test_loss',test_loss,epoch)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    writer.close()


if __name__ == '__main__':
    main()



