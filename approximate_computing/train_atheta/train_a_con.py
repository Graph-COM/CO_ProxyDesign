import numpy as np
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
from model_con import PNA_con2, PNA_con
from torch_geometric.utils import degree
from loss import ErdosLoss
import time
import random
from tqdm import tqdm

def train(model, train_loader,criterion, optimizer,device):
    model.train()
    
    for data in train_loader:
        data.to(device)
        # alpha is (0.01, 0.5)
        alpha = torch.randint(1,51,(torch.max(data.batch)+1, 1))/ 100
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
            #alpha is (0.01, 0.5)
            alpha = torch.randint(1,51,(torch.max(data.batch)+1, 1)) /100
            alpha = alpha.to(device)
            out, fixed_feature = model(data.x, alpha, data.edge_index, data.batch)
            loss = criterion(out, fixed_feature, alpha,  data.edge_index, data.batch)
            fault_mse = fault_mse + loss*(torch.max(data.batch)  +1 )
    return fault_mse / len(test_loader.dataset) 

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    
    parser = argparse.ArgumentParser(description='this is the arg parser for application dataset 2')
    parser.add_argument('--save_path', dest = 'save_path',default = './train_files/con/new_train')
    parser.add_argument('--proxy_path', dest = 'proxy_path',default = '../train_proxy/train_files/con/train_con_100lr0001/best_val_model.pth')
    parser.add_argument('--gpu', dest = 'gpu',default = '7')

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # save the model and config for this training
    old_model_path = r'./model_con.py'
    new_model_path = os.path.join(args.save_path,'model_con.py')
    shutil.copyfile(old_model_path,new_model_path)

    old_config_path = r'../build_dataset/configs/config.yaml'
    new_config_path = os.path.join(args.save_path,'config.yaml')
    shutil.copyfile(old_config_path,new_config_path)

    old_train_path = r'./train_a_con.py'
    new_train_path = os.path.join(args.save_path,'train_a_con.py')
    shutil.copyfile(old_train_path,new_train_path)

    #setup_seed(123)

    cfg = Path("../build_dataset/configs/config.yaml")
    cfg_dict = yaml.safe_load(cfg.open('r'))
    dataset = Application_2_Dataset(cfg_dict['data'])
    data_splits = dataset.get_idx_split()
    train_dataset = dataset[data_splits['train']]
    val_dataset = dataset[data_splits['val']]
   

    train_loader = DataLoader(train_dataset, batch_size = 2048, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 2048, shuffle = False)
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
   
    model = PNA_con2(args.gpu, args.save_path)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    proxy = PNA_con(args.gpu, args.save_path)
    state_dict = torch.load(args.proxy_path,map_location=torch.device('cpu'))
    proxy.load_state_dict(state_dict)
    proxy.to(device)
    proxy.eval()
    #criterion = torch.nn.MSELoss() 
    #criterion = torch.nn.L1Loss()
    criterion = ErdosLoss(args.gpu, proxy)

    tensor_path = os.path.join(args.save_path,'tensor_log')

    writer = SummaryWriter(log_dir = tensor_path)
    best_loss_val = 10000
    best_loss_train = 10000
    start_time = time.time()
    for epoch in tqdm(range(1,201)):
        train(model,train_loader, criterion, optimizer,device)
        
        train_loss = test(model, train_loader,device,criterion)
        val_loss  = test(model, val_loader,device,criterion)
        if (val_loss<best_loss_val):
            best_loss_val = val_loss
            best_val_path = os.path.join(args.save_path,'best_val_model.pth')
            torch.save(model.state_dict(), best_val_path)
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
        writer.add_scalar('scalar/val_loss',val_loss,epoch)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    writer.close()


if __name__ == '__main__':
    main()



