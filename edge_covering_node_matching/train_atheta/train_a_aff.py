import torch
from pathlib import Path
import yaml
import os
from torch_geometric.data import DataLoader
import argparse
import shutil


import sys
sys.path.append("..")
from build_dataset.build_data import Synthetic_Mnist_Dataset
from tensorboardX import SummaryWriter
from model_aff import Aff_Mnist2, ResBlock, Aff_Mnist
from loss_aff import ErdosLoss

from tqdm import tqdm

def train(model, train_loader,criterion_1, criterion_2, optimizer,device, schedular):
    model.train()

    for data in train_loader:
        data.to(device)
        x_input = data.x
        x_input = x_input.unsqueeze(1) # x: B C H W
        node_label = data.node_feature_label.float()
        #the_edge_attr = torch.ones_like(data.edge_attr)
        #the_edge_attr = the_edge_attr.to(device)

        res_out, edge_feature = model(x_input,data.edge_index,data.edge_attr)
        loss_resnet = criterion_2(res_out, node_label)
        loss_optimizer = criterion_1(data.x, data.edge_index, edge_feature, data.batch)
        
        loss = loss_resnet + loss_optimizer
        loss.backward()
        optimizer.step()
        schedular.step()
        optimizer.zero_grad()

def test(model, test_loader,device,criterion_1, criterion_2):
    model.eval()
    fault_mse_res = 0
    optimizer_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data.to(device)
            x_input = data.x
            x_input = x_input.unsqueeze(1) 
            node_label = data.node_feature_label.float()
            #the_edge_attr = torch.ones_like(data.edge_attr)
            #the_edge_attr = the_edge_attr.to(device)
            res_out, edge_feature = model(x_input, data.edge_index, data.edge_attr)
            loss_resnet = criterion_2(res_out, node_label)
            loss_opt = criterion_1(data.x, data.edge_index, edge_feature, data.batch)
            optimizer_loss = optimizer_loss + loss_opt * (torch.max(data.batch)+1)
            fault_mse_res = fault_mse_res + loss_resnet * (torch.max(data.batch)+1)
    return optimizer_loss/len(test_loader.dataset), fault_mse_res/len(test_loader.dataset)



def main():

    parser = argparse.ArgumentParser(description='this is the arg parser for synthetic dataset 2')
    parser.add_argument('--save_path', dest = 'save_path',default = 'train_files/new_train')
    parser.add_argument('--gpu', dest = 'gpu',default = '7')
    parser.add_argument('--proxy_path', dest = 'proxy_path',default = '../train_aff/train_files/aff/train_aff_lr0005/best_val_model.pth')


    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # save the model and config for this training
    old_model_path = r'./model_aff.py'
    new_model_path = os.path.join(args.save_path,'model_aff.py')
    shutil.copyfile(old_model_path,new_model_path)

    old_config_path = r'../build_dataset/configs/config.yaml'
    new_config_path = os.path.join(args.save_path,'config.yaml')
    shutil.copyfile(old_config_path,new_config_path)

    cfg = Path("../build_dataset/configs/config.yaml")
    cfg_dict = yaml.safe_load(cfg.open('r'))
    dataset = Synthetic_Mnist_Dataset(cfg_dict['data'])
    data_splits = dataset.get_idx_split()
    train_dataset = dataset[data_splits['train']]
    val_dataset = dataset[data_splits['valid']]
    test_dataset = dataset[data_splits['test']]
   

    train_loader = DataLoader(train_dataset, batch_size = 70, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 70, shuffle = True)
    #test_loader = DataLoader(test_dataset, batch_size = 70, shuffle = False)
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")


    model = Aff_Mnist2(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    schedular=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.999)

    proxy = Aff_Mnist(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
    state_dict = torch.load(args.proxy_path, map_location = torch.device("cpu"))
    proxy.load_state_dict(state_dict)
    proxy.to(device)
    proxy.eval()


    criterion_1 = ErdosLoss(args.gpu, proxy)
    criterion_2 = torch.nn.MSELoss()
    model.to(device)

    tensor_path = os.path.join(args.save_path,'tensor_log')

    writer = SummaryWriter(log_dir = tensor_path)
    best_mse_test = 100000
    best_mse_train = 100000
    for epoch in tqdm(range(1,202)):
        train(model,train_loader, criterion_1, criterion_2, optimizer, device, schedular)
        train_mse_optimizer, train_mse_res = test(model, train_loader,device,criterion_1, criterion_2)
        val_mse_optimizer, val_mse_res  = test(model, val_loader,device,criterion_1, criterion_2)
        if (val_mse_optimizer<best_mse_test):
            best_mse_val = val_mse_optimizer
            best_test_path = os.path.join(args.save_path,'best_val_model.pth')
            torch.save(model.state_dict(), best_test_path)
        if (train_mse_optimizer<best_mse_train):
            best_mse_train = train_mse_optimizer
            best_train_path = os.path.join(args.save_path,'best_train_model.pth')
            torch.save(model.state_dict(), best_train_path)
        if (epoch%40==1):
            PATH = os.path.join(args.save_path,'epoch'+str(epoch)+'.pth')
            torch.save(model.state_dict(), PATH)
        writer.add_scalar('scalar/train_mseloss_optimizer',train_mse_optimizer,epoch)
        writer.add_scalar('scalar/val_mseloss_optimizer',val_mse_optimizer,epoch)
        writer.add_scalar('scalar/train_mseloss_res',train_mse_res,epoch)
        writer.add_scalar('scalar/test_mseloss_res',train_mse_res,epoch)
        print('Epoch:'+str(epoch) +'Train MSE Opt: '+str(train_mse_optimizer.item())+'Test MSE Opt: '+str(val_mse_optimizer.item())+'Train MSE Res: '+str(train_mse_res.item())+'Test MSE Res: '+str(val_mse_res.item()))
    writer.close()


if __name__ == '__main__':
    main()