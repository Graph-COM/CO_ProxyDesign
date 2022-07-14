import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pathlib import Path
import yaml
import re
import os
import pickle
import sys
sys.path.append("../../..")
from build_dataset.build_inference import Application_1_Inference
from train_g.model import SAGE_dsp, SAGE_lut, SAGE_model2test, PNA_model2test
from train_g_linear.model import LINEAR_dsp, LINEAR_lut, LINEAR_model2, PNA_linear2

from torch_geometric.data import DataLoader

def get_rounded(out_linear, fixed_feature, edge_index, batch, predictor_lut_linear, predictor_dsp_linear, alpha):
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    out_linear.to(device)
    fixed_feature = torch.tensor(fixed_feature).to(device)
    mask = fixed_feature[:,9].to(device)
    edge_index = torch.tensor(edge_index).to(device)
    batch = torch.tensor(batch).to(device)
    predictor_lut_linear.to(device)
    predictor_dsp_linear.to(device)
    alpha = torch.tensor(alpha).to(device)
    num_nodes = out_linear.shape[0]
    for i in range(num_nodes):
        node_feature_tmp_0 = torch.clone(out_linear).to(device)
        node_feature_tmp_0[i] = 0.0
        node_feature_tmp_1 = torch.clone(out_linear).to(device)
        node_feature_tmp_1[i] = 10.0
        #import pdb; pdb.set_trace()
        in_put_tmp_1 = torch.cat([fixed_feature, node_feature_tmp_1], 1).to(device)
        in_put_tmp_0 = torch.cat([fixed_feature, node_feature_tmp_0], 1).to(device)
        
        predicted_dsp_0 = predictor_dsp_linear(in_put_tmp_0, edge_index, batch)
        predicted_lut_0 =predictor_lut_linear(in_put_tmp_0, edge_index, batch)
        predicted_dsp_1 = predictor_dsp_linear(in_put_tmp_1, edge_index, batch)
        predicted_lut_1 =predictor_lut_linear(in_put_tmp_1, edge_index, batch)
        ones = torch.ones(predicted_dsp_0.shape[0],1).to(device)
        if mask[i] == 1:
            #if (predicted_lut_0 + alpha * torch.log(ones + torch.exp(predicted_dsp_0 - bar))) >= (predicted_lut_1 + alpha * torch.log(ones + torch.exp(predicted_dsp_1 - bar))):
            if (predicted_lut_0 + alpha * predicted_dsp_0) >= (predicted_lut_1 + alpha * predicted_dsp_1):    
                out_linear = node_feature_tmp_1
            else:
                out_linear = node_feature_tmp_0
        else:
            out_linear = node_feature_tmp_0
    #print(out_linear)
    predictor_lut_linear.to(torch.device("cpu"))
    predictor_dsp_linear.to(torch.device("cpu"))
    return out_linear.cpu()

def main():
    # load the test dataset
    cfg = Path("path to config")
    cfg_dict = yaml.safe_load(cfg.open('r'))
    testset = Application_1_Inference(cfg_dict['inference'])
    test_splits = testset.get_idx_split()
    test_dataset = testset[test_splits['inference']]
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    # load the model
    model = SAGE_model2test(in_channels = 11, hidden_channels = 128, out_channels = 512)
    #model = PNA_model2test()

    model.load_state_dict(torch.load('path ',torch.device("cpu")))
    model.eval()

    predictor_dsp_linear = LINEAR_dsp(in_channels = 10, hidden_channels = 128, out_channels = 512)
    predictor_dsp_linear.load_state_dict(torch.load('path ',torch.device("cpu")))
    predictor_dsp_linear.eval()

    predictor_lut_linear = LINEAR_lut(in_channels = 10, hidden_channels = 128, out_channels = 512)
    predictor_lut_linear.load_state_dict(torch.load('path ',torch.device("cpu")))
    predictor_lut_linear.eval()

    predictor_dsp = SAGE_dsp(in_channels = 11, hidden_channels = 32, out_channels = 128)
    predictor_dsp.load_state_dict(torch.load('path ',torch.device("cpu")))
    predictor_dsp.eval()

    predictor_lut = SAGE_lut(in_channels = 11, hidden_channels = 32, out_channels = 128)
    predictor_lut.load_state_dict(torch.load('path ',torch.device("cpu")))
    predictor_lut.eval()

    sum_lut_rounded = np.zeros(21)
    sum_dsp_rounded = np.zeros(21)
    sum_lut_rounded_linear = np.zeros(21)
    sum_dsp_rounded_linear = np.zeros(21)
 
    data_index = 0
    for data in test_loader:
        data_index = data_index+1
        #print("testing index:"+str(data_index))
        fixed_feature = data.x[:,:10]
        
        '''
        alpha_list_1 = np.arange(10,200,70)
        alpha_list_2 = np.arange(200, 400, 40)
        alpha_list_3 = np.arange(400, 600, 80)
        alpha_list = np.concatenate((alpha_list_1, alpha_list_2, alpha_list_3))
        '''
        #alpha_list = [10,40,70,100,130,160,190]
        alpha_list = np.array([5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,110,120,140,160,180,200])
        #alpha_list = np.array([10,20,30,40,50])

        # allocate the output
        out_ = torch.zeros((data.x.shape[0], 21))
        lowest_lut = np.zeros(21) + 10000
        lowest_dsp = np.zeros(21)
        lut_linear = np.zeros(21)
        dsp_linear = np.zeros(21)

        alpha_index = 0
        for alpha in alpha_list:
            alpha = torch.tensor(alpha).reshape(-1,1)
            #output of the optimizer output of the optimizer, and turn it into the inlut_roundedput of the predictor
            node_feature_,fixed_feature = model(data.x, alpha, data.edge_index, data.batch)
                
            # round the out_ , and turn it into the input of the predictor
            node_feature__rounded = get_rounded(node_feature_, fixed_feature, data.edge_index, data.batch, predictor_lut, predictor_dsp, alpha)
            in_put_rounded = torch.cat([fixed_feature, node_feature__rounded], 1)
            # get the lut and dsp from the predictor
            lut_rounded = predictor_lut(in_put_rounded, data.edge_index, data.batch)
            dsp_rounded = predictor_dsp(in_put_rounded, data.edge_index, data.batch)
            lut_linear_rounded = predictor_lut_linear(in_put_rounded, data.edge_index, data.batch)
            dsp_linear_rounded = predictor_dsp_linear(in_put_rounded, data.edge_index, data.batch)
            #print("bar: "+str(bars)+"  alpha: "+str(alpha.item())+"  lut: "+str(lut_rounded.item())+"  dsp: "+str(dsp_rounded.item()))
            # if it's smaller, save it into the save_file_list
            out_[:,alpha_index] = node_feature__rounded.reshape(-1)
            lowest_lut[alpha_index] = lut_rounded
            lowest_dsp[alpha_index] = dsp_rounded
            lut_linear[alpha_index] = lut_linear_rounded
            dsp_linear[alpha_index] = dsp_linear_rounded
            alpha_index = alpha_index + 1

        '''we need to save the:
        meta information of the instance: case_index, instance_index, 
        information if the instance: edge_index, fixed_feature
        output (rounded) of optimizer: output_linear
        the output of our objective (rounded): predicted_linear with threshold = 3, 5, 9, 11, 13, 17'''

        save_path = 'path to save'
        list_to_save = {'case_index': data.case_index, 'instance_index': data.instance_index,
                        'edge_index': data.edge_index, 'node_feature': data.x,
                        'mask': data.x[:,9].reshape(-1, 1), 'alpha_list': alpha_list,
                        'output_rounded':out_,
                        'lut_gumbel': lowest_lut,
                        'dsp_gumbel': lowest_dsp,
                        'dsp_linear': dsp_linear, 'lut_linear': lut_linear}
        save_file = open(save_path+str(data.case_index.item())+'.pkl', 'wb')
        pickle.dump(list_to_save, save_file)
        
        print("case:" + str(data.case_index.item()))
        print("DSP: "+str(alpha_list[0])+":"+str(lowest_dsp[0])+"   "+str(alpha_list[1])+":"+str(lowest_dsp[1])+"   "+str(alpha_list[2])+":"+str(lowest_dsp[2])+"   "+str(alpha_list[3])+":"+str(lowest_dsp[3])+"   "+str(alpha_list[4])+":"+str(lowest_dsp[4]))
        print("LUT: "+str(alpha_list[0])+":"+str(lowest_lut[0])+"   "+str(alpha_list[1])+":"+str(lowest_lut[1])+"   "+str(alpha_list[2])+":"+str(lowest_lut[2])+"   "+str(alpha_list[3])+":"+str(lowest_lut[3])+"   "+str(alpha_list[4])+":"+str(lowest_lut[4]))
        print("DSP_LINEAR: "+str(alpha_list[0])+":"+str(dsp_linear[0])+"   "+str(alpha_list[1])+":"+str(dsp_linear[1])+"   "+str(alpha_list[2])+":"+str(dsp_linear[2])+"   "+str(alpha_list[3])+":"+str(dsp_linear[3])+"   "+str(alpha_list[4])+":"+str(dsp_linear[4]))
        print("LUT_LINEAR: "+str(alpha_list[0])+":"+str(lut_linear[0])+"   "+str(alpha_list[1])+":"+str(lut_linear[1])+"   "+str(alpha_list[2])+":"+str(lut_linear[2])+"   "+str(alpha_list[3])+":"+str(lut_linear[3])+"   "+str(alpha_list[4])+":"+str(lut_linear[4]))

        sum_lut_rounded = sum_lut_rounded + lowest_lut
        sum_dsp_rounded = sum_dsp_rounded + lowest_dsp
        sum_lut_rounded_linear = sum_lut_rounded_linear + lut_linear
        sum_dsp_rounded_linear = sum_dsp_rounded_linear + dsp_linear
        
    print("the test dataset has: "+str(data_index)+"instances")
    print("the average lut of linear optimizer rounded: " + str(sum_lut_rounded/data_index))
    print("the average dsp of linear optimizer rounded: " + str(sum_dsp_rounded/data_index))
    print("the average lut by linear: " + str(sum_lut_rounded_linear / data_index))
    print("the average dsp by linear: " + str(sum_dsp_rounded_linear / data_index))

if __name__ == '__main__':
    main()