import numpy as np
import torch
import sys
sys.path.append("..")
from train_atheta.model_aff import Aff_Mnist, Aff_Mnist2, ResBlock
import pickle
from tqdm import tqdm
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

def paint_edge(save_path, index, node_feature, node_feature_label, edge_feature, score,  mode):
    if not os.path.exists(save_path + str(index)):
        os.mkdir(save_path + str(index))
    if mode == 'plain':
        pic_path = save_path + str(index) + '/plain.pdf'
    if mode == 'OPT':
        pic_path = save_path + str(index) + '/gt.pdf'
    if mode == 'GS-Tr+R':
        pic_path = save_path + str(index) + '/gumbel.pdf'
    if mode == 'AFF':
        pic_path = save_path + str(index) + '/linear.pdf'
   
    nodes_16 = node_feature.reshape(-1, 28, 56)

    fig = plt.figure(figsize=(28,14)) # Notice the equal aspect ratio
    ax = [fig.add_subplot(4,4,i+1) for i in range(16)]
    #fig, ax = plt.subplots(nrows=4, ncols=4, sharex='all', sharey='all')
    #ax = ax.flatten()
    #ax[0].set_xticks([])
    #ax[0].set_yticks([])
    i = 0
    for index in range(4):
        for j in range(4):
            #ax[i].set_xlim(0, 56)
            #ax[i].set_ylim(28,0)
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            ax[i].set_aspect('equal')
            img = nodes_16[i].reshape(28, 56)
            ax[i].imshow(img)
            i = i + 1
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()

    if mode == 'plain':
        plt.suptitle("a plain picture", fontsize = 28, color = 'y')
        plt.savefig(pic_path)
        
    else:
        #x = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]).reshape(1,-1)
        #y = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]).reshape(1,-1)
        #xy = np.transpose((np.append(x,y,0))) # 16 * 1 coordinate
        xy = (28, 14)
        edge_direc1 = torch.tensor([[0,0,1,1,2,2,3,4,4,5,5,6,6, 7, 8,8, 9, 9, 10,10,11,12,13,14],
                                [1,4,2,5,3,6,7,5,8,6,9,7,10,11,9,12,10,13,11,14,15,13,14,15]])
        edge_list = np.transpose(edge_direc1.numpy())
        chosen_index = np.where(edge_feature==1)
        chosen_edge = edge_list[chosen_index,:].squeeze(0)
        
        for p in range(len(chosen_edge)):
            con = ConnectionPatch(
            xyA=xy, coordsA=ax[chosen_edge[p][0]].transData,
            xyB=xy, coordsB=ax[chosen_edge[p][1]].transData,
            arrowstyle="-", shrinkB=5)
            con.set_linewidth(12)
            con.set_color([255/255, 64/255, 64/255])
            fig.add_artist(con)
        plt.suptitle(str(mode)+" score: "+ str(score), fontsize = 28, color = 'y')
        plt.savefig(pic_path)

def f(z1, z2, x):
    # z1: node feature in one side, E * 1
    # z2: node feature in the other side, E * 1
    # x: edge feature, optimization variable, E * 1
    g1 = (z1 + z2) / 3 + (z1 * z2) / 100
    g2 = (z1 / (z2+1) + z2 / (z1+1)) * 5
    ans = torch.mean(g1 * x + g2)
    return ans


def test_an_instance(predictor, model, edge_index, node_feature, node_feature_label, device):
    node_feature_label = node_feature_label.to(device)
    node_feature = torch.tensor(node_feature.reshape(-1,28,56)).float()
    node_feature = node_feature.unsqueeze(1)
    node_feature = node_feature.to(device)
    #import pdb; pdb.set_trace()
    edge_feature_direc1 = np.random.randint(0,2,24)
    edge_feature_direc1_reshape = edge_feature_direc1.reshape(-1,1)
    edge_feature_direc1_tensor = torch.from_numpy(edge_feature_direc1_reshape)
    edge_feature = torch.cat((edge_feature_direc1_tensor,edge_feature_direc1_tensor),0)
    edge_feature = torch.tensor(edge_feature).float()
    edge_feature = edge_feature.to(device)  
     
    
    _, model_out = model(node_feature, edge_index, edge_feature)
    out_1 = model_out[:24,0].detach()
    #edge_feature_model = np.zeros(24)
    edge_feature_model = out_1

    # set each digit to either 0 or 1
    for i in range(24):
        edge_feature_tmp_0 = edge_feature_model.clone()
        edge_feature_tmp_0[i] = 0.0
        edge_feature_tmp_1 = edge_feature_model.clone()
        edge_feature_tmp_1[i] = 1.0
        punishment_0 = (   (1-edge_feature_tmp_0[0])*(1-edge_feature_tmp_0[1])  
                           + (1-edge_feature_tmp_0[2])*(1-edge_feature_tmp_0[3])*(1-edge_feature_tmp_0[0])
                           + (1-edge_feature_tmp_0[4])*(1-edge_feature_tmp_0[5])*(1-edge_feature_tmp_0[2])
                           + (1-edge_feature_tmp_0[6])*(1-edge_feature_tmp_0[4])   
                           + (1-edge_feature_tmp_0[7])*(1-edge_feature_tmp_0[8])*(1-edge_feature_tmp_0[1]) 
                           + (1-edge_feature_tmp_0[9])*(1-edge_feature_tmp_0[10])*(1-edge_feature_tmp_0[3])*(1-edge_feature_tmp_0[7]) 
                           + (1-edge_feature_tmp_0[11])*(1-edge_feature_tmp_0[12])*(1-edge_feature_tmp_0[5])*(1-edge_feature_tmp_0[9])
                           + (1-edge_feature_tmp_0[13])*(1-edge_feature_tmp_0[6])*(1-edge_feature_tmp_0[11])
                           + (1-edge_feature_tmp_0[15])*(1-edge_feature_tmp_0[8])*(1-edge_feature_tmp_0[14])
                           + (1-edge_feature_tmp_0[16])*(1-edge_feature_tmp_0[17])*(1-edge_feature_tmp_0[10])*(1-edge_feature_tmp_0[14])
                           + (1-edge_feature_tmp_0[18])*(1-edge_feature_tmp_0[19])*(1-edge_feature_tmp_0[12])*(1-edge_feature_tmp_0[16])
                           + (1-edge_feature_tmp_0[20])*(1-edge_feature_tmp_0[13])*(1-edge_feature_tmp_0[18])
                           + (1-edge_feature_tmp_0[21])*(1-edge_feature_tmp_0[15])
                           + (1-edge_feature_tmp_0[22])*(1-edge_feature_tmp_0[17])*(1-edge_feature_tmp_0[21])
                           + (1-edge_feature_tmp_0[23])*(1-edge_feature_tmp_0[19])*(1-edge_feature_tmp_0[22])
                           + (1-edge_feature_tmp_0[20])*(1-edge_feature_tmp_0[23])
                        ) * 10

        punishment_1 = (   (1-edge_feature_tmp_1[0])*(1-edge_feature_tmp_1[1])  
                           + (1-edge_feature_tmp_1[2])*(1-edge_feature_tmp_1[3])*(1-edge_feature_tmp_1[0])
                           + (1-edge_feature_tmp_1[4])*(1-edge_feature_tmp_1[5])*(1-edge_feature_tmp_1[2])
                           + (1-edge_feature_tmp_1[6])*(1-edge_feature_tmp_1[4])   
                           + (1-edge_feature_tmp_1[7])*(1-edge_feature_tmp_1[8])*(1-edge_feature_tmp_1[1]) 
                           + (1-edge_feature_tmp_1[9])*(1-edge_feature_tmp_1[10])*(1-edge_feature_tmp_1[3])*(1-edge_feature_tmp_1[7]) 
                           + (1-edge_feature_tmp_1[11])*(1-edge_feature_tmp_1[12])*(1-edge_feature_tmp_1[5])*(1-edge_feature_tmp_1[9])
                           + (1-edge_feature_tmp_1[13])*(1-edge_feature_tmp_1[6])*(1-edge_feature_tmp_1[11])
                           + (1-edge_feature_tmp_1[15])*(1-edge_feature_tmp_1[8])*(1-edge_feature_tmp_1[14])
                           + (1-edge_feature_tmp_1[16])*(1-edge_feature_tmp_1[17])*(1-edge_feature_tmp_1[10])*(1-edge_feature_tmp_1[14])
                           + (1-edge_feature_tmp_1[18])*(1-edge_feature_tmp_1[19])*(1-edge_feature_tmp_1[12])*(1-edge_feature_tmp_1[16])
                           + (1-edge_feature_tmp_1[20])*(1-edge_feature_tmp_1[13])*(1-edge_feature_tmp_1[18])
                           + (1-edge_feature_tmp_1[21])*(1-edge_feature_tmp_1[15])
                           + (1-edge_feature_tmp_1[22])*(1-edge_feature_tmp_1[17])*(1-edge_feature_tmp_1[21])
                           + (1-edge_feature_tmp_1[23])*(1-edge_feature_tmp_1[19])*(1-edge_feature_tmp_1[22])
                           + (1-edge_feature_tmp_1[20])*(1-edge_feature_tmp_1[23])
                        ) * 10
        # generate the f prediction of the model's decision
        
        batch = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        edgef_feed0 = torch.tensor(edge_feature_tmp_0.reshape(-1,1))
        edgef_feed0 = torch.cat([edgef_feed0,edgef_feed0],0)
        edgef_feed1 = torch.tensor(edge_feature_tmp_1.reshape(-1,1))
        edgef_feed1 = torch.cat([edgef_feed1,edgef_feed1],0)
        _, y_0 = predictor(node_feature, edge_index, edgef_feed0, batch)
        _, y_1 = predictor(node_feature, edge_index, edgef_feed1, batch)
       
        if (y_0 + punishment_0) <= (y_1 + punishment_1):
            
            edge_feature_model = edge_feature_tmp_0
        else: 
            edge_feature_model = edge_feature_tmp_1
    
    lift_index1 = edge_index[0,:24]
    lift_index2 = edge_index[1,:24]
    node_feature_label_lifted1 = node_feature_label[lift_index1]
    node_feature_label_lifted2 = node_feature_label[lift_index2]
    y = f(node_feature_label_lifted1, node_feature_label_lifted2, edge_feature_model)
    edge_num = sum(edge_feature_model)
    edge_f = torch.tensor(edge_feature_model)
    f_feed = torch.cat([edge_f, edge_f],0).reshape(-1,1).float()
    _, model_out_rounded = predictor(node_feature, edge_index, f_feed, batch)

    del node_feature
    del edge_feature
    del node_feature_label
    del edge_feature_direc1_tensor
    del model_out 
    del out_1
    del edge_feature_tmp_0
    del edge_feature_tmp_1
    del punishment_0
    del punishment_1
    del batch
    del edgef_feed0
    del edgef_feed1
    del y_0
    del y_1
    del node_feature_label_lifted1
    del node_feature_label_lifted2

    output_edge_feature_model = edge_feature_model.cpu().numpy()
    return y.item(), output_edge_feature_model, edge_num.item(), model_out_rounded.item()

    

def whether_match(edge_index, edge_feature):
    # return whether it's a perfect match
    node_flag = np.zeros(16)
    for i in range(24):
        if edge_feature[i]== 1:
            node_flag[edge_index[0,i]] = node_flag[edge_index[0,i]] + 1
            node_flag[edge_index[1,i]] = node_flag[edge_index[1,i]] + 1
    if sum(node_flag)==16:
        return 1
    else:
        return 0


def main():

    parser = argparse.ArgumentParser(description='this is the arg parser for synthetic dataset 2 with mnist')
    parser.add_argument('--proxy_path', dest = 'proxy_path',default = '../train_proxy/train_files/aff/train_aff_lr0005/best_val_model.pth')
    parser.add_argument('--model_path', dest = 'model_path',default = '../train_atheta/train_files/aff/train_a_aff/best_val_model.pth')
    parser.add_argument('--testset_path', dest = 'testset_path',default = '../build_dataset/testset_cover/')
    parser.add_argument('--save_paint_path', dest = 'save_paint_path',default = './paint/test_aff/')
    parser.add_argument('--gpu', dest = 'gpu',default = '7')
    args = parser.parse_args()


    device = torch.device('cuda:'+str(args.gpu))

    #load the model
    model = Aff_Mnist2(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
    model_state_dict = torch.load(args.model_path, map_location = torch.device("cpu"))

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    # load the proxy
    predictor = Aff_Mnist(Res_Block = ResBlock, in_channels = 16, hidden_channels = 32, out_channels = 128)
    predictor_state_dict =  torch.load(args.proxy_path, map_location = torch.device("cpu"))
    predictor.load_state_dict(predictor_state_dict)
    
    predictor.to(device)
    predictor.eval()

    # initialize the edge index
    edge_direc1 = torch.tensor([[0,0,1,1,2,2,3,4,4,5,5,6,6, 7, 8,8, 9, 9, 10,10,11,12,13,14],
                                [1,4,2,5,3,6,7,5,8,6,9,7,10,11,9,12,10,13,11,14,15,13,14,15]])
    edge_direc2_index = [1,0]                                      
    edge_direc2 = edge_direc1[edge_direc2_index]
    edge_index = torch.cat((edge_direc1, edge_direc2),1)
    edge_index = edge_index.to(device)
    batch = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    # load the dataset and test
    test_folder = args.testset_path

    avg_model_rounded = 0
    avg_score = 0
    gt_avg_score = 0
    avg_edge = 0
    
    num_match = 0
    linear_list = []
    gt_list = []
    max_list = []
    for index in tqdm(range(1,501)):
        test_path = test_folder+str(index)+'.pkl'
        test_file = open(test_path, 'rb')
        test_instance = pickle.load(test_file)
        node_feature = test_instance['node_feature']
        
        node_feature_label = test_instance['node_feature_label']
        node_feature_label = torch.tensor(node_feature_label)
        score = test_instance['score']
        this_best = 10000
        this_model_rounded = 0
        this_edge_num =0 
        

        for i in range(8):
            score_by_model, output_edge_feature_model, edge_num, model_out_rounded = test_an_instance(predictor,model, edge_index, node_feature, node_feature_label, device)
            if score_by_model < this_best:
                this_best = score_by_model
                this_model_rounded = model_out_rounded
                this_edge_num = edge_num
                this_edge_feature_model = output_edge_feature_model
            del model_out_rounded

        paint_edge(args.save_paint_path, index, node_feature, node_feature_label, this_edge_feature_model, this_best,  mode = 'AFF')
        
        print("instance "+str(index)+"num_edge: "+str(this_edge_num))
        print("instance "+str(index)+"this_score: "+str(this_best))
        print("instance "+str(index)+"gt_score: "+str(score))
        print("instance "+str(index)+"model roudned: "+str(this_model_rounded))

        avg_score = avg_score + this_best / 500
        gt_avg_score = gt_avg_score + score / 500
        avg_edge = avg_edge + this_edge_num / 500
        avg_model_rounded = avg_model_rounded + this_model_rounded / 500
       
        
    print("the overall average score is:"+str(avg_score))
    print("the overall average edge is:"+str(avg_edge))
    print("the overall ground truth average score is:"+str(gt_avg_score))
    print("the overall model out rounded is"+str(avg_model_rounded))



if __name__ == '__main__':
    main()