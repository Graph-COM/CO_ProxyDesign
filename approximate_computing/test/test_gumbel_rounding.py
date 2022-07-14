import pickle
import numpy as np
import sys
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
sys.path.append("..")
import torch
#from train_g.model import PNA_model, PNA_model2test
from train_g_naive.model import PNA_model, PNA_model2test

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
    #import pdb; pdb.set_trace()
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
    
def test_alpha_in_instance(threshold, alpha, operation, edge_index, optimizer, predictor):
    initial_assignment = torch.randint(0,2,(15,1))
    operation = torch.tensor(operation).reshape(-1,1)
    input_optimizer = torch.cat([operation, initial_assignment], 1).float()
    batch = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    alpha = torch.tensor(alpha).reshape(-1,1)
    optimizer_output,_ = optimizer(input_optimizer, alpha, edge_index, batch)
    rounded_node_feature = optimizer_output.detach().numpy()

    soft_feature = torch.cat([operation, optimizer_output.reshape(-1,1)],1)
    soft_score = predictor(soft_feature, edge_index, batch) + (threshold - sum(optimizer_output)).relu()
    #for index in range(15):
    for index in range(15):
        tmp_feature_0 = torch.clone(torch.tensor(rounded_node_feature).reshape(-1,1))
        tmp_feature_1 = torch.clone(torch.tensor(rounded_node_feature).reshape(-1,1))
        tmp_feature_0[index] = 0.0
        tmp_feature_1[index] = 1.0
        tmp_predict_input_0 = torch.cat([operation, tmp_feature_0],1)
        tmp_predict_input_1 = torch.cat([operation, tmp_feature_1],1)
        predict_0 = predictor(tmp_predict_input_0, edge_index, batch)
        predict_1 = predictor(tmp_predict_input_1, edge_index, batch) 
        if (predict_0 - alpha * sum(tmp_feature_0)) > (predict_1 - alpha * sum(tmp_feature_1)):
        #if (predict_0 - alpha * sum(tmp_feature_0)) > (predict_1 - alpha * sum(tmp_feature_1)):
            rounded_node_feature[index] = 1
        else:
            rounded_node_feature[index] = 0
    #import pdb; pdb.set_trace()
    tensor_node_feature = torch.tensor(rounded_node_feature).reshape(-1,1).float()
    predictor_input = torch.cat([operation, tensor_node_feature],1)
    predictor_output = predictor(predictor_input, edge_index, batch)
    return predictor_output, rounded_node_feature, soft_score

def test_with_initial(num_input, initial_input, operation, min_node_feature, edge_index):
    assignment_noerr = np.zeros(15)
    if num_input == 14:
        edge_index_pre = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,0,1],[14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21]])
        edge_index = edge_index + 14
        edge_index = torch.cat([edge_index_pre, edge_index],1)
    else:
        edge_index_pre = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]])
        edge_index = edge_index + 16
        edge_index = torch.cat([edge_index_pre, edge_index], 1)
    result_noerr = cal_result(initial_input, operation, assignment_noerr, edge_index, 100, 10)
    num_iteration = 500
    result_list = np.zeros((2000, num_iteration))
    for index_iteration in range(num_iteration):
        result = cal_result(initial_input, operation, min_node_feature, edge_index, 100, 10).reshape(-1)
        result_list[:,index_iteration] = result
    error_ = abs(result_list - result_noerr)
    relative_error = error_ / result_noerr
    avg_relative_error = np.mean(relative_error)
    return avg_relative_error

def test_an_instance(test_path, index, threshold, alpha_list, optimizer, predictor):
    test_path = test_path + str(index) + '.pkl'
    test_file = open(test_path, 'rb')
    test_data = pickle.load(test_file)
    operation = test_data['operation']
    edge_index = test_data['edge_index']
    score = test_data['score_'+str(threshold)]
    min_node_feature = np.zeros(15)
    min_score = 100
    chosen_alpha = 0
    min_soft_score = 0
    for alpha in alpha_list:
        score_alpha, node_feature_alpha, soft_score = test_alpha_in_instance(threshold, alpha, operation, edge_index, optimizer, predictor)
        if score_alpha < min_score:
            min_score = score_alpha
            chosen_alpha = alpha
            min_node_feature = node_feature_alpha
            min_soft_score = soft_score
    num_input = 14 + np.random.randint(0,2)*2
    initial_input = np.random.randint(1,100,(2000, num_input))/10
    linear_min_score = test_with_initial(num_input, initial_input, operation, min_node_feature, edge_index)
    back_node_feature = np.zeros(15)
    back_node_feature[15-threshold:] = 1
    back_score = test_with_initial(num_input, initial_input, operation, back_node_feature, edge_index)
    front_node_feature = np.zeros(15)
    front_random_index = np.random.randint(0,8,3)
    front_node_feature[front_random_index] = 1
    front_score = test_with_initial(num_input, initial_input, operation, back_node_feature, edge_index)
    print("instance "+str(index)+"threshold: "+str(threshold) + "all back is:" + str(back_score))
    print("instance "+str(index)+"threshold: "+str(threshold) + "all front is:" + str(front_score))
    print("instance "+str(index)+"threshold: "+str(threshold) + " linear error is: " +str(linear_min_score))
    print("instance "+str(index)+"threshold: "+str(threshold) + " gt error is: " +str(score))
    print("instance "+str(index)+"threshold: "+str(threshold) + " model out rounded is: " +str(min_score))
    print("instance "+str(index)+"threshold: "+str(threshold) + " model soft score is: " +str(min_soft_score))
    print("")
    return linear_min_score, back_score, front_score, score, min_score, min_soft_score

def main():
    
    #load the optimizer
    optimizer = PNA_model2test()
    optimizer_state_dict = torch.load('path ', map_location = torch.device('cpu'))
    optimizer.load_state_dict(optimizer_state_dict)
    optimizer.eval()

    #load the predictor
    predictor = PNA_model()
    predictor_state_dict = torch.load('path ', map_location = torch.device('cpu'))
    predictor.load_state_dict(predictor_state_dict)
    predictor.eval()


    #set how many threshold to test
    threshold_list = [3, 5, 8]

    #set how many alpha to test
    alpha_list_1 = np.arange(0.01,1,0.05)
    #alpha_list_2 = np.arange(1,10,0.5)
    alpha_list = alpha_list_1

    test_path = './test/testset/'
    avg_linear_error = [0, 0, 0]
    avg_gt_error = [0, 0, 0]
    avg_back_error = [0, 0, 0]
    avg_front_error = [0, 0, 0]
    avg_model_rounded = [0, 0, 0]
    avg_soft_score = [0, 0, 0]

    for index in range(1,201):
        num_t = 0
        for threshold in threshold_list:
            linear_error, back_error, front_error, gt_error, model_out_rounded, soft_score = test_an_instance(test_path, index, threshold, alpha_list, optimizer, predictor)
            avg_linear_error[num_t] = avg_linear_error[num_t] + linear_error / 200
            avg_back_error[num_t] = avg_back_error[num_t] + back_error / 200
            avg_front_error[num_t] = avg_front_error[num_t] + front_error / 200
            avg_gt_error[num_t] = avg_gt_error[num_t] + gt_error / 200
            avg_model_rounded[num_t] = avg_model_rounded[num_t] + model_out_rounded / 200
            avg_soft_score[num_t] = avg_soft_score[num_t] + soft_score / 200
            num_t = num_t + 1

    print('linear threshold = '+str(threshold_list[0]) + 'avg reltive error: '+str(avg_linear_error[0]))
    print('back threshold = '+str(threshold_list[0]) + 'avg reltive error: '+str(avg_back_error[0]))
    print('front threshold = '+str(threshold_list[0]) + 'avg reltive error: '+str(avg_front_error[0]))
    print('gt threshold = '+str(threshold_list[0]) + 'avg relative error: '+str(avg_gt_error[0]))
    print('model out rounded = '+str(threshold_list[0]) + 'avg model out rounded: '+str(avg_model_rounded[0].item()))
    print('soft score = '+str(threshold_list[0]) + 'avg soft score: '+str(avg_soft_score[0].item()))
    
    print('linear threshold = '+str(threshold_list[1]) + 'avg reltive error: '+str(avg_linear_error[1]))
    print('back threshold = '+str(threshold_list[1]) + 'avg reltive error: '+str(avg_back_error[1]))
    print('front threshold = '+str(threshold_list[1]) + 'avg reltive error: '+str(avg_front_error[1]))
    print('gt threshold = '+str(threshold_list[1]) + 'avg relative error: '+str(avg_gt_error[1]))
    print('model out rounded = '+str(threshold_list[1]) + 'avg model out rounded: '+str(avg_model_rounded[1].item()))
    print('soft score = '+str(threshold_list[1]) + 'avg soft score: '+str(avg_soft_score[1].item()))

    print('linear threshold = '+str(threshold_list[2]) + 'avg reltive error: '+str(avg_linear_error[2]))
    print('back threshold = '+str(threshold_list[2]) + 'avg reltive error: '+str(avg_back_error[2]))
    print('front threshold = '+str(threshold_list[2]) + 'avg reltive error: '+str(avg_front_error[2]))
    print('gt threshold = '+str(threshold_list[2]) + 'avg relative error: '+str(avg_gt_error[2]))
    print('model out rounded = '+str(threshold_list[2]) + 'avg model out rounded: '+str(avg_model_rounded[2].item()))
    print('soft score = '+str(threshold_list[2]) + 'avg soft score: '+str(avg_soft_score[2].item()))
    

if __name__ == '__main__':
    main()