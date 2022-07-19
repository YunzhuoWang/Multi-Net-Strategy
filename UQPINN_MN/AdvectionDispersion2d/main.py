# -*- coding: UTF-8 -*-
import numpy as np
import time
import torch
import scipy.io
import csv
from AdvectionDispersion2d_UQPINN import AdvectionDispersion2d_UQPINN

import pandas as pd
def main():
    #读入测试集
    np.random.seed(42)
    text = scipy.io.loadmat('./Data/AdvectionDispersion_exact.mat')
    _text_x = text['coordinate']
    _text_y = text['z']
    _text_t = text['t']

    text_x = np.array([ [_text_x[i][j] for i in range(_text_x.shape[0]) for j in range(_text_x.shape[1])] for k in range(_text_t.shape[0])])
    text_t = np.array([ [k for i in range(_text_x.shape[0]) for j in range(_text_x.shape[1])] for k in _text_t])
    text_y = np.array([ [_text_y[k][i][j] for i in range(_text_x.shape[0]) for j in range(_text_x.shape[1])] for k in range(_text_t.shape[0])])

    N_text = 100
    idx = np.random.choice(text_x.shape[1], N_text, replace=False)

    text_x=text_x[:,idx]
    text_x = torch.FloatTensor(text_x)
    text_x = torch.tensor(text_x, dtype=torch.float32)

    text_y=text_y[:,idx]
    text_y = torch.FloatTensor(text_y)
    text_y = torch.tensor(text_y, dtype=torch.float32)

    text_t=text_t[:,idx]
    text_t = torch.FloatTensor(text_t)
    text_t = torch.tensor(text_t, dtype=torch.float32)

    #读入训练集
    N_u = 3000
    N_f = 30000
    training_Nf = scipy.io.loadmat('./Data/AdvectionDispersion_training_Nf.mat')
    Nf_coordinate =  training_Nf['coordinate']
    idx = np.random.choice(Nf_coordinate.shape[0], N_f, replace=False)
    Nf_coordinate = Nf_coordinate[idx, :]
    Nf_coordinate = torch.FloatTensor(Nf_coordinate)
    Nf_dataset = torch.tensor(Nf_coordinate, dtype=torch.float32)

    training_Nu = scipy.io.loadmat('./Data/AdvectionDispersion_training_Nu.mat')
    Nu_coordinate =  training_Nu['coordinate']
    Nu_value =  training_Nu['value']
    # 噪声处理
    np.random.seed(42)
    noise = 0.10
    Nu_value = Nu_value + noise * np.std(Nu_value) * np.random.randn(Nu_value.shape[0], Nu_value.shape[1])
    Nu_dataset = np.hstack((Nu_coordinate,Nu_value))
    idx = np.random.choice(Nu_dataset.shape[0], N_u, replace=False)
    Nu_dataset = Nu_dataset[idx, :]
    Nu_dataset = torch.FloatTensor(Nu_dataset)
    Nu_dataset = torch.tensor(Nu_dataset, dtype=torch.float32)
    run_times=1
    _AdvectionDispersion2d_error_evermoment=[[i*0.01,0.0] for i in range(100)]
    _Inference_time=0
    _AdvectionDispersion2d_error=0
    _Training_time=0
    _value_4_exact = [[text_x[40][i][0],text_x[40][i][1], text_y[40][i]] for i in range(N_text)]
    _value_8_exact = [[text_x[80][i][0],text_x[80][i][1], text_y[80][i]] for i in range(N_text)]
    _value_4_approximate = [[text_x[40][i][0],text_x[40][i][1], 0.0] for i in range(N_text)]
    _value_8_approximate = [[text_x[80][i][0],text_x[80][i][1], 0.0] for i in range(N_text)]
    for i in range(run_times):
        AdvectionDispersion2d_start_time = time.time()
        AdvectionDispersion2d_error_evermoment, AdvectionDispersion2d_error,Inference_time, value_4, value_8= AdvectionDispersion2d_UQPINN(Nf_dataset, Nu_dataset,text_x,text_t,text_y)
        AdvectionDispersion2d_time = time.time()-AdvectionDispersion2d_start_time
        value_4 = value_4.squeeze()
        value_8 = value_8.squeeze()
        _value_4_approximate = [[_value_4_approximate[i][0],_value_4_approximate[i][1], _value_4_approximate[i][2] + value_4[i]] for i in range(N_text)]
        _value_8_approximate = [[_value_8_approximate[i][0],_value_8_approximate[i][1] ,_value_8_approximate[i][2] + value_8[i]] for i in range(N_text)]
        _AdvectionDispersion2d_error_evermoment = [[_AdvectionDispersion2d_error_evermoment[i][0],_AdvectionDispersion2d_error_evermoment[i][1]+AdvectionDispersion2d_error_evermoment[i][1]] for i in range(100)]
        _Inference_time += Inference_time
        _AdvectionDispersion2d_error += AdvectionDispersion2d_error
        _Training_time += AdvectionDispersion2d_time-Inference_time
        print({'AdvectionDispersion2d_time ': AdvectionDispersion2d_time , 'AdvectionDispersion2d_error':AdvectionDispersion2d_error})
    _AdvectionDispersion2d_error_evermoment = [[_AdvectionDispersion2d_error_evermoment[i][0], _AdvectionDispersion2d_error_evermoment[i][1]/run_times] for i in range(100)]
    _value_4_approximate = [[_value_4_approximate[i][0],_value_4_approximate[i][1], _value_4_approximate[i][2]/run_times] for i in range(N_text)]
    _value_8_approximate = [[_value_8_approximate[i][0],_value_8_approximate[i][1] ,_value_8_approximate[i][2]/run_times] for i in range(N_text)]

    dataframe = pd.DataFrame({'t':[_AdvectionDispersion2d_error_evermoment[i][1]for i in range(100)], 'loss': [_AdvectionDispersion2d_error_evermoment[i][0] for i in range(100)]})
    dataframe.to_csv("./Results/UQPINN_MN_AdvectionDispersion2d_noise10_evermoment_loss.csv", index=False,header=False, sep=',')

    # dataframe = pd.DataFrame({'x': [[_value_4_approximate[i][0],_value_4_approximate[i][1]] for i in range(N_text)], 'value': [_value_4_approximate[i][2] for i in range(N_text)]})
    # dataframe.to_csv("./Results/UQPINN_MN_AdvectionDispersion2d_noise10_value_4_approximate.csv", index=False, header=False, sep=',')
    #
    # dataframe = pd.DataFrame({'x': [[_value_8_approximate[i][0],_value_8_approximate[i][1]] for i in range(N_text)], 'loss': [_value_8_approximate[i][2] for i in range(N_text)]})
    # dataframe.to_csv("./Results/UQPINN_MN_AdvectionDispersion2d_noise10_value_8_approximate.csv", index=False, header=False, sep=',')
    #
    # dataframe = pd.DataFrame({'x':  [[_value_4_exact[i][0],_value_4_exact[i][1]] for i in range(N_text)], 'value': [_value_4_exact[i][2] for i in range(N_text)]})
    # dataframe.to_csv("./Results/UQPINN_MN_AdvectionDispersion2d_noise10_value_4_exact.csv", index=False, header=False, sep=',')
    #
    # dataframe = pd.DataFrame({'x': [[_value_8_exact[i][0],_value_8_exact[i][1]] for i in range(N_text)], 'loss': [_value_8_exact[i][2] for i in range(N_text)]})
    # dataframe.to_csv("./Results/UQPINN_MN_AdvectionDispersion2d_noise10_value_8_exact.csv", index=False, header=False, sep=',')

    with open('./Results/UQPINN_MN_AdvectionDispersion2d_loss_noise10.txt', 'w') as f:  # 设置文件对象
        f.write("loss:{},'Training_time:{},'Inference_time:'{}".format(_AdvectionDispersion2d_error/run_times,_Training_time/run_times,_Inference_time/run_times))
        # print({'AdvectionDispersion2d_time ': AdvectionDispersion2d_time , 'AdvectionDispersion2d_error':AdvectionDispersion2d_error})

if __name__=='__main__':
    main()