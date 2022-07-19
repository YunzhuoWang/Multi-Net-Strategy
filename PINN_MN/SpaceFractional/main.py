# -*- coding: UTF-8 -*-
import numpy as np
import time
import torch
import scipy.io
import csv
from SpaceFractional_PINN import SpaceFractional_PINN

import pandas as pd
def main():
    #读入测试集

    np.random.seed(42)
    text = scipy.io.loadmat('./Data/SpaceFractional_exact.mat')
    _text_x = text['coordinate']
    _text_y = text['z']
    _text_t = text['t']

    text_x = np.array([ [_text_x[i] for i in range(_text_x.shape[0])]  for k in range(_text_t.shape[0])])
    text_t = np.array([ [k for i in range(_text_x.shape[0])] for k in _text_t])
    text_y = np.array ([[_text_y[k][i] for i in range(_text_x.shape[0])] for k in range(_text_t.shape[0])])

    N_text = 256  # N_text at each moment
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
    training_Nf = scipy.io.loadmat('./Data/SpaceFractional_training_Nf.mat')
    Nf_coordinate =  training_Nf['coordinate']
    Nf_coordinate = torch.FloatTensor(Nf_coordinate)
    Nf_dataset = torch.tensor(Nf_coordinate, dtype=torch.float32)

    training_Nu = scipy.io.loadmat('./Data/SpaceFractional_training_Nu.mat')
    Nu_coordinate =  training_Nu['coordinate']
    Nu_value =  training_Nu['value']
    # 噪声处理
    np.random.seed(42)
    noise = 0.10
    Nu_value = Nu_value + noise * np.std(Nu_value) * np.random.randn(Nu_value.shape[0], Nu_value.shape[1])
    Nu_dataset = np.hstack((Nu_coordinate,Nu_value))
    Nu_dataset = torch.FloatTensor(Nu_dataset)
    Nu_dataset = torch.tensor(Nu_dataset, dtype=torch.float32)
    run_times=1
    _SpaceFractional_error_evermoment=[[i*0.01,0.0] for i in range(100)]
    _Inference_time=0
    _SpaceFractional_error=0
    _Training_time=0
    _value_4_exact = [[text_x[40][i][0], text_y[40][i]] for i in range(N_text)]
    _value_8_exact = [[text_x[80][i][0], text_y[80][i]] for i in range(N_text)]
    _value_4_approximate = [[text_x[40][i][0], 0.0] for i in range(N_text)]
    _value_8_approximate = [[text_x[80][i][0], 0.0] for i in range(N_text)]
    for i in range(run_times):
        SpaceFractional_start_time = time.time()
        SpaceFractional_error_evermoment, SpaceFractional_error,Inference_time, value_4, value_8= SpaceFractional_PINN(Nf_dataset, Nu_dataset,text_x,text_t,text_y)
        SpaceFractional_time = time.time()-SpaceFractional_start_time
        print({'SpaceFractional_time ': SpaceFractional_time , 'SpaceFractional_error':SpaceFractional_error})
        value_4 = value_4.squeeze()
        value_8 = value_8.squeeze()
        # for i in range(256):
        #     _value_4_approximate[i][1] = _value_4_approximate[i][1]+value_4[i]
        # _value_4_approximate = [[_value_4_approximate[i][0], _value_4_approximate[i][1] + value_4[i]] for i in range(N_text)]
        # _value_8_approximate = [[_value_8_approximate[i][0], _value_8_approximate[i][1] + value_8[i]] for i in range(N_text)]
        _SpaceFractional_error_evermoment = [[_SpaceFractional_error_evermoment[i][0],_SpaceFractional_error_evermoment[i][1]+SpaceFractional_error_evermoment[i][1]] for i in range(100)]
        _Inference_time += Inference_time
        _SpaceFractional_error += SpaceFractional_error
        _Training_time += SpaceFractional_time-Inference_time
    _SpaceFractional_error_evermoment = [[_SpaceFractional_error_evermoment[i][0], _SpaceFractional_error_evermoment[i][1]/run_times] for i in range(100)]
    # _value_4_approximate = [[_value_4_approximate[i][0], _value_4_approximate[i][1]/run_times] for i in range(N_text)]
    # _value_8_approximate = [[_value_8_approximate[i][0], _value_8_approximate[i][1] / run_times] for i in range(N_text)]


    dataframe = pd.DataFrame({'t':[_SpaceFractional_error_evermoment[i][1]for i in range(100)], 'loss': [_SpaceFractional_error_evermoment[i][0] for i in range(100)]})
    dataframe.to_csv("./Results/PINN_MN_SpaceFractiona_noise10_alpha18_evermoment_loss.csv", index=False,header=False, sep=',')

    # dataframe = pd.DataFrame({'x': [_value_4_approximate[i][0] for i in range(N_text)], 'value': [_value_4_approximate[i][1] for i in range(N_text)]})
    # dataframe.to_csv("./Results/PINN_MN_SpaceFractional_noise10_alpha18_value_4_approximate.csv", index=False, header=False, sep=',')
    #
    # dataframe = pd.DataFrame({'x': [_value_8_approximate[i][0] for i in range(N_text)], 'loss': [_value_8_approximate[i][1] for i in range(N_text)]})
    # dataframe.to_csv("./Results/PINN_MN_SpaceFractional_noise10_alpha18_value_8_approximate.csv", index=False, header=False, sep=',')
    #
    # dataframe = pd.DataFrame({'x': [_value_4_exact[i][0] for i in range(N_text)], 'value': [_value_4_exact[i][1] for i in range(N_text)]})
    # dataframe.to_csv("./Results/PINN_MN_SpaceFractional_noise10_alpha18_value_4_exact.csv", index=False, header=False, sep=',')
    #
    # dataframe = pd.DataFrame({'x': [_value_8_exact[i][0] for i in range(N_text)], 'loss': [_value_8_exact[i][1] for i in range(N_text)]})
    # dataframe.to_csv("./Results/PINN_MN_SpaceFractional_noise10_alpha18_value_8_exact.csv", index=False, header=False, sep=',')

    with open('./Results/PINN_MN_SpaceFractional_loss_noise10_alpha18.txt', 'w') as f:  # 设置文件对象
        f.write("loss:{},'Training_time:{},'Inference_time:'{}".format(_SpaceFractional_error/run_times,_Training_time/run_times,_Inference_time/run_times))
        # print({'Burgers1d_time ': Burgers1d_time , 'Burgers1d_error':Burgers1d_error})

if __name__=='__main__':
    main()