# -*- coding: UTF-8 -*-
import numpy as np
import time
import torch
import scipy.io
import csv
from Kdv1d_UQPINN import Kdv1d_UQPINN

import pandas as pd
def main():
    #读入测试集
    np.random.seed(42)
    text = scipy.io.loadmat('./data/Kdv1d_exact.mat')
    text_x = text['coordinate']
    text_y = text['z']
    text_t = text['t']

    text_x,text_t = np.meshgrid(text_x,text_t)
    text_x = text_x.flatten()[:,None]
    text_t = text_t.flatten()[:None]
    text_y = text_y.T
    text_y = text_y.flatten()[:None]

    text_x = torch.FloatTensor(text_x)
    text_x = torch.tensor(text_x, dtype=torch.float32)

    text_y = torch.FloatTensor(text_y)
    text_y = torch.tensor(text_y, dtype=torch.float32)

    text_t = torch.FloatTensor(text_t)
    text_t = torch.tensor(text_t, dtype=torch.float32)

    #读入训练集
    training_Nf = scipy.io.loadmat('./data/Kdv_training_Nf.mat')
    Nf_coordinate =  training_Nf['coordinate']
    Nf_coordinate = torch.FloatTensor(Nf_coordinate)
    Nf_dataset = torch.tensor(Nf_coordinate, dtype=torch.float32)

    training_Nu = scipy.io.loadmat('./data/Kdv_training_Nu.mat')
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
    _Kdv1d_error_evermoment=[[i*0.01,0.0] for i in range(100)]
    _Inference_time=0
    _Kdv1d_error=0
    _Training_time=0
    _value_4_exact = [[text_x[40 * 256 + i], text_y[40 * 256 + i]] for i in range(256)]
    _value_8_exact = [[text_x[80 * 256 + i], text_y[80 * 256 + i]] for i in range(256)]
    _value_4_approximate = [[text_x[40 * 256 + i], 0.0] for i in range(256)]
    _value_8_approximate = [[text_x[80 * 256 + i], 0.0] for i in range(256)]
    for i in range(run_times):
        Kdv1d_start_time = time.time()
        Kdv1d_error_evermoment, Kdv1d_error,Inference_time, value_4, value_8= Kdv1d_UQPINN(Nf_dataset, Nu_dataset,text_x,text_t,text_y)
        Kdv1d_time = time.time()-Kdv1d_start_time
        value_4 = value_4.squeeze()
        value_8 = value_8.squeeze()
        # for i in range(256):
        #     _value_4_approximate[i][1] = _value_4_approximate[i][1]+value_4[i]
        _value_4_approximate = [[_value_4_approximate[i][0], _value_4_approximate[i][1] + value_4[i]] for i in range(256)]
        _value_8_approximate = [[_value_8_approximate[i][0], _value_8_approximate[i][1] + value_8[i]] for i in range(256)]
        _Kdv1d_error_evermoment = [[_Kdv1d_error_evermoment[i][0],_Kdv1d_error_evermoment[i][1]+Kdv1d_error_evermoment[i][1]] for i in range(100)]
        _Inference_time += Inference_time
        _Kdv1d_error += Kdv1d_error
        _Training_time += Kdv1d_time-Inference_time
    _Kdv1d_error_evermoment = [[_Kdv1d_error_evermoment[i][0], _Kdv1d_error_evermoment[i][1]/run_times] for i in range(100)]
    _value_4_approximate = [[_value_4_approximate[i][0], _value_4_approximate[i][1]/run_times] for i in range(256)]
    _value_8_approximate = [[_value_8_approximate[i][0], _value_8_approximate[i][1] / run_times] for i in range(256)]


    dataframe = pd.DataFrame({'t':[_Kdv1d_error_evermoment[i][1]for i in range(100)], 'loss': [_Kdv1d_error_evermoment[i][0] for i in range(100)]})
    dataframe.to_csv("./Results/UQPINN_SN_Kdv1d_noise10_evermoment_loss.csv", index=False,header=False, sep=',')

    # dataframe = pd.DataFrame({'x': [_value_4_approximate[i][0] for i in range(256)], 'value': [_value_4_approximate[i][1] for i in range(256)]})
    # dataframe.to_csv("./Results/UQPINN_SN_Kdv1d_noise10_value_4_approximate.csv", index=False, header=False, sep=',')
    #
    # dataframe = pd.DataFrame({'x': [_value_8_approximate[i][0] for i in range(256)], 'loss': [_value_8_approximate[i][1] for i in range(256)]})
    # dataframe.to_csv("./Results/UQPINN_SN_Kdv1d_noise10_value_8_approximate.csv", index=False, header=False, sep=',')
    #
    # dataframe = pd.DataFrame({'x': [_value_4_exact[i][0] for i in range(256)], 'value': [_value_4_exact[i][1] for i in range(256)]})
    # dataframe.to_csv("./Results/UQPINN_SN_Kdv1d_noise10_value_4_exact.csv", index=False, header=False, sep=',')
    #
    # dataframe = pd.DataFrame({'x': [_value_8_exact[i][0] for i in range(256)], 'loss': [_value_8_exact[i][1] for i in range(256)]})
    # dataframe.to_csv("./Results/UQPINN_SN_Kdv1d_noise10_value_8_exact.csv", index=False, header=False, sep=',')

    with open('./Results/UQPINN_SN_Kdv1d_loss_noise10.txt', 'w') as f:  # 设置文件对象
        f.write("loss:{},'Training_time:{},'Inference_time:'{}".format(_Kdv1d_error/run_times,_Training_time/run_times,_Inference_time/run_times))
        # print({'Kdv1d_time ': Kdv1d_time , 'Kdv1d_error':Kdv1d_error})

if __name__=='__main__':
    main()