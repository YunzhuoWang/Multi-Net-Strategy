# File      :PIGAN_AllenCahn.py
# Time      :2021/5/21--20:32
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import scipy.io
import torch
from PIGAN_AllenCahn_model import AllenCahn


def main():
    # 读入测试集
    np.random.seed(42)
    test = scipy.io.loadmat('./Data/AllenCahn_exact.mat')
    test_point_t = test['t']
    test_point_xy = test['coordinate']
    test_point_x = test_point_xy[:, :, 0]
    test_point_y = test_point_xy[:, :, 1]
    test_real_value = test['z']

    test_point_x, test_point_t1 = np.meshgrid(test_point_x, test_point_t)
    test_point_y, test_point_t = np.meshgrid(test_point_y, test_point_t)
    del test_point_t1
    test_point_x = test_point_x.flatten()[:, None]
    test_point_y = test_point_y.flatten()[:, None]
    test_point_t = test_point_t.flatten()[:, None]
    test_real_value = test_real_value.flatten()[:, None]

    test_point_t = torch.tensor(test_point_t, dtype = torch.float32)
    test_point_x = torch.tensor(test_point_x, dtype = torch.float32)
    test_point_y = torch.tensor(test_point_y, dtype = torch.float32)
    test_real_value = torch.tensor(test_real_value, dtype = torch.float32)
    test_point = torch.cat((test_point_t, test_point_x, test_point_y), dim = 1)

    # 读入训练集
    training_Nf = scipy.io.loadmat('./Data/AllenCahn_training_Nf.mat')
    Nf_coordinate = training_Nf['coordinate']
    Nf_dataset = torch.tensor(Nf_coordinate, dtype = torch.float32)

    # 读入边界条件
    training_Nu = scipy.io.loadmat('./Data/AllenCahn_training_Nu.mat')
    Nu_coordiante = training_Nu['coordinate']
    Nu_value = training_Nu['value']
    # 噪声处理
    noise = 0.1
    Nu_value = Nu_value + noise * np.std(Nu_value) * np.random.randn(Nu_value.shape[0], Nu_value.shape[1])
    Nu_dataset = np.hstack((Nu_coordiante, Nu_value))
    Nu_dataset = torch.tensor(Nu_dataset, dtype = torch.float32)

    # 调用模型
    run_times = 1
    MSE, RMSE, MAE = 0, 0, 0
    train_time, test_time = 0, 0
    for i in range(run_times):
        PIGAN_error, PIGAN_time = AllenCahn(Nf_dataset, Nu_dataset, test_point, test_real_value)
        MSE += PIGAN_error['MSE']
        RMSE += PIGAN_error['RMSE']
        MAE += PIGAN_error['MAE']
        train_time += PIGAN_time['train time']
        test_time += PIGAN_time['test time']
        print({'PIGAN_error': PIGAN_error, 'PIGAN_time': PIGAN_time})
    # 写入文件
    with open('./Results/results.txt', 'w') as f:
        f.write('MSE:{}\nRMSE:{}\nMAE:{}\ntraining time:{}\ntesting time:{}\n'.format(MSE / run_times, RMSE / run_times,
                                                                                      MAE / run_times,
                                                                                      train_time / run_times,
                                                                                      test_time / run_times))


if __name__ == '__main__':
    main()
