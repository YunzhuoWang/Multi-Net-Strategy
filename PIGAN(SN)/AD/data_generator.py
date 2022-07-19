# File      :data_generator.py
# Time      :2021/5/17--12:57
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import scipy.io


def main():
    np.random.seed(42)
    Nf = 30000
    Nu = 3000
    Nt = 100

    # 重构训练集
    training_Nf = scipy.io.loadmat('./Data/AdvectionDispersion_training_Nf1.mat')
    Nf_coordinate = training_Nf['coordinate']
    Nf_idx = np.random.choice(Nf_coordinate.shape[0], Nf, replace = False)
    Nf_point = Nf_coordinate[Nf_idx, :]
    scipy.io.savemat('./Data/AdvectionDispersion_training_Nf.mat', {'coordinate': Nf_point})

    # 重构边界集
    training_Nu = scipy.io.loadmat('./Data/AdvectionDispersion_training_Nu1.mat')
    Nu_coordinate = training_Nu['coordinate']
    Nu_value = training_Nu['value']
    Nu_idx = np.random.choice(Nu_coordinate.shape[0], Nu, replace = False)
    Nu_point = Nu_coordinate[Nu_idx, :]
    Nu_point_value = Nu_value[Nu_idx, :]
    scipy.io.savemat('./Data/AdvectionDispersion_training_Nu.mat', {'coordinate': Nu_point, 'value': Nu_point_value})

    # 重构测试集
    training_test = scipy.io.loadmat('./Data/AdvectionDispersion_exact1.mat')
    test_t = training_test['t']
    test_coordinate = training_test['coordinate']
    test_z = training_test['z']
    Nt_idx_x = np.random.choice(test_coordinate.shape[0], Nt, replace = False)
    Nt_idx_y = np.random.choice(test_coordinate.shape[1], Nt, replace = False)
    test_point = test_coordinate[Nt_idx_x, :, :]
    test_point = test_point[:, Nt_idx_y, :]
    test_point_value = test_z[:, Nt_idx_x, :]
    test_point_value = test_point_value[:, :, Nt_idx_y]
    scipy.io.savemat('./Data/AdvectionDispersion_exact.mat',
                     {'t': test_t, 'coordinate': test_point, 'z': test_point_value})


if __name__ == '__main__':
    main()
