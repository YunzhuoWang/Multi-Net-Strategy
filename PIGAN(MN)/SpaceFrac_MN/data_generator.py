# File      :data_generator.py
# Time      :2021/5/17--12:57
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import scipy.io


def main():
    # 重构测试集
    training_test = scipy.io.loadmat('./Data/SpaceFractional_exact1.mat')
    test_t = training_test['t']
    test_coordinate = training_test['coordinate']
    test_z = training_test['z']
    test_t = test_t[:100, :]
    test_z = test_z[:100, :]
    scipy.io.savemat('./Data/SpaceFractional_exact1.mat',
                     {'t': test_t, 'coordinate': test_coordinate, 'z': test_z})


if __name__ == '__main__':
    main()
