# File      :data_generator.py
# Time      :2021/5/17--12:57
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import scipy.io
from pyDOE import lhs


def main():
    np.random.seed(42)
    lb_x, ub_x = -1, 1
    lb_y, ub_y = 0, 0  # 不包含维度y
    lb_t, ub_t = 0, 0.99
    Nf = 10000
    Nu = 200
    Ntest = 0  # 不构造测试集

    # 构造训练集
    lb, ub = np.array([lb_t, lb_x]), np.array([ub_t, ub_x])
    Nf_coordinate = lb + (ub - lb) * lhs(2, Nf)
    scipy.io.savemat('./Data/Burgers_training_Nf.mat', {'coordinate': Nf_coordinate})

    # 构造边界集
    # 上边界
    x = np.ones(Nu).reshape(Nu, 1)
    t = lb_t + (ub_t - lb_t) * lhs(1, Nu)
    value = np.zeros(Nu).reshape(Nu, 1)
    boundary_up = np.hstack((t, x, value))
    # 下边界
    x = np.array([-1 for i in range(Nu)]).reshape(Nu, 1)
    t = lb_t + (ub_t - lb_t) * lhs(1, Nu)
    value = np.zeros(Nu).reshape(Nu, 1)
    boundary_down = np.hstack((t, x, value))
    # 初始时间边界
    t = np.zeros(Nu).reshape(Nu, 1)
    x = lb_x + (ub_x - lb_x) * lhs(1, Nu)
    value = -np.sin(np.pi * x)
    boundary_time = np.hstack((t, x, value))
    # 合并边界集
    Nu_coordinate = np.vstack((boundary_up, boundary_down, boundary_time))
    Nu_idx = np.random.choice(Nu_coordinate.shape[0], Nu, replace=False)
    Nu_coordinate = Nu_coordinate[Nu_idx, :]
    scipy.io.savemat('./Data/Burgers_training_Nu.mat',
                     {'coordinate': Nu_coordinate[:, 0:2], 'value': Nu_coordinate[:, -1].reshape(Nu, 1)})


if __name__ == '__main__':
    main()
