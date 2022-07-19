# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as sio
def sech (x):
    return 1/np.cosh(x)
def main():
    np.random.seed(42)
    downboundary_x, upboundary_x = -10, 10
    downboundary_t, upboundary_t = 0, 0.99
    #构造Nf训练集
    Nf = 896
    #构造t的测试机
    Nf_t_set = downboundary_t+(upboundary_t-downboundary_t)*np.random.rand(Nf).reshape(Nf,1)
    #构造coordinate的测试集
    Nf_x_set = downboundary_x+(upboundary_x-downboundary_x)*np.random.rand(Nf).reshape(Nf,1)
    Nf_coordinate = np.hstack((Nf_t_set,Nf_x_set))

    sio.savemat('./data/Kdv_training_Nf.mat', {'coordinate': Nf_coordinate})
    #构造Nu训练集
    Nu = 128
    #初值条件
    x_set = downboundary_x+(upboundary_x-downboundary_x)*np.random.rand(2*Nu).reshape(2*Nu,1)
    t_set = np.zeros(2*Nu).reshape(2*Nu,1)
    a,h,g=1,6,9.8
    z_set = a*np.power(sech(0.5*np.power(3*a/np.power(h,3),1/2)*x_set),2)
    init_Nu_coordinat = np.hstack((t_set,x_set,z_set))

    Nu_coordinate = init_Nu_coordinat

    idx = np.random.choice(Nu_coordinate.shape[0], Nu, replace=False)
    Nu_coordinate = Nu_coordinate[idx, :]

    # z_set=z_set.transpose((1,2,0))
    sio.savemat('./data/Kdv_training_Nu.mat', {'coordinate': Nu_coordinate[:,0:2],'value':Nu_coordinate[:,-1].reshape(Nu,1)})

if __name__=='__main__':
    main()