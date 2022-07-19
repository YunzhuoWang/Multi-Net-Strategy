# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import scipy.io as sio
def sech (x):
    return 1/np.cosh(x)
def main():
    downboundary, upboundary = -10, 10
    #构造t的测试机
    t_set = np.linspace(0, 0.99, 100).reshape(100,1)
    #构造coordinate的测试集
    x_set = np.linspace(downboundary, upboundary, 256)
    coordinate = x_set.reshape(256,1)
    #构造z的测试集
    a,h,g=1,6,9.8
    z_set = np.empty([0,256])
    for t in t_set:
        z = [np.power(sech(0.5*np.power(3/np.power(h,3),1/2)*(x_set-np.ones_like(x_set)*np.power(g*h,1/2)*(1+1/(2*h))*t)),2)]
        z_set = np.concatenate((z_set, z), axis=0)
    # z_set=z_set.transpose((1,2,0))
    sio.savemat('./data/Kdv1d_exact.mat', {'t': t_set, 'coordinate': coordinate, 'z': z_set})
    # z = np.exp(-(np.power(x,2)+np.power(y,2))/0.02)
    # dataframe = pd.DataFrame({'x': x, 'y': y,'z': z})
    # dataframe.to_csv("./data./AdvectionDispersion2d_init.csv", index=False,header=False, sep=',')
    # a = np.empty([0, 3])
    # b = np.array([[1, 2, 3]])
    # c = [[7, 8, 9]]
    # a = np.append(a, b, axis=0)
    #
    # a = np.append(a, c, axis=0)
    # O=1

if __name__=='__main__':
    main()