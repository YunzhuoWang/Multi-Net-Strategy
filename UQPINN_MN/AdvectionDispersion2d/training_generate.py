# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as sio
def main():
    np.random.seed(42)
    downboundary_x, upboundary_x = 0, 1
    downboundary_y, upboundary_y = 0, 1
    downboundary_t, upboundary_t = 0, 1
    #构造Nf训练集
    Nf = 91750400
    #构造t的测试机
    Nf_t_set = downboundary_t+(upboundary_t-downboundary_t)*np.random.rand(Nf).reshape(Nf,1)
    #构造coordinate的测试集
    Nf_x_set = downboundary_x+(upboundary_x-downboundary_x)*np.random.rand(Nf).reshape(Nf,1)
    Nf_y_set = downboundary_y+(upboundary_y-downboundary_y)*np.random.rand(Nf).reshape(Nf,1)
    Nf_coordinate = np.hstack((Nf_t_set,Nf_x_set,Nf_y_set))

    sio.savemat('./Data/AdvectionDispersion_training_Nf.mat', {'coordinate': Nf_coordinate})
    #构造Nu训练集
    Nu = 13107200
    #边界1
    x_set = np.zeros(2*Nu).reshape(2*Nu,1)
    y_set = np.zeros(2*Nu).reshape(2*Nu,1)
    t_set = downboundary_t + (upboundary_t - downboundary_t) * np.random.rand(2*Nu).reshape(2*Nu, 1)
    z_set = (1/(4*t_set+1))*np.exp(-(np.power(x_set-np.cos(np.pi/8)*t_set,2)+np.power(y_set-np.sin(np.pi/8)*t_set,2))/(0.02*(4*t_set+1)))
    boundary1_Nu_coordinat = np.hstack((t_set,x_set,y_set,z_set))
    #边界2
    x_set = np.ones(2*Nu).reshape(2*Nu,1)
    y_set = np.zeros(2*Nu).reshape(2*Nu,1)
    t_set = downboundary_t + (upboundary_t - downboundary_t) * np.random.rand(2*Nu).reshape(2*Nu, 1)
    z_set = (1/(4*t_set+1))*np.exp(-(np.power(x_set-np.cos(np.pi/8)*t_set,2)+np.power(y_set-np.sin(np.pi/8)*t_set,2))/(0.02*(4*t_set+1)))
    boundary2_Nu_coordinat = np.hstack((t_set,x_set,y_set,z_set))
    #边界3
    x_set = np.zeros(2*Nu).reshape(2*Nu,1)
    y_set = np.ones(2*Nu).reshape(2*Nu,1)
    t_set = downboundary_t + (upboundary_t - downboundary_t) * np.random.rand(2*Nu).reshape(2*Nu, 1)
    z_set = (1/(4*t_set+1))*np.exp(-(np.power(x_set-np.cos(np.pi/8)*t_set,2)+np.power(y_set-np.sin(np.pi/8)*t_set,2))/(0.02*(4*t_set+1)))
    boundary3_Nu_coordinat = np.hstack((t_set,x_set,y_set,z_set))
    #边界4
    x_set = np.ones(2*Nu).reshape(2*Nu,1)
    y_set = np.ones(2*Nu).reshape(2*Nu,1)
    t_set = downboundary_t + (upboundary_t - downboundary_t) * np.random.rand(2*Nu).reshape(2*Nu, 1)
    z_set = (1/(4*t_set+1))*np.exp(-(np.power(x_set-np.cos(np.pi/8)*t_set,2)+np.power(y_set-np.sin(np.pi/8)*t_set,2))/(0.02*(4*t_set+1)))
    boundary4_Nu_coordinat = np.hstack((t_set,x_set,y_set,z_set))
    #初值条件
    x_set = downboundary_x+(upboundary_x-downboundary_x)*np.random.rand(2*Nu).reshape(2*Nu,1)
    y_set = downboundary_y+(upboundary_y-downboundary_y)*np.random.rand(2*Nu).reshape(2*Nu,1)
    t_set = np.zeros(2*Nu).reshape(2*Nu,1)
    z_set = (1/(4*t_set+1))*np.exp(-(np.power(x_set-np.cos(np.pi/8)*t_set,2)+np.power(y_set-np.sin(np.pi/8)*t_set,2))/(0.02*(4*t_set+1)))
    init_Nu_coordinat = np.hstack((t_set,x_set,y_set,z_set))

    Nu_coordinate = np.vstack((boundary1_Nu_coordinat,boundary2_Nu_coordinat,boundary3_Nu_coordinat,boundary4_Nu_coordinat,init_Nu_coordinat))

    idx = np.random.choice(Nu_coordinate.shape[0], Nu, replace=False)
    Nu_coordinate = Nu_coordinate[idx, :]

    # z_set=z_set.transpose((1,2,0))
    sio.savemat('./Data/AdvectionDispersion_training_Nu.mat', {'coordinate': Nu_coordinate[:,0:3],'value':Nu_coordinate[:,-1].reshape(Nu,1)})

if __name__=='__main__':
    main()