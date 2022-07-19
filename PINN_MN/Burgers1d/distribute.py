# -*- coding: UTF-8 -*-
import numpy as np
import time
import torch
import scipy.io
import csv
from Burgers1d_PINN import Burgers1d_PINN

import pandas as pd
def main():
    #读入测试集
    np.random.seed(42)
    text = scipy.io.loadmat('./Data/burgers_shock.mat')
    text_x = text['x']
    text_y = text['usol']
    text_t = text['t']
    idx = np.random.choice(text_x.shape[0], text_x.shape[0], replace=False)
    text_x=text_x[idx]
    text_y=text_y[idx]
    oo=1


if __name__=='__main__':
    main()