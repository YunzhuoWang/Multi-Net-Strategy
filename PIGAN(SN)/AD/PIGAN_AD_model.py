# File      :PIGAN_AD_model.py
# Time      :2021/5/21--14:38
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import torch
import random
import argparse
import os
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


class NN(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim, num_layers):
        super(NN, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.middle_layer = nn.Sequential()
        for i in range(num_layers - 2):
            self.middle_layer.add_module('hidden_{}'.format(i), nn.Linear(hidden_dim, hidden_dim))
            self.middle_layer.add_module('activation_{}'.format(i), nn.ReLU())
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.middle_layer(out)
        out = self.output_layer(out)
        return out


def adjust_lr(optimizer, itr):
    if itr == 3000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-6
    if itr == 4000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-6


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(generator, discriminator, g_optimizer, d_optimizer, device, num_iter, Nf_dataset, Nu_dataset, test_point,
          test_real_value, discriminator_iteration, generator_iteration):
    # 处理测试点
    test_real_value = Variable(test_real_value, requires_grad = False).to(device)
    test_point = Variable(test_point, requires_grad = False).to(device)

    # 训练时间
    all_train_time = 0
    testing_time = 0

    # 训练神经网络
    for itr in range(num_iter):
        dataloader_Nf = DataLoader(Nf_dataset, batch_size = 14)
        dataloader_Nu = DataLoader(Nu_dataset, batch_size = 2)
        begin_time = time.time()
        count = 0
        for batch_Nf, batch_Nu in zip(dataloader_Nf, dataloader_Nu):
            count += 1
            # 处理训练集
            train_point_interior_t = Variable(batch_Nf[:, 0], requires_grad = True).unsqueeze(-1).to(device)
            train_point_interior_x = Variable(batch_Nf[:, 1], requires_grad = True).unsqueeze(-1).to(device)
            train_point_interior_y = Variable(batch_Nf[:, 2], requires_grad = True).unsqueeze(-1).to(device)

            # 处理边界集
            train_point_boundary = Variable(batch_Nu[:, 0:3], requires_grad = True).to(device)
            train_value_boundary = Variable(batch_Nu[:, 3], requires_grad = True).unsqueeze(-1).to(device)

            # 计算生成器预测值
            train_point_boundary_pred = generator(train_point_boundary).to(device)
            train_point_interior_pred = generator(
                torch.cat([train_point_interior_t, train_point_interior_x, train_point_interior_y], dim = 1)).to(device)

            # 对非抽样点的内部点生成结果进行后处理(post-processing)
            u_t = autograd.grad(outputs = train_point_interior_pred, inputs = train_point_interior_t,
                                grad_outputs = torch.ones(train_point_interior_pred.size()).to(device),
                                create_graph = True,
                                )[0]
            u_x = autograd.grad(outputs = train_point_interior_pred, inputs = train_point_interior_x,
                                grad_outputs = torch.ones(train_point_interior_pred.size()).to(device),
                                create_graph = True,
                                )[0]
            u_xx = autograd.grad(outputs = u_x, inputs = train_point_interior_x,
                                 grad_outputs = torch.ones(u_x.size()).to(device),
                                 create_graph = True,
                                 )[0]
            u_y = autograd.grad(outputs = train_point_interior_pred, inputs = train_point_interior_y,
                                grad_outputs = torch.ones(train_point_interior_pred.size()).to(device),
                                create_graph = True,
                                )[0]
            u_yy = autograd.grad(outputs = u_y, inputs = train_point_interior_y,
                                 grad_outputs = torch.ones(u_y.size()).to(device),
                                 create_graph = True,
                                 )[0]
            # 方程残差
            residual = u_t - 0.02 * (u_xx + u_yy) + u_x * np.cos(np.pi / 8) + u_y * np.sin(np.pi / 8)
            # 判别器的预测值
            discriminator_pred_fake = discriminator(residual)
            discriminator_pred_real = discriminator(torch.zeros_like(residual))
            # 计算生成器误差
            G_loss_boundary = torch.mean(torch.pow(train_point_boundary_pred - train_value_boundary, 2))
            G_loss_interior = torch.mean(F.binary_cross_entropy_with_logits(discriminator_pred_fake,
                                                                            torch.ones_like(discriminator_pred_fake)))
            G_total_loss = G_loss_boundary + G_loss_interior
            # 计算判别器误差
            D_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(discriminator_pred_fake,
                                                                        torch.zeros_like(discriminator_pred_fake)))
            D_loss_real = torch.mean(F.binary_cross_entropy_with_logits(discriminator_pred_real,
                                                                        torch.ones_like(discriminator_pred_real)))
            D_total_loss = D_loss_fake + D_loss_real
            # 梯度更新
            for i in range(discriminator_iteration):
                d_optimizer.zero_grad()
                D_total_loss.backward(retain_graph = True)
                d_optimizer.step()
                # adjust_lr(d_optimizer, itr)
            for i in range(generator_iteration):
                g_optimizer.zero_grad()
                G_total_loss.backward(retain_graph = True)
                g_optimizer.step()
                # adjust_lr(g_optimizer, itr)
            # log
            if count % 10 == 0:
                print({'count': count, 'G loss': G_total_loss, 'D loss': D_total_loss})
        training_time = time.time() - begin_time
        all_train_time += training_time
        # 预测
        test_begin_time = time.time()
        with torch.no_grad():
            test_pred_value = generator(test_point).to(device)
            test_error_MSE = torch.sum(torch.pow(test_real_value - test_pred_value, 2)) / test_point.shape[0]
            test_error_RMSE = torch.sqrt(test_error_MSE)
            test_error_MAE = torch.sum(torch.abs(test_real_value - test_pred_value)) / test_point.shape[0]
        testing_time = time.time() - test_begin_time
        # log
        print({'itr': itr, 'G loss': G_total_loss, 'D loss': D_total_loss, 'test loss': test_error_MSE,
               'train time': training_time, 'test time': testing_time})

    return {'MSE': test_error_MSE, 'RMSE': test_error_RMSE, 'MAE': test_error_MAE}, {
        'train time': all_train_time, 'test time': testing_time}


def AD(Nf_dataset, Nu_dataset, test_point, test_real_value):
    # 初始设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type = int, default = 0)
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--num_iter', type = int, default = 2)
    parser.add_argument('--discriminator_iteration', type = int, default = 1)
    parser.add_argument('--generator_iteration', type = int, default = 5)
    args = parser.parse_args()
    reset_random_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 模型定义
    generator = NN(30, 3, 1, 15)  # 生成器
    discriminator = NN(30, 1, 1, 15)  # 判别器
    generator.to(device)
    discriminator.to(device)
    # 优化器设置
    if args.optimizer == 'Adam':
        g_optimizer = torch.optim.Adam(generator.parameters(), lr = 1e-5)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 1e-5)
    else:
        g_optimizer = torch.optim.SGD(generator.parameters(), lr = 0.001, momentum = 0.9)
        d_optimizer = torch.optim.SGD(discriminator.parameters(), lr = 0.001, momentum = 0.9)
    # 模型训练
    loss, times = train(
        generator = generator,
        discriminator = discriminator,
        g_optimizer = g_optimizer,
        d_optimizer = d_optimizer,
        device = device,
        num_iter = args.num_iter,
        Nf_dataset = Nf_dataset,
        Nu_dataset = Nu_dataset,
        test_point = test_point,
        test_real_value = test_real_value,
        discriminator_iteration = args.discriminator_iteration,
        generator_iteration = args.generator_iteration
    )
    return loss, times
