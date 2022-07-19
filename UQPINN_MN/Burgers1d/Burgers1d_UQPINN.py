import torch
import argparse
import random
import numpy as np
import os
import math
import copy
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import time
# class NN(nn.Module):
#     def __init__(self, hidden_dim, input_dim, output_dim, num_layers):
#         super(NN, self).__init__()
#         self.num_layers = num_layers
#         self.output_dim = output_dim
#         self.input_dim = input_dim
#         self.input_layer = nn.Linear(input_dim, hidden_dim)
#         self.middle_layer = nn.Linear(hidden_dim, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         out = torch.tanh(self.input_layer(x))
#
#         for i in range(self.num_layers - 2):
#             out = torch.tanh(self.middle_layer(out))
#
#         out = self.output_layer(out)
#         return out

class NN(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim, num_layers ):
        super(NN, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        # self.devive = device
        self.input_layer = nn.Linear(input_dim, hidden_dim)  # 对输入数据做线性变换，y=Az+b
        self.middle_layer = nn.Sequential()
        for i in range(num_layers - 2):
            self.middle_layer.add_module('hidden_{}'.format(i), nn.Linear(hidden_dim, hidden_dim))
            self.middle_layer.add_module('activation_{}'.format(i), nn.Tanh())
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))

        out = self.middle_layer(out)

        out = self.output_layer(out)
        return out

class AVEMSE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, output_Nu,Nu,u,u_t,u_x,u_xx):
        loss_b = torch.sum(torch.pow(output_Nu-Nu,2))/output_Nu.size()[0]
        loss_f = torch.sum(torch.pow(u_t+u*u_x-(0.01/math.pi)*u_xx,2))/u_t.size()[0]
        loss = loss_b+loss_f
        return loss


def adjust_learning_rate(optimizer, epoch,lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_r (model_P_tgradient,model_P_0gradient,model_P_1gradient,model_P_2gradient,inp_t_Nf, inp_x_Nf, inp_z_Nf,device,Jacobian_X,Jacobian_T):
    u = model_P_0gradient(torch.cat([inp_t_Nf, inp_x_Nf, inp_z_Nf], dim=1))

    u_x = model_P_1gradient(torch.cat([inp_t_Nf, inp_x_Nf, inp_z_Nf], dim=1))

    u_xx = model_P_2gradient(torch.cat([inp_t_Nf, inp_x_Nf, inp_z_Nf], dim=1))

    u_t = model_P_tgradient(torch.cat([inp_t_Nf, inp_x_Nf, inp_z_Nf], dim=1))

    r = Jacobian_T*u_t+Jacobian_X*u*u_x-(0.01/math.pi)*Jacobian_X*Jacobian_X*u_xx

    return r

def compute_discriminator_loss(model_P_0gradient,model_Q,model_T,inp_x_Nu, inp_t_Nu, value_Nu, inp_z_Nu):

    value_Nu_pred = model_P_0gradient(torch.cat([inp_t_Nu, inp_x_Nu, inp_z_Nu], dim = 1))

    T_real = model_T(torch.cat([inp_t_Nu, inp_x_Nu, value_Nu], dim = 1))
    T_fake = model_T(torch.cat([inp_t_Nu, inp_x_Nu, value_Nu_pred], dim = 1))

    T_real = torch.sigmoid(T_real)
    T_fake = torch.sigmoid(T_fake)

    T_loss = -torch.mean(torch.log(1.0 - T_real ) + \
                             torch.log(T_fake ))

    return T_loss
# T_real = self.net_T(X, T, Y)
# T_fake = self.net_T(X, T, Y_pred)
#
# T_real = tf.sigmoid(T_real)
# T_fake = tf.sigmoid(T_fake)
#
# T_loss = -tf.reduce_mean(tf.log(1.0 - T_real + 1e-8) + \
#                          tf.log(T_fake + 1e-8))
#
# return T_loss


def compute_generator_loss(model_P_0gradient,model_Q,model_T,inp_x_Nu,inp_t_Nu,value_Nu,value_Nu_pred,inp_x_Nf,inp_t_Nf,Residual,inp_z_Nu,inp_z_Nf,lam,beta):

    inp_z_Nu_pred = model_Q(torch.cat([inp_t_Nu,inp_x_Nu,value_Nu_pred], dim = 1))
    inp_z_Nf_pred = model_Q(torch.cat([inp_t_Nf,inp_x_Nf,Residual], dim = 1))

    value_pred = model_P_0gradient(torch.cat([inp_t_Nu,inp_x_Nu,inp_z_Nu], dim = 1))
    T_pred = model_T(torch.cat([inp_t_Nu, inp_x_Nu, value_pred], dim = 1))

    KL = torch.mean(T_pred)

    log_q = -torch.mean(torch.pow(inp_z_Nu-inp_z_Nu_pred,2))

    loss_f = torch.mean(torch.pow(Residual,2))

    loss = KL + (1.0 - lam) * log_q + beta * loss_f

    return loss, KL, (1.0 - lam) * log_q,  loss_f

def train(model_P_tgradient,model_P_0gradient,model_P_1gradient,model_P_2gradient,model_Q,model_T,Nf_dataset,Nu_dataset,loss_fn,device,num_epochs,optimizer_G,optimizer_T,text_x,text_t,text_y,text_z,lam = 1.5, beta = 1.,k1=1,k2=5):
    inp_z_text = Variable(text_z, requires_grad=False).to(device)
    inp_x_text = Variable(text_x, requires_grad=False).to(device)
    inp_t_text = Variable(text_t, requires_grad=False).unsqueeze(-1).to(device)
    value_text = Variable(text_y, requires_grad=False).unsqueeze(-1).to(device)

    for epoch_idx in range(num_epochs):
        print(epoch_idx)
        data_loader_Nf = DataLoader(Nf_dataset,num_workers=0, batch_size=14,)
        data_loader_Nu = DataLoader(Nu_dataset,num_workers=0, batch_size=2,)
        for batch_Nf,batch_Nu in zip(data_loader_Nf,data_loader_Nu):
            inp_z_Nf = Variable(batch_Nf[:, 2], requires_grad=True).unsqueeze(-1).to(device)
            inp_x_Nf = Variable(batch_Nf[:, 1], requires_grad=True).unsqueeze(-1).to(device)
            inp_t_Nf= Variable(batch_Nf[:, 0], requires_grad=True).unsqueeze(-1).to(device)

            inp_z_Nu = Variable(batch_Nu[:, 3], requires_grad=True).unsqueeze(-1).to(device)
            inp_x_Nu = Variable(batch_Nu[:, 1], requires_grad=True).unsqueeze(-1).to(device)
            inp_t_Nu= Variable(batch_Nu[:,  0], requires_grad=True).unsqueeze(-1).to(device)
            value_Nu= Variable(batch_Nu[:,  2], requires_grad=True).unsqueeze(-1).to(device)

            # inp_x_Nf_mean,inp_x_Nf_std = torch.mean(inp_x_Nf),torch.std(inp_x_Nf)
            # inp_t_Nf_mean,inp_t_Nf_std = torch.mean(inp_t_Nf),torch.std(inp_t_Nf)
            # value_Nu_mean,value_Nu_std = torch.mean(value_Nu),torch.std(value_Nu)
            #
            #
            # inp_x_Nf = (inp_x_Nf-inp_x_Nf_mean)/inp_x_Nf_std
            # inp_x_Nu = (inp_x_Nu-inp_x_Nf_mean)/inp_x_Nf_std
            # inp_t_Nf = (inp_t_Nf-inp_t_Nf_mean)/inp_t_Nf_std
            # inp_t_Nu = (inp_t_Nu-inp_t_Nf_mean)/inp_t_Nf_std
            #
            # Jacobian_X = 1 / inp_x_Nf_std
            # Jacobian_T = 1 / inp_t_Nf_std

            # inp_z_Nf = torch.randn(14,requires_grad=True).unsqueeze(-1).to(device)
            # inp_z_Nu = torch.randn(2,requires_grad=True).unsqueeze(-1).to(device)

            Jacobian_X=1
            Jacobian_T=1

            value_Nu_pred = model_P_0gradient(torch.cat([inp_t_Nu,inp_x_Nu,inp_z_Nu],dim=1))
            Residual = get_r(model_P_tgradient,model_P_0gradient,model_P_1gradient,model_P_2gradient,inp_t_Nf, inp_x_Nf, inp_z_Nf,device,Jacobian_X,Jacobian_T)

            G_loss, KL_loss, recon_loss, PDE_loss=compute_generator_loss(model_P_0gradient,model_Q,model_T,inp_x_Nu,inp_t_Nu,value_Nu,value_Nu_pred,inp_x_Nf,inp_t_Nf,
                                                                         Residual,inp_z_Nu,inp_z_Nf,lam,beta)

            T_loss = compute_discriminator_loss(model_P_0gradient,model_Q,model_T,inp_x_Nu, inp_t_Nu, value_Nu, inp_z_Nu)

            for i in range(k1):
                model_T.zero_grad()
                T_loss.backward(retain_graph=True)
                optimizer_T.step()

            for i in range(k2):
                model_P_tgradient.zero_grad()
                model_P_0gradient.zero_grad()
                model_P_1gradient.zero_grad()
                model_P_2gradient.zero_grad()
                model_Q.zero_grad()
                G_loss.backward(retain_graph=True)
                optimizer_G.step()
    with torch.no_grad():
        Inference_time = time.time()
        u_text = model_P_0gradient(torch.cat([inp_t_text, inp_x_text,inp_z_text], dim=1)).to(device)
        Inference_time = time.time() - Inference_time
        loss_text = torch.sum(torch.pow(u_text - value_text, 2)) / u_text.size()[0]
        print(['training_loss', PDE_loss, 'text_loss', loss_text])
    loss_=torch.pow(u_text - value_text, 2)
    loss_evermoment = []
    for i in range(100):
        loss_evermoment.append([i*0.01,float(torch.sum(loss_[i*256:(i+1)*256])/256)])
    return loss_evermoment,loss_text,Inference_time,u_text[40*256:41*256],u_text[80*256:81*256]
    # with torch.no_grad():
    #     Inference_time = time.time()
    #     u_text = model(torch.cat([inp_t_text, inp_x_text], dim=1)).to(device)
    #     Inference_time = time.time()-Inference_time
    #     loss_text = torch.sum(torch.pow(u_text-value_text,2))/u_text.size()[0]
    #     print(['training_loss',loss,'text_loss',loss_text])
    # loss_=torch.pow(u_text - value_text, 2)
    # loss_evermoment = []
    # for i in range(100):
    #     loss_evermoment.append([i*0.01,float(torch.sum(loss_[i*256:(i+1)*256])/256)])
    # return loss_evermoment,loss_text,Inference_time,u_text[40*256:41*256],u_text[80*256:81*256]

def Burgers1d_UQPINN(Nf_dataset, Nu_dataset,text_x,text_t,text_y,num_epochs=150):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--results_path', type=str, default='')
    parser.add_argument('--optimizer', type=str, default='Adam')
    args = parser.parse_args()
    reset_random_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_P_tgradient = NN(20, 3, 1, 5)
    model_P_0gradient = NN(20, 3, 1, 8)
    model_P_1gradient = NN(20, 3, 1, 5)
    model_P_2gradient = NN(20, 3, 1, 5)
    model_Q = NN(20, 3, 1, 8)
    model_T = NN(20, 3, 1, 8)
    model_P_tgradient.to(device)
    model_P_0gradient.to(device)
    model_P_1gradient.to(device)
    model_P_2gradient.to(device)
    model_Q.to(device)
    model_T.to(device)
    if args.optimizer =='Adam':
        optimizer_G = torch.optim.Adam(   [{'params':model_P_tgradient.parameters(),'lr':1e-4},
                                           {'params': model_P_0gradient.parameters(), 'lr': 1e-4},
                                           {'params': model_P_1gradient.parameters(), 'lr': 1e-4},
                                           {'params': model_P_2gradient.parameters(), 'lr': 1e-4},
                                        {'params': model_Q.parameters(), 'lr': 1e-4},])
    else:
        optimizer_G = torch.optim.SGD([{'params':model_P_tgradient.parameters(),'lr':1e-4},
                                           {'params': model_P_0gradient.parameters(), 'lr': 1e-4},
                                           {'params': model_P_1gradient.parameters(), 'lr': 1e-4},
                                           {'params': model_P_2gradient.parameters(), 'lr': 1e-4},
                                        {'params': model_Q.parameters(), 'lr': 1e-4},], lr=1e-2, momentum=0.9)

    if args.optimizer =='Adam':
        optimizer_T = torch.optim.Adam(  [{'params': model_T.parameters(), 'lr': 1e-4},])
    else:
        optimizer_T = torch.optim.SGD([{'params': model_T.parameters(), 'lr': 1e-4},], lr=1e-2, momentum=0.9)
    # t=optimizer.param_groups
    loss_fn = AVEMSE()
    Z_f = torch.randn(Nf_dataset.size()[0]).unsqueeze(-1)
    Z_u = torch.randn(Nu_dataset.size()[0]).unsqueeze(-1)
    text_z = torch.randn(text_x.shape[0]).unsqueeze(-1)
    Nf_dataset = torch.cat((Nf_dataset,Z_f),dim=1)
    Nu_dataset = torch.cat((Nu_dataset,Z_u),dim=1)
    loss_evermoment,loss_text,Inference_time,value_4,value_8 = train(
        model_P_tgradient=model_P_tgradient,
        model_P_0gradient=model_P_0gradient,
        model_P_1gradient=model_P_1gradient,
        model_P_2gradient=model_P_2gradient,
        model_Q=model_Q,
        model_T=model_T,
        Nf_dataset=Nf_dataset,
        Nu_dataset=Nu_dataset,
        loss_fn = loss_fn,
        device = device,
        num_epochs = num_epochs,
        optimizer_G=optimizer_G,
        optimizer_T=optimizer_T,
        text_x = text_x,
        text_t = text_t,
        text_y = text_y,
        text_z=text_z,
        lam = 1.5,
        beta = 1.,
        k1=1,
        k2=2,
    )
    return loss_evermoment,loss_text,Inference_time,value_4,value_8



