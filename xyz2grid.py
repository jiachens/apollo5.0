from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import sys


class flexibleMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        output = 500 * tensor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = 100 * grad_output 
        return grad_input


def xyz2grid(x_var, y_var, z_var, X_RES=672, Y_RES=672, H_RES=120):

    # X_RES *= 3
    # Y_RES *= 3
    # H_RES *= 3

    # i_float = x_var * X_RES / 120 + X_RES / 2
    # j_float = y_var * Y_RES / 120 + Y_RES / 2
    # k_float = z_var * H_RES / 10 + H_RES / 2
    inv_res_x = 0.5 * float(X_RES) / 70

    i_float = (70 - (0.707107 * (x_var - y_var))) * inv_res_x
    j_float = (70 - (0.707107 * (x_var + y_var))) * inv_res_x

    # i_float = (70 - x_var) * X_RES / 140
    # j_float = (70 - y_var) * Y_RES / 140
    k_float = z_var * H_RES / 10 + H_RES / 2

    i_float = i_float.float()
    j_float = j_float.float()
    k_float = k_float.float()

    i_float = torch.clamp(i_float, 0, X_RES - 1) # i_float in [0, X_RES-1]
    j_float = torch.clamp(j_float, 0, Y_RES - 1) # i_float in [0, Y_RES-1]
    k_float = torch.clamp(k_float, 0, H_RES - 1) # i_float in [0, H_RES-1]

    # smaller in [0, RES - 2]
    i_smallerf = torch.clamp(torch.floor(i_float), 0, X_RES-2) # in [0, X_RES-2]
    j_smallerf = torch.clamp(torch.floor(j_float), 0, Y_RES-2)
    k_smallerf = torch.clamp(torch.floor(k_float), 0, H_RES-2)

    i_smaller = i_smallerf.long()
    j_smaller = j_smallerf.long()
    k_smaller = k_smallerf.long()

    # bigger in [1, RES - 1]
    i_bigger = i_smaller + 1
    j_bigger = j_smaller + 1
    k_bigger = k_smaller + 1

    # The distance to the smaller grid line

    # mu = tf.constant(1)
    # mu = 5

    # alpha = 0.5 + 0.5*tf.tanh(mu*(i_float - i_smallerf-0.5))
    # beta = 0.5 + 0.5*tf.tanh(mu*(j_float - j_smallerf-0.5))
    # gamma = 0.5 + 0.5*tf.tanh(mu*(k_float - k_smallerf-0.5))

    fmul = flexibleMul.apply
    alpha = 0.5 + 0.5*torch.tanh(fmul(i_float - i_smallerf - 1))
    beta = 0.5 + 0.5*torch.tanh(fmul(j_float - j_smallerf - 1))
    gamma = 0.5 + 0.5*torch.tanh(fmul(k_float - k_smallerf - 1))

    # alpha = tf.tanh(i_float - i_smallerf)
    # beta = tf.tanh(j_float - j_smallerf)
    # gamma = tf.tanh(k_float - k_smallerf)

    alpha_ = 1 - alpha
    beta_ = 1 - beta
    gamma_ = 1 - gamma

    # 8 vertices in a cube
    weight000 = alpha_ * beta_ * gamma_
    weight001 = alpha_ * beta_ * gamma
    weight010 = alpha_ * beta * gamma_
    weight011 = alpha_ * beta * gamma
    weight100 = alpha * beta_ * gamma_
    weight101 = alpha * beta_ * gamma
    weight110 = alpha * beta * gamma_
    weight111 = alpha * beta * gamma

    grids = Variable(torch.zeros((X_RES, Y_RES, H_RES), dtype=x_var.dtype)).cuda()
    grids = grids.index_put((i_smaller, j_smaller, k_smaller),weight000, accumulate = True)
    grids = grids.index_put((i_smaller, j_smaller, k_bigger),weight001, accumulate = True)
    grids = grids.index_put((i_smaller, j_bigger , k_smaller),weight010, accumulate = True)
    grids = grids.index_put((i_smaller, j_bigger , k_bigger),weight011, accumulate = True)
    grids = grids.index_put((i_bigger , j_smaller, k_smaller),weight100, accumulate = True)
    grids = grids.index_put((i_bigger , j_smaller, k_bigger),weight101, accumulate = True)
    grids = grids.index_put((i_bigger , j_bigger , k_smaller),weight110, accumulate = True)
    grids = grids.index_put((i_bigger , j_bigger , k_bigger),weight111, accumulate = True)

    
    return grids

def xyzi2grid(x_var, y_var, z_var, i_var, X_RES=672, Y_RES=672, H_RES=120):

    # X_RES *= 3
    # Y_RES *= 3
    # H_RES *= 3

    # i_float = x_var * X_RES / 120 + X_RES / 2
    # j_float = y_var * Y_RES / 120 + Y_RES / 2
    # k_float = z_var * H_RES / 10 + H_RES / 2
    inv_res_x = 0.5 * float(X_RES) / 70

    i_float = (70 - (0.707107 * (x_var - y_var))) * inv_res_x
    j_float = (70 - (0.707107 * (x_var + y_var))) * inv_res_x

    # i_float = (70 - x_var) * X_RES / 140
    # j_float = (70 - y_var) * Y_RES / 140
    k_float = z_var * H_RES / 10 + H_RES / 2

    i_float = i_float.float()
    j_float = j_float.float()
    k_float = k_float.float()

    i_float = torch.clamp(i_float, 0, X_RES - 1) # i_float in [0, X_RES-1]
    j_float = torch.clamp(j_float, 0, Y_RES - 1) # i_float in [0, Y_RES-1]
    k_float = torch.clamp(k_float, 0, H_RES - 1) # i_float in [0, H_RES-1]

    # smaller in [0, RES - 2]
    i_smallerf = torch.clamp(torch.floor(i_float), 0, X_RES-2) # in [0, X_RES-2]
    j_smallerf = torch.clamp(torch.floor(j_float), 0, Y_RES-2)
    k_smallerf = torch.clamp(torch.floor(k_float), 0, H_RES-2)

    i_smaller = i_smallerf.long()
    j_smaller = j_smallerf.long()
    k_smaller = k_smallerf.long()

    # bigger in [1, RES - 1]
    i_bigger = i_smaller + 1
    j_bigger = j_smaller + 1
    k_bigger = k_smaller + 1

    # The distance to the smaller grid line

    # mu = tf.constant(1)
    # mu = 5

    # alpha = 0.5 + 0.5*tf.tanh(mu*(i_float - i_smallerf-0.5))
    # beta = 0.5 + 0.5*tf.tanh(mu*(j_float - j_smallerf-0.5))
    # gamma = 0.5 + 0.5*tf.tanh(mu*(k_float - k_smallerf-0.5))

    fmul = flexibleMul.apply
    alpha = 0.5 + 0.5*torch.tanh(fmul(i_float - i_smallerf - 1))
    beta = 0.5 + 0.5*torch.tanh(fmul(j_float - j_smallerf - 1))
    gamma = 0.5 + 0.5*torch.tanh(fmul(k_float - k_smallerf - 1))

    # alpha = tf.tanh(i_float - i_smallerf)
    # beta = tf.tanh(j_float - j_smallerf)
    # gamma = tf.tanh(k_float - k_smallerf)

    alpha_ = 1 - alpha
    beta_ = 1 - beta
    gamma_ = 1 - gamma

    grids = Variable(torch.zeros((2, X_RES, Y_RES, H_RES), dtype=x_var.dtype)).cuda()
    # 8 vertices in a cube
    weight000 = alpha_ * beta_ * gamma_
    weight001 = alpha_ * beta_ * gamma
    weight010 = alpha_ * beta * gamma_
    weight011 = alpha_ * beta * gamma
    weight100 = alpha * beta_ * gamma_
    weight101 = alpha * beta_ * gamma
    weight110 = alpha * beta * gamma_
    weight111 = alpha * beta * gamma

    grids[0,:,:,:] = grids[0,:,:,:].index_put((i_smaller, j_smaller, k_smaller),weight000, accumulate = True)
    grids[0,:,:,:] = grids[0,:,:,:].index_put((i_smaller, j_smaller, k_bigger),weight001, accumulate = True)
    grids[0,:,:,:] = grids[0,:,:,:].index_put((i_smaller, j_bigger , k_smaller),weight010, accumulate = True)
    grids[0,:,:,:] = grids[0,:,:,:].index_put((i_smaller, j_bigger , k_bigger),weight011, accumulate = True)
    grids[0,:,:,:] = grids[0,:,:,:].index_put((i_bigger , j_smaller, k_smaller),weight100, accumulate = True)
    grids[0,:,:,:] = grids[0,:,:,:].index_put((i_bigger , j_smaller, k_bigger),weight101, accumulate = True)
    grids[0,:,:,:] = grids[0,:,:,:].index_put((i_bigger , j_bigger , k_smaller),weight110, accumulate = True)
    grids[0,:,:,:] = grids[0,:,:,:].index_put((i_bigger , j_bigger , k_bigger),weight111, accumulate = True)

    weight000 = weight000 * i_var
    weight001 = weight001 * i_var
    weight010 = weight010 * i_var
    weight011 = weight011 * i_var
    weight100 = weight100 * i_var
    weight101 = weight101 * i_var
    weight110 = weight110 * i_var
    weight111 = weight111 * i_var

    grids[1,:,:,:] = grids[1,:,:,:].index_put((i_smaller, j_smaller, k_smaller),weight000, accumulate = True)
    grids[1,:,:,:] = grids[1,:,:,:].index_put((i_smaller, j_smaller, k_bigger),weight001, accumulate = True)
    grids[1,:,:,:] = grids[1,:,:,:].index_put((i_smaller, j_bigger , k_smaller),weight010, accumulate = True)
    grids[1,:,:,:] = grids[1,:,:,:].index_put((i_smaller, j_bigger , k_bigger),weight011, accumulate = True)
    grids[1,:,:,:] = grids[1,:,:,:].index_put((i_bigger , j_smaller, k_smaller),weight100, accumulate = True)
    grids[1,:,:,:] = grids[1,:,:,:].index_put((i_bigger , j_smaller, k_bigger),weight101, accumulate = True)
    grids[1,:,:,:] = grids[1,:,:,:].index_put((i_bigger , j_bigger , k_smaller),weight110, accumulate = True)
    grids[1,:,:,:] = grids[1,:,:,:].index_put((i_bigger , j_bigger , k_bigger),weight111, accumulate = True)

    #grids[1,:,:,:] = torch.div(grids[1,:,:,:],grids[0,:,:,:]+sys.float_info.epsilon)
    
    return grids

def gridi2feature(grids):

    MIN_H = -5
    MAX_H = 5
    H_RES = 120
    input_cnt_3d = grids[0,:,:,:]
    input_int_3d = grids[1,:,:,:]
    input_int_3d_mean = torch.div(grids[1,:,:,:],grids[0,:,:,:]+sys.float_info.epsilon)
    

    h_map = np.linspace(MIN_H,MAX_H,H_RES)
    height_map = torch.Tensor(np.array(h_map)).cuda()
    height_map_full = torch.ones(input_cnt_3d.shape).cuda() * height_map
    
    # filter out interpolation side effects
    # mix_input_cnt_3d = tf.where(tf.greater(input_cnt_3d,0.9),input_cnt_3d,tf.zeros_like(input_cnt_3d))
    # mix_input_cnt_3d = tf.tanh(50*(input_cnt_3d-1))+input_cnt_3d
    mix_input_cnt_3d = input_cnt_3d
    thresh_cnt = 0
    scale = 1
    total_cnt = torch.sum(mix_input_cnt_3d,-1)
    temp,idx = torch.max(torch.clamp(torch.sign(scale*(mix_input_cnt_3d-thresh_cnt)),0,1)*(height_map_full+5)-5,-1)
    max_height = torch.unsqueeze(torch.mul(temp,torch.sign(total_cnt)),-1)
    mean_height = torch.unsqueeze(torch.mul(torch.div(torch.sum(mix_input_cnt_3d*height_map_full,-1),total_cnt+sys.float_info.epsilon),torch.sign(total_cnt)),-1) # adding 1e-7 for avoiding DIVISION BY ZERO 

    max_int = torch.unsqueeze(torch.sign(total_cnt),-1)*torch.gather(input_int_3d_mean,2,torch.unsqueeze(idx,-1))  # will be replaced
    mean_int = torch.unsqueeze(torch.mul(torch.div(torch.sum(input_int_3d,-1),total_cnt+sys.float_info.epsilon),torch.sign(total_cnt)),-1)
    cnt = torch.unsqueeze(torch.log(total_cnt+1),-1)

    nonempty = torch.clamp(torch.sign((torch.expm1(cnt)) - 1e-3),0,1)

    FM = torch.stack([max_height, mean_height, cnt, max_int, mean_int, nonempty], -1).permute(2,3,0,1)
    
    return FM


def grid2feature(grids):

    MIN_H = -5
    MAX_H = 5
    H_RES = 120
    input_cnt_3d = grids

    h_map = np.linspace(MIN_H,MAX_H,H_RES)
    height_map = torch.Tensor(np.array(h_map)).cuda()
    height_map_full = torch.ones(input_cnt_3d.shape).cuda() * height_map
    
    # filter out interpolation side effects
    # mix_input_cnt_3d = tf.where(tf.greater(input_cnt_3d,0.9),input_cnt_3d,tf.zeros_like(input_cnt_3d))
    # mix_input_cnt_3d = tf.tanh(50*(input_cnt_3d-1))+input_cnt_3d
    mix_input_cnt_3d = input_cnt_3d
    thresh_cnt = 0
    scale = 1
    total_cnt = torch.sum(mix_input_cnt_3d,-1)
    temp,_ = torch.max(torch.clamp(torch.sign(scale*(mix_input_cnt_3d-thresh_cnt)),0,1)*(height_map_full+5)-5,-1)
    max_height = torch.unsqueeze(torch.mul(temp,torch.sign(total_cnt)),-1)
    mean_height = torch.unsqueeze(torch.mul(torch.div(torch.sum(mix_input_cnt_3d*height_map_full,-1),total_cnt+sys.float_info.epsilon),torch.sign(total_cnt)),-1) # adding 1e-7 for avoiding DIVISION BY ZERO 
    
    max_int = torch.unsqueeze(torch.sign(total_cnt)*144./255,-1)  # will be replaced
    mean_int = max_int # will be replaced
    cnt = torch.unsqueeze(torch.log(total_cnt+1),-1)

    nonempty = torch.clamp(torch.sign((torch.expm1(cnt)) - 1e-3),0,1)

    FM = torch.stack([max_height, mean_height, cnt, max_int, mean_int, nonempty], -1).permute(2,3,0,1)
    
    return FM
