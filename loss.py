import torch
import numpy as np
import sys

def lossPassiveAttack(outputPytorch,x_var,y_var,z_var,scale):
    
    inv_res_x = 0.5 * float(672) / 70

    fx = torch.floor((70 - (0.707107 * (x_var - y_var))) * inv_res_x).long()
    fy = torch.floor((70 - (0.707107 * (x_var + y_var))) * inv_res_x).long()

    fx_og = torch.floor((70 - (0.707107 * (torch.div(x_var,scale) - torch.div(y_var,scale)))) * inv_res_x).long()
    fy_og = torch.floor((70 - (0.707107 * (torch.div(x_var,scale) + torch.div(y_var,scale)))) * inv_res_x).long()
    
    mask = torch.zeros((672,672)).cuda().index_put((fx_og,fy_og),torch.ones(fx_og.shape).cuda())
    mask = mask.index_put((fx,fy),torch.ones(fx.shape).cuda())
    
    mask1 = torch.where(torch.mul(mask,outputPytorch[1]) >= 0.5,torch.ones_like(mask),torch.zeros_like(mask))

    # mask.cpu().detach().numpy().tofile('mask.bin')

    #loss_object = torch.max(torch.max(torch.mul(mask,outputPytorch[2]) - 0.2), torch.tensor(0.0).cuda()) #/ torch.sum(mask)
    #loss_object = torch.max(torch.sum(torch.mul(mask,outputPytorch[2])) / torch.sum(mask) - 0.1, torch.tensor(0.0).cuda())
    loss_object = torch.sum(torch.mul(mask1,outputPytorch[2])) / (torch.sum(mask1) + sys.float_info.epsilon)

    mu = 0.05

    loss_distance = torch.mean(torch.sqrt(torch.pow(x_var - torch.div(x_var,scale) + sys.float_info.epsilon,2) + 
                            torch.pow(y_var - torch.div(y_var,scale) + sys.float_info.epsilon,2) +
                            torch.pow(z_var - torch.div(z_var,scale) + sys.float_info.epsilon,2)))
    # + sys.float_info.epsilon to prevent zero gradient

    loss = mu * loss_distance + loss_object

    return loss,loss_object,loss_distance

def FMCountLoss(FM_og,FM_prox,x_var,y_var,z_var,scale):

    inv_res_x = 0.5 * float(672) / 70

    fx = torch.floor((70 - (0.707107 * (torch.div(x_var,scale) - torch.div(y_var,scale)))) * inv_res_x).long()
    fy = torch.floor((70 - (0.707107 * (torch.div(x_var,scale) + torch.div(y_var,scale)))) * inv_res_x).long()
    mask = torch.zeros((672,672)).cuda().index_put((fx,fy),torch.ones(fx.shape).cuda())

    loss = torch.sum(torch.mul(mask,torch.abs(FM_og - FM_prox)))
    return loss


def lossFeatureAttack(outputPytorch,x_var,y_var,z_var,FM_og,FM):
    
    inv_res_x = 0.5 * float(672) / 70

    fx = torch.floor((70 - (0.707107 * (x_var - y_var))) * inv_res_x).long()
    fy = torch.floor((70 - (0.707107 * (x_var + y_var))) * inv_res_x).long()

    
    mask = torch.zeros((672,672)).cuda().index_put((fx,fy),torch.ones(fx.shape).cuda())

    # print torch.nonzero(mask)
    # mask.cpu().detach().numpy().tofile('mask.bin')

    # loss_object = torch.max(torch.max(torch.mul(mask,outputPytorch[1]) - 0.5), torch.tensor(0.0).cuda()) #/ torch.sum(mask)
    loss_object = torch.max(torch.sum(torch.mul(mask,outputPytorch[2])) / torch.sum(mask) - 0.1, torch.tensor(0.0).cuda())

    mu = 0.5

    loss_distance = torch.sum(torch.sqrt(torch.pow(FM_og - FM + sys.float_info.epsilon,2))) / torch.sum(mask)
    # + sys.float_info.epsilon to prevent zero gradient

    loss = mu * loss_distance + loss_object

    return loss,loss_object,loss_distance