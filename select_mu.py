from __future__ import division
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
import pypcd
import torch
import caffe
import sys
import math
sys.path.append('./pytorch-caffe')
from caffenet import *
# root_path = '/z/apollo_sec/apollo/modules/perception/production/data/perception/lidar/models/cnnseg/velodyne64/'
root_path = './cnnseg/velodyne64/'
from torchsummary import summary
# import hiddenlayer as hl
from torch.autograd import Variable
# import torchvision.utils as vutils
# from tensorboardX import SummaryWriter
import argparse
import cluster
from ground_detector_simple import *
from perturbed_FM_generator import *
from torch import autograd
import inputTransformation
from xyz2grid import *
import loss
import c2p_segmentation

height_ = 672
width_ = 672
range_ = 70
min_height_ = -5.0
max_height_ = 5.0



if __name__ == '__main__':

    pclpath = '/home/jiachens/Kitti/object/training/velodyne/'
    protofile = root_path + 'deploy.prototxt'
    weightfile = root_path + 'deploy.caffemodel'
    pytorchModels_hard = c2p_segmentation.generatePytorch(protofile, weightfile)
    pytorchModels_soft = c2p_segmentation.generatePytorch(protofile, weightfile)
    dataIdx = 36
    pclfile = '%06d.bin'%(dataIdx)
    PCL_path = pclpath + pclfile

    _,PCL_except_car,target_car = c2p_segmentation.preProcess(PCL_path,'./target_car.bin')

    i_var = torch.cuda.FloatTensor(target_car[:,3])

    list_soft = []
    list_hard = []

    for alpha in np.linspace(0.8,1.2,20): 
        for sample_rate in np.linspace(0.0,1,20):

            mask = np.random.randint(low = 0, high = i_var.shape[0], size = int(round(sample_rate * i_var.shape[0])))
            scale = torch.ones_like(i_var).cuda() 
            scale[mask] *= alpha

            x_var = torch.mul(scale,torch.cuda.FloatTensor(target_car[:,0]))
            y_var = torch.mul(scale,torch.cuda.FloatTensor(target_car[:,1]))
            z_var = torch.mul(scale,torch.cuda.FloatTensor(target_car[:,2]))

            x_final = torch.cuda.FloatTensor(PCL_except_car[:,0])
            y_final = torch.cuda.FloatTensor(PCL_except_car[:,1])
            z_final = torch.cuda.FloatTensor(PCL_except_car[:,2])
            i_final = torch.cuda.FloatTensor(PCL_except_car[:,3])

            x_final = torch.cat([x_final,x_var],dim = 0)
            y_final = torch.cat([y_final,y_var],dim = 0)
            z_final = torch.cat([z_final,z_var],dim = 0)
            i_final = torch.cat([i_final,i_var],dim = 0)


            grids = xyz2grid(x_final, y_final, z_final, 100)
            FM = grid2feature(grids)
            outputPytorch_soft = pytorchModels_soft(FM)
            _,loss_object_soft,_ = loss.lossPassiveAttack(outputPytorch_soft,x_var,y_var,z_var,scale)
            list_soft.append(loss_object_soft.item())
            
            PCL = torch.stack([x_final,y_final,z_final,i_final]).permute(1,0).cpu().numpy()
            PCLConverted = c2p_segmentation.mapPointToGrid(PCL)
            featureM = c2p_segmentation.generateFM(PCL, PCLConverted)
            featureM = np.array(featureM).astype('float32')
            featureM = torch.cuda.FloatTensor(featureM)
            featureM = featureM.view(1,6,672,672)
            outputPytorch_hard = pytorchModels_hard(featureM)
            _,loss_object_hard,_ = loss.lossPassiveAttack(outputPytorch_hard,x_var,y_var,z_var,scale)
            list_hard.append(loss_object_hard.item()) 

    np.save('./list_hard2.npy',np.array(list_hard).reshape(-1,20))
    np.save('./list_soft2.npy',np.array(list_soft).reshape(-1,20))


