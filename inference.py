from __future__ import division
import torch.nn.functional as F
import os
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
from torch.autograd import Variable
import argparse
import cluster
from ground_detector_simple import *
from perturbed_FM_generator import *
from torch import autograd
import inputTransformation
from xyz2grid import *
import loss
import c2p_segmentation

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

height_ = 672
width_ = 672
range_ = 70
min_height_ = -5.0
max_height_ = 5.0



if __name__ == '__main__':

    pclpath = '/home/jiachens/Kitti/object/training/velodyne/'
    protofile = root_path + 'deploy.prototxt'
    weightfile = root_path + 'deploy.caffemodel'
    pytorchModel = c2p_segmentation.generatePytorch(protofile, weightfile)
    dataIdx = 7
    pclfile = '%06d.bin'%(dataIdx)
    PCL_path = pclpath + pclfile

    PCL = c2p_segmentation.loadPCL(PCL_path)
    PCLConverted = c2p_segmentation.mapPointToGrid(PCL)

    x_final = torch.cuda.FloatTensor(PCL[:,0])
    y_final = torch.cuda.FloatTensor(PCL[:,1])
    z_final = torch.cuda.FloatTensor(PCL[:,2])
    i_final = torch.cuda.FloatTensor(PCL[:,3])

    grid = xyzi2grid(x_final,y_final,z_final,i_final)
    featureM = gridi2feature(grid)
    # featureM.requires_grad = True

        
    pytorchModel.zero_grad()
    outputPytorch = pytorchModel(featureM)


    obj, label_map = cluster.cluster(outputPytorch[1].data.cpu().numpy(), outputPytorch[2].data.cpu().numpy(), 
        outputPytorch[3].data.cpu().numpy(),outputPytorch[0].data.cpu().numpy(),
        outputPytorch[5].data.cpu().numpy())


    obstacle, cluster_id_list = c2p_segmentation.twod2threed(obj, label_map, PCL, PCLConverted)


    # if obstacle != []:
    #     pcd = np.array(obstacle[0].getPCL())
    #     if len(obstacle) == 1:
    #         pcd.tofile('./obs/'+ str(dataIdx) +'_obs.bin')
    #     else:
    #         for obs in obstacle[1:]:
    #             pcd = np.concatenate((pcd,np.array(obs.getPCL())),axis=0)
    #         pcd.tofile('./obs/'+ str(dataIdx) +'_obs.bin')
    # else:
    #     print('No Obstacle Detected.')


