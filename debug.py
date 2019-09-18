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
    pytorchModels_hard = c2p_segmentation.generatePytorch(protofile, weightfile)
    dataIdx = 7
    pclfile = '%06d.bin'%(dataIdx)
    PCL_path = pclpath + pclfile

    _,PCL_except_car,target_car = c2p_segmentation.preProcess(PCL_path,'./7/18_obs.bin')

    i_var = torch.cuda.FloatTensor(target_car[:,3])

    scale = torch.ones_like(i_var).cuda()
    #scale = torch.cuda.FloatTensor(np.fromfile('./genetic_best_scale_multicross_1000_7_18_cyl_09_11.bin',dtype=np.float32))

    #print(scale)

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

    PCL = torch.stack([x_final,y_final,z_final,i_final]).permute(1,0).cpu().numpy()
    PCLConverted = c2p_segmentation.mapPointToGrid(PCL)
    featureM = c2p_segmentation.generateFM(PCL, PCLConverted)
    featureM = np.array(featureM).astype('float32')
    #featureM.tofile('./featureM_og.bin')
    featureM = torch.cuda.FloatTensor(featureM)
    featureM = featureM.view(1,6,672,672)
    # FM = featureM
    # grid = xyzi2grid(x_final,y_final,z_final,i_final)
    # featureM = gridi2feature(grid)
    #featureM[0,[3,4],:,:] = FM[0,[3,4],:,:]

    featureM.requires_grad = True
    with torch.no_grad():
        featureM_og = featureM.clone()

    # PGD attack
    bestLoss = 1e5
    for i in range(2):

        # featureM[0,0,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = -2.
        # featureM[0,1,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = -2.
        # featureM[0,2,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = 0
        # featureM[0,3,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = 0.0
        # featureM[0,4,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = 0.0
        # featureM[0,5,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = 0
        # print featureM[0,[3],torch.nonzero(mask)]

        # grids = xyz2grid(x_final, y_final, z_final, 100)
        # FM = grid2feature(grids)
        # featureM[0,[3,4],:,:] = 144./255.
        # FM[0,[3,4],:,:] = featureM[0,[3,4],:,:]

        # outputPytorch_soft = pytorchModels_soft(FM)
        # _,loss_object_soft,_ = loss.lossPassiveAttack(outputPytorch_soft,x_var,y_var,z_var,scale)
        # list_soft.append(loss_object_soft.item())
        
        pytorchModels_hard.zero_grad()
        outputPytorch_hard = pytorchModels_hard(featureM)

        lossValue,loss_object,loss_distance = loss.lossFeatureAttack(outputPytorch_hard,x_var,y_var,z_var,featureM_og,featureM)
        print('{} {} {}'.format(lossValue,loss_object,loss_distance))
        if lossValue < bestLoss:
            bestLoss = lossValue
            bestfm = featureM
            bestout = outputPytorch_hard

        lossValue.backward() 
        data_grad = featureM.grad.data

        inv_res_x = 0.5 * float(672) / 70

        fx = torch.floor((70 - (0.707107 * (x_var - y_var))) * inv_res_x).long()
        fy = torch.floor((70 - (0.707107 * (x_var + y_var))) * inv_res_x).long()
        mask = torch.zeros((672,672)).cuda().index_put((fx,fy),torch.ones(fx.shape).cuda())

        with torch.no_grad():
            featureM[0,0,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] -= 1e-2 * torch.sign(data_grad[0,0,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]])
            featureM[0,1,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] -= 1e-2 * torch.sign(data_grad[0,1,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]])
            
            featureM[0,3,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = 0.5#0.01 * torch.sign(data_grad[0,3,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]])
            featureM[0,4,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = 0.5#0.01 * torch.sign(data_grad[0,4,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]])
            
            #featureM[0,0,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = torch.clamp(featureM[0,0,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]],-1.2,0.5)
            #featureM[0,1,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = torch.clamp(featureM[0,1,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]],min=-1.6)

            #condition = torch.gt(featureM[0,0,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]],featureM[0,1,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]])
            #featureM[0,1,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = torch.where(condition,featureM[0,1,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]],featureM[0,0,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]])

            #featureM[0,3,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = torch.clamp(featureM[0,3,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]],0.,1)
            #featureM[0,4,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]] = torch.clamp(featureM[0,3,torch.nonzero(mask)[:,0],torch.nonzero(mask)[:,1]],0.,1)

    obj, label_map = cluster.cluster(bestout[1].data.cpu().numpy(), bestout[2].data.cpu().numpy(), 
        bestout[3].data.cpu().numpy(),bestout[0].data.cpu().numpy(),
        bestout[5].data.cpu().numpy())

    # bestfm.data.cpu().numpy().tofile('./featureM.bin')
    # bestout[2].data.cpu().numpy().tofile('./objectiveness.bin')
    # bestout[1].data.cpu().numpy().tofile('./categoryness.bin')

    obstacle, cluster_id_list = c2p_segmentation.twod2threed(obj, label_map, PCL, PCLConverted)

    for obs in obstacle:
        print(obs.numPT())
        # if obs.numPT() == 1463:
        #   target_obs = np.array(obs.getPCL())
        #   target_obs.tofile('./target_obs.bin') #### HACK THE DATA

    if obstacle != []:
        pcd = np.array(obstacle[0].getPCL())
        if len(obstacle) == 1:
            pcd.tofile('./obs/'+ str(dataIdx) +'_obs.bin')
        else:
            for obs in obstacle[1:]:
                pcd = np.concatenate((pcd,np.array(obs.getPCL())),axis=0)
            pcd.tofile('./obs/'+ str(dataIdx) +'_obs.bin')
    else:
        print('No Obstacle Detected.')


