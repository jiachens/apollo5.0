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
import torch.distributions as tdist
import cluster
from ground_detector_simple import *
from perturbed_FM_generator import *
from torch import autograd
import inputTransformation
from xyz2grid import *
import loss
import c2p_segmentation
import render_new

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

height_ = 672
width_ = 672
range_ = 70
min_height_ = -5.0
max_height_ = 5.0

n = tdist.uniform.Uniform(torch.Tensor([-0.05]),torch.Tensor([0.05]))

if __name__ == '__main__':

    pclpath = '/home/jiachens/Kitti/object/training/velodyne/'
    protofile = root_path + 'deploy.prototxt'
    weightfile = root_path + 'deploy.caffemodel'
    pytorchModel = c2p_segmentation.generatePytorch(protofile, weightfile)

    for dataIdx in range(1):
        dataIdx = 1
        pclfile = '%06d.bin'%(dataIdx)
        PCL_path = pclpath + pclfile

        for lrate in [0.002]:#,0.0001,0.0005,0.00001]:
            for mu in [0.003]:#,0.002,0.003,0.004,0.005,0.006]:
    
                vertex,face = render_new.loadmesh('./cube248.ply')
                vertex_og = vertex.clone()
                #vertex = n.sample(vertex.shape).reshape(-1,3).squeeze().cuda() + vertex
                vertex.requires_grad = True

                with torch.no_grad():
                    meshes_og = torch.nn.functional.embedding(face,vertex_og)
                    edge_1_og = meshes_og[:,1] - meshes_og[:,0]
                    edge_2_og = meshes_og[:,2] - meshes_og[:,0]
                    edge_3_og = meshes_og[:,1] - meshes_og[:,2]

                    dis_og = torch.stack([torch.sqrt(torch.pow(edge_1_og[:,0],2) + 
                                torch.pow(edge_1_og[:,1],2) +
                                torch.pow(edge_1_og[:,2],2)), torch.sqrt(torch.pow(edge_2_og[:,0],2) + 
                                torch.pow(edge_2_og[:,1],2) +
                                torch.pow(edge_2_og[:,2],2)), torch.sqrt(torch.pow(edge_3_og[:,0],2) + 
                                torch.pow(edge_3_og[:,1],2) +
                                torch.pow(edge_3_og[:,2],2))],dim = 1)

                PCL = c2p_segmentation.loadPCL(PCL_path,True)
                

                x_final = torch.cuda.FloatTensor(PCL[:,0])
                y_final = torch.cuda.FloatTensor(PCL[:,1])
                z_final = torch.cuda.FloatTensor(PCL[:,2])
                i_final = torch.cuda.FloatTensor(PCL[:,3])
                ray_direction, length = render_new.get_ray(x_final,y_final,z_final)

                NUM_ITER = 500
                best_iter = None
                best_loss = 1e5 # initialize as a big number
                best_out = None

                opt_Adam = torch.optim.Adam([vertex], lr=lrate)
                for iter_num in range(NUM_ITER):


                    point_cloud = render_new.render(ray_direction,length,vertex,face,i_final)

                    grid = xyzi2grid(point_cloud[:,0],point_cloud[:,1],point_cloud[:,2],point_cloud[:,3])
                    featureM = gridi2feature(grid)
                    # featureM.requires_grad = True

                    point_cloud.data.cpu().numpy().tofile('test.bin')
                    pytorchModel.zero_grad()
                    outputPytorch = pytorchModel(featureM)

                    lossValue,loss_object,loss_distance = loss.lossRenderAttack(outputPytorch,vertex,vertex_og,face,mu,dis_og) 
                    if lossValue < best_loss:
                        best_point_cloud = point_cloud
                        best_loss = lossValue
                        best_iter = iter_num
                        best_out = outputPytorch
                    print('Iteration {} of {}: loss={} object_loss={} distance_loss={}'.format(iter_num,NUM_ITER,lossValue,loss_object,loss_distance))
                    lossValue.backward()
                    opt_Adam.step()
                    del outputPytorch,grid,featureM,lossValue,loss_object,loss_distance
                    torch.cuda.empty_cache()

                #
                best_point_cloud.data.cpu().numpy().tofile('best_test.bin')
                print('learning rate: {}, mu: {}, best_loss: {}'.format(lrate,mu,best_loss))
                with torch.no_grad():
                       PCLConverted = c2p_segmentation.mapPointToGrid(point_cloud.data.cpu().numpy())

                obj, label_map = cluster.cluster(best_out[1].data.cpu().numpy(), best_out[2].data.cpu().numpy(), 
                    best_out[3].data.cpu().numpy(),best_out[0].data.cpu().numpy(),
                    best_out[5].data.cpu().numpy())

                with torch.no_grad():
                    obstacle, cluster_id_list = c2p_segmentation.twod2threed(obj, label_map, point_cloud.data.cpu().numpy(), PCLConverted)

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


