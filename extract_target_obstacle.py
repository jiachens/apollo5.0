from __future__ import division, print_function
import argparse
import torch.nn.functional as F
import numpy as np
import torch
import caffe
import sys
import math
sys.path.append('./pytorch-caffe')
from caffenet import *
# root_path = '/z/apollo_sec/apollo/modules/perception/production/data/perception/lidar/models/cnnseg/velodyne64/'
root_path = './cnnseg/velodyne64/'
from torch.autograd import Variable
import cluster
from ground_detector_simple import *
from perturbed_FM_generator import *
import loss
import c2p_segmentation
import os
import inject_cube

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, help='idx is which frame')
FLAGS = parser.parse_args()

pclpath = '/home/jiachens/Kitti/object/training/velodyne/'
protofile = root_path + 'deploy.prototxt'
weightfile = root_path + 'deploy.caffemodel'
pytorchModels = c2p_segmentation.generatePytorch(protofile, weightfile)

dataIdx = FLAGS.idx
pclfile = '%06d.bin'%(dataIdx)
PCL_path = pclpath + pclfile
PCL = c2p_segmentation.loadPCL(PCL_path,True)
#PCL = inject_cube.injectCylinder(PCL,1.8,0.15,7,0,-1.73)
#PCL = inject_cube.injectCylinder(PCL[:,:4].astype('float32'),1,0.05,7,0,-1.73)
#PCL = inject_cube.injectPyramid(PCL,0.5,1,0.1,7,0,-1.73)
#PCL = inject_cube.injectCube(PCL,0.5,7,-.25,-1.73)
#PCL = inject_cube.injectCylinder(PCL,0.5,0.25,7,0,-1.73)

PCL = PCL[:,:4].astype('float32')

# injected_trace = np.array([[6.8,0.2,-0.73,0.55],
#                            [6.8,0.1,-0.73,0.55],
#                            [6.8,0.0,-0.73,0.55],
#                            [6.8,-0.1,-0.73,0.55],
#                            [6.8,-0.2,-0.73,0.55]]).astype('float32')

injected_trace = np.load('/home/jiachens/AML/test1.npy')
PCL = np.vstack([PCL,injected_trace])



PCLConverted = c2p_segmentation.mapPointToGrid(PCL)
featureM = c2p_segmentation.generateFM(PCL, PCLConverted)
featureM = np.array(featureM).astype('float32')
featureM = torch.cuda.FloatTensor(featureM)
featureM = featureM.view(1,6,672,672)
outputPytorch = pytorchModels(featureM)

obj, label_map = cluster.cluster(outputPytorch[1].data.cpu().numpy(), outputPytorch[2].data.cpu().numpy(), 
            outputPytorch[3].data.cpu().numpy(),outputPytorch[0].data.cpu().numpy(),
            outputPytorch[5].data.cpu().numpy())

outputPytorch[2].data.cpu().numpy().tofile('./confidence.bin')
outputPytorch[1].data.cpu().numpy().tofile('./category.bin')
obstacle, cluster_id_list = c2p_segmentation.twod2threed(obj, label_map, PCL, PCLConverted)

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

os.mkdir('./%d'%(dataIdx))
for idx,obs in enumerate(obstacle):
    pcd = np.array(obs.getPCL())
    pcd.tofile('./'+ str(dataIdx) +'/' + str(idx) + '_obs.bin')

