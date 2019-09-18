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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# parser = argparse.ArgumentParser()
# parser.add_argument('--idx', type=int, help='idx is which frame')
# FLAGS = parser.parse_args()

pclpath = '/home/jiachens/Kitti/object/training/velodyne/'
protofile = root_path + 'deploy.prototxt'
weightfile = root_path + 'deploy.caffemodel'
pytorchModels = c2p_segmentation.generatePytorch(protofile, weightfile)

# dataIdx = FLAGS.idx

succ = [0] * 7481

for dataIdx in range(7481):
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

    obstacle, cluster_id_list = c2p_segmentation.twod2threed(obj, label_map, PCL, PCLConverted)

    if obstacle != []:
        for idx,obs in enumerate(obstacle):
            pcd = np.array(obs.getPCL())
            # test
            # if (np.mean(pcd[:,0]) > 7 and np.mean(pcd[:,0]) < 10) and (np.mean(pcd[:,1]) > -1 and np.mean(pcd[:,1]) < 1):
            # test 1
            if (np.mean(pcd[:,0]) > 7.5 and np.mean(pcd[:,0]) < 11) and (np.mean(pcd[:,1]) > -1.5 and np.mean(pcd[:,1]) < 0.5):
                succ[dataIdx] = 1
                break

    print('current frame: {}, succ: {}'.format(dataIdx, succ[dataIdx])) 

rate = sum(succ) / 7481.
print('succ rate is: {}'.format(rate))

of_path = os.path.join('./', 'succ_1.txt')
with open(of_path, 'w+') as f:
    for idx,item in enumerate(succ):
        f.write(str(idx) + ',' + str(item) + '\n')
    f.write('succ rate is: ' + str(rate))

