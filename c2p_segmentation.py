from __future__ import division, print_function
import os
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
from torch.autograd import Variable
import argparse
import cluster
from ground_detector_simple import *
from perturbed_FM_generator import *
from torch import autograd
import inputTransformation
from xyz2grid import *
import loss
import torch.distributions as tdist
import inject_cube

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

height_ = 672
width_ = 672
range_ = 70
min_height_ = -5.0
max_height_ = 5.0

outputs = ['instance_pt', 'category_score', 'confidence_score',
			'height_pt', 'heading_pt', 'class_score']

#writer = SummaryWriter('runs/exp-1')

def rc2grid(r,c):
    return r*672+c


class Obstacle:
	
	def __init__(self, idx):
		self.id = idx
		self.PCL = []

	def add(self,pt):
		self.PCL.append(pt)

	def numPT(self):
		return len(self.PCL)

	def getPCL(self):
		return self.PCL


def mapPointToGrid(PCL):

	point2grid = []

	pos_x = -1
	pos_y = -1
	inv_res_x = 0.5 * float(width_) / range_
	size = PCL.shape[0]
	for idx in range(size):
		if (PCL[idx,2] <= min_height_ or PCL[idx,2] >= max_height_):
			point2grid.append(-1)
			continue
		pos_x, pos_y = groupPc2Pixel(PCL[idx,:], inv_res_x, range_); 
		if (pos_y < 0 or pos_y >= height_ or pos_x < 0 or pos_x >= width_):
			point2grid.append(-1)
			continue
		point2grid.append(pos_y * width_ + pos_x)

	return point2grid

def groupPc2Pixel(PT, inv_res_x, range_):

	fx = (range_ - (0.707107 * (PT[0] + PT[1]))) * inv_res_x
	fy = (range_ - (0.707107 * (PT[0] - PT[1]))) * inv_res_x
	x = -1 if fx < 0 else int(fx)
	y = -1 if fy < 0 else int(fy)
	return x,y


def loadPCL(PCL,flag):

	if flag:
		PCL = np.fromfile(PCL, dtype=np.float32)
		PCL = PCL.reshape((-1,4))

		# spoofed_obs = np.fromfile('spoof_obs.bin', dtype=np.float32)
		# spoofed_obs = spoofed_obs.reshape((-1,4))
		# spoofed_obs[:,3] /= 255
		# scale = 0.5

		# trans_matrix = [[math.cos(0),-math.sin(0),0,delta],
		# 		[math.sin(0),math.cos(0),0,0],
		# 		[0,0,scale,0],
		# 		[0,0,0,1]]

		# for i in range(spoofed_obs.shape[0]):
		# 	temp = spoofed_obs[i,3]
		# 	spoofed_obs[i,3] = 1
		# 	spoofed_obs[i,:] = np.matmul(trans_matrix, spoofed_obs[i,:])
		# 	spoofed_obs[i,3] = temp

		# PCL = np.vstack((PCL,spoofed_obs))

		# deleteline = []
		# for i in range(PCL.shape[0]):
		# 	if (PCL[i,2] < 0.3 and PCL[i,0] > x_min and PCL[i,1] > y_min and PCL[i,1] < y_max):
		# 		deleteline.append(i)
		# PCL = np.delete(PCL, deleteline, axis = 0)


	# ############# Dummy Inject #####################
	# 	size = 60
	# 	PC = []
	# 	for i in range(size):
	# 		x = random.uniform(0, 1)
	# 		y = random.uniform(0, 1)
	# 		z = random.uniform(-1, 1)
	# 		PC.append([x,y,z,1])

	# 	PC = np.array(PC)

	# 	theta = math.pi / 3
	# 	delta = 20
	# 	scale = 1
	# 	trans_matrix = [[math.cos(theta),-math.sin(theta),0,delta],
	# 			[math.sin(theta),math.cos(theta),0,0],
	# 			[0,0,scale,0],
	# 			[0,0,0,1]]
	# 	for i in range(size):
	# 		PC[i,:] = np.matmul(trans_matrix, PC[i,:])
	# 		PC[i,3] = 0.99

	# ################################################
	# 	PCL = np.vstack((PCL,PC))

	else:
		PCL = pypcd.PointCloud.from_path(PCL)
		PCL = np.array(tuple(PCL.pc_data.tolist()))
		PCL = np.delete(PCL, -1, axis = 1)

	return PCL

def generateFM(PCL, PCLConverted):

	max_height_data_ = []
	mean_height_data_ = []
	count_data_ = []
	top_intensity_data_ = []
	mean_intensity_data_ = []
	nonempty_data_ = []

	mapSize = height_ * width_
	size = PCL.shape[0]
	for i in range(mapSize):
		max_height_data_.append(-5.0)
		mean_height_data_.append(0.0)
		count_data_.append(0.0)
		top_intensity_data_.append(0.0)
		mean_intensity_data_.append(0.0)
		nonempty_data_.append(0.0)

	for i in range(size):
		idx = PCLConverted[i]
		if idx == -1:
			continue
		pz = PCL[i,2]
		pi = PCL[i,3] #/ 255.0
		if max_height_data_[idx] < pz:
			max_height_data_[idx] = pz
			top_intensity_data_[idx] = pi;
		mean_height_data_[idx] += float(pz)
		mean_intensity_data_[idx] += float(pi);
		count_data_[idx] += 1.0;

	for i in range(mapSize):
		if count_data_[i] <= sys.float_info.epsilon:
			max_height_data_[i] = 0.0
			count_data_[i] = math.log(1)
		else:
			mean_height_data_[i] /= count_data_[i]
			mean_intensity_data_[i] /= count_data_[i]
			nonempty_data_[i] = 1.0
			count_data_[i] = math.log(int(count_data_[i])+1)

	max_height_data_ = np.array(max_height_data_).reshape(-1,672)
	mean_height_data_ = np.array(mean_height_data_).reshape(-1,672)
	count_data_ = np.array(count_data_).reshape(-1,672)
	top_intensity_data_ = np.array(top_intensity_data_).reshape(-1,672)
	mean_intensity_data_ = np.array(mean_intensity_data_).reshape(-1,672)
	nonempty_data_ = np.array(nonempty_data_).reshape(-1,672)

	FM = [max_height_data_, mean_height_data_, count_data_, top_intensity_data_, 
		mean_intensity_data_, nonempty_data_]

	return FM


def generatePytorch(protofile, weightfile):
	net = CaffeNet(protofile, phase='TEST')
	torch.cuda.set_device(0)
	net.cuda()
	net.load_weights(weightfile)
	net.set_train_outputs(outputs)
	net.set_eval_outputs(outputs)
	net.eval()
	return net

def generateCaffe(protofile, weightfile):
	caffe.set_device(0)
	caffe.set_mode_gpu()
	net = caffe.Net(protofile, weightfile, caffe.TEST)
	return net

def ccs_attack():
    pass

def NIPS_attack():
	pass

def generateNormalMask(target_center):

	mask = np.zeros((672,672))
	x = np.linspace(norm.ppf(0.01),norm.ppf(0.5), 672)

	for r in range(672):
		for c in range(672):
			mask[r,c] = norm.pdf(x[671 - abs(r - target_center[0])]) * norm.pdf(x[671 - abs(c - target_center[1])])
			#mask[r,c] = 1 # for test
	return mask

def preProcess(PCLmain,PCLtarget):

	PCL = loadPCL(PCLmain,True)
	PCL = inject_cube.injectCylinder(PCL,0.5,0.25,7,0,-1.73)
	PCL = PCL[:,:4].astype('float32')

	target_obs = loadPCL(PCLtarget,True)

	aset = set([tuple(x) for x in PCL])
	bset = set([tuple(x) for x in target_obs])

	PCL_except_obs = np.array([x for x in aset - bset])

	return PCL,PCL_except_obs,target_obs

def twod2threed(obj, label_map, PCL, PCLConverted):

	obstacle = []
	cluster_id_list = []
	for obs in obj:
		#if len(obs) >= 4:
		cluster_id_list.append(obs[-1][1])
		obstacle.append(Obstacle(obs[-1][1]))

	size = PCL.shape[0]
	for i in range(size):
		idx = PCLConverted[i]
		if idx < 0:
			continue
		pt = PCL[i,:]
		label = label_map[idx]
		if label < 1: ## means if ==0
			continue
		if label in cluster_id_list and pt[2] <= obj[cluster_id_list.index(label)][-1][-1] + 0.5:
			obstacle[cluster_id_list.index(label)].add(pt) 
			
	obstacle = [obs for obs in obstacle if obs.numPT() >= 3]

	ground = []
	_,_,ground_model,_ = my_ransac(np.delete(PCL, -1, axis = 1))
	for i in range(PCL.shape[0]):
		z = (-ground_model[0] * PCL[i,0] - -ground_model[1] * PCL[i,1] - ground_model[3]) / ground_model[2]
		if PCL[i,2] < (z + 0.25):
			# can directly remove ground here
			ground.append(PCL[i,:])
	ground = np.array(ground)

	index = []
	for idx,obs in enumerate(obstacle):
		num_pt = obs.numPT()
		pc = np.array(obs.getPCL())
		for i in range(pc.shape[0]):
			z = (-ground_model[0] * pc[i,0] - -ground_model[1] * pc[i,1] - ground_model[3]) / ground_model[2]
			if pc[i,2] < (z + 0.25):
				num_pt -= 1
		if num_pt < 3:
			index.append(idx)

	obstacle = [obs for obs in obstacle if obstacle.index(obs) not in index]

	return obstacle, cluster_id_list





def gradattack():
	pclpath = '/home/jiachens/Kitti/object/training/velodyne/'
	protofile = root_path + 'deploy.prototxt'
	weightfile = root_path + 'deploy.caffemodel'
	pytorchModels = generatePytorch(protofile, weightfile)
	n = tdist.Normal(torch.tensor([1.0]), torch.tensor([0.05]))

	for dataIdx in range(1):
		dataIdx = 7
		pclfile = '%06d.bin'%(dataIdx)
		PCL_path = pclpath + pclfile
		#PCL = '1.pcd'
		_, PCL_except_obs,target_obs = preProcess(PCL_path,'./7/16_obs.bin')
		#PCL = inputTransformation.KNNsmooth(PCL)
		# PCL.tofile('PC.bin')
		i_var = torch.cuda.FloatTensor(target_obs[:,3])
		#scale_og = Variable(n.sample(i_var.shape).reshape(1,-1).squeeze().cuda() , requires_grad = True)
		scale_og = Variable(torch.ones_like(i_var).cuda(), requires_grad = True)
		opt_Adam = torch.optim.Adam([scale_og], lr=0.05)
		#scheduler = torch.optim.lr_scheduler.StepLR(opt_Adam, step_size=10, gamma=0.1)

		NUM_ITER = 250
		best_iter = None
		best_loss = 1e5 # initialize as a big number
		best_out = None

		for iter_num in range(NUM_ITER):

			scale = torch.clamp(scale_og,0.95,1.05)
			x_var = torch.mul(scale,torch.cuda.FloatTensor(target_obs[:,0]))
			y_var = torch.mul(scale,torch.cuda.FloatTensor(target_obs[:,1]))
			z_var = torch.mul(scale,torch.cuda.FloatTensor(target_obs[:,2]))

			x_final = torch.cuda.FloatTensor(PCL_except_obs[:,0])
			y_final = torch.cuda.FloatTensor(PCL_except_obs[:,1])
			z_final = torch.cuda.FloatTensor(PCL_except_obs[:,2])
			i_final = torch.cuda.FloatTensor(PCL_except_obs[:,3])

			x_final = torch.cat([x_final,x_var],dim = 0)
			y_final = torch.cat([y_final,y_var],dim = 0)
			z_final = torch.cat([z_final,z_var],dim = 0)
			i_final = torch.cat([i_final,i_var],dim = 0)

			grids = xyz2grid(x_final, y_final, z_final)
			FM = grid2feature(grids)


			PCL = torch.stack([x_final,y_final,z_final,i_final]).permute(1,0).cpu().detach().numpy()
			PCLConverted = mapPointToGrid(PCL)
			featureM = generateFM(PCL, PCLConverted)
			# featureM = inputTransformation.MatrixEstimation(featureM,0.9,method = 'usvt')
			featureM = np.array(featureM).astype('float32')
			# featureM.tofile('featureM.bin')
			featureM = torch.cuda.FloatTensor(featureM)
			featureM = featureM.view(1,6,672,672)
			featureM.requires_grad = True

			FM[0,[3,4],:,:] = featureM[0,[3,4],:,:]

			###### INITIALIZE TRANSFORMATION PARAM #########
			# theta = torch.tensor(0.0)
			# translation = torch.tensor(0.0)
			# scale1 = torch.tensor(1.0)
			# adv_height = torch.tensor(1.0)

			########## Dummy Spoofed #############
			# spoofed = torch.zeros_like(featureM)
			# spoofed[:,:2,:,:] = -5.0
			######################################
			# FM[0,[3,4],:,:] = featureM[0,[3,4],:,:]
			# FM_numpy = FM.detach().cpu().numpy()
			# FM_numpy.tofile('featureM_prox.bin')

			# #####FOR SPOOFING ATTACK############
			# final = combineFeatureM(FM,spoofed)
			# final.retain_grad()
			# outputPytorch = pytorchModels(final)
			# transformation(featureM.cpu(),theta,translation,scale1,adv_height)

			# injectedFeatureM = generateFM(PCL, PCLConverted)

			############ DEFINE AND CALCULATE LOSS ###########
			opt_Adam.zero_grad()
			outputPytorch = pytorchModels(FM)
			outputPytorch_hard = pytorchModels(featureM)
			lossValue,loss_object,loss_distance = loss.lossPassiveAttack(outputPytorch,x_var,y_var,z_var,scale) 
			lossValue_hard,_,_ = loss.lossPassiveAttack(outputPytorch_hard,x_var,y_var,z_var,scale) 
			print('Iteration {} of {}: loss={} object_loss={} distance_loss={} loss diff={}'.format(iter_num,NUM_ITER,lossValue,loss_object,loss_distance,lossValue_hard - lossValue))
			# print('{} {}'.format(lossValue, lossValue_hard))
			if lossValue < best_loss:
				best_loss = lossValue
				best_scale = scale.clone()
				best_iter = iter_num
				best_out = outputPytorch_hard
			lossValue.backward()
			# print(scale.grad.data)
			opt_Adam.step()
			# target_center = (330,336)
			# normal_mask = torch.tensor(generateNormalMask(target_center),dtype=torch.float32).cuda()
			# loss = torch.add(torch.mul(-outputPytorch[2],outputPytorch[1]),1)
			# loss = torch.mul(loss, normal_mask)
			# loss = torch.sum(loss)
			# pytorchModels.zero_grad()
			# loss.backward()
			# data_grad_x = x_var.grad.data
			##################################################


		#print(outputPytorch)

		# #compare with Caffe model
		# caffeModels = generateCaffe(protofile, weightfile)
		# caffeModels.blobs['data'].reshape(1, 6, height_, width_)
		# caffeModels.blobs['data'].data[...] = np.asarray(featureM.cpu())
		# outputCaffe = caffeModels.forward()
		# print(outputCaffe)

		#summary(pytorchModels, (6,672,672))

		# layer_names = pytorchModels.models.keys()
		# blob_names = outputs

		# print('------------ Output Difference ------------')
		# for idx,blob_name in enumerate(blob_names):
		# 	pytorch_data = outputPytorch[idx].data.cpu().numpy()
		# 	caffe_data = caffeModels.blobs[blob_name].data
		# 	diff = abs(pytorch_data - caffe_data).sum()
		# 	print('%-30s pytorch_shape: %-20s caffe_shape: %-20s output_diff: %f' % (blob_name, pytorch_data.shape, caffe_data.shape, diff/pytorch_data.size))
		print('best iter: {}'.format(best_iter))
		best_scale.cpu().detach().numpy().tofile('best_scale.bin')
		x_distord = torch.mul(best_scale,torch.cuda.FloatTensor(target_obs[:,0]))
		y_distord = torch.mul(best_scale,torch.cuda.FloatTensor(target_obs[:,1]))
		z_distord = torch.mul(best_scale,torch.cuda.FloatTensor(target_obs[:,2]))
		#torch.stack([x_distord,y_distord,z_distord,i_var]).permute(1,0).cpu().detach().numpy().tofile('distord_obs.bin')
		######################


		print ('------------  Pytorch Output ------------')
		obj, label_map = cluster.cluster(best_out[1].data.cpu().numpy(), best_out[2].data.cpu().numpy(), 
			best_out[3].data.cpu().numpy(),best_out[0].data.cpu().numpy(),
			best_out[5].data.cpu().numpy())

		# print ('------------  Caffe Output ------------')

		# obj2 = cluster.cluster(caffeModels.blobs['category_score'].data, caffeModels.blobs['confidence_score'].data, 
		# 	caffeModels.blobs['height_pt'].data,caffeModels.blobs['instance_pt'].data,
		# 	caffeModels.blobs['class_score'].data)
		best_out[1].data.cpu().numpy().tofile('./objectness.bin')
		

		####### PROCESS 2D TO 3D #############
		obstacle, cluster_id_list = twod2threed(obj, label_map, PCL, PCLConverted)

		for obs in obstacle:
			print(obs.numPT())
			# if obs.numPT() == 1463:
			# 	target_obs = np.array(obs.getPCL())
			# 	target_obs.tofile('./target_obs.bin') #### HACK THE DATA

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


if __name__ == '__main__':
	gradattack()

