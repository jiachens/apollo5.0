from __future__ import division
import numpy as np 
import torch
import sys

def transformation(featureM,theta,translation,scale,adv_height):

	center = (336,336)

	featureM = featureM[:,:5,:,:]
	featureM = featureM.permute(0,2,3,1)

	# angle = angle * 180.
	# translation = translation * 256.
	# adv_height = adv_height * 5.

	# theta = np.pi/180.*angle

	trans = torch.tensor([[1.,0,center[0]],[0,1,center[1]],[0,0,1]], 
		dtype = torch.float)
	trans_r = torch.tensor([[1,0,-center[0]],[0,1,-center[1]],[0,0,1]], 
		dtype = torch.float)
	rot = torch.tensor([[torch.cos(theta),-torch.sin(theta),translation],
		[torch.sin(theta),torch.cos(theta),0],[0,0,1]], dtype = torch.float, requires_grad = True)
	scale_mt = torch.tensor([[torch.div(1,scale),0,0],[0,torch.div(1,scale),0],[0,0,1]], 
		dtype = torch.float, requires_grad = True)
	rot_matrix = torch.matmul(torch.matmul(trans,torch.matmul(scale_mt,rot)),trans_r)

	ix,iy = torch.meshgrid(torch.arange(0,672), torch.arange(0,672))
	ix = torch.reshape(ix,[-1])
	iy = torch.reshape(iy,[-1])
	ixy = torch.stack([ix,iy,torch.ones_like(ix)]).float()
	rxy = torch.matmul(rot_matrix,ixy) - ixy
	rxy = torch.transpose(rxy,0,1)
	rxy = torch.reshape(rxy,[1,3,672,672])
	flows = rxy[:,:2,:,:]

	basegrid = torch.stack(
					torch.meshgrid(torch.arange(0,672), torch.arange(0,672))
				)
	batched_basegrid = basegrid.repeat(1,1,1,1)
	sampling_grid = batched_basegrid.float() + flows

	sampling_grid_x = torch.clamp(
		sampling_grid[:, 1], 0., 671.)
	sampling_grid_y = torch.clamp(
		sampling_grid[:, 0], 0., 671.)

	x0 = torch.floor(sampling_grid_x).int()
	x1 = x0 + 1
	y0 = torch.floor(sampling_grid_y).int()
	y1 = y0 + 1

	x0 = torch.clamp(x0, 0, 672 - 2)
	x1 = torch.clamp(x1, 0, 672 - 1)
	y0 = torch.clamp(y0, 0, 672 - 2)
	y1 = torch.clamp(y1, 0, 672 - 1)

	b = torch.reshape(
			torch.arange(0, 1), (1, 1, 1)
		).repeat(1,672,672).int()

	idx1, idx2, idx3 = torch.stack([b, y0, x0], 3).long().chunk(3, dim=3)
	Ia = featureM[idx1, idx2, idx3].squeeze().view(1, 672, 672, 5)
	idx1, idx2, idx3 = torch.stack([b, y1, x0], 3).long().chunk(3, dim=3)
	Ib = featureM[idx1, idx2, idx3].squeeze().view(1, 672, 672, 5)
	idx1, idx2, idx3 = torch.stack([b, y0, x1], 3).long().chunk(3, dim=3)
	Ic = featureM[idx1, idx2, idx3].squeeze().view(1, 672, 672, 5)
	idx1, idx2, idx3 = torch.stack([b, y1, x1], 3).long().chunk(3, dim=3)
	Id = featureM[idx1, idx2, idx3].squeeze().view(1, 672, 672, 5)

	x0 = x0.float()
	x1 = x1.float()
	y0 = y0.float()
	y1 = y1.float()

	wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
	wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
	wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
	wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

	wa = torch.unsqueeze(wa, dim=3)
	wb = torch.unsqueeze(wb, dim=3)
	wc = torch.unsqueeze(wc, dim=3)
	wd = torch.unsqueeze(wd, dim=3)

	perturbed_image = sum([wa * Ia, wb * Ib, wc * Ic, wd * Id])

	print(perturbed_image)

	trans_cnt = perturbed_image[:,:,:,2] / (scale*scale)
	trans_round = torch.round(trans_cnt)
	#round_loss = torch.norm(trans_round - trans_cnt)
	trans_cnt = trans_round

	trans_h = perturbed_image[:,:,:,1]
	trans_h = torch.mul(trans_h,adv_height)
	trans_h = torch.clamp(trans_h, -5.0 ,5.0)

	trans_max_h = perturbed_image[:,:,:,0]
	trans_max_h = torch.mul(trans_h,trans_max_h)
	
	trans_mean_int = perturbed_image[:,:,:,4]

	trans_max_int = perturbed_image[:,:,:,3]
	
	condition = torch.eq(trans_cnt, torch.zeros_like(trans_cnt))
	case_true = torch.ones_like(trans_h) * (-5)
	trans_h_min = torch.where(condition, case_true, trans_max_h)


def combineFeatureM(featureM,spoofedFeatureM):

	final_cnt = torch.log(featureM[:,2,:,:] + spoofedFeatureM[:,2,:,:] + 1)

	final_avg_height = torch.mul(torch.mul(featureM[:,1,:,:], featureM[:,2,:,:]) 
						+ torch.mul(spoofedFeatureM[:,1,:,:], spoofedFeatureM[:,2,:,:]), 
						1.0 / (featureM[:,2,:,:] + spoofedFeatureM[:,2,:,:] + sys.float_info.epsilon))


	final_max_height = torch.max(featureM[:,0,:,:], spoofedFeatureM[:,0,:,:]) ## correct

	final_avg_int = torch.mul(torch.mul(featureM[:,4,:,:], featureM[:,2,:,:]) 
						+ torch.mul(spoofedFeatureM[:,4,:,:], spoofedFeatureM[:,2,:,:]), 
						1.0 / (featureM[:,2,:,:] + spoofedFeatureM[:,2,:,:] + sys.float_info.epsilon))

	condition = torch.gt(featureM[:,0,:,:], spoofedFeatureM[:,0,:,:])
	final_max_int = torch.where(condition, featureM[:,3,:,:], spoofedFeatureM[:,3,:,:])
	
	condition = torch.gt(featureM[:,2,:,:] + spoofedFeatureM[:,2,:,:], torch.zeros_like(featureM[:,2,:,:]))
	final_nonempty = torch.where(condition, torch.ones_like(featureM[:,2,:,:]), torch.zeros_like(featureM[:,2,:,:]))


	final = torch.stack([final_max_height,final_avg_height,final_cnt,final_max_int,final_avg_int,final_nonempty], 1)

	return final