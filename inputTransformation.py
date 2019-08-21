from __future__ import division
from scipy import ndimage
from scipy.spatial import cKDTree
from fancyimpute import SoftImpute, BiScaler
import numpy as np
from cvxpy import *

def MedianSmooth(FM,size = 3):

	FM_med = []
	for i in range(len(FM) - 1):
		fm_med = ndimage.median_filter(FM[i], size)
		FM_med.append(fm_med)

	FM_med.append(FM[-1])

	return FM_med

def squeeze(FM):
	
	FM_squeeze = []

	max_height_data_ = FM[0]
	mean_height_data_ = FM[1]
	count_data_ = FM[2]
	top_intensity_data_ = FM[3]
	mean_intensity_data_ = FM[4]
	nonempty_data_ = FM[5]

	# max height is from -5 to 5 -> use round()
	# mean height is from -5 to 5 -> use round()
	# top intensity has 255 levels -> 10 levels
	#

def KNNsmooth(PCL,epsilon = 0.2):

	tree = cKDTree(PCL[:,:3], leafsize=PCL.shape[0]+1)
	distances,_ = tree.query(PCL[:,:3], k=4, n_jobs = 4)
	mean = np.mean(distances[:,1:], axis=1)
	deleteline = []
	thres = np.mean(mean) + epsilon * np.std(mean)
	for i in range(PCL.shape[0]):
		if mean[i] > thres:
			deleteline.append(i)

	PCL = np.delete(PCL, deleteline, 0)
	return PCL

def Randomsmooth(PCL,p = 0.8):

	deleteline = np.random.randint(low = 0, high = PCL.shape[0], size = int(round((1 - p) * PCL.shape[0])))
	PCL = np.delete(PCL, deleteline, 0)
	return PCL


def MatrixEstimation(FM, p, method):

	convertedFM = transFromFM(FM)

	if method == 'usvt':
		ME_FM = usvt(convertedFM,p)
	elif method == 'softimp':
		ME_FM = softimp(convertedFM,p)
	elif method == 'nucnorm':
		ME_FM = nucnorm(convertedFM,p)
	
	return restore(ME_FM,FM)

def transFromFM(FM):

	convertedFM = []
	
	for i in range(5):
		#FM[i] = FM[i].astype('float32')
		scale = np.amax(FM[i]) - np.amin(FM[i])
		convertedFM.append((FM[i] - np.amin(FM[i])) * 2 / scale - 1)
	
	return np.array(convertedFM)

def restore(ME_FM,FM):

	New_FM = []

	for i in range(5):
		scale = np.amax(FM[i]) - np.amin(FM[i])
		New_FM.append((ME_FM[i] + 1) * scale / 2 + np.amin(FM[i]))
	
	New_FM.append(FM[-1])
	return New_FM

def usvt(img, maskp):
	"""Preprocessing with universal singular value thresholding (USVT) approach.
	Data matrix is scaled between [-1, 1] before matrix estimation (and rescaled back after ME)
	[Chatterjee, S. et al. Matrix estimation by universal singular value thresholding. 2015.]
	:param img: original image
	:param maskp: observation probability of each entry in mask matrix
	:return: preprocessed image
	"""
	c, h, w = img.shape

	mask = np.random.binomial(1, maskp, h * w).reshape(h, w)
	p_obs = len(mask[mask == 1]) / (h * w)

	outputs = np.zeros((c, h, w))
	for channel in range(c):
		u, sigma, v = np.linalg.svd(img[channel, :, :] * mask)
		S = np.zeros((h, h))
		sigma = np.concatenate((sigma, np.zeros(h - len(sigma))), axis=0)
		for j in range(int(0.8 * h)):
			S[j][j] = sigma[j]

		W = np.dot(np.dot(u, S), v) / p_obs
		W[W < -1] = -1
		W[W > 1] = 1
		outputs[channel, :, :] = W

	return outputs

def softimp(img, maskp):
	"""Preprocessing with Soft-Impute approach.
	Data matrix is scaled between [-1, 1] before matrix estimation (and rescaled back after ME)
	[Mazumder, R. et al. Spectral regularization algorithms for learning large incomplete matrices. 2010.]
	:param img: original image
	:param maskp: observation probability of each entry in mask matrix
	:return: preprocessed image
	"""
	c, h, w = img.shape

	mask = np.random.binomial(1, maskp, h * w).reshape(h, w).astype(float)
	mask[mask < 1] = np.nan

	outputs = np.zeros((c, h, w))
	for channel in range(c):
		mask_img = img[channel, :, :] * mask
		W = SoftImpute(verbose=False).fit_transform(mask_img)
		W[W < -1] = -1
		W[W > 1] = 1
		outputs[channel, :, :] = W

	return outputs

def nucnorm(img, maskp):
	"""Preprocessing with nuclear norm algorithm.
	Data matrix is scaled between [-1, 1] before matrix estimation (and rescaled back after ME)
	[Cands, J. and Recht, B. Exact matrix completion via convex optimization. 2009.]
	:param img: original image
	:param maskp: observation probability of each entry in mask matrix
	:return: preprocessed image
	"""
	c, h, w = img.shape

	mask = np.random.binomial(1, maskp, h * w).reshape(h, w)
	outputs = np.zeros((c, h, w))
	for channel in range(c):
		W = nuclear_norm_solve(img[channel, :, :], mask, 1)
		W[W < -1] = -1
		W[W > 1] = 1
		outputs[channel, :, :] = W

	return outputs


def nuclear_norm_solve(A, mask, mu):
	"""Nuclear norm minimization solver.
	:param A: matrix to complete
	:param mask: matrix with entries zero (if missing) or one (if present)
	:param mu: control trade-off between nuclear norm and square loss
	:return: completed matrix
	"""
	X = Variable(shape=A.shape)
	objective = Minimize(mu * norm(X, "nuc") + sum_squares(multiply(mask, X-A)))
	problem = Problem(objective, [])
	problem.solve(solver=SCS)
	return X.value

def totalVarianceMinimization(FM):
	pass
