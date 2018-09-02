# convolution neural network, not yet finished

import numpy as np
from scipy import signal

def relu(x, isConv=True):
	if isConv:
		img_width, img_height, img_depth = x.shape
		for each_row_pixel in range(img_width):
			for each_col_pixel in range(img_height): 
				for each_depth_idx in range(img_depth):
					if x[each_row_pixel, each_col_pixel, each_depth_idx] < 0:
						x[each_row_pixel, each_col_pixel, each_depth_idx] = 0
	else: # when it is for dense layer
		x[x < 0] = 0
	return x

def reluDerivative(x, isConv=True):
	if isConv:
		pass
	else:
		ret_x = np.zeros(len(x))
		ret_x[x>0] = 1
		return ret_x

class cnn(object):

	def __init__(self, X, y, weight_size=64, depth=32, dense_neurons=10, alpha=0.1, max_iter=10):
		if len(X.shape) > 3: # image samples have 3 channels instead of just being grey
			self.isColourful = True
		else: # image samples are grey
			self.isColourful = False
			X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

		self.alpha = alpha

		# label dimensions
		try:
			self.y_dim = len(y[0]) 
		except:
			self.y_dim = 1
			y = y.reshape(X.shape[0], self.y_dim)

		# to init filters and configure weights before moving on
		self.weight_window = []
		self.weight_window.append(2*np.random.rand(weight_size, weight_size, X.shape[3], depth)-1)
		self.weight_window.append(2*np.random.rand(weight_size, weight_size, depth, depth)-1)

		self.weight_dense = []

		# run iterations
		convLayer1_res, convLayer1_A = self.convolutionLayerForward(X, convLayerIdx=0, depth=depth)
		maxPoolLayer1_res = self.maxPoolingLayerForward(convLayer1_res, poolingSize=16)
		convLayer2_res, convLayer2_A = self.convolutionLayerForward(maxPoolLayer1_res, convLayerIdx=1, depth=depth)
		maxPoolLayer2_res = self.maxPoolingLayerForward(convLayer2_res, poolingSize=4)
		denseLayer1_res, denseLayer1_A = self.denseLayerForward(maxPoolLayer2_res, num_neurons=5)
		denseLayer2_res, denseLayer2_A = self.denseLayerForward(denseLayer1_res, num_neurons=self.y_dim, denseLayer=1, isPrevConv=False)

		error_y = y - denseLayer2_res # output error

		self.denseLayerBack(denseLayer2_res, denseLayer2_A)

	def convolutionLayerForward(self, prevX, weight_size=64, convLayerIdx=0, depth=6):

		outputX = np.zeros((prevX.shape[0], prevX.shape[1], prevX.shape[2], prevX.shape[3], depth))
		n_samples, img_width, img_height, prev_depth = prevX.shape 
		# convolution
		for each_sample_idx in range(n_samples):
			print("idx" + str(each_sample_idx))
			for each_prev_depth_idx in range(prev_depth):
				for each_depth_idx in range(depth):
					outputX[each_sample_idx, :, :, each_prev_depth_idx, each_depth_idx] = \
						signal.convolve2d(prevX[each_sample_idx, :, :, each_prev_depth_idx], self.weight_window[convLayerIdx][:, :, each_prev_depth_idx, each_depth_idx], mode='same')
		# activation
		outputX = outputX.reshape(n_samples, img_width, img_height, prev_depth*depth)
		outputA = np.zeros((n_samples, img_width, img_height, prev_depth*depth))
		for each_sample_idx in range(n_samples):
			outputA[each_sample_idx, :, :, :] = relu(outputX[each_sample_idx, :, :, :])

		return outputA, outputX

	def maxPoolingLayerForward(self, prevX, poolingSize=16):
		n_samples, img_width, img_height, prev_depth = prevX.shape
		width_steps = np.arange(0, img_width, poolingSize).astype('int32')
		height_steps = np.arange(0, img_height, poolingSize).astype('int32')
		new_width = len(width_steps)-1
		new_height = len(height_steps)-1

		prev_depth = 1
		outputMaxX = np.zeros((n_samples, new_width, new_height, prev_depth))
		for each_sample_idx in range(n_samples):
			for each_width_step in range(new_width):
				for each_height_step in range(new_height):
					for each_depth_idx in range(prev_depth):
						tmpX = np.ravel(prevX[each_sample_idx, width_steps[each_width_step]: width_steps[each_width_step+1], height_steps[each_height_step]: height_steps[each_height_step+1], each_depth_idx])
						outputMaxX[each_sample_idx, each_width_step, each_height_step, each_depth_idx] = np.amax(tmpX)

		return outputMaxX

	def denseLayerForward(self, prevX, num_neurons=10, denseLayer=0, isPrevConv=True):
		if isPrevConv:
			prevX = prevX.reshape(prevX.shape[0], prevX.shape[1]*prevX.shape[2]*prevX.shape[3])
			if not self.weight_dense: # weights not yet defined
				self.weight_dense.append([2*np.random.rand(prevX.shape[1], num_neurons)-1])
		else:
			assert len(prevX.shape) == 2
			if len(self.weight_dense) == 1:
				self.weight_dense.append([2*np.random.rand(prevX.shape[1], num_neurons)-1])

		outputX = np.matmul(prevX, self.weight_dense[denseLayer][0])
		outputA = np.zeros((prevX.shape[0], num_neurons))
		for each_sample_idx in range(prevX.shape[0]):
			outputA[each_sample_idx, :] = relu(outputX[each_sample_idx, :], isConv=False)

		return outputA, outputX

	def denseLayerBack(self, upperY, actiivationEnergy):
		n_samples, n_labels = upperY.shape
		delta_y = np.zeros((n_samples, n_labels))
		for each_sample_idx in range(n_samples):
			delta_y[each_sample_idx] = reluDerivative(upperY[each_sample_idx])
		np.matmul(upperY, actiivationEnergy) * self.alpha


def database_generate(data_volume=2):
	"""
	Generate gaussian distribution abided data samples for classification, with shape of
	(n_samples, img_width, img_height) and all values ranging from 0 to 255. Images are
	consisted of either one filled triangle or one filled square.
	"""
	img_width = 150
	img_height = 150
	X = np.zeros((data_volume, img_width, img_height))
	y = np.zeros(data_volume)
	whiteNoise = 100 # white noise from 0 to 100
	imgVal = 200 # graphic pixel val ranges form 200 to 255
	changeLabelNode = int(data_volume/2)
	for eachSampleIdx in range(data_volume):
		if eachSampleIdx < changeLabelNode:
			y[eachSampleIdx] = 0 # simulated img is a square
			for width_idx in range(img_width):
				for height_idx in range(img_height):
					if width_idx > img_width/4 and width_idx < img_width*3/4 and height_idx > img_height/4 and height_idx < img_height*3/4:
						X[eachSampleIdx, width_idx, height_idx] = np.random.randint(low=imgVal, high=255)
					else:
						X[eachSampleIdx, width_idx, height_idx] = np.random.randint(low=0, high=whiteNoise)
		else:
			y[eachSampleIdx] = 1 # simulated img is a triangle
			for width_idx in range(img_width):
				for height_idx in range(img_height):
					if width_idx > img_height/2-height_idx and width_idx < img_height/2+height_idx:
						X[eachSampleIdx, width_idx, height_idx] = np.random.randint(low=imgVal, high=255)
					else:
						X[eachSampleIdx, width_idx, height_idx] = np.random.randint(low=0, high=whiteNoise)

	return X, y

if __name__=="__main__":
	X, y = database_generate()
	cnn_obj = cnn(X, y) 