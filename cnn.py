import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

class cnn(object):

    def __init__(self, X, y, weight_size=(64, 64), stride_step=1, depth=3, max_iter=10):
        assert len(X.shape) == 3

        self.weight_size = weight_size
        self.stride_step = stride_step
        self.depth = depth

        padded_X = self._zeroPadding(X)
        padded_X = padded_X.reshape(padded_X.shape[0], padded_X.shape[1], padded_X.shape[2], 1) # to add one more dim to indicate img depth

        for iter in range(max_iter):
            self._convLayerForward(padded_X, depth=self.depth, kernel_size=self.weight_size)

    def _convLayerForward(self, prev_X, depth=None, kernel_size=None):
        prev_depth = prev_X.shape[3]

    def _maxPoolingLayerForward(self, prev_X, depth=None, pooling_size=None):
        pass

    def _fullConectLayerForward(self, prev_X, num_neurons=100):
        pass

    def _zeroPadding(self, X):
        # to fill margins with zeros
        n_samples, img_width, img_height = X.shape
        padded_width = int(self.weight_size[0]/2)
        padded_height = int(self.weight_size[1]/2)

        newX = np.zeros((n_samples, padded_width*2+img_width, padded_height*2+img_height))
        newX[:, padded_width:img_width+padded_width, padded_height:img_height+padded_height] = X

        return newX


def database_generate(data_volume=5):
    """
    Generate gaussian distribution abided data samples for classification, with shape of
    (n_samples, img_width, img_height) and all values ranging from 0 to 255. Images are
    consisted of either one filled triangle or one filled square.
    """
    img_width = 200
    img_height = 200
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