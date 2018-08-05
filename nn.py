"""
This is a vanilla neural network
"""
import numpy as np
import sys

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def sigmoid_derivative(output):
    return output * (1 - output)

def tanh_derivative(output):
    return 1 - np.tanh(output)*np.tanh(output)

class NN(object):

    def __init__(self, X=None, y=None, num_neurons=100, num_iter=2000, alpha=0.05, print_training=False):

        num_samples = X.shape[0]
        num_X_dim = X.shape[1]

        num_y_dim = y.shape[1]

        W_1 = np.random.random((num_neurons, num_X_dim))
        W_y = np.random.random((num_y_dim, num_neurons))

        A_1 = np.zeros((num_neurons, num_samples))
        A_y = np.zeros((num_samples, num_y_dim))

        error_y = np.zeros((num_samples, num_y_dim))
        delta_y = np.zeros((num_neurons, num_y_dim))

        for iter in range(num_iter):
            A_1 = tanh(np.matmul(X, W_1.T))
            A_y = sigmoid(np.matmul(A_1, W_y.T))

            error_y = y - A_y
            ave_error_y = np.sum(error_y, axis=0)/num_samples

            delta_y = error_y * sigmoid_derivative(A_y)

            W_y += np.matmul(delta_y.T, A_1) * alpha

            delta_1 = error_y * tanh_derivative(A_1)
            W_1 += np.matmul(delta_1.T, X) * alpha

            if print_training:
                print("Average error: " + str(ave_error_y) + " at iteration " + str(iter))

        self.W_1 = W_1
        self.W_y = W_y
        self.num_samples = num_samples
        self.num_X_dim = num_X_dim
        self.num_y_dim = num_y_dim

    def predict(self, X, y=None, print_result=False):
        W_1 = self.W_1
        W_y = self.W_y

        A_1 = tanh(np.matmul(X, W_1.T))
        A_y = sigmoid(np.matmul(A_1, W_y.T))

        if print_result:
            if y is not None:
                print("True value: " + str(y))
                print("Predicted value: " + str(A_y))
                print("Total error: " + str(sum(y - A_y)))
            else:
                print("Predicted value: " + str(A_y))

        return A_y

if __name__ == '__main__':
    # to predict y = sin(x)
    val_range = 100
    size_range = 5
    # X can support multi dimensional layer, here only 1 dimensional X is introduced
    X = np.random.randint(val_range, size=size_range).reshape(size_range, 1)
    X = X / max(X) # normalization
    y = np.sin(np.sum(X, axis=1)).reshape(size_range, 1)
    NN_obj = NN(X=X, y=y)
    NN_obj.predict(X, y=y, print_result=True)
