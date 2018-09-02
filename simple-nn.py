# This is a simple implementation of neural network without activation func
# h_i = max(W_1 * X^T, 0)
# p_i = W_y * X

import numpy as np

class NN:

    def __init__(self, X, y, num_neurons=3, num_iter=10, alpha=0.5):
        n_samples, n_features = X.shape
        n_output = y.shape[1]

        relu_zeros = np.zeros((n_samples, num_neurons))# for constructing relu for comparison
        W_1 = np.random.rand(num_neurons, n_features)*2-1
        W_y = np.random.rand(n_output, num_neurons)*2-1

        for iter in range(num_iter):
            H_1_tmp = np.matmul(W_1, X.T).T
            relu_mat = np.array([H_1_tmp, relu_zeros])
            H_1 = np.amax(relu_mat, axis=0) # relu
            p_i = H_y = np.matmul(W_y, H_1.T).T # since this is a vanilla nn, only two layers are considered

            delta_p = np.abs(p_i - y) # delta p

            delta_H = np.matmul(delta_p, W_y) # delta h at the layer 1, that this works for linear func
            H_1_copy = np.copy(H_1)
            H_1_copy[H_1_copy>0] = 1 # derivative of relu, unit step
            delta_X_A = np.multiply(delta_H, H_1_copy)
            delta_X = np.matmul(delta_X_A, W_1)

            W_1 += np.sum(delta_X, axis=0)/n_samples * alpha
            W_y += np.sum(delta_H, axis=0)/n_samples * alpha

            print(np.sum(delta_p))


if __name__ == '__main__':
    # to predict y = sin(x)
    val_range = 100
    size_range = 10
    # X can support multi dimensional layer, here only 1 dimensional X is introduced
    X = np.random.randint(val_range, size=size_range).reshape(size_range, 1)
    X = X / max(X) # normalization
    y = np.sin(np.sum(X, axis=1)).reshape(size_range, 1)
    NN_obj = NN(X=X, y=y)
