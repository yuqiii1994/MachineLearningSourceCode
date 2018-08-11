# Linear Regression

import numpy as np 
import matplotlib.pyplot as plt

class lr(object):

    def __init__(self, X, y):

        self.fit(X, y)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.hstack((np.ones(n_samples).reshape(n_samples, 1), X))

        Phi = np.matmul(X.T, X)
        b = np.matmul(X.T, y).reshape(n_features+1, 1)

        self.w = np.matmul(np.linalg.inv(Phi), b)
        self.b = 0.5

    def predict(self, X, y=None):
        n_samples, n_features = X.shape
        X = np.hstack((np.ones(n_samples).reshape(n_samples, 1), X))

        if y is None:
            return np.matmul(self.w, X) + self.b
        self.w = np.atleast_2d(self.w)
        y_predict =  np.sum(self.w * X.T, axis=0)
        absError = np.abs(y - y_predict)/n_samples

        return y_predict, absError

    def predict_and_plot(self, X, y):
        # plot assumes X only has one dimension with corresponding one numerical output for each observation
        plt.scatter(X[:, 0], y, c='b', alpha=0.5) 
        y_predict, absError = self.predict(X, y)
        plt.plot(X[:, 0], y_predict, "k")
        plt.title("A Simple Linear Regression Experiment")

        plt.show()

def database_generate(n_samples=100):
    noise_scale = 0.1 * n_samples # random noise, larger will make prediction harder

    X = np.arange(float(n_samples)).reshape(n_samples, 1)
    y = np.arange(float(n_samples)) + 1 + np.random.rand(n_samples)*noise_scale

    return X, y

def main():
    X, y = database_generate()
    lr_obj = lr(X, y)
    lr_obj.predict_and_plot(X, y)

if __name__ == '__main__':
    main()