#k-means

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class k_means(object):

    def __init__(self, X, y, n_neighborhoods=2, max_iter=50):
        self.n_neighborhoods = n_neighborhoods

        n_samples, n_features = X.shape
        self.n_features = n_features
        self.n_samples = n_features

        max_val_x = np.max(X, axis=0)
        min_val_x = np.min(X, axis=0)
        range_val_x = max_val_x - min_val_x
        centriod = np.random.random([n_neighborhoods, n_features]) * range_val_x
        
        y_predict = np.zeros(n_samples)

        for _ in range(max_iter):
            for each_sample in range(n_samples):
                distance_array = np.zeros(n_neighborhoods)
                for each_centriod in range(n_neighborhoods):
                    distance_array[each_centriod] = self._EuclideanDistance(X[each_sample], centriod[each_centriod])
                y_predict[each_sample] = np.argmin(distance_array)

            for each_neighborhood in range(n_neighborhoods):
                each_neighborX = X[y_predict==each_neighborhood]
                centriod[each_neighborhood] = self._calculateCenter(each_neighborX)

        self.centriod = centriod

    def _EuclideanDistance(self, x, y):
        return np.sum(np.abs(x-y))

    def _calculateCenter(self, X):
        return np.sum(X, axis=0) / X.shape[0]

    def predict(self, X, y=None):
        n_samples = X.shape[0]
        n_neighborhoods = self.n_neighborhoods
        centriod = self.centriod
        y_predict = np.zeros(n_samples)

        for each_sample in range(n_samples):
            distance_array = np.zeros(n_neighborhoods)
            for each_centriod in range(n_neighborhoods):
                distance_array[each_centriod] = self._EuclideanDistance(X[each_sample], centriod[each_centriod])
            y_predict[each_sample] = np.argmin(distance_array)

        if y is None:
            return y_predict

        return y_predict, accuracy_score(y, y_predict)

    def predict_and_plot(self, X, y):
        centriod = self.centriod

        color_array = ['r' if y[i] == 0 else 'b' for i in range(len(y))]
        plt.scatter(X[:, 0], X[:, 1], c=color_array, alpha=0.5) # only the first two dimensions are shown
        plt.scatter(centriod[:, 0], centriod[:, 1], color='k')

        for each_centroid in range(self.n_neighborhoods):
            plt.text(centriod[each_centroid, 0], centriod[each_centroid, 1], 
            "Centroid (" + str(centriod[each_centroid, 0])[:5] + \
            ", " + str(centriod[each_centroid, 1])[:5] + ")")

        y_predict, accuracy = self.predict(X, y)
        plt.title("A K-means Experiment\n", weight='bold')

        plt.show()

        return y_predict, accuracy


def database_generate(data_volume=100, sigma=2):
    data_volume_each_class = int(data_volume/2)
    mu_class1_dim1 = 7
    mu_class1_dim2 = 3
    X_class1_dim1 = np.random.normal(mu_class1_dim1, sigma, size=data_volume_each_class)
    X_class1_dim2 = np.random.normal(mu_class1_dim2, sigma, size=data_volume_each_class)
    X_class1 = np.concatenate((np.atleast_2d(X_class1_dim1), np.atleast_2d(X_class1_dim2)), axis=1)
    y_class1 = np.zeros(data_volume_each_class)
    mu_class2_dim1 = 3
    mu_class2_dim2 = 7
    X_class2_dim1 = np.random.normal(mu_class2_dim1, sigma, size=data_volume_each_class)
    X_class2_dim2 = np.random.normal(mu_class2_dim2, sigma, size=data_volume_each_class)
    X_class2 = np.concatenate((np.atleast_2d(X_class2_dim1), np.atleast_2d(X_class2_dim2)), axis=1)
    y_class2 = np.ones(data_volume_each_class)

    X = np.concatenate((X_class1, X_class2)).T
    y = np.concatenate((y_class1, y_class2))

    if X.shape[0] != y.shape[0]:
        return None

    return X, y

if __name__=="__main__":
    X, y = database_generate()
    k_means_obj = k_means(X, y) # two dim input space with two neighborhoods
    k_means_obj.predict_and_plot(X, y)