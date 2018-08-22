# Gaussian Mixture Models
# unsupervised learning

import math
import numpy as np 
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score

class gmm(object):

    def __init__(self, X, y, n_gau=None, epsilon=1e-5, max_iter=50):
        n_samples, n_features = X.shape
        classes = np.unique(y)

        if n_gau is not None:
            self.n_gau = n_gau # the number of gaussian components, if None then equal to num of classes
        else:
            self.n_gau = len(classes)
        self.epsilon = epsilon # error gap threshold, below which the algorithm will terminate
        self.max_iter = max_iter # max iteration

        self.__initModel(X, y)
 
        gauNormDensity = np.zeros((n_samples, self.n_gau), dtype=float)
        logLikelihood_old = 0
        for iter in range(n_gau):
            idx_x_count = 0
            for each_sample in X:
                for each_gau_idx in range(self.n_gau): 
                    gauNormDensity[idx_x_count][each_gau_idx] = multivariate_normal.pdf(each_sample, mean=self.mean[each_gau_idx], cov=self.cov[each_gau_idx])
                idx_x_count += 1

            # current log likelihood calculation given the equation: 
            # SUM_i ( log ( MixedProbs * NormDensity_i ) ),
            # where i is an index for samples and MixedProbs * NormDensity_i states overall expectation of multiple Gaussian components.
            logLikelihood_current = np.sum(np.log(np.matmul(np.atleast_2d(self.mixedProbs), gauNormDensity.T)))

            # terminate the running if it is converged
            if np.abs(logLikelihood_current - logLikelihood_old) < self.epsilon:
                break
            logLikelihood_old = logLikelihood_current

            responsibilities = np.zeros((n_samples, self.n_gau), dtype=float) # to clear responsibilities at each iter
            # E-step starts for calculation of responsibilities
            for each_sample_idx in range(n_samples):
                normalizer = np.dot(gauNormDensity[each_sample_idx], self.mixedProbs)
                responsibilities[each_sample_idx] = np.multiply(self.mixedProbs, gauNormDensity[each_sample_idx])/normalizer

            # M-step starts for calculation of means and covariances
            responsibilities = responsibilities.T # transposition here is required as means and covs are calculated by all samples in individual feature
            for each_gau_idx in range(self.n_gau):
                normalizer = np.sum(responsibilities[each_gau_idx])

                # Update mean
                self.mean[each_gau_idx] = np.dot(responsibilities[each_gau_idx], X) / normalizer

                # Update covariance
                difference = X - np.tile(self.mean[each_gau_idx], (n_samples, 1))
                self.cov[each_gau_idx] = np.dot(np.multiply(responsibilities[each_gau_idx].reshape(n_samples,1), difference).T, difference) / normalizer

                # Update mixing probabilities
                self.mixedProbs[each_gau_idx] = normalizer / n_samples


    def predict(self, X, y=None):
        y_predict = np.zeros_like(y)

        distances = np.empty(self.n_gau, np.float)
        cov_inverses = [np.linalg.inv(cov) for cov in self.cov]
        for idx, each_sample in enumerate(X):

            # Compute Mahanalobis distances from each mixture
            for j in np.arange(self.n_gau):
                distances[j] = np.dot(np.dot((each_sample - self.mean[j]).T, cov_inverses[j]), each_sample - self.mean[j])

            # Find index of the minimum distance, and assign to a cluster
            y_predict[idx] = np.argmin(distances)

        if y is None:
            return y_predict

        accu = accuracy_score(y, y_predict)
        return y_predict, accu
                
    # set init parameters (mean and covariance) for multi-dimensional gaussian mixture model
    # the init model method is random, that samples are randomly selected with their means and covs calculated
    def __initModel(self, X, y):
        n_samples, n_features = X.shape
        shuffle_idx = np.arange(n_samples)
        np.random.shuffle(shuffle_idx)
        shuffle_cutoff_idx = np.linspace(0, n_samples, num=self.n_gau+1, dtype=int)

        mean = np.zeros((self.n_gau, n_features))
        cov = np.zeros((self.n_gau, n_features, n_features))

        for each_gau_idx in range(self.n_gau):
            mean[each_gau_idx] = np.mean(X[shuffle_idx[shuffle_cutoff_idx[each_gau_idx]: shuffle_cutoff_idx[each_gau_idx+1]]], axis=0)
            cov[each_gau_idx] = np.cov(X[shuffle_idx[shuffle_cutoff_idx[each_gau_idx]: shuffle_cutoff_idx[each_gau_idx+1]]].T)

        self.mean = mean
        self.cov = cov

        # the init probability for each gaussian component is equal, that assumes each gaussian component is of equal weight
        # it is reasonable to assume that each gaussian component has equal prediction weight of 1/n_gaussian_components as samples are randomly selected
        self.mixedProbs = np.array([1/self.n_gau] * self.n_gau)

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
    gmm_obj = gmm(X, y, n_gau=2) 
    y_predict, accu = gmm_obj.predict(X, y)
    print("predicted values: ")
    print(y_predict)
    print("accuracy: ")
    print(accu)