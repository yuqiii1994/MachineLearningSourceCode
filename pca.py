# principle component analysis

import numpy as np
from numpy.linalg import svd

def myPCA(X, retain_rate=0.9):
    # Covariance Matrix
    n, m = X.shape
    co_ma = 1/n * np.matmul(X.T, X)

    U, s, _ = svd(co_ma)

    total_s = np.sum(s) # s already in descending order
    retain_info = np.array([np.sum(s[:i+1])/total_s for i in range(m)])
    retain_info = retain_info[retain_info>retain_rate]
    retain_info_len = len(retain_info)

    pca_X = np.matmul(U[:retain_info_len], X.T)
    return pca_X.T

if __name__=='__main__':
    np.random.seed(10)
    # samples by row and features by col
    X = np.random.rand(10,5)
    print('original matrix:')
    print(X)
    print('result:')
    print(myPCA(X))
