# Naive Bayes (Probability via Gaussian Probability Density Function)

import numpy as np
import math
from sklearn.metrics import accuracy_score


def GaussianPDF(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

class nb(object):
    """
    For a two dimensional matrix input corresponding to binary labels:
    P(C_k | x_0, x_1) = P(C_k) * P(x_0, x_1 | C_k) / P(x_0, x_1)
                      = P(C_k) * MUL_i( p(x_i | C_k) ) / Z
                      = P(C_k) * P(x_0 | C_k) * P(x_1 | C_k) / P(x_0, x_1)
                      = P(C_k) * P(x_0 | C_k) * P(x_1 | C_k) / SUM_k ( P(C_k) * P(x_0, x_1 | C_k) )

    Z should be 1 as a constant since all x_0 and x_1 observations of various class labels sum up to 1
    """

    def __init__(self, X, y, showGaussParam=False):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        # nb_struct: {class_label: [feature_idx_dict: mean, std_deviation]}
        # to reference: nb_struct[class_label][feature_idx]
        self.nb_struct = {}
        self.classes = classes
        self.n_classes = len(classes)
        self.n_features = n_features
        self.ClassProb = {} # P( C_k )

        for each_class in classes:
            sample_idx = y==each_class
            self.ClassProb[each_class] = len(X[sample_idx])/n_samples
            self.nb_struct[each_class] = []
            for each_feat_idx in range(n_features):
                mean_temp = np.mean(X[sample_idx, each_feat_idx])
                std_temp = np.std(X[sample_idx, each_feat_idx])
                temp = {'mean': mean_temp, 'std': std_temp}
                self.nb_struct[each_class].append(temp)

        if showGaussParam:
            print(self.nb_struct)

    # This function only takes one sample per execution
    def calculateClassProb(self, x):
        prob = np.ones([self.n_classes])
        classes = self.classes.astype('int64')
        for each_class in classes:
            prob[each_class] *= self.ClassProb[each_class]
            for each_feat_idx in range(self.n_features):
                meanVal = self.nb_struct[each_class][each_feat_idx]['mean']
                stdVal = self.nb_struct[each_class][each_feat_idx]['std']
                prob[each_class] *= GaussianPDF(x[each_feat_idx], meanVal, stdVal)

        return np.argmax(prob).astype('int64')

    def predict(self, X, y=None):
        y_predict = np.zeros_like(y)
        for idx, each_sample in enumerate(X):
            y_predict[idx] = self.calculateClassProb(each_sample)
        if y is None:
            return y_predict
        accu = accuracy_score(y, y_predict)

        return y_predict, accu

def database_generate(data_volume=100, sigma=3):
    # sigma controls divations that higher makes prediction harder
    data_volume_each_class = int(data_volume/2)
    mu_class1_dim1 = 10
    mu_class1_dim2 = 5
    X_class1_dim1 = np.random.normal(mu_class1_dim1, sigma, size=data_volume_each_class)
    X_class1_dim2 = np.random.normal(mu_class1_dim2, sigma, size=data_volume_each_class)
    X_class1 = np.concatenate((np.atleast_2d(X_class1_dim1), np.atleast_2d(X_class1_dim2)), axis=1)
    y_class1 = np.zeros(data_volume_each_class, dtype=int)
    mu_class2_dim1 = 20
    mu_class2_dim2 = 10
    X_class2_dim1 = np.random.normal(mu_class2_dim1, sigma, size=data_volume_each_class)
    X_class2_dim2 = np.random.normal(mu_class2_dim2, sigma, size=data_volume_each_class)
    X_class2 = np.concatenate((np.atleast_2d(X_class2_dim1), np.atleast_2d(X_class2_dim2)), axis=1)
    y_class2 = np.ones(data_volume_each_class, dtype=int)

    X = np.concatenate((X_class1, X_class2)).T
    y = np.concatenate((y_class1, y_class2))

    if X.shape[0] != y.shape[0]:
        return None

    return X, y

if __name__=="__main__":
    X, y = database_generate()
    nb_obj = nb(X, y) 
    y_predict, accu = nb_obj.predict(X, y)
    print("predicted val:\n" + str(y_predict))
    print("accuracy val:\n" + str(accu))
