# Decision Tree CART (via Gini Impurity)

import numpy as np 

class dt(object):

    def __init__(self, X, y, max_depth=5):
        self.max_depth = max_depth 

        rootNode = self.constructNode()
        rootNode['content'] = self.splitNode(X, y)
        # Tree built: condition to splid, feature index, tree depth
        self.Tree = [rootNode['content']['cond_record'], rootNode['content']['feat_record'], 0]

        self.growTree(X, y, rootNode, depth=0)

    def calculateGiniImpurity(self, x, y):
        # x should be one dimensional matrix with shape of (n_samples, 1)
        n_samples = len(x)
        classes, class_counts = np.unique(y, return_counts=True)

        gini_temp = 0.0
        for each_class in range(len(classes)):
            p = class_counts[each_class] / n_samples
            gini_temp += p * p
        giniVal = 1 - gini_temp

        return giniVal

    def splitNode(self, X, y):
        n_samples, n_features = X.shape 

        gini_record = np.infty
        cond_record = 0
        feat_record = 0
        sample_record = 0
        leftBranchRowidx_record = None
        rightBranchRowidx_record = None

        for each_feat_idx in range(n_features):
            for each_sample_idx in range(n_samples):
                split_cond = X[each_sample_idx, each_feat_idx]
                leftBranch_x, rightBranch_x = np.array([]), np.array([])
                leftBranch_y, rightBranch_y = np.array([]), np.array([])
                leftBranchRowidx = np.array([])
                rightBranchRowidx = np.array([])
                for _each_sample_idx in range(n_samples): 
                    if X[_each_sample_idx, each_feat_idx] > split_cond:
                        leftBranch_x = np.append(leftBranch_x, X[_each_sample_idx, each_feat_idx])
                        leftBranch_y = np.append(leftBranch_y, y[_each_sample_idx])
                        leftBranchRowidx = np.append(leftBranchRowidx, _each_sample_idx)
                    else:
                        rightBranch_x = np.append(rightBranch_x, X[_each_sample_idx, each_feat_idx])
                        rightBranch_y = np.append(rightBranch_y, y[_each_sample_idx])
                        rightBranchRowidx = np.append(rightBranchRowidx, _each_sample_idx)
                leftBranch_gini = self.calculateGiniImpurity(leftBranch_x, leftBranch_y)
                rightBranch_gini = self.calculateGiniImpurity(rightBranch_x, rightBranch_y)
                gini_temp = rightBranch_gini * (len(rightBranch_y)/n_samples) + leftBranch_gini * (len(leftBranch_y)/n_samples) # get gini relative to their sizes

                if gini_temp < gini_record:
                    gini_record = gini_temp
                    cond_record = split_cond
                    sample_record = each_sample_idx
                    feat_record = each_feat_idx
                    leftBranchRowidx_record = leftBranchRowidx
                    rightBranchRowidx_record = rightBranchRowidx
        
        return {'gini': gini_record, 'cond_record': cond_record, 'sample_record': sample_record, 'feat_record': feat_record,
        'leftBranchRowidx_record': leftBranchRowidx_record, 'rightBranchRowidx_record': rightBranchRowidx_record}

    def getLabel(self, X, y):
        classes, class_counts = np.unique(y, return_counts=True)
        return classes[np.argmax(class_counts)]

    def growTree(self, X, y, node, depth):
        depth += 1
        if len(y) < 1:
            return 
        if depth >= self.max_depth:
            return self.getLabel(X, y)

        leftBranch_idx = node['content']['leftBranchRowidx_record'].astype('int64')
        rightBranch_idx = node['content']['rightBranchRowidx_record'].astype('int64')

        node['left'] = self.constructNode()
        node['left']['content'] = self.splitNode(X[leftBranch_idx], y[leftBranch_idx])
        self.growTree(X[leftBranch_idx], y[leftBranch_idx], node['left'], depth)

        node['right'] = self.constructNode()
        node['right']['content'] = self.splitNode(X[rightBranch_idx], y[rightBranch_idx])
        self.growTree(X[rightBranch_idx], y[rightBranch_idx], node['right'], depth)

    def constructNode(self):
        node = {}
        node['left'] = None
        node['right'] = None
        node['content'] = None
        return node

    def predict(self, X):
        pass
                

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
    dt_obj = dt(X, y)
    dt_obj.predict(X)