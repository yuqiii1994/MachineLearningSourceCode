import numpy as np 
import matplotlib.pyplot as plt
import cvxopt

# For simplicity, only linear kernel is used.
def linearKernel(x1, x2):
    return np.dot(x1, x2)

class SVM(object):
    """
    Theoretical explanation of SVM (based on SVM Tutorial by Zoya Gavrilov):

    The target is to find a hyperplane able to linearly separate data points, and we
    assume the boundary is w^T * x + b = 0. Assume w^T * x + b = y, the y should 
    have a shape of (n_samples, 1) (assume binary classification).
    For a maximal margin, after some mathematical transformations, we can have the margin
    width of 2 / ||w||^2 . The problem can be changed into a minimization problem that is
    min(w^T * w) / 2 .

    However, such an ideal scenario can be hardly found, and in an attempt to prevent overfitting,
    soft-margin should be introduced, aka slack variables e_i >= 0. As a result, the problem 
    becomes 
                min( (w^T * w) / 2 + C * sum(e_i) )
    subject to 
                y_i (w^T x_i + b) >= 1 - e_i and e_i >= 0,
    where C is for tolerance of slacking, usually 0.1 by default. The constriants of above equation 
    says that for an arbitrary data point (represented by index i), the result w^T * x + b = y 
    times y_i should always greater than or equal to 1 subtracting its error.

    Kernel tricks for mapping to a higher dimensional space should implemented to replace x with Pi(x), 
    where Pi() represents the function mapping to a higher dimensional space. The constraint then becomes
    y_i (w^T Pi(x_i) + b) >= 1 - e_i and e_i >= 0.

    The minimization problem under constraints can be solved via Lagrangian multipliers. The prediction
    expression y_i (w^T Pi(x_i) + b) should be approaching to 1 as any misclassification will introduce
    negative value (y_i can only be 1 or -1). Here we introduce a_i to transform the constraint into
                max ( a_i * [1 - (y_i (w^T Pi(x_i) + b))] ).
    This new constraint, given C >= a_i >= 0, will yield max value at 0 when a_i approaches to 0.

    The final target is
                    max ( sum_i(a_i) -1/2 sum_i_j(a_i * a_j * y_i * y_j * Pi(x_i)^T * Pi(x_j)) )
    subject to
                    sum_i(a_i * y_i) = 0
                    0 <= a_i <= C.
    To update w and b:
                    f(x) = sign(w^T x + b) = sum_i ( a_i * y_i * K(x_i; x) + b ),
    where b should be of shape (1) while w be of shape (n_features, 1).

    """

    def __init__(self, X, y, C=0.1):

        n_samples, n_features = X.shape 

        K = np.zeros((n_samples, n_samples))

        # kernel space
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = linearKernel(X[i], X[j])

        """
        cvxopt.solvers:

        Minimize (1/2) x^T * P x + q^T x

        Subject to G x <= h
                   A x = b

        For SVM:

        Maximize sum(a) - sum(1/2 * A * Y * X) 

        Subject to sum(a * y) = 0 
        and        0 <= a <= C

        """

        # Lagrangian multiplier to obtain optimal max value
        # via searching for solution a_i for each observation. 

        P = cvxopt.matrix(-1 * np.outer(y,y) * K) * -1
        q = cvxopt.matrix(np.ones(n_samples)) * -1
        A = cvxopt.matrix(y, (1,n_samples)) 
        b = cvxopt.matrix(0.0) 

        # Since the second constraint term has both low and high bounds, G and h should be stacked
        G_high_bound = np.identity(n_samples)
        h_high_bound = np.ones(n_samples) * C
        G_low_bound = -1 * np.identity(n_samples)
        h_low_bound = np.zeros(n_samples)

        G = cvxopt.matrix(np.concatenate((G_low_bound, G_high_bound)))
        h = cvxopt.matrix(np.concatenate((h_low_bound, h_high_bound)))

        print("Lagrangian Multiplier Convex Optimization Begins")
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.array(solution['x']).reshape(n_samples)

        w = np.zeros(n_features)
        b = 0
        for i in range(n_samples):
            w += X[i] * y[i] * a[i]
            b += y[i]
            b -= np.sum(a * y * K[i, :])
        b /= n_samples

        self.w = w
        self.b = b

    def predict(self, X, y=None):
        if y is None:
            return np.sign(np.dot(X, self.w) + self.b)

        result = np.sign(np.dot(X, self.w) + self.b)
        _result_accu = result == y
        test_len = X.shape[0]
        accuracy = np.sum([1 if _result_accu[i] else 0 for i in range(test_len)]) / test_len

        return result, accuracy

    def predict_and_plot(self, X, y):

        sep_line_func = lambda x, c: (-self.w[0] * x - self.b + c) / self.w[1] # assume the first two dimension considered

        range_x_x = np.max(X)
        range_y_x = np.min(X)

        deci_line_str = "$W^T X + b = 0$"
        range_x_y = sep_line_func(range_x_x, 0)
        range_y_y = sep_line_func(range_y_x, 0)
        plt.plot([range_x_x,range_y_x], [range_x_y,range_y_y], linewidth=2, color="k")
        plt.text(range_x_x, range_x_y, deci_line_str, fontweight='heavy')

        upper_line_str = "$W^T X + b = -1$"
        range_x_y = sep_line_func(range_x_x, 1)
        range_y_y = sep_line_func(range_y_x, 1)
        plt.plot([range_x_x,range_y_x], [range_x_y,range_y_y], "k--")
        plt.text(range_x_x, range_x_y, upper_line_str)

        down_line_str = "$W^T X + b = 1$"
        range_x_y = sep_line_func(range_x_x, -1)
        range_y_y = sep_line_func(range_y_x, -1)
        plt.plot([range_x_x,range_y_x], [range_x_y,range_y_y], "k--")
        plt.text(range_x_x, range_x_y, down_line_str)

        color_array = ['r' if y[i] == -1 else 'b' for i in range(len(y))]
        plt.scatter(X[:, 0], X[:, 1], c=color_array, alpha=0.5) # only the first two dimensions are shown

        result, accuracy = self.predict(X, y)

        plt.title("An SVM Experiment\n" + "Accuracy: " + str(accuracy*100) + "%", weight='bold')

        plt.show()

        return result, accuracy

def database_generate(data_volume=100, sigma=2):
    data_volume_each_class = int(data_volume/2)
    mu_class1_dim1 = 7
    mu_class1_dim2 = 3
    X_class1_dim1 = np.random.normal(mu_class1_dim1, sigma, size=data_volume_each_class)
    X_class1_dim2 = np.random.normal(mu_class1_dim2, sigma, size=data_volume_each_class)
    X_class1 = np.concatenate((np.atleast_2d(X_class1_dim1), np.atleast_2d(X_class1_dim2)), axis=1)
    y_class1 = np.ones(data_volume_each_class)
    mu_class2_dim1 = 3
    mu_class2_dim2 = 7
    X_class2_dim1 = np.random.normal(mu_class2_dim1, sigma, size=data_volume_each_class)
    X_class2_dim2 = np.random.normal(mu_class2_dim2, sigma, size=data_volume_each_class)
    X_class2 = np.concatenate((np.atleast_2d(X_class2_dim1), np.atleast_2d(X_class2_dim2)), axis=1)
    y_class2 = -1 * np.ones(data_volume_each_class)

    X = np.concatenate((X_class1, X_class2)).T
    y = np.concatenate((y_class1, y_class2))

    if X.shape[0] != y.shape[0]:
        return None

    return X, y

if __name__ == '__main__':
    X, y = database_generate()

    svm_obj = SVM(X, y)
    result, accu = svm_obj.predict_and_plot(X, y=y)

    print("Accuracy: " + str(accu))