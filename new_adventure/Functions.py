import numpy as np
from derivative_free_estimation import first_shift_estimator, second_shift_estimator

# We expect X to be a (N, d) array, where d is the dimensionality and N is the number of datapoints. 
# The output is then then (N) dimensional. We are only working with scalar functions. 

# output of f1 is of shape (N, d)

# output of f2 is of shape (N, d, d)

class Quadratic:
    def __init__(self, Q):
        self.Q = Q
        self.Q_inv = np.linalg.inv(Q)
        
    def f(self, X):
        Y = np.dot(self.Q, X.T)
        Y = np.diag(np.dot(X, Y)) # TODO fix. inefficient way to remove x_j^T Q x_i for i != j. 
        return Y
    
    def f1(self, X):
        Y = 2*np.dot(self.Q, X.T)
        return Y.T
    
    def f2(self, X):
        return 2 * np.array([list(self.Q)] * X.shape[0])

    def f2_inv(self, X):
        return 1/2. * np.array([list(self.Q_inv)] * X.shape[0])
        
class Ackley:
    
    def __init__(self):
        pass

    def f(self, X):
        xs = X.T
        out_shape = xs[0].shape
        a = np.exp(-0.2 * np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0))))
        b = - np.exp(1. / len(xs) * np.sum(np.cos(2 * np.pi * xs), axis=0))
        return np.array(-20 * a + b + 20 + np.exp(1)).reshape(out_shape)


    def f1(self, X):
        """del H/del xi = -20 * -0.2 * (xi * 1/n) / sqrt(1/n sum_j xj^2) * a + 2 pi sin(2 pi xi)/n * b"""
        xs = X.T
        out_shape = xs.shape
        a = np.exp(-0.2 * np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0))))
        b = -np.exp(1. / len(xs) * np.sum(np.cos(2 * np.pi * xs), axis=0))
        a_p = -0.2 * (xs * 1. / len(xs)) / np.sqrt(1. / len(xs) * np.square(np.linalg.norm(xs, axis=0)))
        b_p = -2 * np.pi * np.sin(2 * np.pi * xs) / len(xs)
        return np.nan_to_num(
            -20 * a_p * a + b_p * b).reshape(out_shape)  # only when norm(x) == 0 do we have nan and we know the grad is zero there

class Gaussian:
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
        self.cov_inv = np.linalg.inv(self.cov)

    def f(self, x):
        k = x.shape[1]
        diff = (x - self.mu).T
        cov_inv_prod = np.dot(self.cov_inv, diff)
        return 1 / np.sqrt(pow(2 * np.pi, k) * np.linalg.det(self.cov)) * np.exp(
            -0.5 * np.sum(diff*(cov_inv_prod), axis=0))

    def f1(self, x):
        k = x.shape[1]
        diff = (x - self.mu).T

        mg = self.f(x)
        grad_term = - np.dot(np.linalg.inv(self.cov), diff)
        return (mg * grad_term).T

class Log_Liklihood:

    def __init__(self, func):
        self.func = func

    def f(self, x):
        return np.log(self.func.f(x))

    def f1(self, x):
        return self.func.f1(x) / self.func.f(x)

    def f2(self, x):
        grads = self.func.f1(x)
        return self.func.f2(x) / self.func.f(x) - np.array([np.outer(grads[i], grads[i].T) for i in range(len(grads))]) / self.func.f(x)**2

class Gaussian_example2:
    def __init__(self, y, M, cov_x, cov_yx):
        self.cov_x = cov_x
        self.cov_yx = cov_yx
        self.y = y 
        self.M = M
        
    def f(self, theta):
        L_hats = []
        for i in range(len(theta)):
            sampled_X = np.random.multivariate_normal(theta[i], self.cov_x, self.M)
            g = Gaussian(self.y, self.cov_yx)
            L_hat = np.mean(g.f(sampled_X))
            L_hats.append(L_hat)
        return L_hats

class ShiftEstimation():
    def __init__(self, F, tau, cov, N):
        self.F = F
        self.tau = tau
        self.cov = cov
        self.N = N
    
    def f(self, x):
        return F.f(x)

    def f1(self, x):
        return first_shift_estimator(self.F, x, cov, tau, N, control_variate=True)

    def f2(self, x):
        return second_shift_estimator(self.F, x, cov, tau, N, control_variate=True)

    def f2_inv(self, x):
        f2 = self.f2(self.F(x))
        return np.array([np.linalg.inv(f2[i]) for i in range(len(f2))])