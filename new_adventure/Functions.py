import numpy as np
import jax.numpy as jnp
from .derivative_free_estimation import new_beta_second_shift_estimator
import multiprocessing

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



class LinearCombination():

    def __init__(self, obj, barrier, weights):
        self.obj = obj
        self.barrier = barrier
        self.funcs = [self.obj, self.barrier]
        self.weights = weights

    def f(self, X):
        res = jnp.sum(jnp.array([jnp.multiply(w, f.f(X)) for w, f in zip(self.weights, self.funcs)]), axis=0)
        return res

    def f1(self, X):
        res = jnp.sum(jnp.array([jnp.multiply(w, f.f1(X)) for w, f in zip(self.weights, self.funcs)]), axis=0)
        return res

    def f2(self, X):
        res = jnp.sum(jnp.array([jnp.multiply(w, f.f2(X)) for w, f in zip(self.weights, self.funcs)]), axis=0)
        return res

    def f2_inv(self, X):
        pre_inv = jnp.array(self.f2(X))
        return jnp.array([jnp.linalg.inv(pre_inv[i]) for i in range(len(pre_inv))])

    def dir_dists(self, xs, dirs):
        return self.barrier.dir_dists(xs, dirs)

class Linear:
    def __init__(self, c):
        """c.shape = (d)"""
        self.c = c

    def f(self, X):
        return X.dot(self.c) #/ float(len(self.c))

    def f1(self, X):
        return np.tile(self.c, (X.shape[0], 1)) #/ float(len(self.c))

    def f2(self, X):
        return np.tile(np.array([0]), (X.shape[0], X.shape[1], X.shape[1]))  #/ float(len(self.c))

class BetaShiftEstimation():
    def __init__(self, F, N):
        self.F = F
        self.N = N

    def f(self, x):
        return self.F.f(x)

    def f1(self, x):
        num_runs = 1500
        alpha=1000
        return self.F.f1(x) 

    def f2(self, x, num_samples=None, jrandom_key=None):
        if num_samples is None:
            num_samples = 5000
        alpha=1000
        res = np.array([new_beta_second_shift_estimator(self.F, x_i, alpha, num_samples, jrandom_key=jrandom_key) for x_i in x])


    def f2_inv(self, x, num_samples = None, jrandom_key=None):
        f2 = self.f2(x, num_samples, jrandom_key)
        return np.array([np.linalg.inv(f2[i]) for i in range(len(f2))])