import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import multiprocessing

# We expect X to be a (N, d) array, where d is the dimensionality and N is the number of datapoints. 
# The output is then then (N) dimensional. We are only working with scalar functions. 

# output of f1 is of shape (N, d)

# output of f2 is of shape (N, d, d)


class Quadratic:
    def __init__(self, Q, b):
        self.Q = Q
        self.Q_inv = jnp.linalg.inv(Q)
        self.b = b
        
    def f(self, X):
        Y = np.dot(self.Q, X.T)
        Y = jnp.diag(jnp.dot(X, Y)) + X.dot(self.b) # TODO fix. inefficient way to remove x_j^T Q x_i for i != j. 
        return Y
    
    def f1(self, X):
        Y = 2*jnp.dot(self.Q, X.T)
        return Y.T + self.b
    
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

    def __init__(self, obj, barrier, weights, f1_var=None):
        self.obj = obj
        self.barrier = barrier
        self.funcs = [self.obj, self.barrier]
        self.weights = weights
        self.f1_var = f1_var

    def f(self, X):
        res = jnp.sum(jnp.array([jnp.multiply(w, f.f(X)) for w, f in zip(self.weights, self.funcs)]), axis=0)
        return res

    def f1(self, X, jrandom_key=None):
        res = jnp.sum(jnp.array([jnp.multiply(w, f.f1(X, jrandom_key)) for w, f in zip(self.weights, self.funcs)]), axis=0)
        if self.f1_var is not None and jrandom_key is not None:
            res += jrandom.normal(key=jrandom_key, shape=X.shape) * self.f1_var**0.5
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

    def f1(self, X, jrandom_key=None):
        return np.tile(self.c, (X.shape[0], 1)) #/ float(len(self.c))

    def f2(self, X):
        return np.tile(np.array([0]), (X.shape[0], X.shape[1], X.shape[1]))  #/ float(len(self.c))



class GaussianSmoothing:
    def __init__(self, func, N, sigma):
        self.func = func
        self.N = N
        self.sigma = sigma
    
    def f(self, X, jrandom_key, sigma):
        d = X.shape[1]
        cov = sigma**2*jnp.eye(d)
        jrandom_key, subkey = jrandom.split(jrandom_key)
        x_samples = jrandom.multivariate_normal(key=subkey, mean=X, cov=cov, shape=(self.N, len(X), )).transpose(1, 0, 2)
        f_out = self.func.f(x_samples.reshape(-1, d)).reshape(len(X), self.N)
        return jnp.mean(f_out, axis=1)
        
    def f1(self, X, jrandom_key, sigma):
        d = X.shape[1]
        cov = sigma**2*jnp.eye(d)
        jrandom_key, subkey = jrandom.split(jrandom_key)
        x_samples = jrandom.multivariate_normal(key=subkey, mean=X, cov=cov, shape=(self.N, len(X), )).transpose(1, 0, 2)
        
        f_out = self.func.f(x_samples.reshape(-1, d)).reshape(len(X), self.N)
        
        res = []
        for i in range(len(X)):
            x_diff = x_samples[i] - jnp.mean(x_samples[i]) # N, d
            cov_inv = jnp.linalg.inv(x_diff.T.dot(x_diff) / self.N)
            res.append(cov_inv.dot((x_samples[i] - jnp.mean(x_samples[i])).T.dot(f_out[i])) / self.N)
        res = jnp.array(res)
        return res # jnp.linalg.inv(cov).dot(res.T).T
    
    def f2(self, X, jrandom_key, sigma):
        d = X.shape[1]
        cov = sigma**2*jnp.eye(d)
        jrandom_key, subkey = jrandom.split(jrandom_key)
        x_samples = jrandom.multivariate_normal(key=subkey, mean=X, cov=cov, shape=(self.N, len(X), )).transpose(1, 0, 2)
        
        f_out = self.func.f(x_samples.reshape(-1, d)).reshape(len(X), self.N)
        
        res = []
        for i in range(len(X)):
            x_diff = x_samples[i] - jnp.mean(x_samples[i]) # N, d
            cov_inv = jnp.linalg.inv(x_diff.T.dot(x_diff) / self.N)

            res.append( (cov_inv.dot(x_diff.T * f_out[i]).dot(x_diff).dot(cov_inv) - cov_inv * jnp.sum(f_out[i]) ) / self.N) 
        res = jnp.array(res)
        return res
    
class Tesselated:
    def __init__(self, func, N, domain, jrandom_key):
        self.func = func
        self.N = N
        self.domain = domain
        S = jrandom.uniform(jrandom_key, shape=(self.N, domain["dim"]), minval=domain["bound"][0], maxval=domain["bound"][1])
        self.m = func.f1(S)
        self.b = func.f(S) - jnp.sum(self.m * S, axis=1) # we have y = f + grad (x - x_0) = f - grad x_0 + grad x
                    
    def f(self, X):
        return jnp.max(self.b + X.dot(self.m.T), axis=1)
    
    def f_all(self, X):
        """For ploting:
        out_all = t.f_all(X)
        for x in out_all.T:
            plt.scatter(X, x)"""
        return self.b + X.dot(self.m.T)

    def f1(self, X):
        argmax_idxs = jnp.argmax(self.b + X.dot(self.m.T), axis=1)
        return jnp.array([self.m[argmax_idxs[i]] for i in range(len(X))])

    def f2(self, X):
        return 0

class MAXQ:
    def __init__(self):
        pass

    def f(self, X):
        return jnp.max(X**2, axis=1)

    def f1(self, X):
        pass
        # argmax_idxs = jnp.argmax(X**2, axis=1)
        # g = np.zeros(X.shape)
        # for i in range(len(X)):
        #     g[i][argmax_idxs[i]] = 2*X[i][argmax_idxs[i]]
        # return jnp.array(g)

    def f2(self, X):
        pass