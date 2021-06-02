import numpy as np
import time
import multiprocessing
import random 
from jax import random as jrandom
import jax.numpy as jnp

def set_seed(seed):
    if seed is None:
        seed = 0
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

"""Assume mu.shape = (d) and cov.shape = (d, d)"""


def hit_run(x_0, barrier, dim, N, alpha):
    dirs = np.random.normal(size=(N, dim)) # sample gaussian and normalize 
    dirs = dirs/np.linalg.norm(dirs, axis=1).reshape(-1, 1)
    dists = barrier.dir_dists(x_0, dirs) # for each dir get distance to boundary
    radius = np.min(dists[0])
    beta_p = (np.random.beta(alpha, alpha, size=(N, 1)) - 0.5) * 2 * radius 
    return x_0 + dirs * beta_p

def jax_hit_run(x_0, barrier, dim, N, alpha, key):
    key, subkey = randomj.split(key)
    dirs = jrandom.normal(subkey, shape=(N, dim)) # sample gaussian and normalize 
    dirs = dirs/jnp.linalg.norm(dirs, axis=1).reshape(-1, 1)
    dists = barrier.dir_dists(x_0, dirs) # for each dir get distance to boundary
    radius = jnp.min(dists[0])

    key, subkey = randomj.split(key)
    beta_p = (jrandom.beta(subkey, alpha, alpha, shape=(N, 1)) - 0.5) * 2 * radius 
    return x_0 + dirs * beta_p


def is_expectation(xs, weights):
    """xs.shape = (N, d) and weights.shape = (N)"""
    return xs.T.dot(weights) / xs.shape[0]


def beta_first_shift_estimator(F, x_0, alpha, N, control_variate=True):
    sample_points = hit_run(x_0, F, x_0.shape[0], N, alpha)
    out_points = F.f(sample_points)
    is_exp = is_expectation(sample_points - np.mean(sample_points, axis=0), out_points)
    return np.linalg.inv(np.cov(sample_points.T)).dot(is_exp) #* (8 * a + 4) / (2*radius)**3

def new_proper_cov(xs, grads):
    """xs.shape = [N, d], grads = [N, d]"""
    return grads.T.dot(xs - np.mean(xs, axis=0))/len(xs) # np.einsum('ik,ij->jk', xs, grads)/len(xs)

def np_new_cov(xs):
    return np.dot((xs - np.mean(xs, axis=0)).T, xs  - np.mean(xs, axis=0))/len(xs)

def new_beta_second_shift_estimator(F, x_0, alpha, N, control_variate=True, estimated_gradient=False):
    # Note, i could not invert the cov at the end to save inversion of the Hessian.
    sample_points = hit_run(x_0, F, x_0.shape[0], N, alpha)   
    out_grads = F.f1(sample_points)
    second_shift_est = new_proper_cov(sample_points, out_grads)
    return second_shift_est.dot(np.linalg.inv(np_new_cov(sample_points)))


def multi_second_shift_estimator_task(x_0, F, process_N, alpha, L_sample_points, L_grads, pid, seed=0):
    set_seed(seed)
    # Notice, we should be sharing the radius. Lock is needed to synchronise 
    sample_points = hit_run(x_0, F, x_0.shape[0], process_N, alpha)
    out_grads = F.f1(sample_points)
    L_sample_points.append(sample_points)
    L_grads.append(out_grads)

def multi_beta_second_shift_estimator(F, x_0, alpha, N, control_variate=True, num_processes=1, pool=None):
    manager = multiprocessing.Manager()
    if pool is None:
        pool = multiprocessing.Pool(processes=num_processes)
        
    L_sample_points = manager.list()
    L_grads = manager.list()
    pool_workers = []
    for i in range(num_processes):
        curr_seed = random.randint(0, 10000)
        p = pool.apply_async(multi_second_shift_estimator_task, (x_0, F, N // num_processes, alpha, L_sample_points, L_grads, i, curr_seed))
        pool_workers.append(p)
    
    pool_workers = [p.wait() for p in pool_workers]

    pool.close()
    pool.join()

    sample_points = np.vstack(list(L_sample_points))
    out_grads = np.vstack(list(L_grads))
    second_shift_est = new_proper_cov(sample_points, out_grads)
    return second_shift_est.dot(np.linalg.inv(np_new_cov(sample_points)))


# Quadratic Regression
def get_quadratic_data_matrix(xs):
    """xs.shape = (N, d). out.shape = (N, d(d + 1)/2)"""
    one_two_factors = np.array([1] + [2]*(xs.shape[1])) # I should have one less two, but since i cant do slicing with [:-0] for the loop to get X, i need to add one here so i can start at [:-1]
    out = [xs[:, i].reshape((-1, 1)) * xs[:, i:] * one_two_factors[:-(i+ 1)]  for i in range(xs.shape[1])] 
    return np.hstack(out)

def quadratic_regression(xs, y):
    X = get_quadratic_data_matrix(xs)
    return np.linalg.inv(np.dot(np.transpose(X), X)).dot(X.T.dot(y))

def quadratic_regression_ipm(F, x_0, alpha, N, control_variate=True, estimated_gradient=False):
    sample_points = hit_run(x_0, F, x_0.shape[0], N, alpha)    
    Y = 2*(F.f(sample_points) - F.f(np.array([x_0]))[0] - (sample_points - x_0).dot(F.f1(np.array([x_0])).reshape(-1)))
    return quadratic_regression(sample_points, Y)


# BFGS approximation
    
def BFGS_update(F, x_0, x_1, inv_hessian_approx=None):
    if inv_hessian_approx is None:
        H = np.eye(len(x_0))
    else:
        H = inv_hessian_approx

    grad_diff = (F.f1(np.array([x_1])) - F.f1(np.array([x_0])))[0]
    update_step = x_1 - x_0
    
    ys = np.inner(grad_diff, update_step)
    Hy = np.dot(H, grad_diff)
    yHy = np.inner(grad_diff, Hy)
    H += (ys + yHy) * np.outer(update_step, update_step) / ys ** 2
    H -= (np.outer(Hy, update_step) + np.outer(update_step, Hy)) / ys
    return H    