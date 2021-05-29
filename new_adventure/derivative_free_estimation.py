import numpy as np
import time
import multiprocessing
import random 

def set_seed(seed):
    if seed is None:
        seed = 0
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

"""Assume mu.shape = (d) and cov.shape = (d, d)"""

    
def doucet_is_expectation(xs, weights):
    """xs.shape = (N, d) and weights.shape = (N)"""
    return xs.T.dot(weights) / np.sum(weights)


def first_shift_estimator(F, mu, cov, tau, N, control_variate=True):
    sample_points = np.random.multivariate_normal(mu, tau**2 * cov, N)

    out_points = np.exp(F.f(sample_points))
    is_exp = doucet_is_expectation(sample_points, out_points) 
    if control_variate:
        is_exp -= np.mean(sample_points, axis=0)
    else:
        is_exp -= mu
    return 1./(tau**2) * np.linalg.inv(cov).dot(is_exp)

def first_estimator(F, mu, cov, tau, N, control_variate=True):
    sample_points = np.random.multivariate_normal(mu, tau**2 * cov, N)

    out_points = F.f(sample_points)
    if control_variate:
        is_exp = np.mean((sample_points - np.mean(sample_points, axis=0)).T * out_points, axis=1).T
    else:
        is_exp = np.mean((sample_points - mu).T * out_points, axis=1).T

    return 1./(tau**2) * np.linalg.inv(cov).dot(is_exp)


def second_shift_estimator(F, mu, cov, tau, N, control_variate=True):
    sample_points = np.random.multivariate_normal(mu, tau**2 * cov, N)
    out_points = np.exp(F.f(sample_points)) # weights
    # is_var = costum(sample_points, out_points)
    # print(is_var)
    is_var = np.cov(sample_points.T, aweights=out_points, ddof=0).reshape(len(mu), len(mu)) # no normalization constant as in the paper
    # print(is_var)
    if control_variate:
        is_var -= np.cov(sample_points.T, ddof=0)
    else:
        is_var -= tau**2 * cov
    return 1./(tau**4) * np.linalg.inv(cov).dot(is_var).dot(np.linalg.inv(cov))

def second_estimator(F, mu, cov, tau, N, control_variate=True):
    sample_points = np.random.multivariate_normal(mu, tau**2 * cov, N)
    out_points = F.f(sample_points) # weights
    # is_var = costum(sample_points, out_points)
    # print(is_var)
    # is_var = np.cov(sample_points.T, aweights=out_points, ddof=0).reshape(len(mu), len(mu)) # no normalization constant as in the paper
    # print(is_var)
    if control_variate:
        is_var = np.cov((sample_points).T, aweights=out_points, ddof=0).reshape(len(mu), len(mu)) 
    else:
        # print(np.mean(out_points) * tau**2)
        # print(costum2(sample_points, mu, out_points))
        # print(np.cov(sample_points.T, aweights=out_points, ddof=0).reshape(len(mu), len(mu)) * np.sum(out_points) / float(N))
        # print()
        is_var = costum2(sample_points, mu, out_points) - tau**2 * np.cov(sample_points.T, ddof=0) * np.mean(out_points)
        # print(is_var)
        # is_var = np.cov(sample_points.T, aweights=out_points, ddof=0).reshape(len(mu), len(mu)) * np.sum(out_points) / float(N) - tau**2 * np.cov(sample_points.T, ddof=0) * np.mean(out_points)  # tau**2 * cov
        # print(is_var)

    return 1./(tau**4) * np.linalg.inv(cov).dot(is_var).dot(np.linalg.inv(cov))

def costum(xs, weights):
    """xs.shape = (N, d), weights.shape = (N)"""
    weights /= np.sum(weights) 
    xs_adjusted = (xs.T - np.sum(xs.T * weights, axis=1)).T
    outer_adjusted = np.array([np.outer(x_adjusted, x_adjusted) for x_adjusted in xs_adjusted])
    res = None
    for i, o in enumerate(outer_adjusted): 
        if res is None:
            res = weights[i] * o
        else:
            res += weights[i] * o
    return res

def costum2(xs, mu, weights):
    """xs.shape = (N, d), weights.shape = (N), mu.shape = (d)"""
    xs_adjusted = (xs.T - mu).T
    outer_adjusted = np.array([np.outer(x_adjusted, x_adjusted) for x_adjusted in xs_adjusted])
    res = None
    for i, o in enumerate(outer_adjusted): 
        if res is None:
            res = weights[i] * o
        else:
            res += weights[i] * o
    return res/len(weights)

########

def hit_run(x_0, barrier, dim, N, alpha):
    # a = time.time()
    dirs = np.random.normal(size=(N, dim)) # sample gaussian and normalize 
    dirs = dirs/np.linalg.norm(dirs, axis=1).reshape(-1, 1)
    # print("Got dirs", time.time() - a)
    # a = time.time()
    dists = barrier.dir_dists(x_0, dirs) # for each dir get distance to boundary
    radius = np.min(dists[0])
    # print("Got rad", time.time() - a)
    # a = time.time()
    beta_p = (np.random.beta(alpha, alpha, size=(N, 1)) - 0.5) * 2 * radius 
    # print("rest", time.time() - a)
    return x_0 + dirs * beta_p


def is_expectation(xs, weights):
    """xs.shape = (N, d) and weights.shape = (N)"""
    return xs.T.dot(weights) / xs.shape[0] # / np.sum(weights)


def beta_first_shift_estimator(F, x_0, alpha, N, control_variate=True):
    sample_points = hit_run(x_0, F, x_0.shape[0], N, alpha)
    out_points = F.f(sample_points)
    is_exp = is_expectation(sample_points - np.mean(sample_points, axis=0), out_points)
    return np.linalg.inv(np.cov(sample_points.T)).dot(is_exp) #* (8 * a + 4) / (2*radius)**3

def get_A(xs):
    """xs.shape = [N, d]"""
    xs -= np.mean(xs, axis=0)
    curr_m = None
    for i in range(len(xs)):
        if curr_m is None:
            curr_m = np.outer(xs[i]**2, xs[i]**2)
        else:
            curr_m += np.outer(xs[i]**2, xs[i]**2)
    curr_m = curr_m / len(xs)
    return curr_m

def proper_cov(xs, weights):
    """xs.shape = [N, d]"""
    xs -= np.mean(xs, axis=0)
    curr_m = None
    for i in range(len(xs)):
        if curr_m is None:
            curr_m = np.outer(xs[i], xs[i])*weights[i]
        else:
            curr_m += np.outer(xs[i], xs[i])*weights[i]
    curr_m /= len(xs)
    return curr_m


def beta_second_shift_estimator(barrier, x_0, alpha, N, control_variate=True):
    sample_points = hit_run(x_0, barrier, x_0.shape[0], N, alpha)
    out_points = barrier.f(sample_points)
    second_shift_est = proper_cov(sample_points, out_points)
    second_shift_est -= barrier.f(x_0.reshape(1, -1)) * np.cov(sample_points.T)
    A = get_A(sample_points)
    return 2*((second_shift_est - np.eye(len(x_0)) * np.diag(second_shift_est))/(A*2.) + np.eye(len(x_0)) * np.linalg.inv(A).dot(np.diag(second_shift_est)))

def new_proper_cov(xs, grads):
    """xs.shape = [N, d], grads = [N, d]"""
    return grads.T.dot(xs - np.mean(xs, axis=0))/len(xs) # np.einsum('ik,ij->jk', xs, grads)/len(xs)

def np_new_cov(xs):
    return np.dot((xs - np.mean(xs, axis=0)).T, xs  - np.mean(xs, axis=0))/len(xs)

def new_beta_second_shift_estimator(F, x_0, alpha, N, control_variate=True, estimated_gradient=False):
    sample_points = hit_run(x_0, F, x_0.shape[0], N, alpha)
    # print(F.f1(sample_points).shape) 
    # print(beta_first_shift_estimator(F, x_0, alpha, N*2, control_variate=True).shape)
    # if estimated_gradient:
    #     out_grads = [beta_first_shift_estimator(F, x, alpha, 2*N, control_variate=True) for x in sample_points] # 
    # else:
    #     
    out_grads = F.f1(sample_points)
    second_shift_est = new_proper_cov(sample_points, out_grads)
    return second_shift_est.dot(np.linalg.inv(np_new_cov(sample_points)))

def new_beta_inverse_second_shift_estimator(F, x_0, alpha, N, control_variate=True, estimated_gradient=False):
    a = time.time()
    sample_points = hit_run(x_0, F, x_0.shape[0], N, alpha)
    print("sampling points", time.time() - a)
    a = time.time()

    if estimated_gradient:
        out_grads = [beta_first_shift_estimator(F, x, alpha, 2*N, control_variate=True) for x in sample_points] # 
    else:
        out_grads = F.f1(sample_points)
    print("Get grads", time.time() - a) 
    a = time.time()
    second_shift_est = new_proper_cov(sample_points, out_grads)
    print("Get second shift est", time.time() - a)
    a = time.time()

    sample_points_cov = np.cov(sample_points.T)
    
    print("get sample cov", time.time() - a)
    a = time.time()
    second_inv = np.linalg.inv(second_shift_est)
    print("get second inverse", time.time() - a)
    
    return sample_points_cov.dot(second_inv)

def multi_second_shift_estimator_task(x_0, F, process_N, alpha, L_sample_points, L_grads, pid, seed=0):
    set_seed(seed)
    # Notice, we should be sharing the radius. So some lock is probably needed to synchronise 
    sample_points = hit_run(x_0, F, x_0.shape[0], process_N, alpha)
    # print(sample_points)
    out_grads = F.f1(sample_points)
    L_sample_points.append(sample_points)
    L_grads.append(out_grads)

def multi_beta_second_shift_estimator(F, x_0, alpha, N, control_variate=True, num_processes=1):
    manager = multiprocessing.Manager()
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