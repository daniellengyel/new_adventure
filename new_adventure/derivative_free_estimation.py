import numpy as np
import time
def is_expectation(xs, weights):
    """xs.shape = (N, d) and weights.shape = (N)"""
    return xs.T.dot(weights) / np.sum(weights)


def first_shift_estimator(F, mu, cov, tau, N, control_variate=True):
    sample_points = np.random.multivariate_normal(mu, tau**2 * cov, N)
    out_points = np.exp(F.f(sample_points))
    is_exp = is_expectation(sample_points, out_points) 
    if control_variate:
        is_exp -= np.mean(sample_points)
    else:
        is_exp -= mu
    return 1./(tau**2) * np.linalg.inv(cov).dot(is_exp)

def second_shift_estimator(F, mu, cov, tau, N, control_variate=True):
    sample_points = np.random.multivariate_normal(mu, tau**2 * cov, N)
    out_points = np.exp(F.f(sample_points)) # weights
    a = time.time()
    is_var = costum(sample_points, out_points)
    print(time.time() - a)
    print(is_var)
    a = time.time()
    is_var = np.cov(sample_points.T, aweights=out_points, ddof=0).reshape(len(mu), len(mu)) # no normalization constant as in the paper
    if control_variate:
        is_var -= np.cov(sample_points.T, ddof=0)
    else:
        is_var -= tau**2 * cov
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