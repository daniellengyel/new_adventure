import numpy as np
import time
import multiprocessing
import random 
from jax import random as jrandom
import jax.numpy as jnp
import jax

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

def jax_hit_run(x_0, F, dim, N, alpha, new_jrandom_key, chosen_basis_idx=None):
    new_jrandom_key, subkey = jrandom.split(new_jrandom_key)
    # sample gaussian and normalize 
    if chosen_basis_idx is None:
        dirs = jrandom.normal(subkey, shape=(N, dim)) 
    else:
        dirs = jrandom.normal(subkey, shape=(N, len(chosen_basis_idx)))
        temp_dirs = jnp.zeros((dim, N))
        temp_dirs = jax.ops.index_update(temp_dirs, chosen_basis_idx, dirs.T)
        dirs = temp_dirs.T

    dirs = dirs/jnp.linalg.norm(dirs, axis=1).reshape(-1, 1)
    dists = F.dir_dists(x_0, dirs) # for each dir get distance to boundary

    radius = jnp.min(dists[0])
    new_jrandom_key, subkey = jrandom.split(new_jrandom_key)
    beta_p = (jrandom.beta(subkey, alpha, alpha, shape=(N, 1)) - 0.5) * 2 * radius 
    return x_0 + dirs * beta_p, radius


def is_expectation(xs, weights):
    """xs.shape = (N, d) and weights.shape = (N)"""
    return xs.T.dot(weights) / xs.shape[0]


def beta_first_shift_estimator(F, x_0, alpha, N, jrandom_key):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points = jax_hit_run(x_0, F, x_0.shape[0], N, alpha, subkey)
    out_points = F.f(sample_points)
    is_exp = is_expectation(sample_points - jnp.mean(sample_points, axis=0), out_points)
    return jnp.linalg.inv(jnp.cov(sample_points.T)).dot(is_exp) #* (8 * a + 4) / (2*radius)**3

def new_proper_cov(xs, grads):
    """xs.shape = [N, d], grads = [N, d]"""
    return grads.T.dot(xs - jnp.mean(xs, axis=0))/len(xs)

def get_A(xs):
    """xs.shape = [N, d]"""
    xs -= jnp.mean(xs, axis=0)
    curr_m = (xs ** 2).T.dot(xs ** 2)
    curr_m = curr_m / len(xs)
    return curr_m

def np_new_cov(xs):
    return jnp.dot((xs - jnp.mean(xs, axis=0)).T, xs  - jnp.mean(xs, axis=0))/len(xs)

def beta_second_shift_estimator(F, x_0, alpha, N, jrandom_key):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, radius = jax_hit_run(x_0, F, x_0.shape[0], N, alpha, subkey)
    out_points = F.f(sample_points)
    diffs = sample_points -  jnp.mean(sample_points, axis=0) 
    second_shift_est =  diffs.T.dot(diffs * out_points.reshape(-1, 1)) / len(diffs)

    grads_inners = F.f1(x_0.reshape(1, -1))[0].dot(diffs.T) # N

    a = diffs.T.dot(diffs * grads_inners.reshape(-1, 1)) / len(diffs)
    
    second_shift_est -= F.f(x_0.reshape(1, -1))[0] * (radius**2)/float(1 + 2 * alpha) # diffs.T.dot(diffs)/len(diffs) # - a   

    A = jnp.array([[3*radius**4 / float(3 + 8*alpha + 4 * alpha**2)]]) # get_A(sample_points)

    return  2*jnp.eye(len(x_0)) * jnp.linalg.inv(A).dot(jnp.diag(second_shift_est)) # 2*((second_shift_est - jnp.eye(len(x_0)) * jnp.diag(second_shift_est))/(A*2.) +

def new_beta_second_shift_estimator(F, x_0, alpha, N, jrandom_key):
    # Note, i could not invert the cov at the end to save inversion of the Hessian.
    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, radius = jax_hit_run(x_0, F, x_0.shape[0], N, alpha, subkey)   
    jrandom_key, subkey = jrandom.split(jrandom_key)
    out_grads = F.f1(sample_points, subkey)
    second_shift_est = new_proper_cov(sample_points, out_grads)
    return second_shift_est.dot(jnp.linalg.inv(np_new_cov(sample_points)))


def multilevel_inv_estimator(F, x_0, alpha, N, d_prime, jrandom_key):
    d = len(x_0)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, radius = jax_hit_run(x_0, F, x_0.shape[0], N, alpha, subkey)   
    out_grads = F.f1(sample_points)
    grad_X = new_proper_cov(sample_points, out_grads)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
    U = jnp.eye(d)[U_idxs].T
    grad_X_multilevel = U.dot(jnp.linalg.inv(U.T.dot(grad_X).dot(U))).dot(U.T)
    return np_new_cov(sample_points).dot(grad_X_multilevel)

def multilevel_update_direction(F, x_0, alpha, N, d_prime, jrandom_key):
    d = len(x_0)

    jrandom_key, subkey = jrandom.split(jrandom_key)
    U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
    U = jnp.eye(d)[U_idxs].T # (d, d')

    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, radius = jax_hit_run(x_0, F, x_0.shape[0], N, alpha, subkey, chosen_basis_idx=U_idxs)  
    X = (sample_points - jnp.mean(sample_points, axis=0)).T # (d, N)
    out_grads = F.f1(sample_points)

    gradF = F.f1(jnp.array([x_0]))[0]
    grad_X_low_inv = jnp.linalg.inv(out_grads.T[U_idxs].dot(X[U_idxs].T)/float(N))
    return -X.dot(X.T.dot(U.dot(grad_X_low_inv.dot(U.T.dot(gradF))))) / float(N)

def multilevel_estimator_woodbury(F, x_0, alpha, N, d_prime, jrandom_key):
    """Return C_inv, W, V for woodbury"""
    d = len(x_0)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, radius = jax_hit_run(x_0, F, x_0.shape[0], N, alpha, subkey)   
    out_grads = F.f1(sample_points)
    grad_X = new_proper_cov(sample_points, out_grads)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
    U = jnp.eye(d)[U_idxs].T
    XXt = np_new_cov(sample_points)
    C_inv = U.T.dot(XXt).dot(U)
    W = grad_X.dot(XXt).dot(U)
    V = U.T.dot(XXt)
    return C_inv, W, V

def get_neystrom_inv(A, d_prime, jrandom_key):
    d = len(A)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
    U = jnp.eye(d)[U_idxs].T
    A_inv_approx = U.dot(jnp.linalg.inv(U.T.dot(A).dot(U))).dot(U.T)
    return A_inv_approx

def get_neystrom_inv_direction(A, d_prime, grad, jrandom_key):
    d = len(A)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
    U = jnp.eye(d)[U_idxs].T
    dir_approx = U.dot(jnp.linalg.inv(U.T.dot(A).dot(U)).dot(U.T.dot(grad)))
    return dir_approx

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


# Finite Difference 
def finite_difference_hessian(F, X, h):
    hess = []
    all_out_central = F.f(X)
    for i, x in enumerate(X):
        x_back = x - jnp.eye(x.shape[0]) * h
        x_forward = x + jnp.eye(x.shape[0]) * h
        
        out_back = F.f(x_back)
        out_forward = F.f(x_forward)
        
        curr_hess = jnp.diag((out_back + out_forward - 2 * all_out_central[i]) / h**2) 
        out_back = F.f(x_back)
        out_forward = F.f(x_forward)
        
        for d in range(1, len(x)):
        
            x_forward_forward = x + jnp.eye(x.shape[0]) * h + jnp.eye(x.shape[0], k=d) * h
            x_forward_backward = x + jnp.eye(x.shape[0]) * h - jnp.eye(x.shape[0], k=d) * h
            x_backward_forward = x - jnp.eye(x.shape[0]) * h + jnp.eye(x.shape[0], k=d) * h
            x_backward_backward = x - jnp.eye(x.shape[0]) * h - jnp.eye(x.shape[0], k=d) * h
  
            out_forward_forward = F.f(x_forward_forward)
            out_forward_backward = F.f(x_forward_backward)
            out_backward_forward = F.f(x_backward_forward)
            out_backward_backward = F.f(x_backward_backward)
            
            curr_off_diag = jnp.eye(x.shape[0], k=d) * (out_forward_forward - out_forward_backward - out_backward_forward + out_backward_backward).reshape(-1, 1)/(4.*h**2)
            curr_hess += curr_off_diag + curr_off_diag.T
            
        hess.append(curr_hess)
        
    return jnp.array(hess)