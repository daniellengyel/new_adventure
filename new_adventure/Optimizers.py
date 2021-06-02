import numpy as np
import jax.numpy as jnp
from jax import random as jrandom
import jax
import time
import pickle
from .Functions import LinearCombination, BetaShiftEstimation
from .utils import get_barrier, get_potential
from .derivative_free_estimation import BFGS_update, new_beta_second_shift_estimator
import multiprocessing

def get_optimizer(config):
    if "Newton" == config["optimization_name"]:
        return Newton()
    elif "ISMD" == config["optimization_name"]:
        return ISMD(config["optimization_meta"])
    elif "OLD" == config["optimization_name"]:
        return OLD(config["optimization_meta"])
    elif "Newton_IPM" == config["optimization_name"]:
        return Newton_IPM(config)
    elif "Newton_shift_est_IPM" == config["optimization_name"]:
        return Newton_shift_est_IPM(config)
    elif "BFGS" == config["optimization_name"]:
        return BFGS(config)

class Newton:
    def __init__(self, config=None):
        pass

    def update(self, X, F, time_step=None):
        H_inv = F.f2_inv(X)
        f1 = F.f1(X)
        return X - np.array([H_inv[i].dot(f1[i]) for i in range(len(f1))]), False


def helper_Newton_shift_est_IPM(obj, barrier, estimator):
    def helper(X, jrandom_key, t):
        num_samples = 5000
        alpha=200
        combined_F = LinearCombination(obj, barrier, [1, t])
        return estimator(combined_F, X, alpha, num_samples, jrandom_key)
    return helper


class Newton_shift_est_IPM:
    def __init__(self, config):
        self.config = config
        self.barrier = get_barrier(config)
        self.F = get_potential(config)
        self.c1 = config["optimization_meta"]["c1"]
        self.c2 = config["optimization_meta"]["c2"]
        self.delta = config["optimization_meta"]["delta"]
        self.jrandom_key = jrandom.PRNGKey(config["optimization_meta"]["jrandom_key"])
        self.jited_estimator = jax.jit(helper_Newton_shift_est_IPM(self.F, self.barrier, new_beta_second_shift_estimator))


    def update(self, X, F, time_step, full_path=True):
        t = 4 * (0.75)**(time_step) # (1.5)**(time_step)
        combined_F = LinearCombination(F, self.barrier, [1, t])

        if full_path:
            full_path_arr = [(X.copy(), time.time())]
        
        while True:
            # Get hess inverse
            key, subkey = jrandom.split(self.jrandom_key)

            temp_res = self.jited_estimator(X[0], subkey, t)
            H_inv = jnp.linalg.inv(temp_res)
            self.jrandom_key = key
            
            f1 = combined_F.f1(X)

            search_direction = -H_inv.dot(f1[0])
            newton_decrement_squared = -f1[0].dot(search_direction)
            print(newton_decrement_squared)
            print(F.f(X))

            if newton_decrement_squared < 0:
                if full_path:
                    full_path_arr.append((X.copy(), time.time()))
                continue
            newton_decrement = np.sqrt(np.abs(newton_decrement_squared))

            # Check if completed
            alpha = linesearch(combined_F, X[0], 1/(1 + newton_decrement) * search_direction, self.c1, self.c2)
            if alpha is None:
                break
            X[0] = X[0] + 1/(1 + newton_decrement) * alpha * search_direction
            if full_path:
                full_path_arr.append((X.copy(), time.time()))
                
            if newton_decrement < self.delta:
                break
        # print(F.f(X))
        if full_path:
            return full_path_arr
        return X

save_path = "/Users/daniellengyel/new_adventure/"

class Newton_IPM:
    def __init__(self, config):
        self.config = config
        self.barrier = get_barrier(config)
        self.c1 = config["optimization_meta"]["c1"]
        self.c2 = config["optimization_meta"]["c2"]
        self.delta = config["optimization_meta"]["delta"]

    def update(self, X, F, time_step, full_path=True):
        t = 4 * (0.5)**(time_step) # (1.5)**(time_step)
        combined_F = LinearCombination(F, self.barrier, [1, t])
        
        if full_path:
            full_path_arr = [(X.copy(), time.time())]
        
        while True:
            H_inv = combined_F.f2_inv(X)
            f1 = combined_F.f1(X)

            search_direction = -H_inv[0].dot(f1[0])
            print(-f1[0].dot(search_direction))
            newton_decrement = np.sqrt(-f1[0].dot(search_direction))
            print(newton_decrement**2)
            print(self.barrier.f(X))
            print(F.f(X))

            # Check if completed
            if newton_decrement**2 < self.delta:
                break

            alpha = 1/(1 + newton_decrement) # linesearch(combined_F, X[0], 1/(1 + newton_decrement) * search_direction, self.c1, self.c2) # 1/(1 + newton_decrement) #
            X[0] = X[0] + alpha * search_direction
            if full_path:
                full_path_arr.append((X.copy(), time.time()))

        if full_path:
            return full_path_arr
        return X
        
class BFGS:
    def __init__(self, config):
        self.config = config
        self.barrier = get_barrier(config)
        self.H_inv = np.eye(config["domain_dim"])
        self.c1 = config["optimization_meta"]["c1"]
        self.c2 = config["optimization_meta"]["c2"]
        self.delta = config["optimization_meta"]["delta"]
        self.num_iter = 0

    def update(self, X, F, time_step, full_path=True):
        assert len(X) == 1
        t = 4 * (0.75)**(time_step) # (1.5)**(time_step)
        combined_F = LinearCombination(F, self.barrier, [1, t])

        if time_step == 0:
            self.H_inv = np.eye(self.config["domain_dim"]) #np.linalg.inv(combined_F.f2(X)[0])

        if full_path:
            full_path_arr = [(X.copy(), time.time())]
        
        while True:
        # for _ in range(50):
        

            f1 = combined_F.f1(X)

            search_direction = -self.H_inv.dot(f1[0])
            newton_decrement_squared = -f1[0].dot(search_direction)

            if newton_decrement_squared < 0:
                if full_path:
                    full_path_arr.append((X.copy(), time.time()))
                continue
            
            newton_decrement = np.sqrt(newton_decrement_squared)
            print(newton_decrement)

            # Check if completed
            if newton_decrement**2 < self.delta:
                break

            alpha = linesearch(combined_F, X[0], 1/(1 + newton_decrement) * search_direction, self.c1, self.c2)
            
            print(F.f(X))

            if alpha is None:
                print("Alpha was none.")
                break

            X_prev = X[0].copy()
            X[0] = X[0] + 1/(1 + newton_decrement) * alpha * search_direction
            self.H_inv = BFGS_update(combined_F, X_prev, X[0], self.H_inv)
            
            if full_path:
                full_path_arr.append((X.copy(), time.time()))

            self.num_iter += 1

        if full_path:
            return full_path_arr
        return X

# class ISMD:
#     # For now no direct mirror map
#     def __init__(self, meta=None):
#         self.gamma = meta["gamma"]
#         self.sigma = meta["sigma"]
    
#     def update(self, X, F, time_step=None):
#         interaction = 1/float(len(X)) * np.array([np.sum(X - X[i], axis=0) for i in range(len(X))])
#         noise = np.random.rand(*X.shape)
#         return X - self.gamma * F.f1(X) +  self.gamma * interaction + self.sigma * np.sqrt(self.gamma) * noise
 
# class OLD:
#     # For now no direct mirror map
#     def __init__(self, meta=None):
#         self.gamma = meta["gamma"]
#         self.sigma = meta["sigma"]
    
#     def update(self, X, F, time_step=None):
#         noise = np.random.rand(*X.shape)
#         return X - self.gamma * F.f1(X) + self.sigma * np.sqrt(self.gamma) * noise

# class ISMD:
#     # For now no direct mirror map
#     def __init__(self, meta=None):
#         self.gamma = meta["gamma"]
#         self.sigma = meta["sigma"]
    
#     def update(self, X, F, time_step=None):
#         interaction = 1/float(len(X)) * np.array([np.sum(X - X[i], axis=0) for i in range(len(X))])
#         noise = np.random.rand(*X.shape)
#         return X - self.gamma * F.f1(X) +  self.gamma * interaction + self.sigma * np.sqrt(self.gamma) * noise


def linesearch(F, x_0, search_direction, c1, c2):
    """x0.shape = (dim)"""
    alpha = 1
    
    f0 = F.f(np.array([x_0]))[0]
    f1 = F.f1(np.array([x_0]))[0]
    dg = np.inner(search_direction, f1)
    
    counter = 0
    while True:
        # Check armijo rule 
        if F.f(np.array([x_0 + alpha * search_direction])) <= f0 + c1*alpha*dg:
            return alpha
            # check curvature condition
            if -np.inner(search_direction, F.f1(np.array([x_0 + alpha * search_direction]))) <= -c2 * dg:
                return alpha
        
        alpha = c2*alpha
        
        counter += 1
        if counter > 1000:
            return None
            raise Exception("Too many linesearch iterations. ")


def check_completion(grad, update_direction, delta):
    return np.inner(grad, update_direction)**2 / 2. < delta