import numpy as np
import jax.numpy as jnp
from jax import random as jrandom
import jax
from functools import partial
from jax import lax
import time, sys
import pickle
from .Functions import LinearCombination, GaussianSmoothing
from .utils import get_barrier, get_potential
import os, psutil
process = psutil.Process(os.getpid())

def get_optimizer(config):
    if "GaussianSmoothing" == config["optimization_name"]:
        return GaussianSmoothingOptimization(config)
    elif "Gradient_Descent" == config["optimization_name"]:
        return Gradient_Descent(config)

class UnconstrainedOptimization:
    def __init__(self, config):
        self.config = config
        self.obj = get_potential(config)
        self.c1 = config["optimization_meta"]["c1"]
        self.c2 = config["optimization_meta"]["c2"]
        self.delta = config["optimization_meta"]["delta"]
        self.with_neystrom = config["optimization_meta"]["with_neystrom"]
        self.jrandom_key = jrandom.PRNGKey(config["optimization_meta"]["jrandom_key"])
        self.linesearch = helper_linesearch(self.obj, self.c1, self.c2)
        self.loop_steps_remaining = config["num_total_steps"]
        self.verbose = True

    def update(self, X, time_step, full_vals=False, full_path=False):
        assert not (full_vals and full_path)


        if full_path:
            full_path_arr = [(X.copy(), time.time())]
        if full_vals:
            vals_arr = [(self.obj.f(X)[0], time.time())]
        
        t = 0
        while self.loop_steps_remaining > 0:
            self.loop_steps_remaining -= 1
            t += 1

            # get search direction
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            search_direction, f1 = self.step_getter(X, subkey, t)
            newton_decrement_squared = -f1.dot(search_direction)
            
            # check if valid search direction
            if newton_decrement_squared < 0:
                if full_path:
                    full_path_arr.append((X.copy(), time.time()))
                if full_vals:
                    vals_arr.append((self.obj.f(X)[0], time.time()))
                continue
            newton_decrement = np.sqrt(np.abs(newton_decrement_squared))

            if self.verbose:
                print("Newton Decrement Squared", newton_decrement_squared)
                print("Obj", float(self.obj.f(X)[0]))
                print("Steps Remaining", self.loop_steps_remaining)
                print()

            # Check if completed
            if newton_decrement**2 < self.delta:
                break

            # do line search
            alpha = self.linesearch(X[0], search_direction, f1, t) 

            # update step #1/(1 + newton_decrement) *
            X[0] = X[0] +  alpha * search_direction
            if full_path:
                full_path_arr.append((X.copy(), time.time()))
            if full_vals:
                vals_arr.append((self.obj.f(X)[0], time.time()))
            # clean up after update (i.e. BFGS update)
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            self.post_step(X, subkey, t)

        if full_path:
            return X, full_path_arr
        if full_vals:
            return X, vals_arr
        return X, None


    def step_getter(self, X, jrandom_key, t):
        pass

    def post_step(self, X, jrandom_key, t):
        pass


class Gradient_Descent(UnconstrainedOptimization):
    def __init__(self, config):
        super().__init__(config)
    
    def step_getter(self, X, jrandom_key, t):
        return -self.obj.f1(X)[0], self.obj.f1(X)[0]


class GaussianSmoothingOptimization(UnconstrainedOptimization):
    def __init__(self, config):
        super().__init__(config)
        self.sigma = config["optimization_meta"]["sigma"]
        self.d_prime = config["optimization_meta"]["d_prime"]
        self.smoothing = GaussianSmoothing(self.obj, config["optimization_meta"]["num_samples"], config["optimization_meta"]["sigma"])

    @partial(jax.jit, static_argnums=(0,))
    def step_getter(self, X, jrandom_key, t):
        sigma = self.sigma
        # sigma = sigma * jnp.e**(- 0.0001 * t)

        jrandom_key, subkey = jrandom.split(jrandom_key)
        approx_H = self.smoothing.f2(X, subkey, sigma)[0]
        jrandom_key, subkey = jrandom.split(jrandom_key)
        f1 = self.smoothing.f1(X, subkey, sigma)[0]
        search_direction = -jnp.linalg.inv(approx_H).dot(f1)

        return search_direction, f1


# class BFGS(UnconstrainedOptimization):
#     def __init__(self, config):
#         super().__init__(config)
#         self.H_inv = np.eye(config["domain_dim"])
#         self.X_prev = None
        
#     def step_getter(self, X, jrandom_key, t):
#         self.X_prev = X[0].copy()
#         f1 = self.combined_F.f1(X)
#         return -self.H_inv.dot(f1[0])
    
#     def post_step(self, X, jrandom_key, t):
#         self.H_inv = BFGS_update(self.combined_F, self.X_prev, X[0], self.H_inv)


def helper_linesearch(obj, c1, c2):

    def helper(x_0, search_direction, f1, t):

        f0 = obj.f(x_0.reshape(1, -1))[0]
        dg = jnp.inner(search_direction, f1)

        def armijo_rule(alpha):
            return obj.f((x_0 + alpha * search_direction).reshape(1, -1))[0] > f0 + c1*alpha*dg
        
        def armijo_update(alpha):
            return c2*alpha
            
        alpha = 1
        while armijo_rule(alpha):
            alpha = armijo_update(alpha)

        return alpha

    return helper
