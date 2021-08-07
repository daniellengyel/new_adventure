import numpy as np
import jax.numpy as jnp
from jax import random as jrandom
import jax
from functools import partial
from jax import lax
import time, sys
import pickle
from .Functions import LinearCombination
from .utils import get_barrier, get_potential
from .derivative_free_estimation import BFGS_update, beta_2E1, multilevel_beta_inv_2E1, neystrom_update_direction, multilevel_beta_newton_update_2E1_1E1, FD_2FD1
import os, psutil
process = psutil.Process(os.getpid())

def get_optimizer(config):
    if "Newton_IPM" == config["optimization_name"]:
        return Newton_IPM(config)
    elif "Newton_2B1_IPM" == config["optimization_name"]:
        return Newton_2B1_IPM(config)
    elif "BFGS" == config["optimization_name"]:
        return BFGS(config)
    elif "Newton_multilevel_2B1_IPM" == config["optimization_name"]:
        return Newton_multilevel_2B1_IPM(config)
    elif "Gradient_Descent" == config["optimization_name"]:
        return Gradient_Descent(config)
    elif "Newton_2FD1_IPM" == config["optimization_name"]:
        return Newton_2FD1_IPM(config)

class OptimizationBlueprint:
    def __init__(self, config):
        self.config = config
        self.barrier = get_barrier(config)
        self.obj = get_potential(config)
        self.dim = config["domain_dim"]
        self.c1 = config["optimization_meta"]["c1"]
        self.c2 = config["optimization_meta"]["c2"]
        self.delta = config["optimization_meta"]["delta"]
        self.with_neystrom = config["optimization_meta"]["with_neystrom"]
        self.automatic_diff = config["optimization_meta"]["automatic_diff"]
        self.jrandom_key = jrandom.PRNGKey(config["optimization_meta"]["jrandom_key"])
        self.linesearch = helper_linesearch(self.obj, self.barrier, self.c1, self.c2)
        self.loop_steps_remaining = config["num_total_steps"]
        self.verbose = True
        self.f1_var = self.config["potential_meta"]["f1_var"]

    def update(self, X, time_step, full_vals=False, full_path=False):
        assert not (full_vals and full_path)

        t = 4 * (0.5)**(time_step) 

        self.combined_F = LinearCombination(self.obj, self.barrier, [1, t], self.f1_var)

        if full_path:
            full_path_arr = [(X.copy(), time.time())]
        if full_vals:
            vals_arr = [(self.obj.f(X)[0], self.barrier.f(X)[0], self.combined_F.f(X)[0], time.time())]
        
        while self.loop_steps_remaining > 0:
            self.loop_steps_remaining -= 1
             
            # get search direction
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            f1 = self.combined_F.f1(X) 
            search_direction = self.step_getter(X, subkey, t)
            newton_decrement_squared = -f1[0].dot(search_direction)
            
            # check if valid search direction
            if newton_decrement_squared < 0:
                if full_path:
                    full_path_arr.append((X.copy(), time.time()))
                if full_vals:
                    vals_arr.append((self.obj.f(X)[0], self.barrier.f(X)[0], self.combined_F.f(X)[0], time.time()))
                continue
            newton_decrement = np.sqrt(np.abs(newton_decrement_squared))

            if self.verbose:
                print("Newton Decrement Squared", newton_decrement_squared)
                print("Obj", self.obj.f(X))
                print("Steps Remaining", self.loop_steps_remaining)
                print()

            # Check if completed
            if newton_decrement**2 < self.delta:
                break

            # do line search
            alpha = self.linesearch(X[0], 1/(1 + newton_decrement) * search_direction, t) 

            # update step
            X[0] = X[0] + 1/(1 + newton_decrement) * alpha * search_direction
            if full_path:
                full_path_arr.append((X.copy(), time.time()))
            if full_vals:
                vals_arr.append((self.obj.f(X)[0], self.barrier.f(X)[0], self.combined_F.f(X)[0], time.time()))
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

class Newton_IPM(OptimizationBlueprint):
    def __init__(self, config):
        super().__init__(config)
        self.d_prime = config["optimization_meta"]["d_prime"]

            

    def step_getter(self, X, jrandom_key, t):
        jrandom_key, subkey = jrandom.split(jrandom_key)
    
        if not self.with_neystrom:
            if self.automatic_diff:
                H_inv = jnp.linalg.inv(jax.hessian(self.combined_F.f)(X).reshape(X.shape[1], X.shape[1]))
                f1 = jax.grad(lambda x: self.combined_F.f(x)[0])(X)[0]
            else:
                H_inv = self.combined_F.f2_inv(X)[0]
                f1 = self.combined_F.f1(X, subkey)[0]

            search_direction = -H_inv.dot(f1)
        else:
            f1 = self.combined_F.f1(X, subkey)
            search_direction = -neystrom_update_direction(self.combined_F.f2(X)[0], self.d_prime, f1[0], jrandom_key)

        return search_direction

class Gradient_Descent(OptimizationBlueprint):
    def __init__(self, config):
        super().__init__(config)
    
    def step_getter(self, X, jrandom_key, t):
        return -self.combined_F.f1(X)[0]

class Newton_2FD1_IPM(OptimizationBlueprint):
    def __init__(self, config):
        super().__init__(config)
        self.d_prime = config["optimization_meta"]["d_prime"]
        self.num_samples = config["optimization_meta"]["num_samples"]
        self.err_bound = config["optimization_meta"]["2FD1_err_bound"]

    @partial(jax.jit, static_argnums=(0,))
    def step_getter(self, X, jrandom_key, t):
        combined_F = LinearCombination(self.obj, self.barrier, [1, t])
        jrandom_key, subkey = jrandom.split(jrandom_key)
        f1 = combined_F.f1(X, subkey)
        jrandom_key, subkey = jrandom.split(jrandom_key)
        approx_H = FD_2FD1(combined_F, X[0], self.err_bound, self.dim, subkey)

        H_inv = jnp.linalg.inv(approx_H)
        search_direction = -H_inv.dot(f1[0])
       
        return search_direction


class Newton_2B1_IPM(OptimizationBlueprint):
    def __init__(self, config):
        super().__init__(config)
        self.d_prime = config["optimization_meta"]["d_prime"]
        self.num_samples = config["optimization_meta"]["num_samples"]
        self.alpha = config["optimization_meta"]["alpha"]

    @partial(jax.jit, static_argnums=(0,))
    def step_getter(self, X, jrandom_key, t):
        combined_F = LinearCombination(self.obj, self.barrier, [1, t])
        jrandom_key, subkey = jrandom.split(jrandom_key)
        f1 = combined_F.f1(X, subkey)
        approx_H = beta_2E1(combined_F, X[0], self.alpha, self.num_samples, jrandom_key)
        
        if not self.with_neystrom:
            H_inv = jnp.linalg.inv(approx_H)
            search_direction = -H_inv.dot(f1[0])
        else:
            jrandom_key, subkey = jrandom.split(jrandom_key)
            search_direction = -neystrom_update_direction(approx_H, self.d_prime, f1[0], jrandom_key) 
            
        return search_direction

class Newton_multilevel_2B1_IPM(OptimizationBlueprint):
    def __init__(self, config):
        super().__init__(config)
        self.d_prime = config["optimization_meta"]["d_prime"]
        self.num_samples = config["optimization_meta"]["num_samples"]
        self.alpha = config["optimization_meta"]["alpha"]

    @partial(jax.jit, static_argnums=(0,))
    def step_getter(self, X, jrandom_key, t):
        combined_F = LinearCombination(self.obj, self.barrier, [1, t])
        return multilevel_beta_newton_update_2E1_1E1(combined_F, X[0], self.alpha, self.num_samples, self.d_prime, jrandom_key)



class BFGS(OptimizationBlueprint):
    def __init__(self, config):
        super().__init__(config)
        self.H_inv = np.eye(config["domain_dim"])
        self.X_prev = None

    def step_getter(self, X, jrandom_key, t):
        self.X_prev = X[0].copy()
        f1 = self.combined_F.f1(X)
        return -self.H_inv.dot(f1[0])
    
    def post_step(self, X, jrandom_key, t):
        self.H_inv = BFGS_update(self.combined_F, self.X_prev, X[0], self.H_inv)




def helper_linesearch(obj, barrier, c1, c2):

    def helper(x_0, search_direction, t):
        combined_F = LinearCombination(obj, barrier, [1, t])
        f0 = combined_F.f(x_0.reshape(1, -1))[0]
        f1 = combined_F.f1(x_0.reshape(1, -1))[0]
        dg = jnp.inner(search_direction, f1)

        def armijo_rule(alpha):
            return combined_F.f((x_0 + alpha * search_direction).reshape(1, -1))[0] > f0 + c1*alpha*dg
        
        def armijo_update(alpha):
            return c2*alpha
            
        alpha = 1
        while armijo_rule(alpha):
            alpha = armijo_update(alpha)

        return alpha

    return helper
