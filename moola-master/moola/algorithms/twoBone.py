from .optimisation_algorithm import *
from math import sqrt
from ..adaptors.adaptor import convert_to_moola_dual_vector
from ..adaptors.dolfin_vector import DolfinPrimalVector

import jax.random as jrandom 
import numpy as np
import jax.numpy as jnp
import jax

# not ideal since we only need it to access the Function method.
from dolfin import *
from dolfin_adjoint import *

import time

class LinearOperator(object):

    def __init__(self, matvec):
        self.matvec = matvec

    def __mul__(self,x):
        return self.matvec(x)

    def __rmul__(self,x):
        return NotImplemented

    def __call__(self,x):
        return self.matvec(x)


def dual_to_primal(x):
    return x.primal()

class BetaApproximations(LinearOperator):
    '''
    This class implements the limit-memory BFGS approximation of the inverse Hessian.
    '''
    def __init__(self, jrandom_key):
        alpha = 10
        N = 1000
        R = 0.1
        self.alpha = alpha
        self.N = N
        self.R = R
        self.d_prime = None

        self.jrandom_key = jrandom_key


    def multi_two_B_one(self, obj, x_0):
        """Makes use of Matrix-Vector Product."""
        alpha = self.alpha
        N = self.N
        R = self.R 
        d_prime = self.d_prime

        x_0_np = x_0.data.vector().get_local()
        f = x_0.copy()
        
        d = len(x_0_np)
        d_prime = d

        jrandom_key, subkey = jrandom.split(self.jrandom_key)
        U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
        U = jnp.eye(d)[U_idxs].T # (d, d')

        jrandom_key, subkey = jrandom.split(jrandom_key)
        sample_points = beta_sampling(x_0_np, x_0_np.shape[0], N, alpha, R, subkey, chosen_basis_idx=U_idxs)  
        ru = (sample_points - jnp.mean(sample_points, axis=0)).T # (d, N)
        
        jrandom_key, subkey = jrandom.split(jrandom_key)
        out_grads = []
        # parallel ? 
        curr_f = Function(x_0.data.function_space())
        curr_f = DolfinPrimalVector(curr_f)

        for i in range(N):
            f.data.vector().set_local(np.array(sample_points[i]))
            curr_f.assign(f) # there is an issue with hashing if i assign directly to f. Weird stuff. 
            obj(curr_f)            
            out_grads.append(obj.derivative(curr_f).data.vector().get_local())
            
        out_grads = jnp.array(out_grads)

        gradF = jnp.array(obj.derivative(x_0).data.vector().get_local())

        grad_X_low_inv = jnp.linalg.inv(out_grads.T[U_idxs].dot(ru[U_idxs].T)/float(N))
        cov = jnp.cov(sample_points.T)

        self.jrandom_key = jrandom_key

        update_dir_np = -cov.dot(U.dot(grad_X_low_inv.dot(U.T.dot(gradF))))
        update_dir = Function(x_0.data.function_space())
        update_dir.vector().set_local(update_dir_np)
        return DolfinPrimalVector(update_dir)


class twoBone(OptimisationAlgorithm):
    """
        Implements the tBone method.
     """
    def __init__(self, problem, initial_point = None, options={}, hooks={}, **args):
        '''
        Initialises the tBone algorithm.

        Valid options are:

         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - jtol: Functional reduction stopping tolerance: |j - j_prev| < tol. Default: 1e-4.
            - rjtol: Relative version of the functional reduction stopping criterion. Default: 1e-6.
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol. Default: 1e-4.
            - rgtol: Relative version of the gradient stopping criterion. Default: 1e-5.
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200.
            - display: dis/enable outputs to screen during the optimisation (higher number yields more output). Default: 2
            - line_search: defines the line search algorithm to use. Default: strong_wolfe
            - line_search_options: additional options for the line search algorithm. The specific options read the help
              for the line search algorithm.
            - an optional callback method which is called after every optimisation iteration.
         * hooks: A dictionariy containing user-defined "hook" functions that are called at certain events during the optimisation.
            - before_iteration: Is called before each iteration.
            - after_iteration: Is called after each each iteration.
          '''
        # Set the default options values
        self.problem = problem
        self.set_options(options)
        jrandom_key = jrandom.PRNGKey(0)
        self.BetaApprox = BetaApproximations(jrandom_key)

        self.linesearch = get_line_search_method(self.options['line_search'], self.options['line_search_options'])
        self.data = {'control'   : initial_point,
                     'iteration' : 0,
                    }


    @classmethod
    def default_options(cls):
        # this is defined as a function to prevent defaults from being changed at runtime.
        default = OptimisationAlgorithm.default_options()
        default.update(
            # generic parameters:
            {"jtol"                   : 1e-4,
             "rjtol"                  : 1e-6,
             "gtol"                   : 1e-4,
             "rgtol"                  : 1e-5,
             "maxiter"                :  200,
             "display"                :    2,
             "line_search"            : "strong_wolfe",
             "line_search_options"    : {"ftol": 1e-3, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1},
             "callback"               : None,
             "record"                 : ("grad_norm", "objective"),
             })
        return default

    def __str__(self):
        s = "tBone method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.options['line_search']
        s += "Maximum iterations:\t %i\n" % self.options['maxiter']
        return s

    def solve(self):
        '''
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem
         '''
        self.display( self.__str__(), 1)

        objective = self.problem.obj
        options = self.options

        xk = self.data['control']
        it = self.data['iteration']

        # compute initial data. 
        J = objective(xk)
        dJ_xk = objective.derivative(xk)
        dJ_norm = dJ_xk.primal_norm()
        self.update({'objective' : J,
                     'initial_J' : J,
                     'grad_norm' : dJ_norm,
                     'initial_grad_norm': dJ_norm})
        self.record_progress()

        # Start the optimisation loop
        while self.check_convergence() == 0:
            self.display(self.iter_status, 2)


            # compute search direction
            # start_time = time.time()
            pk = self.BetaApprox.multi_two_B_one(objective, xk)
            # print(pk.data.vector().get_local())
             # print(repr(pk.data.vector().get_local()))

            # if it == 0:
            #     # then normalize;
            #     pk.scale( 1. / dJ_xk.primal_norm())
            # print(pk.data.vector().get_local())

            # do a line search and update
            xk, ak = self.do_linesearch(objective, xk, pk)
            pk.scale(ak)
            J, oldJ = objective(xk), J

            it += 1

            # store current iteration variables
            self.update({'iteration' : it,
                         'control'   : xk,
                         'grad_norm' : dJ_xk.primal_norm(),
                         'delta_J'   : oldJ-J,
                         'objective' : J,})
            self.record_progress()
        self.display(self.convergence_status, 1)
        self.display(self.iter_status, 1)
        return self.data


def beta_sampling(x_0, dim, N, alpha, radius, new_jrandom_key, chosen_basis_idx=None):
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

    new_jrandom_key, subkey = jrandom.split(new_jrandom_key)
    beta_p = (jrandom.beta(subkey, alpha, alpha, shape=(N, 1)) - 0.5) * 2 * radius 

    res = []
    res += dirs * beta_p

    return x_0 + jnp.array(res)


def beta_covariance(dim, R, alpha):
    """Returns scalar sigma, since the covariance matrix is sigma*I_{dim}. So we are saving space."""
    return (R**2)/(1 + 2 * alpha) * 1/dim


