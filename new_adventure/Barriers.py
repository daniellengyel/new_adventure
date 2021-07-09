"""Here we define the barrier functions and at the same time the domain."""

import numpy as np
import time
import jax.numpy as jnp
import jax
from functools import partial
from jax import jit
import jax.random as jrandom

class LogPolytopeBarrier:

    def __init__(self, ws, bs):
        """ws.shape = (N, d), bs.shape = (N)"""
        self.ws = jnp.array(ws)
        self.bs = jnp.array(bs)
        self.dim = len(ws[0])

        self.jrandom_key = jrandom.PRNGKey(1)

    # @partial(jit, static_argnums=(0,))
    def _get_dists(self, xs):
        """We consider the sum of log barrier (equivalent to considering each barrier to be a potential function).
        Distance to a hyperplane w.x = b is given by | w.x/|w| - b/|w| |. We consider the absolute value of this, which follows the assumption that if we are on the a side of the hyperplane we stay there. 
        However, the signs tell us whether we are on the side of the hyperplane which is closer to the origin. If the sign is negative, then we are closer."""
        
        xs_len_along_ws = xs.dot(self.ws.T)/jnp.linalg.norm(self.ws, axis=1)
        hyperplane_dist = self.bs/jnp.linalg.norm(self.ws, axis=1)
        dists = xs_len_along_ws - hyperplane_dist # dists.shape = (N_x, N_ws)
        signs = 2*(dists * jnp.sign(hyperplane_dist) > 0) - 1
        return jnp.abs(dists), signs
    
    # @partial(jit, static_argnums=(0,))
    def dir_dists(self, xs, dirs):
        # we get the distance of the direction to every boundary (if parallel we have infty). We have w.(x0 + td) = b. Hence, t = (b - w.x0)/(w.d). So t is the scale to apply to d to get to the hyperplane. 
        xs_len_along_ws = xs.dot(self.ws.T)/(dirs.dot(self.ws.T))
        hyperplane_dist = self.bs/(dirs.dot(self.ws.T))
        dists = xs_len_along_ws - hyperplane_dist # dists.shape = (N_x, N_ws)
        signs = 2*(dists * jnp.sign(hyperplane_dist) > 0) - 1
        return jnp.abs(dists), signs
    
    # @partial(jit, static_argnums=(0,))
    def f(self, xs):
        """x.shape = (N, d). Outside of the bounded region around zero we are infinite."""
        dists, signs = self._get_dists(xs) 
        ret = -jnp.sum(jnp.log(dists), axis=1) # shape = (N_x)
        ret = jnp.where(jnp.any(signs > 0, axis=1), jnp.inf, ret)
        return ret

    # @partial(jit, static_argnums=(0,))
    def f1(self, xs, jrandom_key=None):
        dists, signs = self._get_dists(xs)
        grads = (1/dists * signs).dot((-self.ws / jnp.linalg.norm(self.ws, axis=1).T.reshape(-1, 1)))

        return grads

    # @partial(jit, static_argnums=(0,))
    def f2(self, xs):
        normalized_ws = self.ws / jnp.linalg.norm(self.ws, axis=1).reshape(-1, 1)
        dists, signs = self._get_dists(xs)
        hess = []
        for i in range(len(xs)):
            hess.append(jnp.dot(normalized_ws.T, 1/(dists[i].reshape(-1, 1))**2 * normalized_ws))

        
        hess = jnp.array(hess)
        return hess 
    
    # @partial(jit, static_argnums=(0,))
    def f2_inv(self, x):
        f2 = self.f2(x)
        return jnp.array([jnp.linalg.inv(f2[i]) for i in range(len(f2))])


        