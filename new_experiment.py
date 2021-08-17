import jax.numpy as jnp
from jax import random as jrandom
import jax
from jax import jit



def samples(delta, num_samples):
    
    num_tries = 40
    res  = {i + 1: 0 for i in range(num_tries)}
    start_time = time.time()

    x = delta*jnp.ones(num_samples)
    f = jnp.zeros(num_samples)
    for i in range(num_tries):
        x += (-1)**i * jnp.random.uniform(size=(num_samples,))

        res[i + 1] += jnp.sum((jnp.abs(x) > 0.5)*(f == 0))
        f = f + 1*(jnp.abs(x) > 0.5)
    start_time = time.time()

    p1 = jnp.array([[2*i + 1, res[2*i + 1]/float(num_samples)] for i in range(num_tries//2)])
    p2 = jnp.array([[2*i + 2, res[2*i + 2]/float(num_samples)] for i in range(num_tries//2)])
    

    return (sum(p1[:, 1]) - sum(p2[:, 1]))/num_samples

def binary_search(delta, eps, lower_bound, upper_bound, num_samples):
    diff = samples(delta, num_samples)
    while abs(diff) > eps:

        if  diff > 0:
            lower_bound, upper_bound, delta = lower_bound, delta, (lower_bound + delta)/2.
        else:
            lower_bound, upper_bound, delta = delta, upper_bound, (upper_bound + delta)/2.

        diff = binary_search(delta., eps, lower_bound, upper_bound, num_samples)
    return delta

eps = 1e-11
num_samples = int(1e5)
binary_search(-0.25, eps, -0.5, 0, num_samples)
