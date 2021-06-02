import new_adventure as new_adv
import new_adventure.derivative_free_estimation as dfe
import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np
import time

import multiprocessing
from os import getpid



def multi_second_shift_estimator_task(x_0, F, process_N, alpha):
    x_0 = np.ones(dim) / np.linalg.norm(np.ones(dim))
    # Notice, we should be sharing the radius. So some lock is probably needed to synchronise 
    sample_points = dfe.hit_run(x_0, F, x_0.shape[0], process_N, alpha)
    out_grads = F.f1(sample_points)
    return sample_points, out_grads

def new_beta_second_shift_estimator(F, x_0, alpha, N, num_processes=1, pool=None):
    if pool is None:
        pool = multiprocessing.Pool(num_processes)
    args = [[None, F, N//num_processes, alpha] for _ in range(num_processes)]
    ret = pool.starmap(multi_second_shift_estimator_task, args)
    
    sample_points = []
    out_grads = []
    for r in ret:
        sample_points.append(r[0])
        out_grads.append(r[1])
    sample_points = np.vstack(sample_points)
    out_grads = np.vstack(out_grads)

    second_shift_est = dfe.new_proper_cov(sample_points, out_grads)
    return second_shift_est.dot(np.linalg.inv(dfe.np_new_cov(sample_points)))

np.random.seed(10)
dim = 50
num_barriers = 2**12
dirs = np.random.normal(size=(num_barriers, dim)) # sample gaussian and normalize 
ws = dirs/np.linalg.norm(dirs, axis=1).reshape(-1, 1)
bs = np.ones(num_barriers)

barrier = new_adv.Barriers.LogPolytopeBarrier(ws, bs)

F = new_adv.Functions.Linear(np.ones(dim))

F = barrier # new_adv.Functions.LinearCombination(F, barrier, [1, 1])

xs = np.ones(dim) / np.linalg.norm(np.ones(dim))

times = []
for _ in range(1, 32):
    pool = multiprocessing.Pool(_)
    new_beta_second_shift_estimator(F, xs, 200, 5000, num_processes=_, pool=pool)

    start_time = time.time()
    new_beta_second_shift_estimator(F, xs, 200, 5000, num_processes=_, pool=pool)
    times.append(time.time() - start_time)

print(times)