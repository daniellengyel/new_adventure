import new_adventure as new_adv
import new_adventure.derivative_free_estimation as dfe
import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np
import time

import multiprocessing
from os import getpid



def f(i):
    print(i)
    return 1

def process_task(x_0, F, process_N, alpha, L_sample_points, L_grads, pid):
    x_0 = np.ones(dim) / np.linalg.norm(np.ones(dim))

    # print(process_N)
    # print(pid)
    # Notice, we should be sharing the radius. So some lock is probably needed to synchronise 
    sample_points = dfe.hit_run(x_0, F, x_0.shape[0], process_N, alpha)
    out_grads = F.f1(sample_points)
#     L_sample_points.append(sample_points)
#     L_grads.append(out_grads)

def new_beta_second_shift_estimator(F, x_0, alpha, N, control_variate=True, num_processes=1):


#     with multiprocessing.Manager() as manager:
    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool(processes=num_processes)
    L_sample_points = None # [] #manager.list()
    L_grads = None # [] #manager.list()
    pool_workers = []
    for i in range(num_processes):
#             p = pool.apply_async(f, (i,))
        p = pool.apply_async(process_task, (None, F, N // num_processes, alpha, L_sample_points, L_grads, i,))
        pool_workers.append(p)
#         p.wait()
#             print(i)
#         print(pool_workers)
    for pw in pool_workers:
        a = time.time()
        pw.wait()
        print(pw)
        print(time.time() - a)
#     [pw.wait() for pw in pool_workers]
#         print([pw.get(timeout=1) for pw in pool_workers])


#         sample_points = np.vstack(list(L_sample_points))
#         out_grads = np.vstack(list(L_grads))

#     second_shift_est = dfe.new_proper_cov(sample_points, out_grads)
#     return second_shift_est.dot(np.linalg.inv(dfe.np_new_cov(sample_points)))

np.random.seed(10)
dim = 2500
num_barriers = dim * 2
dirs = np.random.normal(size=(num_barriers, dim)) # sample gaussian and normalize 
ws = dirs/np.linalg.norm(dirs, axis=1).reshape(-1, 1)
bs = np.ones(num_barriers)

barrier = new_adv.Barriers.LogPolytopeBarrier(ws, bs)

F = new_adv.Functions.Linear(np.ones(dim))

F = barrier # new_adv.Functions.LinearCombination(F, barrier, [1, 1])

xs = np.ones(dim) / np.linalg.norm(np.ones(dim))

a = time.time()
c = new_beta_second_shift_estimator(F, xs, 200, 1000, control_variate=True, num_processes=8)
print(time.time() - a)

a = time.time()
c = new_beta_second_shift_estimator(F, xs, 200, 1000, control_variate=True, num_processes=4)
print(time.time() - a)