import numpy as np

from .utils import *
from .Functions import *
from .Optimizers import get_optimizer

import time

"""Particle output will have the shape [[[]-particles]-timesteps] or paths[time_step][particle_idx]"""




def optimize(config, verbose=False):

    # init num_particles
    all_paths = []
    p_start = get_particles(config)
    p_curr = np.array([np.array(p) for p in p_start])
    p_num_particles = len(p_start)

    num_steps = config["num_steps"]

    # get optimization method
    opt = get_optimizer(config) # gets the barrier and objective function

    # TODO init saving
    # file_stamp = str(time.time())  # get_file_stamp()
    # writer = SummaryWriter("{}/runs/{}".format(folder_path, file_stamp))

    if config["seed"] is not None:
        np.random.seed(config["seed"])
    

    for t in range(num_steps):
        print(t)

        full_update_path = opt.update(p_curr, t)
        all_paths.append(full_update_path)
        p_curr = full_update_path[-1][0].copy()

        # if (t % 50) == 0 and verbose:
        #     print("Iteration", t)
        #     print("diff", np.abs(func(x_next) - func(x_curr)))
        
    return all_paths
