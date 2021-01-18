import numpy as np
import matplotlib.pyplot as plt
from ray import tune


import sys, os
import pickle
import yaml

from save_load import *
from optimize import *


def experiment_run(config_inp, path):
    # setup saveing and save config
    file_stamp = str(time.time())
    exp_folder = os.path.join(path, file_stamp)
    os.mkdir(exp_folder)
    save_config(path, file_stamp, config_inp)

    # get opt and save
    results = optimize(config_inp, exp_folder)
    save_opt_path(path, file_stamp, results)


config = {}

# TODO add dim parameter

# particle init 
config["domain_dim"] = 1
config["particle_init"] = "uniform"
config["num_particles"] = 25
config["x_range"] = [80, 100]

# function

config["potential_name"] = "quadratic"
config["potential_meta"] = {"Q": [[2]]}

# optimization
config["optimization_name"] = "ISMD"
config["optimization_meta"] = {"gamma": 0.1, "sigma": 0.1}
# config["learning_rate"] = tune.grid_search(list(np.linspace(1, 40, 10)) + list(np.linspace(0.01, 1, 5)))
# config["temperature"] =  tune.grid_search(list(np.linspace(0.01, 4, 5))) # 0.2


# meta parameters (seed, how to save etc.)

config["seed"] = 0
config["return_full_path"] = True
config["num_steps"] = 10

# TODO figure out how to enforce a domain
config["domain_enforcer"] = None # "hyper_cube_enforcer"
config["domain_enforcer_params"] = 0.2



# --- Set up folder in which to store all results ---
folder_name = get_file_stamp(config["optimization_name"])
cwd = os.environ["PATH_TO_ADV_FOLDER"]
folder_path = os.path.join(cwd, "experiments", config["potential_name"], folder_name)
print(folder_path)
os.makedirs(folder_path)

# msg = "Use FSM on the symmetric function."
# with open(os.path.join(folder_path, "description.txt"), "w") as f:
#     f.write(msg)

analysis = tune.run(lambda config_inp:  experiment_run(config_inp, folder_path), config=config)
print(analysis)

