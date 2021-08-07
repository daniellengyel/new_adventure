import numpy as np
import matplotlib.pyplot as plt


import sys, os, time
import pickle
import yaml

import psutil 

import new_adventure as new_adv

# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)
# config.update('jax_disable_jit', True)

# Job specific 
try:
    ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) - 1
except:
    ARRAY_INDEX = -1

def experiment_run(config_inp, path):
    # setup saveing and save config
    file_stamp = str(time.time())
    exp_folder = os.path.join(path, file_stamp)
    os.mkdir(exp_folder)
    new_adv.save_load.save_config(path, file_stamp, config_inp)

    # get opt and save
    a = time.time()
    results = new_adv.main.optimize(config_inp, verbose=True)
    print(time.time() - a)
    # print(results)
    new_adv.save_load.save_opt_path(path, file_stamp, results)


config = {}



# particle init 
dim = 500
config["domain_dim"] = dim
config["particle_init"] = "origin" # "function_specific"
config["num_particles"] = 1
# config["x_range"] = [-0.002, 0.002]

config["domain_name"] = "Polytope"
config["domain_meta"] = {"seed": 10, "num_barriers": dim * 5}

# function

config["potential_name"] = "linear"
config["potential_meta"] = {"seed": 10, "direction_name": "ones", "f1_var": None}

# optimization
config["optimization_type"] = "IPM"
config["optimization_name"] = "Newton_2FD1_IPM"
# "GaussianSmoothing"
# if ARRAY_INDEX == 0:
#     config["optimization_name"] = "Newton_2B1_IPM" # "Newton_shift_est_IPM" # "BFGS" #  "Newton_IPM" #   #"Newton" 
# elif ARRAY_INDEX == 1:
#     config["optimization_name"] = "BFGS"
# else:
#     config["optimization_name"] = "Newton_IPM"
# "Newton_2FD1_IPM"
# "Newton_multilevel_est_IPM"
# "Gradient_Descent"

config["optimization_meta"] = {"c1": 0.001, "c2": 0.7, "barrier_type": "log",
								"delta": 0.5, "jrandom_key": 0,
                                "with_neystrom": False, "d_prime": 1000, "num_samples": 5000, "alpha": 100, "sigma": 0.5, "2FD1_err_bound": 0.0001,
                                "automatic_diff": True}


# meta parameters (seed, how to save etc.)

config["seed"] = 0
config["return_full_path"] = False
config["num_path_steps"] = 15
config["num_total_steps"] = 300 

# --- Set up folder in which to store all results ---
folder_name = new_adv.save_load.get_file_stamp(config["optimization_name"])
cwd = os.environ["PATH_TO_ADV_FOLDER"]
folder_path = os.path.join(cwd, "experiments", config["potential_name"], folder_name)
print(folder_path)
os.makedirs(folder_path)

# msg = "Use FSM on the symmetric function."
# with open(os.path.join(folder_path, "description.txt"), "w") as f:
#     f.write(msg)

analysis = experiment_run(config, folder_path) #tune.run(lambda config_inp:  experiment_run(config_inp, folder_path), config=config)
print(analysis)

# TODO 
# derivative free: choose subspace 
# finite difference implementation 
# look into what panos sent