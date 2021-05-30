import numpy as np
import matplotlib.pyplot as plt


import sys, os, time
import pickle
import yaml

import psutil 

import new_adventure as new_adv

print(psutil.cpu_count(logical=False))

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

# TODO add dim parameter
np.random.seed(11)
dim = 750
num_barriers = dim * 8
dirs = np.random.normal(size=(num_barriers, dim)) # sample gaussian and normalize 
ws = dirs/np.linalg.norm(dirs, axis=1).reshape(-1, 1)
bs = np.ones(num_barriers)

# particle init 
config["domain_dim"] = dim
config["particle_init"] = "origin"
config["num_particles"] = 1
# config["x_range"] = [-0.002, 0.002]

# function

config["potential_name"] = "linear"
config["potential_meta"] = {"c": np.ones(dim)} #{"Q": np.array([[1, 0], [0, 1]]) , "estimation_type": "shift_estimator"} #[[1, 0], [0, 1]]

# optimization
config["optimization_name"] = "Newton_IPM" # "Newton_shift_est_IPM" # "BFGS" #  "Newton_IPM" #   #"Newton" 
config["optimization_meta"] = {"c1": 0.001, "c2": 0.7, 
								"barrier_type": "log", "delta": 0.01}


config["domain_name"] = "Polytope"
config["domain_meta"] = {"ws": ws, "bs": bs}

# config["domain_meta"] = {"ws": np.array([[1, 0], [1, 0], [2, 1], [2, 1]]), "bs": [-0.1, 0.1, 0.3, -0.3]}
# config["learning_rate"] = tune.grid_search(list(np.linspace(1, 40, 10)) + list(np.linspace(0.01, 1, 5)))
# config["temperature"] =  tune.grid_search(list(np.linspace(0.01, 4, 5))) # 0.2


# meta parameters (seed, how to save etc.)

config["seed"] = 0
config["return_full_path"] = True
config["num_steps"] = 15
config["num_processes"] = 1


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

