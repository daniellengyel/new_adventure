import time, os, subprocess, socket


import pickle
import yaml

import datetime


def remove_png(dir_path):
    files = os.listdir(dir_path)
    for item in files:
        if item.endswith(".png"):
            os.remove(os.path.join(dir_path, item))

def save_config(experiment_folder, process_id, config):
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    with open(os.path.join(experiment_folder, process_id, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(experiment_folder, process_id):
    with open(os.path.join(experiment_folder, process_id, "config.yml"), "r") as f:
        config = yaml.load(f)
    return config

def save_opt_path(experiment_folder, process_id, opt_path):
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    with open(os.path.join(experiment_folder, process_id, "results.pkl"), "wb") as f:
        pickle.dump(opt_path, f)

def load_opt_path(experiment_folder, process_id):
    with open(os.path.join(experiment_folder, process_id, "results.pkl"), "rb") as f:
        all_paths = pickle.load(f)
    return all_paths


def get_file_stamp(prefix=None):
    """Return time and hostname as string for saving files related to the current experiment"""
    host_name = socket.gethostname()
    mydate = datetime.datetime.now()
    stamp = "{}_{}".format(mydate.strftime("%b%d_%H-%M-%S"), host_name)
    if prefix is not None:
        stamp = "{}_{}".format(prefix, stamp)
    return stamp

def animation_path_generator(experiment_folder):
    for process_id in os.listdir(experiment_folder):
        if "DS_Store" in process_id:
            continue
        animation_path = os.path.join("{}".format(experiment_folder), process_id)
        yield process_id, animation_path



