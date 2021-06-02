from .Functions import Quadratic, Ackley, Linear
from .Barriers import LogPolytopeBarrier
import numpy as np
import random

def get_potential(config):
    # get potential
    if config["potential_name"] == "quadratic":
        potential_meta = config["potential_meta"]
        F = Quadratic(potential_meta["Q"])
        if potential_meta["estimation_type"] == "shift_estimator":
            F = ShiftEstimation(F, 0.1, np.eye(len(potential_meta["Q"])), 1000)
    if config["potential_name"] == "linear":
        potential_meta = config["potential_meta"]
        # Set Linear Objective
        c = potential_meta["c"]
        F = Linear(c)
        # if potential_meta["estimation_type"] == "shift_estimator":
        #     F = ShiftEstimation(F, 0.1, np.eye(len(c)), 1000)
    elif config["potential_name"] == "Ackley":
        F = Ackley()
    else:
        raise ValueError("Does not support given function {}".format(config["potential_name"]))
    return F    

def get_barrier(config):
    if (config["optimization_meta"]["barrier_type"] == "log") and (config["domain_name"] == "Polytope"):
        domain_meta = config["domain_meta"]
        B = LogPolytopeBarrier(domain_meta["ws"], domain_meta["bs"])
    else: 
        raise ValueError("Does not support given barrier type {} with domain {}".format(config["optimization_meta"]["barrier_type"], config["domain_name"]))
    return B
    
def get_particles(config):
    # get start_pos
    # TODO fix dimension
    if config["particle_init"] == "uniform":
        num_particles = config["num_particles"]
        particles = [[np.random.uniform(config["x_range"][0], config["x_range"][1])] for _ in range(num_particles)]
    elif config["particle_init"] == "2d_uniform": # for now same range for all dimensions
        num_particles = config["num_particles"]
        x_low, x_high = config["x_range"]
        particles = [[np.random.uniform(x_low, x_high), np.random.uniform(x_low, x_high)] for _ in range(num_particles)]
    elif config["particle_init"] == "2d_position": # for now same range for all dimensions
        num_particles = config["num_particles"]
        x_low, x_high = config["x_range"]
        p = np.array(config["particle_init"]["params"]["position"])
        assert ((x_low <= p) & (p <= x_high)).all()
        particles = [p for _ in range(num_particles)]
    elif config["particle_init"] == "origin":
        num_particles = config["num_particles"]
        particles = [np.zeros(config["domain_dim"]) for _ in range(num_particles)]
    else:
        raise ValueError("Does not support given function {}".format(config["particle_init"]))
    return np.array(particles)

def get_config_to_id_map(configs):
    map_dict = {}

    for net_id in configs:
        conf = configs[net_id]
        tmp_dict = map_dict
        for k, v in conf.items():
            if "potential" in k:
                continue
            if isinstance(v, list):
                v = tuple(v)

            if k not in tmp_dict:
                tmp_dict[k] = {}
            if v not in tmp_dict[k]:
                tmp_dict[k][v] = {}
            prev_dict = tmp_dict
            tmp_dict = tmp_dict[k][v]
        prev_dict[k][v] = net_id
    return map_dict

def get_ids(config_to_id_map, config):
    if not isinstance(config_to_id_map, dict):
        return [config_to_id_map]
    p = list(config_to_id_map.keys())[0]

    ids = []
    for c in config_to_id_map[p]:
        if isinstance(config[p], list):
            config_compare = tuple(config[p])
        else:
            config_compare = config[p]
        if (config_compare is None) or (config_compare == c):
            ids += get_ids(config_to_id_map[p][c], config)
    return ids

def different_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    a[:, 14] = 0
    return (a[0] != a[1:]).any(0)

def get_hp(cfs):
    filter_cols = different_cols(cfs)
    hp_names = cfs.columns[filter_cols]
    hp_dict = {hp: cfs[hp].unique() for hp in hp_names}
    return hp_dict


def find_basins(grad_f, bounds, smallest_diameter, eps, plateau_eps):
    """Finds all attractive basisn with smallest diameter given. eps tells us when we can stop finding zero i.e. when we have found x s.t. |f(x)| < eps"""
    n = (bounds[1] - bounds[0]) / (smallest_diameter * 2)
    xs = np.linspace(bounds[0], bounds[1], int(n) + 1)
    outs = grad_f(np.array([xs]))[0]

    basins = []
    left_p = bounds[0]
    passed_minimum = False

    for i in range(1, int(n)):
        out = outs[i]

        if (outs[i - 1] >= 0) and (outs[i] <= 0):
            right_p = binary_search_zero(grad_f, np.array([xs[i - 1]]), np.array([xs[i]]), eps)[0]
            basins.append([left_p, right_p])
            left_p = right_p
        elif (abs(outs[i]) < plateau_eps):
            if not passed_minimum:
                passed_minimum = True
            else:
                right_p = xs[i]
                basins.append([left_p, right_p])
                left_p = right_p
                passed_minimum = False

    if left_p != xs[-1]:
        basins.append([left_p, xs[-1]])
    return np.array(basins)


def binary_search_zero(f, a, b, eps):
    while True:
        new_x = (b + a) / 2.
        if abs(f(new_x)) < eps:
            return new_x
        if any(f(new_x) < 0):
            b = new_x
        else:
            a = new_x
