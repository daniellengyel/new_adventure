from .Functions import Quadratic, Ackley, Linear, Tesselated, MAXQ
from .Barriers import LogPolytopeBarrier
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
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
        if potential_meta["direction_name"] == "ones":
            c = np.ones(config["domain_dim"])
        else:
            raise ValueError("Does not support given function {} with direction {}.".format(config["potential_name"], potential_meta["direction_name"]))
        F = Linear(c)
    elif config["potential_name"] == "Ackley":
        F = Ackley()
    elif config["potential_name"] == "TesselatedQuadratic":
        jrandom_key = jrandom.PRNGKey(config["potential_meta"]["seed"])
        # Get Q and b 
        jrandom_key, subkey = jrandom.split(jrandom_key)
        Q = jrandom.normal(key=subkey, shape=(config["domain_dim"], config["domain_dim"], ))
        jrandom_key, subkey = jrandom.split(jrandom_key)
        b = jrandom.normal(key=subkey, shape=(config["domain_dim"], ))

        q_func = Quadratic(Q, b)
        F = Tesselated(q_func, config["potential_meta"]["num_tesselations"], {"bound": config["potential_meta"]["tesselation_domain"], "dim":config["domain_dim"]}, jrandom_key)
    elif config["potential_name"] == "MAXQ":
        F = MAXQ()

    else:
        raise ValueError("Does not support given function {}".format(config["potential_name"]))
    return F    

def get_barrier(config):
    if (config["optimization_meta"]["barrier_type"] == "log") and (config["domain_name"] == "Polytope"):
        domain_meta = config["domain_meta"]
        np.random.seed(domain_meta["seed"])
        num_barriers = domain_meta["num_barriers"]
        dim = config["domain_dim"]
        dirs = np.random.normal(size=(num_barriers, dim)) # sample gaussian and normalize 
        ws = dirs/np.linalg.norm(dirs, axis=1).reshape(-1, 1)
        bs = np.ones(num_barriers)
        B = LogPolytopeBarrier(ws, bs)
    else: 
        raise ValueError("Does not support given barrier type {} with domain {}".format(config["optimization_meta"]["barrier_type"], config["domain_name"]))
    return B
    
def get_particles(config):
    # get start_pos
    # TODO fix dimension
    # if config["particle_init"] == "uniform":
    #     num_particles = config["num_particles"]
    #     particles = [[np.random.uniform(config["x_range"][0], config["x_range"][1])] for _ in range(num_particles)]
    # elif config["particle_init"] == "2d_uniform": # for now same range for all dimensions
    #     num_particles = config["num_particles"]
    #     x_low, x_high = config["x_range"]
    #     particles = [[np.random.uniform(x_low, x_high), np.random.uniform(x_low, x_high)] for _ in range(num_particles)]
    # elif config["particle_init"] == "2d_position": # for now same range for all dimensions
    #     num_particles = config["num_particles"]
    #     x_low, x_high = config["x_range"]
    #     p = np.array(config["particle_init"]["params"]["position"])
    #     assert ((x_low <= p) & (p <= x_high)).all()
    #     particles = [p for _ in range(num_particles)]
    if config["particle_init"] == "origin":
        num_particles = config["num_particles"]
        particles = [jnp.zeros(config["domain_dim"]) for _ in range(num_particles)]
    elif config["particle_init"] == "function_specific":
        if config["potential_name"] == "MAXQ":
            num_particles = config["num_particles"]
            particles = [jnp.array([-4.08653571, -4.13286989, -3.95076614, -4.15547513, -3.49994787, -3.57026795,
  -3.7938423 , -4.12136015, -3.65786342, -4.07157976, -4.2349544 , -3.428094,
  -4.23294738, -4.23543057, -4.01950864, -4.22205812, -4.1754283 , -3.99612619,
  -3.41018688, -4.0687547 , -3.67556994, -3.84908772, -3.98871223, -3.86126325,
  -3.36466582, -3.3261831 , -4.21375671, -4.29674189, -4.21564043, -4.28309616,
  -4.23919035, -3.89637401, -4.29584853, -3.82188051, -4.1372486 , -4.20578596,
  -3.83789293, -4.23093461, -3.8112112 , -3.7979367 , -4.16727839, -3.93755917,
  -3.86394499, -3.83850989, -4.28515094, -3.93463323, -3.61277935, -3.66578832,
  -3.68904054, -4.07395594, -3.39706506, -3.8093003 , -3.90893972, -4.2108115,
  -4.03895986, -4.23518914, -3.68259048, -3.69635024, -3.28611546, -4.30379517,
  -4.07922212, -3.50977417, -4.1484698 , -3.53270307, -4.06869231, -3.41841916,
  -4.05838022, -3.68581409, -4.07195305, -3.90964769, -3.96420945, -4.09866475,
  -4.06873767, -3.74638503, -4.12283425, -3.56686407, -3.47421458, -3.45025409,
  -3.65747365, -3.70107743, -4.15962439, -4.29960385, -4.11018416, -3.18124287,
  -3.56384593, -3.65123884, -4.00761417, -4.25759951, -3.88235651, -3.69468319,
  -4.03985722, -4.06003586, -4.08625979, -4.11675827, -4.16695674, -3.65293747,
  -4.25515595, -3.83481097, -3.9418322 , -3.30547283,])] # [jnp.array([1. * i * (1 - 2*int(i + 1 > config["domain_dim"]/2.)) for i in range(config["domain_dim"])]) for _ in range(num_particles)]
    else:
        raise ValueError("Does not support given function {}.".format(config["particle_init"]))
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

# @jit
def woordbury_update(A_inv, C_inv, U, V):
    """(A + UCV)^{-1} = A_inv - A_inv U (C_inv + V A_inv U)^{-1} V A_inv"""
    mid_inv = jnp.linalg.inv(C_inv + jnp.dot(V, jnp.dot(A_inv, U)))
    return A_inv - jnp.dot(A_inv, jnp.dot(U, jnp.dot(mid_inv, jnp.dot(V, A_inv))))