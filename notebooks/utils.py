import numpy as np


def convert_opt_path(opt_path):
    res = []
    start_time = None
    for t in range(len(opt_path)):
        p = opt_path[t]
        for i in range(len(p)):
            if start_time is None:
                start_time = p[i][-1]
            res.append([p[i][0], p[i][-1] - start_time])
    return np.array(res).reshape(-1, 2)