import numpy as np

def get_optimizer(config):
    if "Newton" == config["optimization_name"]:
        return Newton()
    elif "ISMD" == config["optimization_name"]:
        return ISMD(config["optimization_meta"])
    elif "OLD" == config["optimization_name"]:
        return OLD(config["optimization_meta"])

class Newton:
    def __init__(self, config=None):
        pass

    def update(self, X, F, time_step=None):
        H_inv = F.f2_inv(X)
        f1 = F.f1(X)
        return X - np.array([H_inv[i].dot(f1[i]) for i in range(len(f1))])


class ISMD:
    # For now no direct mirror map
    def __init__(self, meta=None):
        self.gamma = meta["gamma"]
        self.sigma = meta["sigma"]
    
    def update(self, X, F, time_step=None):
        interaction = 1/float(len(X)) * np.array([np.sum(X - X[i], axis=0) for i in range(len(X))])
        noise = np.random.rand(*X.shape)
        return X - self.gamma * F.f1(X) +  self.gamma * interaction + self.sigma * np.sqrt(self.gamma) * noise

class OLD:
    # For now no direct mirror map
    def __init__(self, meta=None):
        self.gamma = meta["gamma"]
        self.sigma = meta["sigma"]
    
    def update(self, X, F, time_step=None):
        noise = np.random.rand(*X.shape)
        return X - self.gamma * F.f1(X) + self.sigma * np.sqrt(self.gamma) * noise

class ISMD:
    # For now no direct mirror map
    def __init__(self, meta=None):
        self.gamma = meta["gamma"]
        self.sigma = meta["sigma"]
    
    def update(self, X, F, time_step=None):
        interaction = 1/float(len(X)) * np.array([np.sum(X - X[i], axis=0) for i in range(len(X))])
        noise = np.random.rand(*X.shape)
        return X - self.gamma * F.f1(X) +  self.gamma * interaction + self.sigma * np.sqrt(self.gamma) * noise