'''
  Solves min_x 1/2 * (x[0]**2 + x[1]**2 + ... +x[n-1]**2),
     x in R^n
'''
from moola import *
import numpy as np
from numpy.random import random

class MyFunctional(Functional):
    def __call__(self, x):
        arr = x.data
        return 0.5*sum(arr**2)

    def derivative(self, x):
        arr = x.data
        return NumpyDualVector(arr)


def test():
    init_control = NumpyPrimalVector(random(5))

    obj = MyFunctional()
    prob = Problem(obj)

    # Solve with steepest descent
    options = {}
    solver = SteepestDescent(prob, init_control, options=options)
    sol = solver.solve()
    assert sol["control"].norm() < 1e-10
    assert sol["iteration"] == 1

    # Solve with Fletcher-Reeves method 
    options = {'gtol': 1e-10, "beta_rule": "fletcher-reeves"}
    solver = NonLinearCG(prob, init_control, options=options)
    sol = solver.solve()
    assert sol["control"].norm() < 1e-10
    assert sol["iteration"] == 1
