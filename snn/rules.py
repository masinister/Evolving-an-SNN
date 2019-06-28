import numpy as np


def STDP(m,n):
    return np.random.random_sample((m,n))

rules = {"STDP": STDP,}

def get(name):
        return rules[name]
