import numpy as np


def random(m,n):
    return np.random.random_sample((m,n))

schemes = {"random": random,}

def get(name):
        return schemes[name]
