import numpy as np

class Schemes:

    def random(m,n):
        return np.random.random_sample(m,n)

    schemes = {"random": random,}

    @staticmethod
    def get(name):
        return schemes[name]
