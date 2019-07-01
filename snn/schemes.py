import numpy as np

def random(m,n):
    return np.random.random_sample((m,n))

def all2all(m,n):
    return np.ones((m,n))

def allBut1(m,n,i,j):
    x = np.ones((m,n))
    x[i,j] = 0
    return x

schemes = {"random": random, "all2all": all2all,"allBut1": allBut1,}

def get(name):
        return schemes[name]
