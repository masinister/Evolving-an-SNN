import numpy as np

def random(m,n):
    s = np.random.random_sample((m,n)).flat
    z = [np.random.choice([0, x]) for x in s]
    return np.array(z).reshape((m,n))

def all2all(m,n):
    return np.random.random_sample((m,n))

def allBut1(m):
    # it is assumed that you are connecting the same layer to itself
    # remove connection along the diagonal to disallow self connected neurons
    x = (np.ones((m,m)) - np.eye(m))*np.random.random_sample((m,m))
    return x

def grid(m,n):
    g = np.zeros((m*n,m*n))
    for i in range(m):
        for j in range(n-1):
            g[m*i + j][m*i + j+1] = g[m*i + j+1][m*i + j] = 1
    for i in range(m-1):
        for j in range(n):
            g[n*(i+1) + j][n*i + j] = g[n*i + j][n*(i+1) + j] = 1
    return g


schemes = {"random": random, "all2all": all2all,"allBut1": allBut1, "grid": grid}


def get(name):
        return schemes[name]
