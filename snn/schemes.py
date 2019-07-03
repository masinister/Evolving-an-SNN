import numpy as np

'''
Connection schemes generate an initial weighted adjacency matrix specifying which
presynap neurons are connected to what postsynap neurons
'''

def random(m,n):
    '''
    Assign random weights and randomly choose connections to zero out (i.e. disconnect)
    '''
    s = np.random.random_sample((m,n)).flat
    z = [np.random.choice([0, x]) for x in s]
    return np.array(z).reshape((m,n))

def all2all(m,n):
    '''
    Connect every presynap neruon to every postsynap neuron with random weights
    '''
    return np.random.random_sample((m,n))

def allBut1(m):
    '''
    This scheme is meant to connect a layer to itself, with random weights.
    Connect every neuron to every other neuron except to itself.
    '''
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
