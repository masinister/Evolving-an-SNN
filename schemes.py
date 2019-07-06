import numpy as np
import networkx as nx

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

def one2one(m):
    '''
    This scheme connects 2 layers of equal population size in a one-to-one fashion.
    '''
    return np.eye(m)


def grid(m,n):
    g = np.zeros((m*n,m*n))
    for i in range(m):
        for j in range(n-1):
            g[m*i + j][m*i + j+1] = g[m*i + j+1][m*i + j] = 1
    for i in range(m-1):
        for j in range(n):
            g[n*(i+1) + j][n*i + j] = g[n*i + j][n*(i+1) + j] = 1
    return g




def local(m,n):
    '''
    This scheme emebeds the neurons into a 2D rectangular lattice in which every
    neuron is connected to any neurons within a certain distance from itself.
    '''
    d = {}
    count = 0
    for i in range(m):
        for j in range(n):
            d[count] = np.array([j,i])
            count+=1

    G = nx.Graph()
    G.add_nodes_from(d.keys())
    for n, pos in d.items():
        G.node[n]['position'] = pos

    for i in range(len(d)):
        for j in range(i+1,len(d)):
            dist = np.linalg.norm(d[i]-d[j])
            if dist <= np.sqrt(8)+.001:
                G.add_edge(i,j,weight=1/dist)

    return nx.to_numpy_matrix(G)






schemes = {"random": random, "all2all": all2all,"allBut1": allBut1, \
           "one2one": one2one, "grid": grid, "local": local}


def get(name):
        return schemes[name]
