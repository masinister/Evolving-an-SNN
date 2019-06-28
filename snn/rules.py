import numpy as np


def STDP(trace,adj,post_activ,eta,mu, avg):
    w = 1 - adj
    w = np.power(w,mu)
    w = np.matmul(np.diag(post_activ),w)
    x = trace - avg
    delta_w = eta*np.matmul(np.diag(x),w)
    return delta_w

rules = {"STDP": STDP,}

def get(name):
        return rules[name]
