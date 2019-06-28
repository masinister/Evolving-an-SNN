import numpy as np

<<<<<<< HEAD

def STDP(trace,adj,post_activ,eta,mu, avg):
    w = 1 - adj
    w = np.power(w,mu)
    w = np.matmul(np.diag(post_activ),w)
    x = trace - avg
    delta_w = eta*np.matmul(np.diag(x),w)
    return delta_w
=======
def STDP(m,n):
    return np.random.random_sample((m,n))
>>>>>>> 8c08c8d3f1f7fb300eedbf07032987793162e416

rules = {"STDP": STDP,}

def get(name):
        return rules[name]
