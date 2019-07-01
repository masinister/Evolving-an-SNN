import numpy as np

def STDP(trace, adj, post_activ, eta, mu, avg):
    '''
    STDP: Time and Weight Dependent Rule
    dw = eta*(pre_trace - avg_trace)(1-weight)**mu

           dw : change in synaptic weight
          eta : parameter to scale magnitude of dw (learning rate)
    pre_trace : exp. decaying record of presynaptic spikes
    avg_trace : target average presynaptic trace
       weight : current value of synaptic weight
           mu : parameter to scale the weight dependence
    '''
    unweighted = np.array(adj>0, dtype='int')
    w = 1 - adj
    w *= unweighted # entries that were 0 (i.e. no connection) need to remain 0
    w = np.power(w, mu)
    w = np.matmul(w, np.diag(post_activ)) # only update if there is a postsynaptic spike
    x = trace - avg
    delta_w = eta * np.matmul(np.diag(x), w)
    return delta_w

def random(m,n):
    return np.random.random_sample((m,n))

rules = {"STDP": STDP,}

def get(name):
    return rules[name]
