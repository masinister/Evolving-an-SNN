import numpy as np


'''
STDP: Time and Weight Dependent Rule
delta_w = eta*(pre_trace - avg)(1-weight)**mu

delta_w : change in synaptic weight
eta : parameter to scale magnitude of delta_w (aka learning rate)
pre_trace : record of presynaptic spikes
avg : target average presynaptic trace
weight : current value of synaptic weight
mu : parameter to scale the dependence delta_w on the current weight
    
'''
def STDP(trace, adj, post_activ, eta, mu, avg):

    '''
    unweighted : binary adjacency matrix to keep track of what connections exist
    w : matrix representing the weight dependent part of delta_w 
    '''
    unweighted = np.array(adj>0, dtype='int')
    w = 1 - adj     # subtraction changes 0 entries to 1 (creating new connections),
    w *= unweighted # elementwise multiplication is done with unweighted to fix this
    w = np.power(w, mu)
    # only update if there is a postsynaptic spike (0 out columns where there is no post-synap spike)
    w = np.matmul(w, np.diag(post_activ))
    x = trace - avg
    delta_w = eta * np.matmul(np.diag(x), w)
    return delta_w


def Static(trace, adj, post_activ, eta, mu, avg):
    return 0


rules = {"STDP": STDP,"Static": Static}

def get(name):
    return rules[name]
