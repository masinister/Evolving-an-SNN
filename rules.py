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


def PPrule(pre_tr, post_tr, adj, pre_activ, post_activ, eta, mu):
    '''
    dw = -eta * post_tr * w * (1 - w)    if pre or (pre and post) spike
    dw = mu * pre_tr * w * (1 - w)       if post and (not pre) spike

    In the comments below, pre (post) refers to whether a pre (post) synaptic neuron
    fired.
    dw1 : weight changes according to the first rule
    dw2 : weight changes according to the second rule
    Note there is no overlap between dw1 and dw2, so a nonzero entry for one rule
    is zero for the other.
    '''
    unweighted = np.array(adj>0, dtype='int')
    dw = adj*(1-adj)
    dw *= unweighted # 0 out nonexistent connections
    #  Entries are True iff post fired and pre did not fire
    notPre = np.logical_not(pre_activ)
    # dw(i,j) are nonzero iff pre or (pre and post) == pre is True
    # this affects weights along the row of the adj matrix so we multiply on the right
    # by pre_activ
    dw1 = np.dot( np.diag(pre_activ), dw )
    dw1 = -eta*np.dot( np.diag(pre_tr), dw )
    # Entries are nonzero iff post and (not pre) is True
    # diag( notPre ) * dw * diag( post_activ )
    dw2 = np.dot( notPre, np.dot(dw, np.diag(post_activ)) )
    dw2 = mu*np.dot( dw, np.diag(post_tr) )

    return dw1 + dw2


def static(pre_tr, post_tr, adj, pre_activ, post_activ, eta, mu):
    return 0


rules = {"STDP": STDP,"static": static, "PPrule": PPrule}

def get(name):
    return rules[name]
