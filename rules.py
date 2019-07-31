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
def STDP(pre_tr, post_tr, adj, pre_activ, post_activ, params):

    '''
    unweighted : binary adjacency matrix to keep track of what connections exist
    w : matrix representing the weight dependent part of delta_w
    '''
    w = adj*(1 - adj)
    w = np.power(w, params["mu"])
    # only update if there is a postsynaptic spike (0 out columns where there is no post-synap spike)
    w = np.dot( w, np.diag(post_activ))
    x = pre_tr - params["avg"]
    delta_w = params["eta"] * np.dot( np.diag(x), w)
    return delta_w


def PreAndPost(pre_tr, post_tr, adj, pre_activ, post_activ, params):
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
    dw = np.ones(adj.shape)
    dw1 = -params["eta"] * dw * pre_activ[:,None] * post_tr
    dw2 = params["mu"] * dw * post_activ * pre_tr[:,None]
    return dw1 + dw2


def static(pre_tr, post_tr, adj, pre_activ, post_activ, params):
    return 0


rules = {"STDP": STDP,"static": static, "PreAndPost": PreAndPost}

def get(name):
    return rules[name]
