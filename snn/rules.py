import numpy as np

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


rules = {"static": static, "PreAndPost": PreAndPost}

def get(name):
    return rules[name]
