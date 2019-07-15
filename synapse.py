import rules
import numpy as np

'''
Synapse specifies what type of connection is between two neurons.
i.e. what type of learning rule

pre_trace : a record of presynaptic spikes which decays over time
post_trace : a record of presynaptic spikes which decays over time
decay : decay rate of pre_trace
rule : what rule is used to update synaptic weight

pre_activ : activations of presynaptic population
post_activ : activations of postsynaptic population
'''
class Synapse:

    def __init__(self, params, pre_activ, post_activ, rule):
        self.pre_trace = np.array(pre_activ)
        self.post_trace = np.array(post_activ)
        self.params = params
        self.rule = rule

    def update(self, pre_activ, post_activ):
        # Decay traces that did not fire
        self.pre_trace = self.params["decay_pre"]*self.pre_trace*np.logical_not(pre_activ)
        self.post_trace = self.params["decay_post"]*self.post_trace*np.logical_not(post_activ)
        # traces that did fire are set to 1
        self.pre_trace += pre_activ
        self.post_trace += post_activ

    def delta_w(self, adj, pre_activ, post_activ):
        return rules.get(self.rule)(self.pre_trace, self.post_trace, adj, pre_activ, post_activ, self.params)

    def set_params(self, params):
        self.params = params
