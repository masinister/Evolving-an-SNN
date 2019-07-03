import rules
import numpy as np

'''
Synapse specifies what type of connection is between two neurons.
i.e. what type of learning rule

pre_trace : a record of presynaptic spikes which decays over time
decay : decay rate of pre_trace
rule : what rule is used to update synaptic weight

pre_activ : activations of presynaptic population
'''
class Synapse:

    def __init__(self, decay, pre_activ, rule):
        self.pre_trace = np.array(pre_activ)
        self.decay = decay
        self.rule = rule

    def update(self,pre_activ):
        self.pre_trace = self.decay*self.pre_trace + pre_activ

<<<<<<< HEAD
    def delta_w(self, adj, post_activ):
        return rules.get(self.rule)(self.pre_trace, adj, post_activ, self.params["eta"], self.params["mu"], self.params["avg"])

    def set_params(self, params):
        self.params = params
=======
    def delta_w(self, adj, post_activ, eta, mu, avg):
        return rules.get(self.rule)(self.pre_trace, adj, post_activ, eta, mu, avg)
>>>>>>> parent of dfc3741... pass params to synapse
