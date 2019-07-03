import rules
import numpy as np

class Synapse:

    def __init__(self, params, pre_activ, rule):
        self.pre_trace = np.array(pre_activ)
        self.params = params
        self.rule = rule

    def update(self,pre_activ):
        self.pre_trace = self.params["decay"]*self.pre_trace + pre_activ

    def delta_w(self, adj, post_activ):
        return rules.get(self.rule)(self.pre_trace, adj, post_activ, self.params["eta"], self.params["mu"], self.params["avg"])

    def set_params(self, params):
        self.params = params
