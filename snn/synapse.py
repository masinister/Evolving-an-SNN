import rules

class Synapse:

    def __init__(self, decay, pre_activ, rule):
        self.pre_trace = pre_activ
        self.decay = decay
        self.rule = rule

    def update(self,pre_activ):
        self.pre_trace = decay*self.pre_trace + self.pre_activ

    def delta_w(self, adj, post_activ, eta, mu):
        return rules.get(self.rule)(self.pre_trace, adj, post_activ, eta, mu)
