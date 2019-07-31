from population import Population
import rules
import numpy as np


class Connection:
    '''
    Connection encapsulates the interaction
    between two populations of neurons.

    pre: presynaptic population
    post: postsynaptic population
    adj: weighted adjacency matrix (adj[i][j] is the weight of the connection from pre[i] to post[j])
         (i.e. rows are presynap neurons and columns are postsynap neurons)
    wmin: minimum weight value in adj
    wmax: maximum weight value in adj
    norm: normalize weights such that the average value of weights is (norm/pre.num_neurons)
    rule: method for updating the weight matrix (e.g. STDP learning rule)
    params: hyperparameters for synapse and learning rule
    '''
    def __init__(self, pre, post, adj, params, **kwargs):
        self.pre = pre
        self.post = post
        self.adj = adj
        self.params = params
        self.rule = kwargs.get("rule", "static")
        self.wmin = kwargs.get("wmin", 0)
        self.wmax = kwargs.get("wmax", 1)
        self.norm = kwargs.get("norm", 78.4)
        self.learning = kwargs.get("learning", True)

    def input(self):
        '''
        Pass input from pre to post
        '''
        feed = np.sum(self.adj * self.pre.activation[:,None], axis=0)
        self.post.input(feed)

    def update(self):
        '''
        Update Weights
        '''
        if self.learning:
            self.adj += rules.get(self.rule)(self.pre.trace, self.post.trace, self.adj, self.pre.activation, self.post.activation, self.params)

    def normalize(self):
        self.adj = np.clip(self.adj,self.wmin,self.wmax)
        self.adj /= np.sum(np.absolute(self.adj), axis = 0) + 0.0001
        self.adj *= self.norm
