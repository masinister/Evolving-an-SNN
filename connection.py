from population import Population
from synapse import Synapse
import numpy as np
'''
Connection encapsulates the interaction
between two populations of neurons.
'''
class Connection:
    '''
    pre: presynaptic population
    post: postsynaptic population
    adj: weighted adjacency matrix (adj[i][j] is the weight of the connection from pre[i] to post[j])
         (i.e. rows are presynap neurons and columns are postsynap neurons)
    rule: method for updating the weight matrix (e.g. STDP learning rule)
    params: hyperparameters for synapse and learning rule
    '''
    def __init__(self, pre, post, adj, params, **kwargs):
        self.pre = pre
        self.post = post
        self.adj = adj
        self.params = params
        self.rule = kwargs.get("rule")
        self.wmin = kwargs.get("wmin")
        self.wmax = kwargs.get("wmax")
        self.synapse = Synapse(self.params, self.pre.activations, self.post.activations, self.rule)

    '''
    update synapse and adjacency matrix then transmit weighted sums of spikes along the connection
    '''
    def input(self):
        feed = np.array(np.dot(self.pre.activations, self.adj))
        feed = np.reshape(feed,(np.size(feed),))
        self.post.input(feed)

    def update(self):
        self.synapse.update(self.pre.activations, self.post.activations)
        self.adj += self.synapse.delta_w(self.adj, self.pre.activations, self.post.activations)

    def set_params(self, params):
        self.params = params
        self.synapse.set_params(params)

    def normalize(self):
        # self.adj *= 0.99
        self.adj = np.clip(self.adj,self.wmin,self.wmax)
        self.adj /= np.sum(self.adj, axis = 0) + 0.0001
        self.adj *= 78.4
        print(self.adj)
