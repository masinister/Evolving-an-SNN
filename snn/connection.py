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
    rule: method for updating the weight matrix
    '''
    def __init__(self, pre, post, adj, rule, params):
        self.pre = pre
        self.post = post
        self.adj = adj
        self.params = params
        self.synapse = Synapse(self.params["decay"], self.pre.activations, rule)

    '''
    update synapse and adjacency matrix then transmit weighted sums of spikes along the connection
    '''
    def update(self):
        self.synapse.update(self.pre.activations)
        self.adj = self.adj + self.synapse.delta_w(self.adj, self.post.activations, self.params["eta"], self.params["mu"], self.params["avg"])
        self.adj = self.adj / np.max(self.adj)
        feed = np.matmul(np.array(self.pre.activations), self.adj)
        self.post.input(feed)

    def set_params(self, params):
        self.params = params
        self.synapse.decay = params["decay"]
