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
        self.synapse = Synapse(self.pre.activations, rule)

    '''
    update synapse and adjacency matrix then transmit weighted sums of spikes along the connection
    '''
    def update(self):
        self.synapse.update(pre.activations)
        self.adj = self.adj + self.synapse.delta_w(self.adj, self.post.activations, self.params["eta"], self.params["mu"])
        feed = np.matmul(np.array(pre.activations), adj)
        post.update(feed)
