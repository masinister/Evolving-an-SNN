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
    def __init__(self, pre, post, adj, rule, params):
        self.pre = pre
        self.post = post
        self.adj = adj
        self.params = params
        self.synapse = Synapse(self.params, self.pre.activations, rule)
        self.normalize()

    '''
    update synapse and adjacency matrix then transmit weighted sums of spikes along the connection
    '''
    def update(self):
        # Static connections to not change weights
        self.synapse.update(self.pre.activations)
        self.adj = self.adj + self.synapse.delta_w(self.adj, self.post.activations)
        self.normalize()
        '''
        There is a weird bug happening with numpy where feed is being type-casted
        as a numpy.matrixlib.defmatrix.matrix with shape (1,784) i.e. a 2D array
        instead of (784,) i.e. a 1D array. This causes indexing problems, when feed
        is passed to a Population. This data type cannot be reshaped, so as a work
        around it is turned into a numpy.array and reshaped to (784,)
        '''
        feed = np.array(np.dot(self.pre.activations, self.adj))
        feed = np.reshape(feed,(np.size(feed),))
        self.post.input(feed)

    def set_params(self, params):
        self.params = params
        self.synapse.set_params(params)

    def normalize(self):
        self.adj /= np.max(np.transpose(self.adj), axis = 0)[:, None] + 0.0001
