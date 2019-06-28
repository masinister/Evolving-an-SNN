from population import Population
from schemes import Schemes
'''
Connection encapsulates the interaction
between two populations of neurons.
'''
class Connection:

    '''
    pre: presynaptic population
    post: postsynaptic population
    scheme: connection scheme (e.g. random)
    rule: method for updating the weight matrix
    '''
    def __init__(self, pre, post, adj, rule):
        self.pre = pre
        self.post = post
        self.adj = adj

    def set_adj(self, w):
        self.adj = w

    def propogate(self):
        pass

    def update(self):
        # TODO Rule update adj
