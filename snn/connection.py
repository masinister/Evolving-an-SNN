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
    def __init__(self, pre, post, scheme, rule):
        self.pre = pre
        self.post = post
        self.scheme = Schemes.get(scheme)
        self.adj = scheme(len(pre), len(post))

    def set_adj(self, w):
        self.adj = w

    def propogate(self):
        pass

    def update(self):
        # TODO Rule update adj
