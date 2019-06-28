from population import Population
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
    def __init__(self, pre, post, adj, rule):
        self.pre = pre
        self.post = post
        self.adj = adj

    def set_adj(self, w):
        self.adj = w

    def update(self):
        pass
        # TODO Rule update adj
