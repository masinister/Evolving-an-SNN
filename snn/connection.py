from population import Population

class Connection:

    def __init__(self, pre, post, scheme, rule):
        self.pre = pre
        self.post = post
        self.adj = []

    def set_adj(self, w):
        self.adj = w

    def propogate(self):
        pass

    def update(self):
        # TODO Rule update adj
