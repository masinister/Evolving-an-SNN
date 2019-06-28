class Connection:

    def __init__(self,pre,post,scheme=None):

        assert scheme != None, "Connection scheme required"

        self.connections = None
        self.pre = pre
        self.post = post


        if scheme == 'Local24':
            assert pre == post,"Local connection only valid for interlayer connection"
            assert pre.has_shape, "Population needs geometric structure"

            self.connections = -np.ones((pre.num_neurons,25))
            dim = len(pre.structure)
            # pad structure with -1's
            structure = -np.ones((dim+4,dim+4))
            structure[2:dim+2,2:dim+2] = pre.structure

            ind = 0
            for i in range(2,dim+2):
                for j in range(2,dim+2):
                    s = structure[i-2:i+3,j-2:j+3].flatten()
                    self.connections[ind] = np.delete(s,12)     # index 12 is where middle neuron is
                    ind += 1

            print(self.connections)

    def update(self):
        pass
