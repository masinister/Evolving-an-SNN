import numpy as np

# =============================================================================
# === Neuron Types ============================================================

class ICMNeuron:

    def __init__(self, volt_decay=.7, thresh_decay=.4,
                 thresh_bias=100, volt_init=1, thresh_init=100):

        self.volt_decay = volt_decay        # decay rate of internal voltage
        self.thresh_decay = thresh_decay    # decay rate of threshold
        self.thresh_bias = thresh_bias      # thrshld incrs when spike appears

        self.threshold = thresh_init
        self.voltage = volt_init
        self.activation = int(volt_init > thresh_init)

    def update(self, feed=0):

        self.voltage = self.volt_decay*self.voltage + feed
        self.activation = int(self.voltage > self.threshold)
        self.threshold = self.thresh_decay*self.threshold \
                       + self.thresh_bias*self.activation

    def default_params(self):
        pass

# =============================================================================
# === Population ==============================================================

class Population:

    def __init__(self, num_neurons=0, neuron_type=ICMNeuron, shape=None):

        self.num_neurons = num_neurons
        self.neurons = []
        for i in range(num_neurons):
            neuron = neuron_type()
            self.neurons.append(neuron)

        self.shape = shape # label for the structure


        if shape != None:
            self.has_shape = True

            if shape == '2D Grid':
                # 2D Grid structure in the upper left quadrant
                # For now assume num_neurons is a perfect square for 2D grid
                '''
                2D Grid of 16 neurons number indicates index in Population.neurons
                .      .      .      .      .
                .      .      .      .      .
                0  ... 1  ... 2  ... 3  ........
                .      .      .      .      .
                .      .      .      .      .
                4  ... 5  ... 6  ... 7  ........
                .      .      .      .      .
                .      .      .      .      .
                8  ... 9  ... 10 ... 11 ........
                .      .      .      .      .
                .      .      .      .      .
                12 ... 13 ... 14 ... 15 .......
                '''
                dim = int(np.sqrt(num_neurons))
                self.structure = np.zeros((dim,dim))

                index = 0
                for i in range(dim):
                    for j in range(dim):
                        self.structure[i,j] = index
                        index += 1



    def setAllNeurons(self, vd=.7, td=.4, tb=100):

        for neuron in self.neurons:
            neuron.volt_decay = vd
            neuron.thresh_decay = td
            neuron.thresh_bias = tb

    def updateNeurons(self, feed):
        for neuron in self.neurons:
            neuron.update(feed)

    def print_neuron(self):
        pass

# =============================================================================
# === Connection ==============================================================

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



# =============================================================================
# === Main ====================================================================

Input = Population(100,shape='2D Grid')
con = Connection(Input,Input,'Local24')
print(con.connections[1])




###
