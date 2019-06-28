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
