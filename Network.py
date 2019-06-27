import numpy as np

# ==============================================================================
# === Neuron Types =============================================================

class ICMNeuron:

    def __init__(self, volt_decay=.7, thresh_decay=.4,
                 thresh_bias=100, volt_init=1, thresh_init=100):

        self.volt_decay = volt_decay      # decay rate of internal voltage
        self.thresh_decay = thresh_decay  # decay rate of threshold
        self.thresh_bias = thresh_bias    # thrshld incrs when spike appears

        self.threshold = thresh_init
        self.voltage = volt_init
        self.activation = int(volt_init > thresh_init)


    def update(self, feed=0):

        self.voltage = self.volt_decay*self.voltage + feed
        self.activation = int(self.voltage > self.threshold)
        self.threshold = self.thresh_decay*self.threshold \
                       + self.thresh_bias*self.activation

# ==============================================================================
# === Population ===============================================================

class Population:

    def __init__(self, num_neurons=0, neuron_type=ICMNeuron):

        self.num_neurons = num_neurons
        self.neurons = []
        for i in range(num_neurons):
            neuron = neuron_type()
            self.neurons.append(neuron)


    def setAllNeurons(self, vd=.7, td=.4, tb=100):

        for neuron in self.neurons:
            neuron.volt_decay = vd
            neuron.thresh_decay = td
            neuron.thresh_bias = tb

    def updateNeurons(self, feed):
        for neuron in self.neurons:
            neuron.update(feed)

# ==============================================================================
# === Connection ===============================================================

class Connection:

    def __init__(self,pre,post,scheme,rule):

        self.pre = pre
        self.post = post
        self.adjacency = [] # initialized with scheme

    def propogate(self)
        pass


# ==============================================================================
# === Network ==================================================================

class Network:

    def __init__(self)
        pass

# ==============================================================================
# === Main =====================================================================

Input = Population(100,shape='2D Grid')




###
