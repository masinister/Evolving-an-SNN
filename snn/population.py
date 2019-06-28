class Population:

    def __init__(self, num_neurons=0, neuron_type-"ICMNeuron"):
        self.num_neurons = num_neurons
        self.neurons = []
        for i in range(num_neurons):
            neuron = neuron_type()
            self.neurons.append(neuron)
