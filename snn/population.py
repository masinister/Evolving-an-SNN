class Population:

    def __init__(self, num_neurons=0, neuron_type-"ICMNeuron"):

        self.num_neurons = num_neurons
        self.neurons = [neuron_type() for _ in range(num_neurons)]
        self.activations = [0]*num_neurons


    def update(feed):

        for i in self.num_neurons:
            spike = self.neurons[i].update(feed[i])
            self.activations[i] = spike
