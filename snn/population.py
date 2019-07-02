import neuron

class Population:

    def __init__(self, num_neurons=0, neuron_type=neuron.ICMNeuron):

        self.num_neurons = num_neurons
        self.neurons = [neuron_type() for _ in range(num_neurons)]
        self.activations = [0]*num_neurons

    def input(self, feed):
        for i in range(self.num_neurons):
            self.neurons[i].input(feed[i])

    def update(self):
        '''
        Pass in presynaptic feed to each postsynaptic neuron in the population
        and update the neuron. Record which neurons in the population fired.
        '''
        for i in range(self.num_neurons):
            spike = self.neurons[i].update()
            self.activations[i] = spike

class Image_Input(Population):

    def __init__(self, image):
        self.num_neurons = len(image)*len(image[0])
        self.neurons = []
        for column in image:
            for pixel in column:
                self.neurons.append(neuron.PoissonNeuron(rate = pixel / 255))
        self.activations = [0]*self.num_neurons

    def change_image(self, image):
        i = 0
        for column in image:
            for pixel in column:
                self.neurons[i].rate = pixel / 255.0
                i+=1
