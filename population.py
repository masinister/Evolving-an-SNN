import neuron
import matplotlib.pyplot as plt
import numpy as np

class Population:
    '''
    num_neurons : number of neurons in the population
    neurons : list of neurons
    neuron_type : class of the neurons
    activations : list of which neurons fired at current time step
    '''
    def __init__(self, n_params, num_neurons=0, neuron_type=neuron.ICMNeuron):

        self.num_neurons = num_neurons
        self.neurons = [neuron_type(n_params) for _ in range(num_neurons)]
        self.activations = np.zeros(num_neurons)

    def input(self, feed):
        # pass input feed of to each neuron
        for i in range(self.num_neurons):
            self.neurons[i].input(feed[i])

    def update(self):
        '''
        Update each neuron and record which neurons in the population fired
        '''
        for i in range(self.num_neurons):
            self.activations[i] = self.neurons[i].update()


class Image_Input(Population):
    '''
    Population of input neurons. Each neuron fires randomly with probability
    proportional to pixel intensity.
    '''
    def __init__(self, image):
        self.num_neurons = len(image)*len(image[0])
        self.neurons = []
        self.neurons = list(map(neuron.PoissonNeuron, (image / 255.0 /4).flat))
        self.activations = np.zeros(self.num_neurons)

    def set_input(self, image):
        # change to another image
        self.neurons = list(map(neuron.PoissonNeuron, (image / 255.0 /4).flat))

    def set_blank(self):
        for n in self.neurons:
            n.rate = 0




class Const_Input(Population):
    '''
    Population of input neurons. Each neuron feeds in a constant input proportional
    to the intensity of the cooresponding pixel
    '''
    def __init__(self, image):
        self.num_neurons = len(image)*len(image[0])
        self.activations = np.zeros(self.num_neurons)
        i = 0
        for column in image:
            for pixel in column:
                self.activations[i] = pixel/255
                i += 1

    def update(self):
        # Const neuron / population does not change
        pass

    def set_input(self, image):
        # change to another image
        i = 0
        for column in image:
            for pixel in column:
                self.activations[i] = pixel/255
                i+=1

    def set_blank(self):
        for i in range(self.num_neurons):
            self.activations[i] = 0
