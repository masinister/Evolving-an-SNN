import neuron

class Population:
    '''
    num_neurons : number of neurons in the population
    neurons : list of neurons
    neuron_type : class of the neurons
    activations : list of which neurons fired at current time step
    '''
    def __init__(self, num_neurons=0, neuron_type=neuron.ICMNeuron):

        self.num_neurons = num_neurons
        self.neurons = [neuron_type() for _ in range(num_neurons)]
        self.activations = [0]*num_neurons

    def input(self, feed):
        # pass input feed of to each neuron
        for i in range(self.num_neurons):
            self.neurons[i].input(feed[i])

    def update(self):
        '''
        Update each neuron and record which neurons in the population fired
        '''
        for i in range(self.num_neurons):
            spike = self.neurons[i].update()
            self.activations[i] = spike


class Image_Input(Population):
    '''
    Population of input neurons. Each neuron fires randomly with probability
    proportional to pixel intensity.
    '''
    def __init__(self, image):
        self.num_neurons = len(image)*len(image[0])
        self.neurons = []
        for column in image:
            for pixel in column:
                # set average firing rate of the neurons
                self.neurons.append(neuron.PoissonNeuron(rate = pixel / 255.0 ))
        self.activations = [0]*self.num_neurons

    def set_input(self, image):
        # change to another image
        i = 0
        for column in image:
            for pixel in column:
                self.neurons[i].rate = pixel / 255.0
                i+=1

    def set_blank(self):
        for n in self.neurons:
            n.rate = 0
