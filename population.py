import matplotlib.pyplot as plt
import numpy as np
import random

class Population:
    '''
    num_neurons : number of neurons in the population
    neurons : list of neurons
    neuron_type : class of the neurons
    activations : list of which neurons fired at current time step
    '''
    def __init__(self, **kwargs):
        self.num_neurons = kwargs.get("num_neurons")
        self.voltage = kwargs.get("v_init") * np.ones(self.num_neurons)
        self.v_decay = kwargs.get("v_decay")
        self.threshold = kwargs.get("t_init") * np.ones(self.num_neurons)
        self.min_thresh = kwargs.get("min_thresh")
        self.t_bias = kwargs.get("t_bias")
        self.t_decay = kwargs.get("t_decay")
        self.refrac = kwargs.get("refrac")
        self.v_reset = kwargs.get("v_reset")
        self.v_rest = kwargs.get("v_rest")
        self.dt = self.threshold - self.min_thresh
        self.activation = np.zeros(self.num_neurons)
        self.feed = np.zeros(self.num_neurons)


    def input(self, feed):
        self.feed += feed

    def update(self):
        self.voltage = self.v_rest + self.v_decay * (self.voltage - self.v_rest)
        self.voltage += (self.refrac == 0) * self.feed
        if self.refrac == 0:
            mask = self.voltage >= self.threshold
            self.voltage[mask] = self.v_reset
            self.activation[mask] = 1
            self.dt[mask] += self.t_bias
            self.refrac = 5
        else:
            self.activation *= 0
            self.dt *= self.t_decay
            self.refrac -= 1
        self.threshold = self.min_thresh + self.dt
        self.feed = 0
        return self.activation


class Image_Input(Population):
    '''
    Population of input neurons. Each neuron fires randomly with probability
    proportional to pixel intensity.
    '''
    def __init__(self, image):
        self.num_neurons = len(image)*len(image[0])
        self.activation = np.zeros(self.num_neurons)
        self.rate = 0
        self.set_input(image)

    def set_input(self, image):
        # change to another image
        self.rate = (image / 255 / 4).flat

    def set_blank(self):
        self.rate *= 0

    def update(self):
        self.activation *= 0
        self.activation[random.random() < self.rate] = 1
