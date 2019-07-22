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
        self.refrac_count = np.zeros(self.num_neurons)
        self.v_reset = kwargs.get("v_reset")
        self.v_rest = kwargs.get("v_rest")
        self.one_spike = kwargs.get("one_spike")
        self.dt = self.threshold - self.min_thresh
        self.activation = np.zeros(self.num_neurons)
        self.feed = np.zeros(self.num_neurons)
        self.learning = True

    def input(self, feed):
        self.feed += feed

    def update(self):
        self.voltage = self.v_rest + self.v_decay * (self.voltage - self.v_rest)
        r = self.refrac_count == 0
        self.voltage[r] += self.feed[r]
        self.activation.fill(0)
        s = self.voltage >= self.threshold
        if self.one_spike:
            if s.any():
                a = np.random.choice(np.nonzero(s)[0])
                s.fill(0)
                s[a] = 1
        if (s*r).any():
            self.voltage[s*r] = self.v_reset
            self.activation[s*r] = 1
            if self.learning:
                self.dt[s*r] += self.t_bias * s.astype(np.float).sum(0) + 0.5
            self.refrac_count[s*r] = self.refrac
        if (~s).any() and self.learning:
            self.dt[~s] *= self.t_decay
        if (~r).any():
            self.refrac_count[~r] -= 1
        self.threshold = self.min_thresh + self.dt
        self.feed.fill(0)



class Image_Input(Population):
    '''
    Population of input neurons. Each neuron fires randomly with probability
    proportional to pixel intensity.
    '''
    def __init__(self, image):
        self.num_neurons = len(image)*len(image[0])
        self.activation = np.zeros(self.num_neurons)
        self.rate = np.zeros(self.num_neurons)
        self.set_input(image)

    def set_input(self, image):
        # change to another image
        self.rate = np.array(list((image / (255.0 * 4.0)).flat))

    def set_blank(self):
        self.rate.fill(0)

    def update(self):
        self.activation.fill(0)
        for i in range(len(self.activation)):
            self.activation[i] = int(random.random() < self.rate[i])
