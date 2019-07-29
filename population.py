import matplotlib.pyplot as plt
import numpy as np
import random

class Population:
    '''
    Population of neurons modeled after the neurons in (Diehl and Cook 2015)

    v_decay: fraction of voltage that decays in a time step
    t_bias: amount threshold increases when the neuron fires
    t_decay: fraction of threshold that decays in a time step
    dt: portion of threshold that decays
    refrac: length of refractory period
    refrac_count: records times since neurons last fired
    v_reset: voltage that neurons are reset to after firing
    min_volt: voltage that neurons decay down to
    one_spike: Boolean to only allow one neuron to spike in a time step
    activation: list of which neurons fired in a time step
    feed: voltage inputs that are fed into the neurons
    adapt_thresh: Boolean for whether threshold is adaptive (True => is adaptive)
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
        self.min_volt = kwargs.get("min_volt")
        self.one_spike = kwargs.get("one_spike")
        self.dt = self.threshold - self.min_thresh
        self.activation = np.zeros(self.num_neurons)
        self.feed = np.zeros(self.num_neurons)
        self.adapt_thresh = kwargs.get("adapt_thresh", True)

    def input(self, feed):
        self.feed += feed

    def update(self):
        '''
        Update the internals of the neurons in the population. That is record which neurons fire
        and update their voltages, thresholds, etc
        '''
        self.voltage = self.min_volt + self.v_decay * (self.voltage - self.min_volt)
        not_in_refractory = self.refrac_count == 0
        self.voltage[not_in_refractory] += self.feed[not_in_refractory]
        self.activation.fill(0)
        spikes = self.voltage >= self.threshold
        # if one_spike == True, randomly choose one of the spikes and zero out the others
        if self.one_spike:
            if spikes.any():
                a = np.random.choice(np.nonzero(spikes)[0])
                spikes.fill(0)
                spikes[a] = 1
        # if neurons have built up enough voltage to spike and are not in their refractory period,
        # fire the neurons, reset their voltages, increment their threshold
        if (spikes*not_in_refractory).any():
            self.voltage[spikes*not_in_refractory] = self.v_reset
            self.activation[spikes*not_in_refractory] = 1
            if self.adapt_thresh:
                self.dt[spikes*not_in_refractory] += self.t_bias * spikes.astype(np.float).sum(0)
            self.refrac_count[spikes*not_in_refractory] = self.refrac
        # decay the threshold of any neurons that do not fire
        if (~spikes).any() and self.adapt_thresh:
            self.dt[~spikes] *= self.t_decay
        # decrement refractory counter
        if (~not_in_refractory).any():
            self.refrac_count[~not_in_refractory] -= 1
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
