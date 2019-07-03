from numpy import random

class ICMNeuron:
    '''
    voltage : internal voltage of neuron
    v_decay : decay rate of internal voltage
    threshold : spiking threshold (spike when voltage >= threshold)
    min_thresh : minimum threshold value
    t_bias : how much threshold increase when the neuron spikes
    dt : portion of threshold that decays    
    t_decay : decay rate of threshold
    activation : binary 0 or 1 for whether or not neuron fired
    feed : input voltage coming from data/other neurons
    '''
    def __init__(self, v_decay=.5, t_decay=.9, t_bias=80, t_init=5):

        self.voltage = 0
        self.v_decay = v_decay

        self.threshold = 100
        self.min_thresh = t_init
        self.t_bias = t_bias
        self.dt = 0
        self.t_decay = t_decay

        self.activation = 0
        self.feed = 0

    def input(self,feed):
        self.feed += feed

    def update(self):
        '''
        Update the internals of the neuron to the next time step, given an input
        feed from presynaptic neurons, according to the following rules.
        Returns whether or not the neuron spiked

        V[n+1] = v_decay*V[n] + feed
        Y[n+1] = {1, if    V[n+1] > T[n]
                 {0, else
        T[n+1] = t_decay*T[n] + bias*Y[n+1]

            V: internal voltage
            T: voltage threshold for firing
            Y: boolean for whether neuron did or did not fire
        decay: decay parameters for the internal volatge and threshold
         feed: input from data or other neurons
         bias: how much threshold increases when neuron fires
        '''
        self.voltage = self.v_decay*self.voltage + self.feed
        self.activation = int(self.voltage > self.threshold)
        self.dt = self.t_decay*self.dt + self.t_bias*self.activation
        self.threshold = self.min_thresh + self.dt
        self.feed = 0
        return self.activation



class PoissonNeuron:
    def __init__(self, rate):
        self.rate = rate    # average firing rate of neuron
        self.activation = 0

    def update(self):
        # at each time step fire with probability equal to the avg firing rate
        r = random.rand()
        if r <= self.rate:
            self.activation = 1
            return 1
        else:
            self.activation = 0
            return 0


class TestNeuron:
    def __init__(self, v = 0.5):
        self.voltage = v

    def update(self):
        return 1
