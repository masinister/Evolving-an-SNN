from numpy import random

class ICMNeuron:

    def __init__(self, volt_decay=.4, thresh_decay=.7,
                 thresh_bias=100, thresh_init=1):

        self.volt_decay = volt_decay        # decay rate of internal voltage
        self.thresh_decay = thresh_decay    # decay rate of threshold
        self.thresh_bias = thresh_bias      # how much threshold increase when
                                            # the neuron spikes

        self.threshold = 100
        self.v_thresh = thresh_init
        self.dt = 0
        self.voltage = 0
        self.activation = 0
        self.feed = 0

    def input(self,feed):
        self.feed += feed

    def update(self):
        '''
        Update the internals of the neuron to the next time step, given an input
        feed from presynaptic neurons, according to the following rules.
        Returns whether or not the neuron spiked

        V[n+1] = decay*V[n] + presynap_feed
        Y[n+1] = {1, if    V[n+1] > T[n]
                 {0, else
        T[n+1] = decay*T[n] + bias*Y[n+1]

                    V: internal voltage
                    T: voltage threshold for firing
                    Y: boolean for whether neuron did or did not fire
                decay: decay parameters for the internal volatge and threshold
        presynap_feed: input from data or other neurons
                 bias: how much threshold increases when neuron fires
        '''
        self.voltage = self.volt_decay*self.voltage + self.feed
        self.activation = int(self.voltage > self.threshold)
        self.dt = self.thresh_decay*self.dt \
                + self.thresh_bias*self.activation
        self.threshold = self.v_thresh + self.dt
        self.feed = 0
        return self.activation



class PoissonNeuron:
    def __init__(self, rate):
        self.rate = rate
        self.activation = 0

    def update(self):
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
