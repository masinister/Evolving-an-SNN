class ICMNeuron:

    def __init__(self, volt_decay=.99, thresh_decay=.4,
                 thresh_bias=100, volt_init=1, thresh_init=100):

        self.volt_decay = volt_decay        # decay rate of internal voltage
        self.thresh_decay = thresh_decay    # decay rate of threshold
        self.thresh_bias = thresh_bias      # how much threshold increase when
                                            # the neuron spikes

        self.threshold = thresh_init
        self.voltage = volt_init
        self.activation = int(volt_init > thresh_init)

    def input(self,feed):
        self.voltage += feed

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
        self.voltage = self.volt_decay*self.voltage
        self.activation = int(self.voltage > self.threshold)
        self.threshold = self.thresh_decay*self.threshold \
                       + self.thresh_bias*self.activation
        return self.activation

class TestNeuron:
    def __init__(self, v = 0.5):
        self.voltage = v

    def update(self, feed):
        return 1
