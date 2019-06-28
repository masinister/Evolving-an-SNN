class ICMNeuron:

    def __init__(self, volt_decay=.7, thresh_decay=.4,
                 thresh_bias=100, volt_init=1, thresh_init=100):

        self.volt_decay = volt_decay        # decay rate of internal voltage
        self.thresh_decay = thresh_decay    # decay rate of threshold
        self.thresh_bias = thresh_bias      # thrshld incrs when spike appears

        self.threshold = thresh_init
        self.voltage = volt_init
        self.activation = int(volt_init > thresh_init)

    def update(self, feed=0):

        self.voltage = self.volt_decay*self.voltage + feed
        self.activation = int(self.voltage > self.threshold)
        self.threshold = self.thresh_decay*self.threshold \
                       + self.thresh_bias*self.activation

    def default_params(self):
        pass
