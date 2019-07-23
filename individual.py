import numpy as np
from network import Network
import train

class Individual():

    def __init__(self, params = None):
        self.accuracy = 0.0
        self.params = params
        #self.network = SNN(params)

    def randomize(self):
        temp = []
        for i in range(len(self.params)):
            temp.append((np.random.random_sample(np.shape(self.params[i])) - .5))
        self.params = temp

    def evaluate(self):
        self.accuracy = self.network.score()
