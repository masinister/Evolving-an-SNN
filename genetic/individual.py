import numpy as np

class Individual():

    def __init__(self, params):
        self.accuracy = 0.0
        self.params = params
        #self.network = SNN(params)

    def randomize(self):
        temp = []
        for i in range(len(self.weights)):
            temp.append((np.random.random_sample(np.shape(self.params[i])) - .5))
        self.weights = temp

    def evaluate(self):
        self.accuracy = self.network.score()
