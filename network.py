from connection import Connection
from population import Population
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Network:
    # Assumed that first population in pop list is the input population
    # and the last population in pop list is the output population
    def __init__(self, pop, conn):
        # List of populations/connections
        self.populations = pop
        self.connections = conn
        self.neuron_labels = []

    def run(self, steps):
        t = []
        v = []
        for s in range(steps):
            for c in self.connections:
                c.update()
            for p in self.populations:
                p.update()
            t.append([x.threshold for x in self.populations[1].neurons])
            v.append([x.voltage for x in self.populations[1].neurons])
        plt.plot(t)
        plt.plot(v)
        plt.show()

    def record(self, steps):
        rates = []
        for pop in self.populations[1:]:
            rates.append(np.zeros(pop.num_neurons))

        # present the image, and every time a neuron fires increment rates
        for s in range(steps):
            for c in self.connections:
                c.update()
            i = 0
            for p in self.populations:
                p.update()
                if i > 0:
                    rates[i-1] += p.activations
                i += 1

        # Normalize each population's firing rate
        for i in range(len(rates)):
            rates[i] = rates[i]/(np.sum(rates[i]) + 0.0001)
        return rates

    def enable_learning(self):
        for c in self.connections:
            c.params["training"] = True

    def disable_learning(self):
        for c in self.connections:
            c.params["training"] = False

    def set_params(self, params):
        for c in self.connections:
            c.set_params(params)

    # Return a distribution of probabilities that each label corresponds to the example (set elsewhere)
    def predict(self, steps):
        rates = self.record(steps)
        dist = 0 * rates[0][0]
        for i in range(len(rates)):
            for j in range(len(rates[i])):
                dist += rates[i][j] * self.neuron_labels[i][j]
        dist = dist / (np.sum(dist) + 0.001)
        return dist
