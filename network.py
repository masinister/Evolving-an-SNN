from connection import Connection
from population import Population
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Network:
    # Assumed that first population in pop list is the input population
    # and the last population in pop list is the output population
    def __init__(self, pop, conn, rule):
        # List of populations/connections
        self.populations = pop
        self.connections = conn
        self.neuron_labels = []
        self.rule = rule

    def run(self, steps):
        t = []
        v = []
        w = []
        a = []
        for s in range(steps):
            for c in self.connections:
                c.update()
            for p in self.populations:
                p.update()
            w.append([x for x in self.connections[0].adj.flat[17150:17250]])
            v.append([x.voltage for x in self.populations[1].neurons])
            t.append([x.threshold for x in self.populations[1].neurons])
            a.append([x.activation for x in self.populations[1].neurons])
        return w, t, v, a

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
            rates[i] = [int(x == max(rates[i])) for x in rates[i]]
            rates[i] /= (np.sum(rates[i]) + 0.0001)
            #
        return rates

    def enable_learning(self):
        for c in self.connections:
            c.synapse.rule = self.rule

    def disable_learning(self):
        for c in self.connections:
            c.synapse.rule = "static"

    def set_params(self, params):
        for c in self.connections:
            c.set_params(params)

    # Return a distribution of probabilities that each label corresponds to the example (set elsewhere)
    def predict(self, steps):
        rates = self.record(steps)
        dist = np.zeros(10)
        for i in range(len(rates)):
            for j in range(len(rates[i])):
                dist += rates[i][j] * self.neuron_labels[i][j]
        dist /= (np.sum(dist))
        return dist
