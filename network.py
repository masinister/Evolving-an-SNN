from connection import Connection
from population import Population
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
        w = []
        a = []
        for s in range(steps):
            for p in self.populations:
                p.update()
            for c in self.connections:
                c.input()
            for c in self.connections:
                c.update()
            w.append([x for x in self.connections[0].adj.flat[10000:10100]])
            v.extend([self.populations[1].voltage])
            t.extend([self.populations[1].threshold])
            # a.extend([self.populations[1].activation])
            a.extend([self.connections[1].synapse.pre_trace])
        sw = self.get_square_weights(self.connections[0].adj, 10, 28)
        img = Image.fromarray((sw * 255).astype(np.uint8))
        img.save("img/img.png")
        return w, t, v, a

    def record(self, steps):
        rates = [np.zeros(pop.num_neurons) for pop in self.populations[1:]]
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
            rates[i] = (rates[i] == max(rates[i])).astype(float)
            rates[i] /= (np.sum(rates[i]) + 0.0001)
        return rates

    def rest(self):
        for c in self.connections:
            pre = np.size(c.synapse.pre_trace)
            post = np.size(c.synapse.post_trace)
            c.synapse.pre_trace = np.zeros(pre)
            c.synapse.post_trace = np.zeros(post)
        for p in self.populations[1:]:
            p.voltage *= 0
            p.voltage += p.v_rest
            p.dt.fill(0)
            p.refrac = 0

    def enable_learning(self):
        for c in self.connections:
            c.synapse.rule = c.rule

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
        dist /= (np.sum(dist) + 0.0001)
        return dist

    def get_square_weights(self, weights, n_sqrt, side):
        side = (side, side)
        square_weights = np.zeros((side[0] * n_sqrt, side[1] * n_sqrt))
        for i in range(n_sqrt):
            for j in range(n_sqrt):
                n = i * n_sqrt + j
                x = i * side[0]
                y = (j % n_sqrt) * side[1]
                filter_ = weights[:, n].reshape((*side))
                square_weights[x : x + side[0], y : y + side[1]] = filter_
        return square_weights
