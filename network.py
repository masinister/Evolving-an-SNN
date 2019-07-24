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

    def run(self, steps, **kwargs):
        t = []
        v = []
        w = []
        a = []
        for s in range(steps):
            for c in self.connections:
                c.update()
            for c in self.connections:
                c.input()
            for p in self.populations:
                p.update()
            if kwargs.get("plot", False):
                w.append([x for x in self.connections[0].adj.flat[10000:10100]])
                v.extend([self.populations[1].voltage])
                t.extend([self.populations[1].threshold])
                a.extend([self.connections[1].synapse.pre_trace])
        if kwargs.get("draw_weights", False):
            sw1 = self.get_square_weights(self.connections[0].adj, np.sqrt(self.connections[0].adj.shape[1]).astype(int), 28)
            img1 = Image.fromarray((sw1 * 255).astype(np.uint8))
            img1.save("img/C%d.png" %(kwargs.get("id", 0)))
        return w, t, v, a

    def record(self, steps):
        rates = [np.zeros(pop.num_neurons) for pop in self.populations[1:]]
        # present the image, and every time a neuron fires increment rates
        for s in range(steps):
            for c in self.connections:
                c.update()
            for c in self.connections:
                c.input()
            i = 0
            for p in self.populations:
                p.update()
                if i > 0:
                    rates[i-1] += p.activation
                i += 1
        # Normalize each population's firing rate
        for i in range(len(rates)):
            rates[i] = (rates[i] == max(rates[i])).astype(float)
            rates[i] /= (np.sum(rates[i]) + 0.0001)
        return rates

    def rest(self):
        for c in self.connections:
            c.synapse.pre_trace.fill(0)
            c.synapse.post_trace.fill(0)
        for p in self.populations[1:]:
            p.voltage.fill(p.v_rest)
            p.refrac_count.fill(0)

    def enable_learning(self):
        for c in self.connections:
            c.synapse.rule = c.rule
        for p in self.populations:
            c.learning = True

    def disable_learning(self):
        for c in self.connections:
            c.synapse.rule = "static"
        for p in self.populations:
            c.learning = False

    def set_params(self, params):
        for c in self.connections:
            c.set_params(params)

    # Return a distribution of probabilities that each label corresponds to the example (set elsewhere)
    def predict(self, steps):
        rates = self.record(steps)
        dist = np.zeros(10)
        for i in range(len(rates)):
            dist += np.sum(rates[i][:,None] * self.neuron_labels[i], axis=0)
        dist /= (np.sum(dist) + 0.0001)
        return dist

    def get_square_weights(self, weights, box_w, input_neurons):
        input_neurons = (input_neurons, input_neurons)
        square_weights = np.zeros((input_neurons[0] * box_w, input_neurons[1] * box_w))
        for i in range(box_w):
            for j in range(box_w):
                n = i * box_w + j
                x = i * input_neurons[0]
                y = (j % box_w) * input_neurons[1]
                square_weights[x : x + input_neurons[0], y : y + input_neurons[1]] = weights[:, n].reshape((*input_neurons))
        return square_weights
