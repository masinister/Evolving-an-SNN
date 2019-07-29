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
        self.neuron_labels = np.array([])

    def run(self, steps, **kwargs):
        if kwargs.get("learning", True):
            self.enable_learning()
        else: self.disable_learning()
        t = []
        v = []
        w = []
        a = []
        rates = [np.zeros(pop.num_neurons) for pop in self.populations[1:]]
        prediction = np.zeros(10)
        for s in range(steps):
            for c in self.connections:
                c.update()
            for c in self.connections:
                c.input()
            for p in self.populations:
                p.update()
            if kwargs.get("record", False) or kwargs.get("predict", False):
                for i in range(len(rates)):
                    rates[i] += self.populations[i+1].activation
            if kwargs.get("plot", False):
                w.append([x for x in self.connections[0].adj.flat[10000:10100]])
                v.extend([self.populations[1].voltage])
                t.extend([self.populations[1].threshold])
                a.extend([self.connections[1].synapse.pre_trace])
        if kwargs.get("record", False) or kwargs.get("predict", False):
            for i in range(len(rates)):
                rates[i] = (rates[i] == max(rates[i])).astype(float)
                rates[i] /= (np.sum(rates[i]) + 0.0001)
        if kwargs.get("predict", False):
            for i in range(len(rates)):
                prediction += np.sum(rates[i][:,None] * self.neuron_labels[i], axis=0)
            prediction /= (np.sum(prediction) + 0.0001)
        if kwargs.get("draw_weights", False):
            for i in range(len(self.connections)):
                if self.connections[i].rule != "static":
                    sw = self.get_square_weights(self.connections[i].adj,
                                                  np.sqrt(self.connections[i].adj.shape[1]).astype(int),
                                                  np.sqrt(self.connections[i].adj.shape[0]).astype(int))
                    img = Image.fromarray((sw * 255).astype(np.uint8))
                    img.save("img/C%d - %d.png" %(i, kwargs.get("id", 0)))
        return {"w":w, "t":t, "v":v, "a":a, "rates":rates, "prediction": prediction}

    def rest(self):
        for c in self.connections:
            c.synapse.pre_trace.fill(0)
            c.synapse.post_trace.fill(0)
        for p in self.populations[1:]:
            p.voltage.fill(p.min_volt)
            p.refrac_count.fill(0)
            # p.dt.fill(p.min_thresh)

    def enable_learning(self):
        for c in self.connections:
            c.synapse.rule = c.rule
        for p in self.populations:
            p.learning = True

    def disable_learning(self):
        for c in self.connections:
            c.synapse.rule = "static"
        for p in self.populations:
            p.learning = False

    def set_params(self, params):
        for c in self.connections:
            c.set_params(params)

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

    def normalize(self):
        for c in self.connections:
            if c.rule != "static":
                c.normalize()
