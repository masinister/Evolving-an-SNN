from .connection import Connection
from .population import Population
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Network:
    '''
    Network of populations and the connections between them

    Assumed that first population in pop list is the input population

    learning: Boolean whether or not the network is learning
    draw_weights: Boolean whether or not to plot grid of weights
    plot: Boolean whether or not to plot weights/thresholds/voltages vs time
    '''
    def __init__(self, pop, conn):
        # List of populations/connections
        self.populations = pop
        self.connections = conn
        self.neuron_labels = np.array([])

    def run(self, steps, **kwargs):
        if kwargs.get("learning", True):
            self.enable_learning()
        else: self.disable_learning()

        if kwargs.get("record_spikes", False):
            pop_index = kwargs.get("pop_index", 0)
            spikes = np.zeros((self.populations[pop_index].num_neurons,steps))
        else:
            spikes = []
        # lists to record thresholds/voltage/weights etc
        t = []
        v = []
        w = []
        a = []
        rates = [np.zeros(pop.num_neurons) for pop in self.populations[1:]]
        prediction = np.zeros(10)
        for s in range(steps):
            for c in self.connections:
                c.input()
            for c in self.connections:
                c.update()
            for p in self.populations:
                p.update()

            if kwargs.get("record_spikes", False):
                spikes[:,s] = self.populations[pop_index].activation

            # if labelling or predicting record firing activity
            if kwargs.get("count_spikes", False) or kwargs.get("predict", False):
                for i in range(len(rates)):
                    rates[i] += self.populations[i+1].activation

            if kwargs.get("plot", False):
                w.append([x for x in self.connections[0].adj.flat[10000:10100]])
                v.extend([self.populations[1].voltage])
                t.extend([self.populations[1].threshold])
                a.extend([self.populations[1].trace])
        if kwargs.get("count_spikes", False) or kwargs.get("predict", False):
            for i in range(len(rates)):
                rates[i] = (rates[i] == max(rates[i])).astype(float)
                rates[i] /= (np.sum(rates[i]) + 0.0001)
        if kwargs.get("predict", False):
            for i in range(len(rates)):
                prediction += np.sum(rates[i][:,None] * self.neuron_labels[i], axis=0)
            prediction /= (np.sum(prediction) + 0.0001)
        # if you want to visualize weights in a grid
        if kwargs.get("draw_weights", False):
            for i in range(len(self.connections)):
                if self.connections[i].rule != "static":
                    sw = self.get_square_weights(self.connections[i].adj,
                                                  np.sqrt(self.connections[i].adj.shape[1]).astype(int),
                                                  np.sqrt(self.connections[i].adj.shape[0]).astype(int))
                    img = Image.fromarray((sw * 255 / (self.connections[i].wmax - self.connections[i].wmin)).astype(np.uint8))
                    img.save("img/C%d - %d.png" %(i, kwargs.get("id", 0)))
        return {"w":w, "t":t, "v":v, "a":a, "rates":rates, "prediction": prediction, "spikes": spikes}


    def rest(self):
        '''
        reset internals of neurons (except thresholds) to 0
        '''
        for p in self.populations[1:]:
            p.voltage.fill(p.min_volt)
            p.refrac_count.fill(0)
            p.trace.fill(0)
            p.activation.fill(0)
            p.feed.fill(0)

    def enable_learning(self):
        for c in self.connections:
            c.learning = True
        for p in self.populations:
            p.adapt_thresh = True

    def disable_learning(self):
        for c in self.connections:
            c.learning = False
        for p in self.populations:
            p.adapt_thresh = False

    def get_square_weights(self, weights, box_w, input_neurons):
        '''
        organize weight matrix into a grid for visualization
        '''
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

    def set_weights(self, weights):
        for i in range(len(weights)):
            self.connections[i].adj = weights[i]
