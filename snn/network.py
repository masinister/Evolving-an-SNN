from connection import Connection
from population import Population
from tqdm import tqdm
import numpy as np

class Network:
    # Assumed that last population in pop list is the output layer
    def __init__(self, pop, conn):
        # List of populations/connections
        self.populations = pop
        self.connections = conn
        # firing_rates[i] is a list of average firing rates for each neuron
        # in the ith population
        self.firing_rates = [[] for _ in range(len(pop))]
        i=0
        for p in self.populations:
            self.firing_rates[i] = np.zeros(p.num_neurons)
            i+=1


    def run(self, steps):
        for s in tqdm(range(steps), ncols = 28*3):
            for c in self.connections:
                c.update()
            i=0
            for p in self.populations:
                p.update()
                self.firing_rates[i] += self.populations[i].activations
                i+=1
        for rate in self.firing_rates:
            rate = rate/steps

    def set_params(self, params):
        for c in self.connections:
            c.set_params(params)
