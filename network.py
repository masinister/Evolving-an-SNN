from connection import Connection
from population import Population
from tqdm import tqdm
import numpy as np

class Network:
    # Assumed that first population in pop list is the input population
    # and the last population in pop list is the output population
    def __init__(self, pop, conn):
        # List of populations/connections
        self.populations = pop
        self.connections = conn
        self.neuron_labels = np.zeros(pop[-1].num_neurons)

    def run(self, steps):
        for s in range(steps):
            for c in self.connections:
                c.update()
            for p in self.populations:
                p.update()

    def record(self, steps):
        rates = []
        for pop in self.populations[1:]:
            rates.append( np.zeros(pop.num_neurons) )
        # Assert STDP is turned off
        for c in self.connections:
            assert( c.params["training"] )

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
            rates[i] = rates[i]/np.max(rates[i])

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

    # Return a distribution of probabilities that each label corresponds to the given example
    def predict(self, example):
        pass
