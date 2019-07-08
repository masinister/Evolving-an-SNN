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

    def set_params(self, params):
        for c in self.connections:
            c.set_params(params)        
