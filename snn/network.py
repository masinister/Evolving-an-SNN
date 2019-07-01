from connection import Connection
from population import Population

class Network:

    def __init__(self, pop, conn):
        self.populations = pop
        self.connections = conn

    def run(self, steps):
        for c in self.connections:
            c.update()
        for p in self.populations:
            p.update()
