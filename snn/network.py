from connection import Connection
from population import Population

class Network:

    def __init__(self, pop, conn):
        self.populations = pop
        self.connections = conn

    def run(steps):
        for c in connections:
            c.update()
