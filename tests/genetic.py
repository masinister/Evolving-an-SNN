import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from genetic import Optimizer

def genetic_test():
    generations = 1000
    population = 32
    optimizer = Optimizer(8)
    optimizer.run(generations, population)
