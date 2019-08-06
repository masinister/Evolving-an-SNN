from genetic.individual import Individual
import numpy as np
import multiprocessing
import time
import sys

class Optimizer():

    def __init__(self, num_threads):
        self._trim = 1
        self._mu = 1.2
        self._num_threads = num_threads
        self._populations = []

    def evaluate_all(self, population):
        for individual in population:
            individual.evaluate()

    def random_population(self, n):
        networks = []
        for i in range(n):
            network = Individual(i)
            network.randomize()
            networks.append(network)
        return list(networks)

    def mutate(self, layer):
        return 0

    def evolve(self, sub_population):
        new_pop = [sub_population[0]]
        best_params = sub_population[0].params
        for i in sub_population[1:]:
            p1 = np.random.choice(sub_population).params
            p2 = np.random.choice(sub_population).params
            new_params = [best_params[k] + self._mu * (p1[k] - p2[k]) for k in best_params.keys()]
            new_network = Individual(i.id, new_params)
            new_pop.append(new_network)
        return new_networks

    def next_gen(self, sub_population):
        new_pop = self.evolve(sub_population[:-self._trim])
        new_pop.extend(self.random_population(_trim))
        return new_pop

    def pde(self, generations):
        pop_size = len(self._populations)
        for i in range(generations):
            result = 'Gen %d: '%(i,)
            threads = []

            #Start threads
            for sub_population in self._populations:
                t = multiprocessing.Process(target = self.evaluate_all, args = (sub_population,))
                threads.append(t)
                t.start()
                time.sleep(1)

            #Finish threads
            for j in range(pop_size):
                threads[j].join()
                self._populations[j] = list(self._populations[j])
                self._populations[j].sort(key = lambda n: n.accuracy, reverse=True)
                result += '| %.4f |'%(self._populations[j][0].accuracy,)
                self._populations[j] = self.next_gen(self._populations[j])

            #Migration
            for j in range(pop_size):
                if np.random.rand() < 0.1:
                    populations[(j+1)%pop_size].append(populations[j][0])
                    populations[j].append(populations[(j+1)%pop_size][0])
                    del(populations[(j+1)%pop_size][0])
                    del(populations[j][0])

            sys.stdout.write('\r'+result)
            sys.stdout.flush()

    def run(self, generations, pop_size):
        print("Initializing networks")
        networks = self.random_population(pop_size)
        self._populations = np.array_split(networks, self._num_threads)
        print("Training")
        self.pde(generations)
