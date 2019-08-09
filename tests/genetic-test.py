from genetic.genetic import Optimizer

if __name__ == '__main__':
    generations = 1000
    population = 8
    optimizer = Optimizer(2)
    optimizer.run(generations, population)
