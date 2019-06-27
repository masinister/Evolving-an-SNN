from genetic import Optimizer

def main():
    generations = 1000
    population = 128
    optimizer = Optimizer(8)
    optimizer.run(generations, population)

if __name__ == '__main__':
    main()
