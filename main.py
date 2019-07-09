from genetic import Optimizer
import schemes
from connection import Connection
import population
from synapse import Synapse
import neuron
from network import Network
import train
from tensorflow.keras.datasets import mnist
import numpy as np

def genetic_test():
    generations = 1000
    population = 128
    optimizer = Optimizer(8)
    optimizer.run(generations, population)

def snn_test():
    print("Initializing Network")

    params = {"eta": 0.5, "mu": 1.0, "decay": 0.5, "avg": 0.2, "training": True}

    n_params = {"v_init": 0, "v_decay": .5, "t_init": 5, "min_thresh": 1, "t_bias": 80, "t_decay": .7}

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    Input = population.Image_Input(x_train[0])
    L1 = population.Population(n_params,64, neuron_type=neuron.ICMNeuron)
    L2 = population.Population(n_params, 64, neuron_type=neuron.ICMNeuron)

    rand = schemes.get("random")
    allBut1 = schemes.get("allBut1")
    grid = schemes.get("grid")

    C1 = Connection(Input, L1, rand(784,64), "STDP", params)
    C2 = Connection(L1, L1, grid(8,8), "STDP", params)
    C3 = Connection(L1, L2, grid(8,8), "STDP", params)
    C4 = Connection(L2, L2, grid(8,8), "STDP", params)

    network = Network([Input, L1, L2], [C1, C2, C3, C4])
    network.set_params(params)

    print("Training")
    train.train(network, x_train[0:256], 64, 64)

    print("Labelling")
    train.label_neurons(network, x_test[0:256], y_test[0:256], 10, 64, 64)

    print("Testing")
    train.evaluate(network, x_test[1000:1256], y_test[1000:1256], 64, 64)


if __name__ == '__main__':
    snn_test()
