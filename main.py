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

    '''
    Learning paramters for STDP rule
    '''
    params = {"eta": .2, "mu": 2, "decay_pre": 0.5, "decay_post": 0.5, "avg": 0.7}
    '''
    Learning paramters for PreAndPost rule
    '''
    # params = {"eta": .1, "mu": 0.9, "decay_pre": 0.5, "decay_post": 0.5, "avg": 0.7}
    '''
    Parameters for neuron activity
    '''
    n_params = {"v_init": 0, "v_decay": .6, "t_init": 5, "min_thresh": 1, "t_bias": 80, "t_decay": .9}


    '''
    different connection schemes
    '''
    rand = schemes.get("random")
    allBut1 = schemes.get("allBut1")
    grid = schemes.get("grid")
    all2all = schemes.get("all2all")
    one2one = schemes.get("one2one")
    local = schemes.get("local")


    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    '''
    Initialize populations
    (neuron_params, num_neurons, neuron model)
    '''
    Input = population.Image_Input(x_train[0])
    L1 = population.Population(n_params,49, neuron_type=neuron.ICMNeuron)
    L2 = population.Population(n_params,49, neuron_type=neuron.ICMNeuron)

    '''
    Initialize connections between populations
    (Presynap, Postsynap, connection_scheme, learning_rule, learning_params)
    '''
    C1 = Connection(Input, L1, rand(784,49), "STDP", params)
    C2 = Connection(L1, L2, rand(49,49), "STDP", params)


    '''
    (list of populations, list of connections, learning_rule)
    first population in list is assumed to be the input layer
    '''
    network = Network([Input, L1, L2], [C1, C2,])
    network.set_params(params)


    '''
    train/label/validate
    10: number of labels
    50: number of time steps an image is presented for
    40: number of time steps the network rests for inbetween images
    '''
    print("Training")
    train.train(network, x_train[0:50], 50, 40)
    print("Labelling")
    train.label_neurons(network, x_train[0:200], y_train[0:200], 10, 50, 40)
    print("Testing")
    train.evaluate(network, x_train[50000:50200], y_train[50000:50200], 50, 40)


if __name__ == '__main__':
    snn_test()
