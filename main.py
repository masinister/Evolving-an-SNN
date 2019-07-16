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
    # params = {"eta": .1, "mu": 2, "decay_pre": 0.5, "decay_post": 0.5, "avg": 0.9}
    '''
    Learning paramters for PreAndPost rule
    '''
    params = {"eta": 0.0001, "mu": 0.01, "decay_pre": 0.95, "decay_post": 0.95, "avg": 0.9}
    '''
    Parameters for neuron activity
    '''
    n_params = {"v_init": 0, "v_decay": .4, "t_init": 50, "min_thresh": 0.01, "t_bias": 0.05, "t_decay": .9999}


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
    L1 = population.Population(n_params,100, neuron_type=neuron.ICMNeuron)

    '''
    Initialize connections between populations
    (Presynap, Postsynap, connection_scheme, learning_rule, learning_params)
    '''
    C1 = Connection(Input, L1, rand(784,100), "PreAndPost", params)
    C2 = Connection(L1, L1, rand(100,100), "PreAndPost", params)


    '''
    (list of populations, list of connections, learning_rule)
    first population in list is assumed to be the input layer
    '''
    network = Network([Input, L1,], [C1, C2,])
    network.set_params(params)


    '''
    train/label/validate
    10: number of labels
    50: number of time steps an image is presented for
    40: number of time steps the network rests for inbetween images
    '''
    for i in range(100):
        print("Training", i)
        train.train(network, x_train[50 * i: 50 * (i+1)], 100, 40)
        print("Labelling")
        train.label_neurons(network, x_train[50*i:50*(i+1)], y_train[50*i:50*(i+1)], 10, 100, 40)
        print("Testing")
        train.evaluate(network, x_train[5000:5050], y_train[5000:5050], 100, 40)


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('snn_test()')
    snn_test()
