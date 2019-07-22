from genetic import Optimizer
import schemes
from connection import Connection
import population
from synapse import Synapse
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
    Learning paramters for PreAndPost rule
    '''
    params = {"eta": 0.0001, "mu": 0.01, "decay_pre": 0.95, "decay_post": 0.95}
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
    '''
    Input = population.Image_Input(x_train[0])
    L1 = population.Population(
        num_neurons = 100,
        v_init = -65,
        v_decay = .99,
        t_init = -50,
        min_thresh = -52,
        t_bias = 0.05,
        t_decay = .99999999,
        v_reset = -65,
        v_rest = -65,
        refrac = 5
    )

    '''
    Initialize connections between populations
    '''
    inh = -17.5
    C1 = Connection(Input, L1, 0.3 * all2all(784,100), params, rule = "PreAndPost", wmin = 0, wmax = 1)
    C2 = Connection(L1, L1, allBut1(100) * inh, params, rule = "static", wmin = inh, wmax = 0)

    '''
    (list of populations, list of connections, learning_rule)
    first population in list is assumed to be the input layer
    '''
    network = Network([Input, L1, ], [C1, C2,])
    network.set_params(params)


    '''
    train/label/validate
    10: number of labels
    50: number of time steps an image is presented for
    40: number of time steps the network rests for inbetween images
    '''

    for i in range(100):
        print("Training", i)
        train.train(network, x_train[50 * i: 50 * (i+1)], 300, 300)
        # print("Labelling")
        # train.label_neurons(network, x_train[0: 100], y_train[0: 100], 10, 100, 40)
        # print("Testing")
        # train.evaluate(network, x_train[50000:50100], y_train[50000:50100], 100, 40)
    print("Testing")
    train.evaluate(network, x_train[50000:51000], y_train[50000:51000], 100, 40)

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('snn_test()')
    snn_test()
