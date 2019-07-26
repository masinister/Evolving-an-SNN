from genetic import Optimizer
import schemes
from connection import Connection
import population
from synapse import Synapse
from network import Network
import train
from tensorflow.keras.datasets import mnist
import numpy as np
from copy import deepcopy
from tqdm import tqdm

def genetic_test():
    generations = 1000
    population = 32
    optimizer = Optimizer(8)
    optimizer.run(generations, population)

def snn_test():
    print("Initializing Network")
    '''
    Learning paramters for PreAndPost rule
    '''
    params = {"eta": 0.0005, "mu": 0.05, "decay_pre": 0.95, "decay_post": 0.95}
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
        v_reset = -65,
        v_rest = -65,
        t_init = -50,
        min_thresh = -52,
        t_bias = 0.15,
        t_decay = .9999999,
        refrac = 5,
        one_spike = True
    )
    '''
    Initialize connections
    '''
    inh = -120
    C1 = Connection(Input, L1, 0.3 * all2all(Input.num_neurons, L1.num_neurons), params, rule = "PreAndPost", wmin = 0, wmax = 1)
    C2 = Connection(L1, L1, allBut1(L1.num_neurons) * inh, params, rule = "static", wmin = inh, wmax = 0)

    network = Network([Input, L1,], [C1, C2,])
    network.set_params(params)
    network.neuron_labels = [np.zeros((pop.num_neurons, 10)) for  pop in network.populations[1:]]

    outer = tqdm(total = 100, desc = 'Epochs', position = 0)
    for i in range(100):
        train.all_at_once(network, x_train[500 * i: 500 * (i+1)], y_train[500 * i: 500 * (i+1)], 10, 250)
        # print("Training", i)
        # train.train(network, x_train[500 * i: 500 * (i+1)], 250, draw_weights=False)
        # print("Labelling", i)
        # train.label_neurons(network, x_train[0: 500], y_train[0: 500], 10, 50)
        # print("Testing", i)
        # train.evaluate(network, x_train[50000:50500], y_train[50000:50500], 50)
        outer.update(1)
    print("Testing")
    train.evaluate(network, x_train[50000:51000], y_train[50000:51000], 100)

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('snn_test()')
    snn_test()
    # genetic_test()
