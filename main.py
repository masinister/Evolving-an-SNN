from genetic import Optimizer
import schemes
from connection import Connection
import population
from network import Network
import train
from tensorflow.keras.datasets import mnist
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pickle

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
    params = {"eta": 0.0005, "mu": 0.05, "decay_pre": 0.95, "decay_post": 0.95,}
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
    # data = []
    # batch=300
    # order = [0,3,9,1,6,5,2,8,7]
    # for s in range(3):
    #     i=0
    #     count=0
    #     x = []
    #     while count < batch:
    #         if y_train[i] == order[s*3] or y_train[i] == order[s*3+1] or y_train[i] == order[s*3+2]:
    #             x.append(x_train[i])
    #             count+=1
    #         i+=1
    #     np.random.shuffle(x)
    #     data.extend(x)
    # data = np.array(data)
    #
    #
    '''
    Initialize populations
    '''
    thresh_load = open("thresh.pickle","rb")
    thresh = pickle.load(thresh_load)
    thresh_load.close()

    num_neurons=100
    Input = population.Image_Input(x_train[0])
    thr_init = thresh #-50
    L1 = population.Population(
        num_neurons = num_neurons,
        v_init = -65,
        v_decay = .99,
        v_reset = -65,
        min_volt = -65,
        t_init = thr_init,
        min_thresh = -52,
        t_bias = 0.25,
        t_decay = .9999999,
        refrac = 5,
        one_spike = True
    )


    '''
    Initialize connections
    '''
    weights_load = open("weights.pickle","rb")
    weights = pickle.load(weights_load)
    weights_load.close()
    inh = -120
    w_init = weights #0.3 * all2all(Input.num_neurons, L1.num_neurons)
    C1 = Connection(Input,
                    L1,
                    w_init,
                    params,
                    rule = "PreAndPost",
                    wmin = 0,
                    wmax = 1,
                    norm = 78.4)

    C2 = Connection(L1,
                    L1,
                    allBut1(L1.num_neurons) * inh,
                    params,
                    rule = "static",
                    wmin = inh,
                    wmax = 0)

    network = Network([Input, L1,], [C1, C2,])

    train.train(network, x_train[2000:3000], 350, draw_weights=True)
    weight_dump = open("weights.pickle","wb")
    pickle.dump(C1.adj,weight_dump)
    weight_dump.close()
    thresh_dump = open("thresh.pickle","wb")
    pickle.dump(L1.threshold,thresh_dump)
    thresh_dump.close()



if __name__ == '__main__':
    # import cProfile
    # cProfile.run('snn_test()')
    snn_test()
    # genetic_test()
