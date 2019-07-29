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
    data = []
    batch=300
    order = [0,3,9,1,6,5,2,8,7]
    for s in range(3):
        i=0
        count=0
        x = []
        while count < batch:
            if y_train[i] == order[s*3] or y_train[i] == order[s*3+1] or y_train[i] == order[s*3+2]:
                x.append(x_train[i])
                count+=1
            i+=1
        np.random.shuffle(x)
        data.extend(x)
    data = np.array(data)


    '''
    Initialize populations
    '''
    cell_num = 16
    nn = 100
    Input = population.Image_Input(x_train[0])
    L1 = population.Population(
        num_neurons = cell_num,
        v_init = -65,
        v_decay = .99,
        v_reset = -65,
        v_rest = -65,
        t_init = -50,
        min_thresh = -52,
        t_bias = 0.25,
        t_decay = .9999999,
        refrac = 5,
        one_spike = True
    )
    L2 = population.Population(
        num_neurons = cell_num,
        v_init = -65,
        v_decay = .99,
        v_reset = -65,
        v_rest = -65,
        t_init = -50,
        min_thresh = -52,
        t_bias = 0.25,
        t_decay = .9999999,
        refrac = 5,
        one_spike = True
    )
    L3 = population.Population(
        num_neurons = cell_num,
        v_init = -65,
        v_decay = .99,
        v_reset = -65,
        v_rest = -65,
        t_init = -50,
        min_thresh = -52,
        t_bias = 0.25,
        t_decay = .9999999,
        refrac = 5,
        one_spike = True
    )
    L4 = population.Population(
        num_neurons = nn,
        v_init = -65,
        v_decay = .99,
        v_reset = -65,
        v_rest = -65,
        t_init = -50,
        min_thresh = -52,
        t_bias = 0.25,
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
    C3 = Connection(Input, L2, 0.3 * all2all(Input.num_neurons, L2.num_neurons), params, rule = "static", wmin = 0, wmax = 1)
    C4 = Connection(L2, L2, allBut1(L2.num_neurons) * inh, params, rule = "static", wmin = inh, wmax = 0)
    C5 = Connection(Input, L3, 0.3 * all2all(Input.num_neurons, L3.num_neurons), params, rule = "static", wmin = 0, wmax = 1)
    C6 = Connection(L3, L3, allBut1(L3.num_neurons) * inh, params, rule = "static", wmin = inh, wmax = 0)
    C7 = Connection(Input, L4, 0.3 * all2all(Input.num_neurons, L4.num_neurons), params, rule = "PreAndPost", wmin = 0, wmax = 1)
    C8 = Connection(L4, L4, allBut1(L4.num_neurons) * inh, params, rule = "static", wmin = inh, wmax = 0)


    network = Network([Input, L1, L2, L3, L4], [C1, C2, C3, C4, C5, C6, C7, C8])
    network.set_params(params)

    '''
    train/label/validate
    '''
    batch = 300
    print("Training 1")
    train.train(network, data[0:batch], 250, draw_weights=True)
    network.populations[2].dt.fill(0)
    network.connections[0].rule = "static"
    network.connections[0].synapse.rule = "static"
    network.connections[2].rule = "PreAndPost"
    network.connections[2].synapse.rule = "PreAndPost"
    print("Training 2")
    train.train(network, data[batch:batch*2], 250, draw_weights=True)
    network.populations[3].dt.fill(0)
    network.connections[2].rule = "static"
    network.connections[2].synapse.rule = "static"
    network.connections[4].rule = "PreAndPost"
    network.connections[4].synapse.rule = "PreAndPost"
    print("Training 3")
    train.train(network, data[batch*2:batch*3], 250, draw_weights=True)
    print("Dreaming")
    train.dream(network, 250, reps=1, draw_weights=True)

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('snn_test()')
    snn_test()
    # genetic_test()
