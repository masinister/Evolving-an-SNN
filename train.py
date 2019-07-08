import numpy as np
from tqdm import tqdm

def train(network, train_data, learn_steps, rest_steps):
    network.enable_learning()
    for x in tqdm(train_data):
        network.populations[0].set_input(x)
        network.run(learn_steps)
        network.populations[0].set_blank()
        network.run(rest_steps)

def label_neurons(network, test_data, test_labels, num_labels, show_steps, rest_steps):
    '''
    firing_rates is a list of 2D arrays (of possibly varying shape) such that
    firing_rates[i,j] is an array containing the 10 average firing rates
    (cooresponding to each digit) for the jth neuron in the ith population
    (excluding the input layer)

    So if firing_rates[1,3,8] is high, that means the 3rd neuron in the
    1st population tends to fire a lot when the network is presented with an 8
    '''
    network.disable_learning()
    firing_rates = []
    for pop in network.populations[1:]:
        firing_rates.append( np.zeros((pop.num_neurons, num_labels)) )

    '''
    This turns off STDP. STDP is not turned back on here, so if you go through
    cycles of training and labeling then "Static" needs to be set to False again
    for the appropriate populations
    '''
    # For each image shown, record the number of times that the neurons fire
    i = 0
    for x in tqdm(test_data):
        network.populations[0].set_input(x)
        rates = network.record(show_steps)
        j=0
        for r in firing_rates:
            r[:,test_labels[i]] += rates[j]
            j += 1
        network.populations[0].set_blank()
        network.run(rest_steps)
        i += 1

    '''
    The firing_rates for each label need to be divided by the number of times
    that type of object was shown to the network
    '''
    network.neuron_labels = firing_rates
    # print(firing_rates[0][0:10])
    # print(firing_rates[1][0:10])

def evaluate(network, test_data, test_labels, steps, rest_steps):
    network.disable_learning()
    for i in range(len(test_data)):
        network.populations[0].set_input(test_data[i])
        res = network.predict(steps)
        print(np.argmax(res), res, test_labels[i])
        network.populations[0].set_blank()
        network.run(rest_steps)
