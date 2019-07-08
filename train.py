import numpy as np

def train(network, train_data, learn_steps, rest_steps):
    i=0
    network.params["training"] = True
    for x in train_data:
        network.populations[0].set_input(x)
        network.run(learn_steps)
        network.populations[0].set_blank()
        network.run(rest_steps)
        i+=1

def label_neurons(network, test_data, test_labels, num_labels, steps):
    network.params["training"] = False
    '''
    firing_rates is a list of 2D arrays (of possibly varying shape) such that
    firing_rates[i,j] is an array containing the 10 average firing rates
    (cooresponding to each digit) for the jth neuron in the ith population
    (excluding the input layer)

    So if firing_rates[1,3,8] is high, that means the 3rd neuron in the
    1st population tends to fire when the network is presented with an 8
    '''
    firing_rates = []
    for pop in network.populations[1:]:
        firing_rates.append( np.zeros((pop.num_neurons, num_labels)) )
    #count spikes
    network.neuron_labels = firing_rates

def evaluate(network, test_data, test_labels, num_labels, steps):
    network.accuracy = 0.0
