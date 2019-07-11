import numpy as np
from tqdm import tqdm

def train(network, train_data, learn_steps, rest_steps):
    network.enable_learning()
    for x in tqdm(train_data):
        # print(np.around(network.connections[2].adj, decimals=2))
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

    # For each image shown, record the number of times that the neurons fire
    # and count how many times the network saw each digit
    label_count = np.zeros((num_labels,))
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
        label_count[test_labels[i]] += 1
        i += 1

    # divide each category by the number of times it saw each category
    i = 0
    for r in firing_rates:
        j = 0
        for count in label_count:
            if count != 0:
                firing_rates[i][:,j] /= count
            j += 1
        i += 1

    for pop in firing_rates:
        pop /= (sum(pop) + 0.001)

    network.neuron_labels = firing_rates
    # print(firing_rates[0][0:10])
    # print(firing_rates[1][0:10])

def evaluate(network, test_data, test_labels, steps, rest_steps):
    network.disable_learning()
    correct = 0
    view_count = np.zeros(10)
    correct_count = np.zeros(10)
    for i in tqdm(range(len(test_data))):
        network.populations[0].set_input(test_data[i])
        res = network.predict(steps)
        # print(np.argmax(res), ["%.4f" % r for r in res], test_labels[i])
        if np.argmax(res) == test_labels[i]:
            correct += 1
            correct_count[test_labels[i]] += 1
        view_count[test_labels[i]] += 1
        network.populations[0].set_blank()
        network.run(rest_steps)
    for i in range(10):
        if view_count[i] != 0:
            correct_count[i] /= view_count[i]
    print("Got %.3f correct" % (correct/len(test_labels)), correct_count)
