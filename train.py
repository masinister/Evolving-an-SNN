import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from color import color

def train(network, train_data, steps, **kwargs):
    weight = []
    thresh = []
    volt = []
    act = []

    for x in tqdm(train_data):
        network.populations[0].set_input(x)
        res = network.run(steps, **kwargs)
        network.normalize()
        network.rest()
        if kwargs.get("plot", False):
            weight.extend(res.get("w", []))
            thresh.extend(res.get("t", []))
            volt.extend(res.get("v", []))
            act.extend(res.get("a", []))
    if kwargs.get("plot", False):
        fig, axs = plt.subplots(4,sharex=True,gridspec_kw={'hspace': .5})
        fig.suptitle("Info about 1st Layer (rest times omitted from plot)")
        axs[0].plot(weight)
        axs[0].set_title("100 Synap Weights From Input to L1")
        axs[1].plot(thresh)
        axs[1].set_title("L1 Thresholds")
        axs[2].plot(volt)
        axs[2].set_title("L1 Voltages")
        axs[3].plot(act)
        axs[3].set_title("L1 Activations")
        plt.show()

def label_neurons(network, test_data, test_labels, num_labels, steps, **kwargs):
    '''
    firing_rates is a list of 2D arrays (of possibly varying shape) such that
    firing_rates[i,j] is an array containing the 10 average firing rates
    (cooresponding to each digit) for the jth neuron in the ith population
    (excluding the input layer)

    So if firing_rates[1,3,8] is high, that means the 3rd neuron in the
    1st population tends to fire a lot when the network is presented with an 8
    '''
    pop_rates = [np.zeros((pop.num_neurons, num_labels)) for  pop in network.populations[1:]]

    # For each image shown, record the number of times that the neurons fire
    # and count how many times the network saw each digit
    for i in tqdm(range(len(test_data))):
        network.populations[0].set_input(test_data[i])
        rates = network.run(steps,
                            learning = False,
                            record = True,
                            **kwargs).get("rates")
        network.rest()
        for j in range(len(pop_rates)):
            pop_rates[j][:,test_labels[i]] += rates[j]

    for pop in pop_rates:
        pop /= (sum(pop) + 0.0001)

    network.neuron_labels = pop_rates

def evaluate(network, test_data, test_labels, steps, **kwargs):
    correct = 0
    view_count = np.zeros(10)
    correct_count = np.zeros(10)
    for i in tqdm(range(len(test_data))):
        network.populations[0].set_input(test_data[i])
        res = network.run(steps,
                          learning = False,
                          record = True,
                          predict = True,
                          **kwargs).get("prediction")
        if np.argmax(res) == test_labels[i]:
            correct += 1
            correct_count[test_labels[i]] += 1
        # print(np.argmax(res), test_labels[i], np.around(res,2))
        view_count[test_labels[i]] += 1
        network.rest()
    correct_count /= view_count + 0.0001
    print("Got " + color(np.around(correct / len(test_labels), 3)) + " correct")
    print("Accuracy per digit:")
    print(*[color(f) for f in list(np.around(correct_count, 3))], sep = ', ')
    return correct/len(test_labels)

def all_at_once(network, test_data, test_labels, num_labels, steps, **kwargs):
    network.neuron_labels = [np.zeros((pop.num_neurons, 10)) for  pop in network.populations[1:]]
    correct = 0
    view_count = np.zeros(10)
    correct_count = np.zeros(10)
    bar = tqdm(total = len(test_data), position = 1)
    acc = tqdm(total = 0, position = 2, bar_format ='{desc}')
    d_acc = tqdm(total = 0, position = 3, bar_format ='{desc}')
    for i in range(len(test_data)):
        network.populations[0].set_input(test_data[i])
        res = network.run(steps,
                            learning = True,
                            record = True,
                            predict = True,
                            **kwargs)
        network.rest()
        network.normalize()
        rates = res.get("rates")
        prediction = res.get("prediction")
        for j in range(len(network.neuron_labels)):
            network.neuron_labels[j][:,test_labels[i]] += rates[j]
        if np.argmax(prediction) == test_labels[i]:
            correct += 1
            correct_count[test_labels[i]] += 1
        view_count[test_labels[i]] += 1
        for pop in network.neuron_labels:
            pop /= (sum(pop) + 0.0001)
        acc.set_description_str("Accuracy: " + color(np.around(correct / (i+0.0001), 3)) + "    ")
        d_acc.set_description_str("Accuracy per digit: " + ', '.join([color(f) for f in list(np.around(correct_count /  (view_count + 0.0001), 3))]) + "                ")
        bar.update(1)
