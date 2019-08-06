import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.color import color
import pickle
from tensorflow.keras.models import load_model
from PIL import Image


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
    if kwargs.get("save_weights", False):
        pickle.dump([c.adj for c in network.connections], open("weights.pickle", "wb"))


def dream(network, steps, **kwargs):
    '''
    memory cells are smaller populations of neurons that are meant to store previously seen
    data in the form of the synaptic weights. Dreaming presents this data back to the network
    '''
    num_con = len(network.connections)
    num_pop = len(network.populations)
    inp_size = int(np.sqrt(network.populations[0].num_neurons))
    # assumes populations[1] is a memory cell and that all mem cells have same number of neruons
    ind = list(range(network.populations[1].num_neurons))
    # shuffle the order of what memories to present
    np.random.shuffle(ind)
    # turn off learning for memory cells, assumes that connections[0:num_con-2] are connections to the memory cells
    for conn in network.connections[0:num_con-2]:
        conn.rule = "static"
        conn.synapse.rule = "static"
    # present memories for "reps" number of times
    for r in range(kwargs.get("reps", 1)):
        for i in ind:
            # loop through the connections from input to mem cells, assumes every other connection is from input to a memory cell
            for mem in network.connections[0:num_con-2:2]:
                # set input to the weights of a neuron from a memory cell, assumes memory cell is fully connected to input
                memory = mem.adj[:,i]/np.max(mem.adj[:,i])
                # img = Image.fromarray((memory*255).astype(np.uint8))
                # img.save("img/memory.png")
                network.populations[0].set_input(memory)
                network.run(steps, **kwargs)
                network.normalize()
                network.rest()




def label_neurons(network, test_data, test_labels, num_labels, steps, **kwargs):
    '''
    firing_rates is a list of 2D arrays (of possibly varying shape) such that
    firing_rates[i,j] is an array containing the 10 average firing rates
    (cooresponding to each digit) for the jth neuron in the ith population
    (excluding the input layer)

    So if firing_rates[1,3,8] is high, that means the 3rd neuron in the
    1st population tends to fire a lot when the network is presented with an 8
    '''
    weight = []
    thresh = []
    volt = []
    act = []
    pop_rates = [np.zeros((pop.num_neurons, num_labels)) for  pop in network.populations[1:]]

    # For each image shown, record the number of times that the neurons fire
    # and count how many times the network saw each digit
    for i in tqdm(range(len(test_data))):
        network.populations[0].set_input(test_data[i])
        res = network.run(steps,
                            learning = False,
                            record = True,
                            **kwargs)
        rates = res.get("rates")
        network.rest()
        for j in range(len(pop_rates)):
            pop_rates[j][:,test_labels[i]] += rates[j]
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

    for pop in pop_rates:
        pop /= (sum(pop) + 0.0001)

    network.neuron_labels = pop_rates

def evaluate(network, test_data, test_labels, steps, **kwargs):
    weight = []
    thresh = []
    volt = []
    act = []
    correct = 0
    view_count = np.zeros(10)
    correct_count = np.zeros(10)
    for i in tqdm(range(len(test_data))):
        network.populations[0].set_input(test_data[i])
        res = network.run(steps,
                          learning = False,
                          record = True,
                          predict = True,
                          **kwargs)
        prediction = res.get("prediction")
        if np.argmax(prediction) == test_labels[i]:
            correct += 1
            correct_count[test_labels[i]] += 1
        # print(np.argmax(res), test_labels[i], np.around(res,2))
        view_count[test_labels[i]] += 1
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
    correct_count /= view_count + 0.0001
    print("Got " + color(np.around(correct / len(test_labels), 3)) + " correct")
    print("Accuracy per digit:")
    print(*[color(f) for f in list(np.around(correct_count, 3))], sep = ', ')
    return correct/len(test_labels)

def all_at_once(network, test_data, test_labels, num_labels, steps, **kwargs):
    if not network.neuron_labels:
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
        acc.set_description_str("Accuracy: " + color(np.around(correct / (i+1), 3)) + "    ")
        d_acc.set_description_str("Accuracy per digit: " + ', '.join([color(f) for f in list(np.around(correct_count /  (view_count + 0.0001), 3))]) + "                       ")
        bar.update(1)
    if kwargs.get("save_weights", False):
        pickle.dump([c.adj for c in network.connections], open("weights.pickle", "wb"))
