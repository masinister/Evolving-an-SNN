

def train(network, train_data, learn_steps, rest_steps):
    i=0
    for x in train_data:
        network.populations[0].set_input(x)
        network.run(learn_steps)
        network.populations[0].set_blank()
        network.run(rest_steps)
        i+=1

def associate_neurons(network, test_data, test_labels, steps):
    pass
