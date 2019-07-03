

def train(network, train_data, learn_steps, rest_steps):
    i=0
    for x in train_data:
        print("Training item", i)
        network.populations[0].set_input(x)
        network.run(learn_steps)
        network.populations[0].set_blank()
        network.run(rest_steps)
        i+=1
