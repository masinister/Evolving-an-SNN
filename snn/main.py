import schemes
from connection import Connection
import population
from synapse import Synapse
import neuron
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()
Input = population.Image_Input(x_train[0])
# Input = population.Population(100, neuron_type=neuron.TestNeuron)
L1 = population.Population(784, neuron_type=neuron.ICMNeuron)
rand = schemes.get("random")
allBut1 = schemes.get("allBut1")
params = {"eta": 0.5, "mu": 2, "decay": 0.2, "avg": 1}
C1 = Connection(Input, L1, rand(Input.num_neurons, L1.num_neurons), "STDP", params)
C2 = Connection(L1, L1, allBut1(L1.num_neurons), "STDP", params)

p = []
data = np.zeros((28,28), dtype=np.uint8)

img = Image.fromarray(x_train[0])
img.save("img/base.png")

for i in range(300):
    C1.update()
    C2.update()
    Input.update()
    L1.update()
    # p.append(C2.post.neurons[0].threshold)
    if np.count_nonzero(C2.post.activations) > 0:
        for j in range(len(data)):
            for k in range(len(data[0])):
                data[j][k] = 255*L1.activations[28*j + k]
            # print(C2.post.activations[0])
        img = Image.fromarray(data)
        img.save("img/img %d.png" %(i,))

# plt.plot(p)
# plt.show()
