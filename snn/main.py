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
params = {"eta": 1.5, "mu": 1.5, "decay": 1, "avg": 1}

# Input = population.Population(100, neuron_type=neuron.TestNeuron)
L1 = population.Population(784, neuron_type=neuron.ICMNeuron)

rand = schemes.get("random")
allBut1 = schemes.get("allBut1")
grid = schemes.get("grid")

C1 = Connection(Input, L1, rand(Input.num_neurons, L1.num_neurons), "STDP", params)
C2 = Connection(L1, L1, grid(28,28), "STDP", params)

p = []
v = []
a = []

def train(n):
    print("Training image ", n)
    data = np.zeros((28,28), dtype=np.uint8)
    img = Image.fromarray(x_train[n])
    img.save("img/%d.png" %(n,))
    Input.change_image(x_train[n])
    for i in range(100):
        C1.update()
        # C2.update()
        Input.update()
        L1.update()
        if np.count_nonzero(L1.activations) > 0:
            for j in range(len(data)):
                for k in range(len(data[0])):
                    data[j][k] = 255 * L1.activations[28*j + k]
            img = Image.fromarray(data)
            img.save("img/img %d_%d.png" %(n,i,))
        p.append([x.threshold for x in L1.neurons][::10])
        v.append([x.voltage for x in L1.neurons][::10])

for i in range(1):
    train(i)
#
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(p)
ax2.plot(v)
plt.show()
