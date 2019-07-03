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
params = {"eta": 1.5, "mu": 2.0, "decay": 0.5, "avg": 1}

# Input = population.Population(100, neuron_type=neuron.TestNeuron)
L1 = population.Population(784, neuron_type=neuron.ICMNeuron)
L2 = population.Population(784, neuron_type=neuron.ICMNeuron)

rand = schemes.get("random")
allBut1 = schemes.get("allBut1")
grid = schemes.get("grid")

C1 = Connection(Input, L1, rand(784,784), "STDP", params)
C2 = Connection(L1, L1, grid(28,28), "STDP", params)
C3 = Connection(L1, L2, grid(28,28), "STDP", params)

p = []
v = []
a = []

def train(n):
    print("Training image ", n)
    data = np.zeros((28,28), dtype=np.uint8)
    img = Image.fromarray(x_train[n])
    img.save("img/%d.png" %(n,))
    Input.change_image(x_train[n])
    for i in range(64):
        Input.update()
        L1.update()
        L2.update()
    for i in range(64):
        C1.update()
        C2.update()
        C3.update()
        Input.update()
        L1.update()
        L2.update()
        p.append([x.threshold for x in L2.neurons][::10])
        v.append([x.voltage for x in L2.neurons][::10])
        if np.count_nonzero(L1.activations) > 0:
            for j in range(len(data)):
                for k in range(len(data[0])):
                    data[j][k] += L2.activations[28*j + k]
    img = Image.fromarray(data * 50)
    img.save("img/img %d_%d.png" %(n,i,))

for i in range(10):
    train(i)
#
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(p)
ax2.plot(v)
plt.show()
