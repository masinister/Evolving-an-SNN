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

Input = population.Image_Input(x_train[1])
L1 = population.Population(100, neuron_type=neuron.ICMNeuron)

params = {"eta": 1.5, "mu": 1.5, "decay": 0.5, "avg": 1}

rand = schemes.get("random")
allBut1 = schemes.get("allBut1")
grid = schemes.get("grid")

C1 = Connection(Input, L1, rand(Input.num_neurons, L1.num_neurons), "STDP", params)

nn = 3
t = [[] for _ in range(nn)]
v = [[] for _ in range(nn)]
a = [[] for _ in range(nn)]

ai = [[] for _ in range(nn)]

for j in range(nn):
    t[j].append(L1.neurons[j*20].threshold)
    v[j].append(L1.neurons[j*20].voltage)
    a[j].append(L1.neurons[j*20].activation*30)


for i in range(2000):
    C1.update()
    Input.update()
    L1.update()

    for j in range(nn):
        t[j].append(L1.neurons[j*20].threshold)
        v[j].append(L1.neurons[j*20].voltage)
        a[j].append(L1.neurons[j*20].activation*30)
        ai[j].append(Input.neurons[j*140].activation)


for i in range(nn):
    t[i].insert(0,499)
t = np.array(t)

fig, axs = plt.subplots(nn,sharex=True)
for i in range(nn):
    axs[i].plot(t[i,:])
    axs[i].plot(v[i])
    axs[i].plot(a[i])
plt.show()






