import schemes
from connection import Connection
import population
from synapse import Synapse
import neuron
from network import Network
import train
from tensorflow.keras.datasets import mnist
import numpy as np

params = {"eta": 1.5, "mu": 2.0, "decay": 0.5, "avg": 1}

(x_train, y_train), (x_test, y_test) = mnist.load_data()

Input = population.Image_Input(x_train[0])
L1 = population.Population(784, neuron_type=neuron.ICMNeuron)
L2 = population.Population(784, neuron_type=neuron.ICMNeuron)

rand = schemes.get("random")
allBut1 = schemes.get("allBut1")
grid = schemes.get("grid")

C1 = Connection(Input, L1, rand(784,784), "STDP", params)
C2 = Connection(L1, L1, grid(28,28), "STDP", params)
C3 = Connection(L1, L2, rand(784,784), "STDP", params)
C4 = Connection(L2, L2, grid(28,28), "STDP", params)

network = Network([Input, L1, L2], [C1, C2, C3, C4])
network.set_params(params)
train.train(network, x_train[0:10], 100, 28)
