import schemes
from connection import Connection
import population
from synapse import Synapse
import neuron
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
Input = population.Image_Input(x_train[0])
# Input = population.Population(100, neuron_type=neuron.TestNeuron)
Output = population.Population(100, neuron_type=neuron.ICMNeuron)
rand = schemes.get("random")
params = {"eta": 0.5, "mu": 0.5, "decay": 0.5, "avg": 0.5}
con = Connection(Input, Output, rand(Input.num_neurons, Output.num_neurons), "STDP", params)

p = []
v = []
for i in range(100):
    con.update()
    Input.update()
    Output.update()
    p.append(con.post.neurons[0].threshold)
    v.append(con.post.neurons[0].voltage)
    # print(con.post.neurons[0].threshold, con.post.neurons[0].voltage)
plt.plot(p)
plt.plot(v)
plt.show()

# print(x_train[0].shape)
