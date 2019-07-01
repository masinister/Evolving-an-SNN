import schemes
from connection import Connection
from population import Population
from synapse import Synapse
import neuron
import matplotlib.pyplot as plt

Input = Population(2, neuron_type=neuron.TestNeuron)
Output = Population(2, neuron_type=neuron.ICMNeuron)
rand = schemes.get("random")
params = {"eta": 0.5, "mu": 0.5, "decay": 0.5, "avg": 0.5}
con = Connection(Input, Output, rand(Input.num_neurons, Output.num_neurons), "STDP", params)
#
# print(con.adj, con.params)
# p = []
# v = []
# for i in range(100):
#     con.update()
#     Input.update()
#     Output.update()
#     p.append(con.post.neurons[0].threshold)
#     v.append(con.post.neurons[0].voltage)
#     print(con.adj)
#
# plt.plot(p)
# plt.plot(v)
# plt.show()

print(schemes.get("grid")(6,3))
