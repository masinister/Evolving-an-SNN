import schemes
from connection import Connection
from population import Population
from synapse import Synapse

Input = Population(100)
rand = schemes.get("random")
params = {"eta": 0.5, "mu": 1}
con = Connection(Input,Input, rand(Input.num_neurons, Input.num_neurons), , params)
print(con.adj)
