import schemes
from connection import Connection
from population import Population

Input = Population(100)
rand = schemes.get("random")
params = {"eta": 0.5, "mu": 1}
con = Connection(Input,Input, rand(Input.num_neurons, Input.num_neurons), 'Local24', params)
print(con.adj)
