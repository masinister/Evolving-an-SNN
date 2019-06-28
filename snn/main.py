import schemes
from connection import Connection
from population import Population

Input = Population(100)
rand = schemes.get("random")
con = Connection(Input,Input, rand(Input.num_neurons, Input.num_neurons), 'Local24')
print(con.adj)
