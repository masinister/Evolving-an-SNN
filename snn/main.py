from schemes import Schemes

Input = Population(100,shape='2D Grid')
con = Connection(Input,Input, Schemes.get("random")(Input.num_neurons, Input.num_neurons), 'Local24')
print(con.adj)
