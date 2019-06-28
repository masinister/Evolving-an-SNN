Input = Population(100,shape='2D Grid')
con = Connection(Input,Input,'Local24')
print(con.connections[1])
