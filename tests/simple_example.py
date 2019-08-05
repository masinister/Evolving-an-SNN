from snn import schemes
from snn.connection import Connection
from snn import population
from snn.network import Network
from snn import train
from tensorflow.keras.datasets import mnist
import numpy as np
from tqdm import tqdm

print("Initializing Network")
'''
Learning paramters for PreAndPost rule
'''
params = {"eta": 0.0005, "mu": 0.05}
'''
different connection schemes
'''
rand = schemes.get("random")
allBut1 = schemes.get("allBut1")
grid = schemes.get("grid")
all2all = schemes.get("all2all")
one2one = schemes.get("one2one")
local = schemes.get("local")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

'''
Initialize default populations
'''

Input = population.Image_Input()
L1 = population.Population()

'''
Initialize connections
'''
inh = -120
C1 = Connection(Input,
                L1,
                0.3 * all2all(Input.num_neurons, L1.num_neurons),
                params,
                rule = "PreAndPost",
                wmin = 0,
                wmax = 1)

C2 = Connection(L1,
                L1,
                allBut1(L1.num_neurons) * inh,
                params,
                rule = "static",
                wmin = inh,
                wmax = 0)

network = Network([Input, L1,], [C1, C2,])

outer = tqdm(total = 100, desc = 'Epochs', position = 0)
for i in range(100):
    train.all_at_once(network, x_train[1000 * i: 1000 * (i+1)], y_train[1000 * i: 1000 * (i+1)], 10, 300, draw_weights = False)
    outer.update(1)
