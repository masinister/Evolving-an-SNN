from snn import schemes
from snn.connection import Connection
from snn import population
from snn.network import Network
from snn import train
from tensorflow.keras.datasets import mnist
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pickle
from utils.plot import Plotter


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
Initialize populations
'''

Input = population.Image_Input()
L1 = population.Population(
    num_neurons = 150,
    v_init = -65,
    v_decay = .99,
    v_reset = -60,
    min_volt = -65,
    t_init = -52,
    min_thresh = -52,
    t_bias = 0.25,
    t_decay = .9999999,
    refrac = 5,
    trace_decay = .95,
    one_spike = True
)
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
                wmax = 1,
                norm = 78.4)

C2 = Connection(L1,
                L1,
                allBut1(L1.num_neurons) * inh,
                params,
                rule = "static",
                wmin = inh,
                wmax = 0)

network = Network([Input, L1,], [C1, C2,])

plotter = Plotter(["trace"])

# outer = tqdm(total = 100, desc = 'Epochs', position = 0)
for i in range(100):
    # train.all_at_once(network, x_train[500 * i: 500 * (i+1)], y_train[500 * i: 500 * (i+1)], 10, 250, draw_weights = True)
    print("Training", i)
    train.train(network, x_train[500 * i: 500 * (i+1)], 250, save_weights = True, plot = plotter)
    print("Labelling", i)
    train.label_neurons(network, x_train[0: 100], y_train[0: 100], 10, 300)
    print("Testing", i)
    train.evaluate(network, x_train[50000:50100], y_train[50000:50100], 300)
    # outer.update(1)
print("Testing")
train.evaluate(network, x_train[50000:51000], y_train[50000:51000], 100)
