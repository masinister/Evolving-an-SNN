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
'''
2-layer Diehl and Cook network
'''


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

for x in x_train:
    x = x.astype(float) * 2

'''
Initialize populations
'''

Input = population.Image_Input()
L1 = population.Population(
    num_neurons = 100,
    v_init = -65,
    v_decay = .99,
    v_reset = -60,
    min_volt = -65,
    t_init = -52,
    min_thresh = -52,
    t_bias = 0.05,
    t_decay = .9999999,
    refrac = 5,
    trace_decay = .95,
    one_spike = True
)
L2 = population.Population(
    num_neurons = 100,
    v_init = -65,
    v_decay = .9,
    v_reset = -45,
    min_volt = -60,
    t_init = -40,
    adapt_thresh = False,
    refrac = 2,
    trace_decay = .95,
)
'''
Initialize connections
'''

inh = -17.5
exc = 22.5
C1 = Connection(Input,
                L1,
                0.3 * all2all(Input.num_neurons, L1.num_neurons),
                params,
                rule = "PreAndPost",
                wmin = 0,
                wmax = 1,
                norm = 78.4)

C2 = Connection(L1,
                L2,
                one2one(L1.num_neurons) * exc,
                params,
                rule = "static",
                wmin = 0,
                wmax = exc)

C3 = Connection(L2,
                L1,
                allBut1(L1.num_neurons) * inh,
                params,
                rule = "static",
                wmin = inh,
                wmax = 0)

network = Network([Input, L1, L2], [C1, C2, C3])

plotter = Plotter(["voltage", "trace", "activation"])
outer = tqdm(total = 100, desc = 'Epochs', position = 0)
for i in range(100):
    train.all_at_once(network, x_train[1000 * i: 1000 * (i+1)], y_train[1000 * i: 1000 * (i+1)], 10, 300, draw_weights = True, plot = plotter)
    outer.update(1)
