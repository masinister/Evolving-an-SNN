from snn.population import Image_Input, Population
from snn.connection import Connection
from snn.network import Network
from snn import schemes
from utils.correlation import correlation
from tensorflow.keras.datasets import mnist
import numpy as np
from tqdm import tqdm
import pickle
import time
import gudhi

'''
Perform TDA on a pretrained network
'''


(x_train, y_train), (x_test, y_test) = mnist.load_data()

params = {"eta": 0.0005, "mu": 0.05}

allBut1 = schemes.get("allBut1")
all2all = schemes.get("all2all")

''' Initialize populations '''
Input = Image_Input(x_train[0])
L1 = Population(
    num_neurons = 100,
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

''' Initialize connections '''
inh = -120
C1 = Connection(Input,
                L1,
                0.3 * all2all(Input.num_neurons, L1.num_neurons),
                params,
                rule = "static",
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

network = Network([Input, L1], [C1, C2])

''' Import and load saved weights '''
weight_file = open('weights.pickle', 'rb')
weights = pickle.load(weight_file)
weight_file.close()
network.set_weights(weights)

''' Record spiking activity of Layer 1 while showing it a bunch of images '''
print("Recording Spikes")
spikes = network.run(50, learning=False, record_spikes=True, pop_index=1).get("spikes")
for i in tqdm(range(1,20)):
    network.populations[0].set_input(x_train[i])
    spikes = np.append(spikes, network.run(50, learning=False, record_spikes=True, pop_index=1).get("spikes"), axis = 1)
    network.rest()


''' Calculate correlation matrix from spiking activity '''
corr = np.zeros((len(spikes), len(spikes)))

print("Calculating correlation matrix")
for i in tqdm(range(len(spikes)-1)):
    for j in range(i+1,len(spikes)):
        corr[i,j] = correlation(spikes[i,:], spikes[j,:])
        corr[i,j] = 1/(corr[i,j]+.01)

corr = corr + np.transpose(corr)

''' Compute and plot barcodes based on correlation matrix '''
print("Computing Barcodes")
rips_complex = gudhi.RipsComplex(distance_matrix=corr)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
diag_Rips = simplex_tree.persistence()
plt = gudhi.plot_persistence_barcode(diag_Rips)
plt.show()
