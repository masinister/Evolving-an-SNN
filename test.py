import population
from network import Network
from correlation import correlation_coeff
from tensorflow.keras.datasets import mnist
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import time
import gudhi


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# Input = population.Image_Input(x_train[1])
# network = Network([Input],[])
# spikes = network.record_spikes(250)


# for i in range(250):
#     img = Image.fromarray((np.reshape(spikes[:,i]*255,(28,28))).astype(np.uint8))
#     img.save("spikes.png")
#     time.sleep(.5)


# corr = np.zeros((784,784))
#
# for i in tqdm(range(783)):
#     for j in range(i+1,784):
#         corr[i,j] = correlation(spikes[i,:], spikes[j,:])
#
# corr_out = open('corr.pickle','wb')
# pickle.dump(corr,corr_out)
# corr_out.close()

corr_in = open('corr.pickle','rb')
corr = pickle.load(corr_in)

for i in tqdm(range(783)):
    for j in range(i+1,784):
        corr[i,j] = 1/corr[i,j] if corr[i,j]!=0 else 100

corr = corr + np.transpose(corr)

rips_complex = gudhi.RipsComplex(distance_matrix=corr)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
diag_Rips = simplex_tree.persistence()
plt = gudhi.plot_persistence_barcode(diag_Rips)
plt.show()

# plt.imshow(corr,cmap="gist_ncar")
# plt.colorbar()
# plt.show()
