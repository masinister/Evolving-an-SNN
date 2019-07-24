import numpy as np
from network import Network
import train
from tensorflow.keras.datasets import mnist
import schemes
import population
from connection import Connection

class Individual():

    def __init__(self, id, params = {"eta": 0.0005, "mu": 0.05, "decay_pre": 0.95, "decay_post": 0.95, "t_bias": 0.25, "t_decay": 0.9999999, "v_decay": 0.99}):
        self.id = id
        self.accuracy = 0.0
        self.params = params
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train[0:200]
        self.y_train = y_train[0:200]
        self.x_test = x_test[0:200]
        self.y_test = y_test[0:200]
        self.network = None


    def randomize(self):
        ## Only use on decay parameters (things between 0 and 1)
        temp = {}
        for i in self.params.keys():
            temp[i] = np.random.random_sample(np.shape(self.params[i]))
        self.params = temp
        self.initialize_network(self.params)

    def evaluate(self):
        train.train(self.network, self.x_train, 250, id = self.id)
        train.label_neurons(self.network, self.x_train[0:200], self.y_train[0:200], 10, 100)
        self.accuracy = train.evaluate(self.network, self.x_test, self.y_test, 100)

    def initialize_network(self, params):
        allBut1 = schemes.get("allBut1")
        all2all = schemes.get("all2all")
        Input = population.Image_Input(self.x_train[0])
        c_params = {"eta": params.get("eta", 0.0005),
                    "mu": params.get("mu", 0.05),
                    "decay_pre": params.get("decay_pre", 0.95),
                    "decay_post": params.get("decay_post", 0.95)}
        L1 = population.Population(
            num_neurons = params.get("num_neurons", 16),
            v_init = params.get("v_init", -65),
            v_decay = params.get("v_decay", .99),
            v_reset = params.get("v_reset", -65),
            v_rest = params.get("v_rest", -65),
            t_init = params.get("t_init", -50),
            min_thresh = params.get("min_thresh", -52),
            t_bias = params.get("t_bias", 0.25),
            t_decay = params.get("t_decay", .9999999),
            refrac = params.get("refrac", 5),
            one_spike = params.get("one_spike", True)
        )
        inh = params.get("inh", -120)
        C1 = Connection(Input, L1, 0.3 * all2all(Input.num_neurons, L1.num_neurons), params, rule = "PreAndPost", wmin = 0, wmax = 1)
        C2 = Connection(L1, L1, allBut1(L1.num_neurons) * inh, params, rule = "static", wmin = inh, wmax = 0)
        self.network = Network([Input, L1, ], [C1, C2,])
        self.network.set_params(c_params)
