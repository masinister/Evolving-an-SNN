import matplotlib.pyplot as plt
from collections import Iterable

class Plotter:

    def __init__(self, fields):
        fig, axs = plt.subplots(len(fields),gridspec_kw={'hspace': .5})
        self.fields = fields
        self.fig = fig
        self.axs = [axs] if not isinstance(axs, Iterable) else axs
        plt.ion()
        plt.show()

    def plot(self, data):
        for i in range(len(self.axs)):
            self.axs[i].clear()
            self.axs[i].set_title(self.fields[i])
            self.axs[i].plot(data[self.fields[i]])
        plt.pause(0.05)
