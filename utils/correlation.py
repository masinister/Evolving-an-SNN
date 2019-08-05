import numpy as np

def shift(spikes, k):
    num = len(spikes)
    shifted = np.zeros(num)
    shifted[0:num-k] = spikes[k:]
    return shifted

def correlation(spike1,spike2):
    assert len(spike1) == len(spike2), "Spike trains different length"
    num = len(spike1)
    scale = (np.sum(spike1)*np.sum(spike2))
    if scale != 0:
        scale = 1/scale
    # tau_max = T = number time steps image is shown for
    ccg12 = np.zeros(num+1)
    ccg21 = np.zeros(num+1)

    for k in range(num+1):
        ccg12[k] = np.dot( spike1, shift(spike2,k) )
        ccg21[k] = np.dot( spike2, shift(spike1,k) )

    return scale*np.max([ np.sum(ccg12), np.sum(ccg21) ])
