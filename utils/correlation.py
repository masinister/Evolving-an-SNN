import numpy as np

def shift(spikes, k):
    num = len(spikes)
    shifted = np.zeros(num)
    shifted[0:num-k] = spikes[k:]
    return shifted

def correlation(spike1, spike2, window):
    assert len(spike1) == len(spike2), "Spike trains have different length"
    num = len(spike1) # number of time steps in spike trains
    assert window <= num, "Window too large, will cause out of range index"
    scale = np.sum(spike1)*np.sum(spike2)*window
    if scale != 0:
        scale = 1/scale * num

    ccg12 = np.zeros(window+1)
    ccg21 = np.zeros(window+1)

    for k in range(window+1):
        ccg12[k] = np.dot( spike1, shift(spike2,k) )
        ccg21[k] = np.dot( spike2, shift(spike1,k) )

    return scale*np.max([ np.sum(ccg12), np.sum(ccg21) ])
