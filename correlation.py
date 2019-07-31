import numpy as np

def shift(spikes, k):
    num = len(spikes)
    shifted = np.zeros(num)
    shifted[0:num-k] = spikes[k:]
    return shifted

def correlation_coeff(spike1,spike2):
    assert len(spike1) == len(spike2), "Spike trains different length"
    num = len(spike1)
    # tau_max = T = number time steps image is shown for
    ccg12 = np.zeros(num+1)
    ccg21 = np.zeros(num+1)

    for k in range(num+1):
        ccg12[k] = np.dot( spike1, shift(spike2,k) )
        ccg21[k] = np.dot( spike2, shift(spike1,k) )

    return np.max([ np.sum(ccg12), np.sum(ccg21) ])


s1 = [1,0,1,0,1,1]
s2 = [1,0,1,1,0,0]

print(correlation_coeff(s1,s2))
