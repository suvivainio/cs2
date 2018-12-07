
# This function calculates the effective sample size for the input array
# Formula is 11.5 from BDA3, p. 286
# Input: np.array[n,j,k], where 
# - n is the number of observations,
# - j is the number of parameters and 
# - k is the number of chains
import numpy as np
import ChainMix
import importlib
importlib.reload(ChainMix)

def Neff(input0):
    varPsi=ChainMix.ChainMix(input0, True)
    # Calculate squared differences
    diff0=np.zeros([input0.shape[0],input0.shape[0],input0.shape[1],input0.shape[2]])
    for k in range(input0.shape[2]):
        for j in range(input0.shape[1]):
            for n0 in range(input0.shape[0]):
                for n1 in range(input0.shape[0]):
                    if n1 < n0:
                        diff0[n0,n1,j,k]=(input0[n0,j,k]-input0[n1,j,k])**2
    vt0=np.sum(diff0, axis=0)
    print('vt0', vt0.shape)
    vt1=np.sum(vt0,axis=2)
    m=input0.shape[2]
    n=input0.shape[0]
    tt=1/(m*(n-np.array(range(n))))
    print('tt', tt.shape)
    # Vector with autocorrelations for each lag and parameter
    vt=tt[:,np.newaxis]*vt1
    # varPsi has variances for each parameter
    # autocorr has dimensions lag and parameter
    autocorr=1-vt/(2*varPsi)
    # Effective sample size for each parameter
    nEff=m*n/(1+2*np.sum(autocorr, axis=0))
    return nEff
