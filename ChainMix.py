
# This function assigns the mixing of the chains
# Formula is 11.3 from BDA3, p. 284
# Input: np.array[n,j,k], where 
# - n is the number of observations,
# - j is the number of parameters and 
# - k is the number of chains
import numpy as np
def ChainMix(input0, nEff=False):
    # m: number of chains
    # n: length of chains
    # b: between chain variance
    # w: within chain variance
    m=input0.shape[2]
    n=input0.shape[0]
    meanChain=np.mean(input0, axis=0)
    meanAllChains=np.mean(meanChain, axis=1)
    b=n/(m-1) * np.sum((meanChain-meanAllChains[:, np.newaxis])**2, 1)
    w=np.mean(np.std(input0, axis=0, ddof=1)**2,1)
    varPsi=(n-1)/n*w+1/n*b
    if nEff==True: return varPsi
    return np.sqrt(varPsi/w)
