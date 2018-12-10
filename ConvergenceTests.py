


# This is a program that performs convergence testing
# Input: np.array[n,j,k], where 
# - n is the number of observations,
# - j is the number of parameters and 
# - k is the number of chains
# Tests: 
# - between and within-chain variance (Rhat)
# - Effective sample size

import ChainMix
import Neff
import importlib
import numpy as np
import matplotlib.pyplot as plt
importlib.reload(ChainMix)
importlib.reload(Neff)

def ConvergenceTest(input0):
    n0=input0.shape[0]
    j0=input0.shape[1]
    k0=input0.shape[2]
    # Divide chains into 2; continue processing 2k chains
    input1=np.zeros([n0//2, j0, k0*2])
    for i in range(4):
        input1[:,:,i]=input0[:n0//2,:,i]
        input1[:,:,i+1]=input0[n0//2:,:,i]

    rHat=ChainMix.ChainMix(input1)
    print('RHat statistics for the parameters are: ', rHat)
    plt.figure(0)
    plt.bar(range(len(rHat)),height=rHat)
    plt.axhline(y=1.1, color='tab:orange')
    plt.title('Rhat-statistic (target value 1.1)')
    plt.figure(1)
    inputEff=Neff.Neff(input1)
    print('Effective sample size: ', inputEff)
    plt.bar(range(len(inputEff)),height=inputEff)
    plt.axhline(y=10.0, color='tab:orange')
    plt.title('Effective sample size, target value, target 10 independent observations')


