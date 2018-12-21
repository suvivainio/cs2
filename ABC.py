import autograd.numpy as np
import autograd.numpy.random as npr
import autograd
import matplotlib.pyplot as plt
from scipy.special import binom
from scipy.misc import derivative
npr.seed(42)

# Parameters
# Phi0: original values of phi
# nIterations: number of iterations in the main program
# fTarget: target function
# fNu: function to simulate nu
# nSample: sample size (=length of Nu)
# fU: function to simulate u
# nParticle: number of particles (=length of u)
# rho_0, t0: parameters defining the step length
def avabc(phi0,nIterations,fTarget, fNu, nSample, fU, nParticle,CalculateTheta, printInfo=False,rho_0=0.1, t0 = 100):
    phiHist = np.zeros([nIterations, len(phi0)])
    phiHist[0]=phi0
    lambdaTarget=lambda x: fTarget(x, nu, u)
    gradTarget=autograd.grad(lambdaTarget)
    # lower bound indormation
    lbHist=np.empty(nIterations)
    gradHist=np.empty(nIterations)
    thetaHist=np.empty(nIterations)
    for t in range(1, nIterations):
        if printInfo: print('********** Round: ', t)
        nu = fNu(nSample)
        u=fU(nParticle)
        g=gradTarget(phiHist[t-1])
        if printInfo: print('avabc: g', g)
        phiHist[t]=phiHist[t-1]+rho_0/(t0+t)*g
        if printInfo: print('*** avabc, phiHist, rho, g: ', phiHist[t], rho_0, g)
        # Collect information on sampler performance
        lbHist[t]=lambdaTarget(phiHist[t])
        gradHist[t]=derivative(lambdaTarget, phiHist[t])
        thetaHist[t]=CalculateTheta(phiHist[t], nu)
    print('Returning values')
    return phiHist, lbHist, gradHist, thetaHist