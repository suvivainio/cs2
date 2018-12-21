import autograd.numpy as np
import autograd.numpy.random as npr
import autograd
import matplotlib.pyplot as plt

npr.seed(42)

"""
Parameters:
theta0: starting value of the parameters that are estimated
epsilon: step size
nSteps: denoted L in the paper; number of steps to take to produce a new candidate for theta
fLikelihood: log-likelihood function, that can be differentiated 
nSamples: tells the number of samples to take; burn in period is also included in this parameter. 
            Burn-in period is 0.5*nSamples.

Return:
basicHmc retuns last half of the sample, theta values and corresponding r values.
"""
def basicHmc(theta0, epsilon, nSteps, fLikelihood, nSamples):
    """
    Initialize:
    empty vectors for theta (thetaHist) and r (rHist) values.
    thetaHist: vector is the size of nSamples+1, where first value is the parameter theta0
    gradLikelihood: gradient of the likelihood parameter
    nAccepted: tells the acceptance rate after the burn-in period.
    """
    thetaHist=np.empty([nSamples+1, len(theta0)])
    thetaHist[0]=theta0
    rHist=np.empty([nSamples, len(theta0)])
    gradLikelihood=autograd.grad(fLikelihood)
    nAccepted=0.0
    allAccepted=0.0
    """
    In the beginning of each round momentum variable r is initialized with a sample from normal distribution
    that has the same length as theta-vector.
    """
    for i in range(nSamples):
        r0=npr.normal(size=len(theta0))
        rCur=np.copy(r0)
        rHist[i]=r0
        thetaCur=thetaHist[i]
        thetaHist[i+1]=thetaHist[i]
        """
        Take the predefined number of leapfrog steps of size epsilon 
        """
        for j in range(nSteps):
            thetaCur, rCur = Leapfrog(thetaCur, rCur, epsilon, gradLikelihood)
        """
        Accept the proposed theta with propability alpha
        """
        cond0=fLikelihood(thetaCur)-1/2*rCur.T@rCur
        cond1=fLikelihood(thetaHist[i])-1/2*r0.T@r0
        alpha=np.log(npr.rand()) 
        if alpha < min(0,(cond0 - cond1)):
            thetaHist[i+1]=thetaCur
            rHist[i]=-rCur
            allAccepted+=1.0
            if i > nSamples/2:
                nAccepted+=1.0
    print('All in all accepted: ',allAccepted,', proportion: ', allAccepted/nSamples)
    print('Accepted: ', nAccepted)
    print('Acceptance rate: ', nAccepted / (nSamples/2.0))
    return thetaHist[nSamples//2+1:], rHist[nSamples//2:]

def Leapfrog(parTheta, parR, parEpsilon, thetaGradient):
    parR0 = parR +(parEpsilon/2)*thetaGradient(parTheta)
    parTheta0 = parTheta+parEpsilon*parR0
    parR0 = parR0+(parEpsilon/2)*thetaGradient(parTheta0)
    return parTheta0, parR0

