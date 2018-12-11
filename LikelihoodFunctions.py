import autograd.numpy as np


"""
TEST CASE: LOGISTIC REGRESSION, EQUATION 21 FROM HOFFMAN AND GELMAN
theta: coefficients that are estimated
creditX: independent variables; matrix from the data
creditY: dependent variable; matrix from the data
sigma2: theta (intercept at index 0 and coefficients) is given zero mean
        normal priors and variance of 100
output: lProb is the log probability for theta
"""
def lBayesLR(theta, x, y, sigma2=100):
    # Discard small values since they only conribute little to the sum of exponents
    # that is discard all values below -700.0
    exponent0=(-y*(x@theta))[-y*(x@theta) > -700.0]
    # Implement log-sum-trick to avoid crass overflows
    term1=-np.sum(exponent0+np.log(1.0+np.exp(-exponent0)))
    term2=-1/(2*sigma2)*theta@theta
    lProb=term1+term2
    return lProb

    
"""
TEST CASE HIERARCHICAL LOGISTIC REGRESSION, EQ. 22
"""
# First parameter in theta is variance
# Rest are beta coefficients
def lBayesHLR(theta, x, y, lambda0=0.01):
    sigma2=np.exp(theta[0])
    beta=theta[1:]
    prob1=-np.sum(np.log(1+np.exp(-(y[:, np.newaxis]*x)@beta)))-1/(2*sigma2)*beta@beta-len(y)/2*np.log(sigma2)-lambda0*sigma2
    return prob1