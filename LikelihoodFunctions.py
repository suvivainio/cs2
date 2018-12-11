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
    lProb=-np.sum(np.log(1+np.exp(-y*(x@theta))))-1/(2*sigma2)*theta@theta
    return lProb

"""
def lBayesLR(theta, x, y, sigma2=100):
    # Delete very small terms, since they don't contribute to the sum
    # and would cause underflow
    exponent0=-y*(x@theta)
    exponent1=exponent0[exponent0>-700.0]
    if len(exponent0[exponent0<-700.0])>0: 
        print('Capping small likelihood values!')
        print('original vs capped length', len(exponent0), len(exponent1))
        print('discarded # of values: ', len(exponent0)-len(exponent1))
    maxExp=np.max(exponent1)
    if maxExp > 700:
        # max(Exponent) is large use logsum-trick to avoid overflows
        # Ignore term np.log(1), since it is small
        lProb=-np.sum(maxExp+np.log(np.exp(exponent1-maxExp)))-1/(2*sigma2)*theta@theta
    else:
        
        lProb=-np.sum(np.log(1+np.exp(exponent1)))-1/(2*sigma2)*theta@theta

    return lProb
"""