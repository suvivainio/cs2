# Computational statistics 2 homework

## Problem 1
Hoffman & Gelman 2011: The No U-Turn Sampler
* implement algorithms 1 (basic HMC) & 6 (NUTS with dual averaging)
   * test that your implementations work by replicating the following tests in Section 4.1:
      * 250-dimensional Gaussian
      * Bayesian logistic regression & hierarchical Bayesian logistic regression on the UCI german credit data
* compare the results given by basic HMC & NUTS
* compare your results with the same models run in Stan using the provided NUTS sampler (see e.g. mc-stan.org/users/interfaces/pystan.html)
        
### Relevant files
Notebooks:
* The No-U-Turn Sampler.ipynb
* pystan tests.ipynb
Samplers:
* noUturnSampler.py
* HMC.py
Functions and other files:
* ChainMix.py
* ConvergenceTests.py
* LikelihoodFunctions.py
* Neff.py
* dataGerman.tab

## Problem 2
Moreno et al. 2016: Automatic Variational ABC
* implement algorithm 1
* test your implementation by replicating the simulated tests (4.1-4.2)
* compare your results to the results in the paper
    
### Relevant files
Automatic Variational ABC.ipynb
ABC.py
