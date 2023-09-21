import math
import emcee
import numpy as np

"""
MCMCStarSolver.py

This module contains the implementation of a usecase for the emcee generator

Methods
    - cost_function: a generic cost function to estimate chi2, and the acceptance of walkers
    - log_likelihood: the function to determine the likelihood for each value
    - log_prior: defines prior knowledge of values
    - log_probability: computes the log_prior + log_likelihood for each cycle
    - estimateStellarParameters: The method that should be called from this module to estimate parameters
                                returns the chain sampled for analyzing and plots, along with median estimated
                                values for the parameters iterated 
                                
"""

def cost_function(x, t, e):
    s = 0.0
    for i in range(len(x)):
        s += math.pow(x[i] - t[i], 2) / math.pow(e[i], 2)
    return s

def log_likelihood(theta, values, error, model):
    """
    
    The log_likelihood function computes the log-likelihood given certain parameters 
    (theta), observed values, and their respective errors. The higher the 
    log-likelihood, the more probable the observed data is under the assumed model.
    
    """
    # Get the model stellar parameters based on the predicted values
    # Here we use theta directly, as theta holds the predicted parameters used in the model, that
    # are extracted directly in the get_stellar_params method
    # TODO generalize this
    target = model(theta)
    
    # Compute the likelihood using the cost function, which compares observed and model values
    return -0.5 * cost_function(values, target, error)

def log_prior(theta, ranges):
    """
    The log_prior function computes the prior probability for given parameters (theta). 
    
    Currently using a UNIFORM prior of p=0 for parameters inside these bounds 
    Negative infinity for values outside bounds. (forbidden values)
    
    """
    for i, value in enumerate(theta):
        if not (ranges[i][0] <= value <= ranges[i][1]):
            return -np.inf
    return 0.0

def log_probability(theta, values, error, ranges, model):
    """
    The log_probability function computes the overall probability (posterior) 
    for given parameters (theta) by summing the log prior and the log likelihood.
    """
    lp = log_prior(theta, ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, values, error, model)

def estimateStellarParameters(parameters, nwalkers, runs, burnin, model, progress = False):
    """
    This method is designed to estimate the stellar parameters of a star using a MCMC method.
    
    - parameters["initialValues"] : The initial or observed values array of the star.
    - parameters["errorValues"] : The observational uncertainty or error in the corresponding values
    - parameters["ranges"] : An array of tuples containing the minimum and maximum possible values for each of the predicted parameters
    - ndim : The number of dimensions or parameters being estimated. 
    - burnin : The number of steps to discard from the beginning of the MCMC chain
    - nwalkers: the amount of walkers to use for the emcee algorithm
    - model: the method to be used to compare the cost function in the log_likelihood method
    - progress: show progress bar or not (disabled by default)

    Returns:
    - The samples from the sampler chain (to be improved)
    """
    
    initValues = parameters["initialValues"]
    errorValues = parameters["errorValues"]
    ranges = parameters["ranges"]

    ndim = len(initValues)

    # randomize the initial values of the walkers
    pos = np.empty((nwalkers, ndim))
    for i in range(ndim):
        pos[:, i] = np.random.uniform(*ranges[i], size=nwalkers)

    # Passing scalar values to the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(initValues, errorValues, ranges, model))
    sampler.run_mcmc(pos, runs, progress=progress)


    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    M_median = np.median(samples[:, 0])
    Xc_median = np.median(samples[:, 1])
    
    return samples, (M_median, Xc_median)