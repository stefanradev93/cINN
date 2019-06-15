from numba import jit, prange
import numpy as np
import scipy
import tensorflow as tf


@jit(nopython=True)
def simulate_ricker_single(t_obs, r, sigma, phi):
    """
    Simulate a dataset from the Ricker model 
    describing dynamics of the number of individuals over time.

    Arguments:
    t_obs : int -- length of the observed time-series
    r     : float -- the r parameter of the Ricker model
    sigma : float -- the sigma parameter of the Ricker model
    phi   : float -- the phi parameter of the Ricker model          
    """
    
    x = np.zeros(t_obs)
    N = 10
    for t in range(t_obs):
        x[t] = np.random.poisson(phi * N)
        N = r * N * np.exp(-N + np.random.normal(loc=0, scale=sigma))
    return x

@jit(nopython=True, cache=True, parallel=True)
def simulate_ricker_batch(X, params, n_batch, t_obs):
    """
    Simulates a batch of Ricker processes in parallel.
    """
    for i in prange(n_batch):
        X[i, :] = simulate_ricker_single(t_obs, params[i, 0], params[i, 1], params[i, 2])
    

def simulate_ricker(batch_size=64, t_obs=500, low_r=1, high_r=90, 
                    low_sigma=0.05, high_sigma=0.7, low_phi=0, high_phi=20, to_tensor=True):
    """
    Simulates and returns a batch of 1D timeseries obtained under the Ricker model.
    ----------

    Arguments:
    batch_size : int -- number of Ricker processes to simulate
    t_obs      : int -- length of the observed time-series
    low_r      : float -- lower bound for the uniform prior on r
    high_r     : float -- upper bound for the uniform prior on r
    low_sigma  : float -- lower bound for the uniform prior on sigma
    high_sigma : float -- upper bound for the uniform prior on sigma
    low_phi    : float -- lower bound for the uniform prior on phi
    high_phi   : float -- upper bound for the uniform prior on phi
    to_tensor  : bool  -- a flag indicating whether to return numpy arrays or tf tensors
    ----------

    Returns:
    (X, theta)  : (np.array of shape (batch_size, t_obs, 1), np.array of shape (batch_size, 3)) or
              (tf.Tensor of shape (batch_size, t_obs, 1), tf.Tensor of shape (batch_size, 3)) --
              a batch or time series generated under a batch of Ricker parameters
    """
    
    # Prepare placeholders
    theta = np.random.uniform(low=[low_r, low_sigma, low_phi], 
                          high=[high_r, high_sigma, high_phi], size=(batch_size, 3))
    X = np.zeros((batch_size, t_obs))

    # Simulate a batch from the Ricker model
    simulate_ricker_batch(X, theta, batch_size, t_obs)
    if to_tensor:
        return tf.convert_to_tensor(X[:, :, np.newaxis], dtype=tf.float32), tf.convert_to_tensor(theta, dtype=tf.float32)
    return X[:, :, np.newaxis]


@jit(nopython=True)
def deriv_sir(y, N, beta, gamma):
    """
    The SIR model differential equations.
    ----------

    Arguments:
    y     : tuple (S, I, R) - holding the different # of instances in each compartment
    N     : int -- the total # of instances in the population
    beta  : float -- beta parameter of the SIR model
    gamma : float --- gamma parameter of the SIR model 
    """
    
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

@jit
def simulate_sir_single(beta, gamma, N=1000, I0=1, R0=0, t_max=200):
    """Simulates a single SIR scenario by numerical intergatrion."""
    
    t = np.linspace(0, t_max, t_max)
    S0 = N - I0 - R0
    y0 = S0, I0, R0 
    ret = scipy.integrate.odeint(deriv_sir, y0, t, args=(N, beta, gamma))
    return ret

@jit(nopython=True, cache=True, parallel=True)
def simulate_sir_batch(X, params, n_batch, N=1000, I0=1, R0=0, t_max=200):
    """
    Simulates a batch of SIR timeseries in parallel.
    """

    for i in prange(n_batch):
        X[i] = simulate_sir_single(params[i, 0], params[i, 1], N, I0, R0, t_max)
        
def simulate_sir(batch_size=64, N=1000, low_beta=0.05, high_beta=2., 
                 low_gamma=0.01, I0=1, R0=0, t_max=200, to_tensor=True):
    """
    Simulates and returns a batch of timeseries obtained under the SIR model.
    ----------

    Arguments:
    batch_size : int -- number of Ricker processes to simulate
    N          : int -- the size of the population
    low_beta   : float -- lower bound for the uniform prior on beta
    high_beta  : float -- upper bound for the uniform prior on beta
    low_gamma  : float -- lower bound for the uniform prior on gamma
    I0         : int -- initial number of infected members
    R0         : int -- initial number of recovered members
    t_max      : int -- the length of the observed time-series
    to_tensor  : bool  -- a flag indicating whether to return numpy arrays or tf tensors
    ----------

    Returns:
    (X, theta)  : (np.array of shape (batch_size, t_obs, 1), np.array of shape (batch_size, 3)) or
              (tf.Tensor of shape (batch_size, t_obs, 1), tf.Tensor of shape (batch_size, 3)) --
              a batch or time series generated under a batch of Ricker parameters
    """
    
    # Draw from priors
    beta_samples = np.random.uniform(low=low_beta, high=high_beta, size=batch_size)
    gamma_samples = np.random.uniform(low=low_gamma, high=beta_samples)
    theta = np.c_[beta_samples, gamma_samples]

    # Simulate with drawn parameters
    X = np.zeros((batch_size, t_max, 3))
    simulate_sir_batch(X, theta, batch_size, N, I0, R0, t_max)

    if to_tensor:
        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(theta, dtype=tf.float32)
    return X, theta
