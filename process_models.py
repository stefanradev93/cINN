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
    phi   : float -- the phi (rho) parameter of the Ricker model          
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
    

def simulate_ricker(batch_size=64, n_points=None, t_obs_min=100, t_obs_max=500, low_r=1, high_r=90, 
                    low_sigma=0.05, high_sigma=0.7, low_phi=0, high_phi=15, to_tensor=True):
    """
    Simulates and returns a batch of 1D timeseries obtained under the Ricker model.
    ----------

    Arguments:
    batch_size : int -- number of Ricker processes to simulate
    n_points   : int -- length of the observed time-series
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
    
    # Sample t_obs, if None given
    if n_points is None:
        n_points = np.random.randint(low=t_obs_min, high=t_obs_max+1)

    # Prepare placeholders
    theta = np.random.uniform(low=[low_r, low_sigma, low_phi], 
                          high=[high_r, high_sigma, high_phi], size=(batch_size, 3))
    X = np.zeros((batch_size, n_points))

    # Simulate a batch from the Ricker model
    simulate_ricker_batch(X, theta, batch_size, n_points)
    if to_tensor:
        return tf.convert_to_tensor(X[:, :, np.newaxis], dtype=tf.float32), tf.convert_to_tensor(theta, dtype=tf.float32)
    return X[:, :, np.newaxis]


def sir(u, beta, gamma, t, N=1000, dt=0.1, iota=0.5):
    """
    Implements the stochastic SIR equations.
    """
    
    S, I, R = u
    lambd = beta *(I+iota)/N
    ifrac = 1.0 - np.exp(-lambd*dt)
    rfrac = 1.0 - np.exp(-gamma*dt)
    infection = np.random.binomial(S, ifrac)
    recovery = np.random.binomial(I, rfrac)
    return [S-infection, I+infection-recovery, R+recovery]


def simulate_sir_single(beta, gamma, t_max=500, N=1000):
    """Simulates a single SIR process."""

    tf = 200
    t = np.linspace(0, tf, t_max)
    
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    u = [N-1,1,0]
    S[0], I[0], R[0] = u
    for j in range(1, t_max):
        u = sir(u, beta, gamma, t[j], N)
        S[j],I[j],R[j] = u
    return np.array([S, I, R]).T

@jit
def simulate_sir(batch_size, n_points=None, low_beta=0.01, high_beta=1., low_gamma=0., 
                 t_min=200, t_max=500, N=1000, to_tensor=True):
    """
    Simulates and returns a batch of timeseries obtained under the SIR model.
    ----------

    Arguments:
    batch_size : int -- number of Ricker processes to simulate
    low_beta   : float -- lower bound for the uniform prior on beta
    high_beta  : float -- upper bound for the uniform prior on beta
    low_gamma  : float -- lower bound for the uniform prior on gamma
    t_max      : int -- the length of the observed time-series
    N          : int -- the size of the population
    to_tensor  : bool  -- a flag indicating whether to return numpy arrays or tf tensors
    ----------

    Returns:
    (X, theta)  : (np.array of shape (batch_size, t_obs, 1), np.array of shape (batch_size, 3)) or
              (tf.Tensor of shape (batch_size, t_obs, 1), tf.Tensor of shape (batch_size, 3)) --
              a batch or time series generated under a batch of Ricker parameters
    """

    # Select T
    if n_points is None:
        n_points = np.random.randint(low=t_min, high=t_max+1)
    
    # Prepare X and theta
    X = np.zeros((batch_size, n_points, 3))
    beta_samples = np.random.uniform(low=low_beta, high=high_beta, size=batch_size)
    gamma_samples = np.random.uniform(low=low_gamma, high=beta_samples)
    theta = np.c_[beta_samples, gamma_samples]
    
    # Run the SIR simulator for # batch_size
    for j in prange(batch_size):
        X[j] = simulate_sir_single(theta[j, 0], theta[j, 1], t_max=n_points, N=N)
    
    if to_tensor:
        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(theta, dtype=tf.float32)
    return X, theta