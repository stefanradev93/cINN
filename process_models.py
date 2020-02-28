from numba import jit, prange
import ctypes
from numba.extending import get_cython_function_address
from scipy import integrate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf


# Get a pointer to the C function diffusion.c
try:
    addr_diffusion = get_cython_function_address("diffusion", "diffusion_trial")
    functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                ctypes.c_int)

    diffusion_trial = functype(addr_diffusion)
except ModuleNotFoundError:
    print('Warning: You need to compile the diffusion.pyx file via Cython in order to simulate from the LFM model.')
    pass


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

    # Ensure sigma and phi are positive
    phi = phi if phi >= 0 else 0
    sigma = sigma if sigma >= 0 else 0

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
    return X[:, :, np.newaxis], theta


def simulate_ricker_params(theta, n_points=500, to_tensor=True):
    """
    Simulates a batch of Ricker datasets given parameters.
    """


    theta = np.atleast_2d(theta)
    X = np.zeros((theta.shape[0], n_points))

    # Simulate a batch from the Ricker model
    simulate_ricker_batch(X, theta, theta.shape[0], n_points)

    X = X[:, :, np.newaxis]

    if to_tensor:
        return tf.convert_to_tensor(X, dtype=tf.float32)
    return X



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

                 t_min=200, t_max=500, N=1000, normalize=True, to_tensor=True):
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

    if normalize:
        X /= N
    if to_tensor:
        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(theta, dtype=tf.float32)
    return X, theta


@jit(nopython=True, cache=True, parallel=True)
def simulate_batch_diffusion_p(x, params):
    """
    Simulate a batch from the diffusion model.
    ----------
    INPUT:
    x - np.array of shape (n_batch, n_trials, n_cond)
    params - np.array of shape (n_batch, n_params):
        param index 0 - drift_rate1
        param index 1 - drift_rate2
        param index 2 - variability in drift rate
        param index 3 - relative starting point
        param index 4 - variability in starting point
        param index 5 - threshold
		param index 6 - NDT
		param index 7 - variability in NDT
        param index 8 - alpha
    """

    # For each batch
    for i in prange(x.shape[0]):
        # For each trial
        for j in prange(x.shape[1]):
            # First condition
            x[i, j, 0] = diffusion_trial(params[i, 0], params[i, 2], params[i, 3],
                                         params[i, 4], params[i, 5], params[i, 6],
                                         params[i, 7], params[i, 8], 0.001, 5000)
            # Second condition
            x[i, j, 1] = diffusion_trial(params[i, 1], params[i, 2], params[i, 3],
                                         params[i, 4], params[i, 5], params[i, 6],
                                         params[i, 7], params[i, 8], 0.001, 5000)


def lotka_volterra_forward(params, n_obs, T, x0, y0):
    """Performs one forward simulation from the LV model"""

    def dX_dt(X, t=0):
        """Return the growth rate of fox and rabbit populations."""
        return np.array([ a*X[0] -   b*X[0]*X[1], -c*X[1] + d*b*X[0]*X[1] ])

    t = np.linspace(0, T,  n_obs)
    X0 = np.array([10, 5])
    a, b, c, d = params

    # a - pray birth rate
    # b - predation rate
    # c - predator death rate
    # d - predator birth rate
    X = integrate.odeint(dX_dt, X0, t)

    # Clip inf (divergent sims)
    X[np.isneginf(X)] = 0
    X[np.isposinf(X)] = -1
    return X


def simulate_lotka_volterra(batch_size, p_lower=-2, p_upper=2, n_points=None, x0=10, y0=5,
                            T=15, to_tensor=True, n_min=200, n_max=1000, summary=False):

    """
    Simulates batch_size datasets from the LV model. Code inspired by:
    https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html
    """

    # Sample number of trials, if None given
    if n_points is None:
        n_points = np.random.randint(n_min, n_max+1)

    theta_batch = np.exp(np.random.uniform(low=p_lower, high=p_upper, size=(batch_size, 4)))
    X_batch = np.zeros((batch_size, n_points, 2))

    for j in range(batch_size):
        X_batch[j] = lotka_volterra_forward(theta_batch[j], n_points, T, x0, y0)

    # Clip large and small values
    X_batch = np.clip(X_batch, 0., 100000.)
    X_batch[np.isneginf(X_batch)] = -1.
    X_batch[np.isposinf(X_batch)] = 100000.
    X_batch[np.isnan(X_batch)] = -1.

    # Compute summaries, if given
    if summary:
        x = X_batch
        lag1 = int(0.2 * (n_points / T))
        lag2 = int(0.4 * (n_points / T))
        # Means
        x_means = np.mean(x, axis=1)
        # Logvars
        x_logvars = np.log1p(np.var(x, axis=1))
        # Autocorrelations at lag 0.2 and 0.4 time units
        x_auto = np.array([np.corrcoef(np.c_[x[i][:-lag1], x[i][lag1:]], rowvar=False) for i in range(x.shape[0])])
        x_auto11_1 = x_auto[:, 0, 2]
        x_auto12_2 = x_auto[:, 1, 3]
        x_auto = np.array([np.corrcoef(np.c_[x[i][:-lag2], x[i][lag2:]], rowvar=False) for i in range(x.shape[0])])
        x_auto21_1 = x_auto[:, 0, 2]
        x_auto22_2 = x_auto[:, 1, 3]
        # Cross-correlation
        x[:, :, 0] = (x[:, :, 0] - np.mean(x[:, :, 0], axis=1)[:, np.newaxis]) / (np.std(x[:, :, 0], axis=1)[:, np.newaxis] * x.shape[1])
        x[:, :, 1] = (x[:, :, 1] - np.mean(x[:, :, 1], axis=1)[:, np.newaxis]) / (np.std(x[:, :, 1], axis=1)[:, np.newaxis])
        x_cross = np.array([np.correlate(x[i, :, 0] , x[i, :, 1]) for i in range(x.shape[0])])
        X_batch = np.c_[x_means, x_logvars, x_auto11_1, x_auto12_2, x_auto21_1, x_auto22_2, x_cross]

    if to_tensor:
        X_batch, theta_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32), tf.convert_to_tensor(theta_batch, dtype=tf.float32)

    return X_batch, theta_batch

def simulate_lv_params(theta, n_points=500, x0=10, y0=5, T=15, 
                       to_tensor=True, n_min=200, n_max=1000, summary=False):
    """
    Simulates a batch of Ricker datasets given parameters.
    """

    theta = np.atleast_2d(theta)
    X = np.zeros((theta.shape[0], n_points, 2))

    for j in range(theta.shape[0]):
        X[j] = lotka_volterra_forward(theta[j], n_points, T, x0, y0)

    if to_tensor:
        return tf.convert_to_tensor(X, dtype=tf.float32)
    return X

def simulate_diffusion(batch_size, pbounds, n_points=None, n_cond=2,
                       to_tensor=True, cond_coding=False, n_trials_min=100, n_trials_max=1000):
    """Simulates batch_size datasets from the full Ratcliff diffusion model."""

    # Get number of parameters
    n_params = len(pbounds)

    # Sample number of trials, if None given
    if n_points is None:
        n_points = np.random.randint(n_trials_min, n_trials_max+1)

    # Extract parameter bounds
    lower_bounds = [pbounds['v1'][0], pbounds['v2'][0], pbounds['sv'][0], pbounds['zr'][0],
                    pbounds['szr'][0], pbounds['a'][0], pbounds['ndt'][0], pbounds['sndt'][0],
                    pbounds['alpha'][0]]

    upper_bounds = [pbounds['v1'][1], pbounds['v2'][1], pbounds['sv'][1], pbounds['zr'][1],
                    pbounds['szr'][1], pbounds['a'][1], pbounds['ndt'][1], pbounds['sndt'][1],
                    pbounds['alpha'][1]]

    # Draw from priors
    theta_batch  = np.random.uniform(low=lower_bounds, high=upper_bounds,  size=(batch_size, n_params)).astype(np.float32)
    X_batch = np.zeros((batch_size, n_points, n_cond), dtype=np.float32)
    simulate_batch_diffusion_p(X_batch, theta_batch)

    # Return in specified format (condition coding or just stack, tf.Tensor or np.array)
    if cond_coding:
        X_batch = np.stack(([np.c_[X_batch[:, :, 0], X_batch[:, :, 1]],
                             np.c_[np.zeros((batch_size, n_points)), np.ones((batch_size, n_points))]]), axis=-1)
    if to_tensor:
        X_batch, theta_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32), tf.convert_to_tensor(theta_batch, dtype=tf.float32)
    return X_batch, theta_batch

def simulate_diffusion_params(params, n_cond=2, to_tensor=True, n_points=None, n_trials_min=100, n_trials_max=1000):
    """Simulates batch_size datasets from the full Ratcliff diffusion model."""

    # Sample number of trials, if None given
    if n_points is None:
        n_points = np.random.randint(n_trials_min, n_trials_max+1)

    X_batch = np.zeros((1, n_points, n_cond), dtype=np.float32)
    simulate_batch_diffusion_p(X_batch, params)

    # Return in specified format (condition coding or just stack, tf.Tensor or np.array)
    if to_tensor:
        X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
    return X_batch


def plot_sir(beta, gamma, n_points=500, figsize=(8, 4), N=1000, filename=None):
    """
    Simulates a single SIR process.
    """

    # Generate SIR dataframes
    X = simulate_sir_single(beta, gamma, t_max=n_points, N=N)
    t = np.arange(1, n_points+1)

    # Prepare figure
    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(t, X[:, 0], ax=ax, label='Susceptible')
    sns.lineplot(t, X[:, 1], ax=ax, label='Infected')
    sns.lineplot(t, X[:, 2], ax=ax, label='Recovered')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'Number of time points ($T$)', fontsize=12)
    ax.set_ylabel('Number of individuals', fontsize=12)
    ax.legend(fontsize=10)
    f.tight_layout()

    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_plot_multiple.png".format(filename), dpi=600)


def plot_ricker_multiple(T=500, figsize=(8, 3), filename='Ricker'):
    """Plots example datasets from the Ricker model."""

    X, theta = simulate_ricker(10, n_points=T, to_tensor=False)
    t = np.arange(1, T+1)

    f, axarr = plt.subplots(2, 5, figsize=figsize)

    for i, ax in enumerate(axarr.flat):


        sns.lineplot(t, X[i, :, 0], ax=ax)
        if i == 0:
            ax.set_xlabel(r"Generation number $t$", fontsize=10)
            ax.set_ylabel("Number of individuals", fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    f.tight_layout()

    if filename is not None:
        f.savefig("figures/{}_plot_multiple.png".format(filename), dpi=300, bbox_inches='tight')


def plot_sir_multiple(T=500, figsize=(10, 3), filename='SIR'):
    """Plots example datasets from the SIR model."""

    X, theta = simulate_sir(10, n_points=T, to_tensor=False)
    t = np.arange(1, T+1)

    f, axarr = plt.subplots(2, 5, figsize=figsize)

    for i, ax in enumerate(axarr.flat):


        sns.lineplot(t, X[i, :, 0], ax=ax, label='Susceptible')
        sns.lineplot(t, X[i, :, 1], ax=ax, label='Infected')
        sns.lineplot(t, X[i, :, 2], ax=ax, label='Recovered')

        if i == 0:
            ax.set_xlabel(r'Number of time points ($T$)', fontsize=10)
            ax.set_ylabel('Number of individuals', fontsize=10)
            f.legend(loc='lower center', bbox_to_anchor=(0.5, -0.02), shadow=True, ncol=3, fontsize=6, borderaxespad=1)
        ax.get_legend().remove()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    f.tight_layout()

    if filename is not None:
        f.savefig("figures/{}_plot_multiple.png".format(filename), dpi=300, bbox_inches='tight')


def plot_diffusion_multiple(n=1000, figsize=(10, 3), filename='levy'):
    """Plots example datasets from the SIR model."""

    parameter_bounds = {
        'v1': [0.0, 6.0],
        'v2': [-6.0, 0.0],
        'sv': [0.0, 0.0],
        'zr': [0.3, 0.7],
        'szr': [0.0, 0.0],
        'a': [0.6, 3.0],
        'ndt': [0.3, 1.0],
        'sndt': [0.0, 0.0],
        'alpha': [1.0, 2.0],
    }

    X, theta = simulate_diffusion(10, parameter_bounds, n_points=n, to_tensor=False)
    n = np.arange(1, n+1)

    f, axarr = plt.subplots(2, 5, figsize=figsize)

    for i, ax in enumerate(axarr.flat):

        # Plot just a single condition
        upper = X[i, :, 0][X[i, :, 0] >= 0]
        lower = X[i, :, 0][X[i, :, 0] < 0]
        sns.distplot(upper, ax=ax, label='Upper threshold', rug=True, color="#4873b8")
        sns.distplot(lower, ax=ax, label='Lower threshold', rug=True, color="#b8485e")

        if i == 0:
            ax.set_xlabel(r'Number of trials ($n$)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            f.legend(loc='lower center', bbox_to_anchor=(0.5, -0.02), shadow=True, ncol=2, fontsize=6, borderaxespad=1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    f.tight_layout()

    if filename is not None:
        f.savefig("figures/{}_multiple.png".format(filename), dpi=300, bbox_inches='tight')


def load_test_ricker(to_tensor=True):
    """
    A utility for loading the Ricker test data.
    """

    X_test = np.load('sim_data/ricker_test500/ricker_X.npy')
    theta_test = np.load('sim_data/ricker_test500/ricker_theta.npy')

    if to_tensor:
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        theta_test = tf.convert_to_tensor(theta_test, dtype=tf.float32)

    return X_test, theta_test
