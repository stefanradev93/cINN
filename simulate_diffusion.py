from numba import jit, prange
import ctypes
from numba.extending import get_cython_function_address
import tensorflow as tf
import numpy as np


# Get a pointer to the C function diffusion.c
addr_diffusion = get_cython_function_address("diffusion", "diffusion_trial")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, 
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_int)

diffusion_trial = functype(addr_diffusion)


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
        # For each condition
        for j in prange(x.shape[1]):
            # First condition
            x[i, j, 0] = diffusion_trial(params[i, 0], params[i, 2], params[i, 3], 
                                         params[i, 4], params[i, 5], params[i, 6], 
                                         params[i, 7], params[i, 8], 0.001, 5000)
            # Second condition
            x[i, j, 1] = diffusion_trial(params[i, 1], params[i, 2], params[i, 3], 
                                         params[i, 4], params[i, 5], params[i, 6], 
                                         params[i, 7], params[i, 8], 0.001, 5000)

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


def plot_diffusion_multiple(n=1000, figsize=(20, 20), filename='levy'):
    """Plots example datasets from the SIR model."""

    X, theta = simulate_diffusion(25, n_points=n, to_tensor=False)
    n = np.arange(1, n+1)

    f, axarr = plt.subplots(5, 5, figsize=figsize)

    for i, ax in enumerate(axarr.flat):

        # Plot just a single condition
        sns.distplot(X[i, :, 0], ax=ax, label='Upper threshold', rug=True, color="#4873b8")

        if i == 0:
            ax.set_xlabel(r'Number of trials ($n$)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=10)
        else:
            ax.get_legend().remove()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    f.tight_layout()

    if filename is not None:
        f.savefig("figures/{}_levy_multiple.png".format(filename), dpi=600, bbox_inches='tight')
