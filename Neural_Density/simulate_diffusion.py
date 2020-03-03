from numba import jit, prange
import ctypes
from numba.extending import get_cython_function_address
import numpy as np


# Get a pointer to the C function diffusion.c
addr_diffusion = get_cython_function_address("diffusion", "diffusion_trial")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_int)

diffusion_trial = functype(addr_diffusion)


@jit(nopython=True, cache=True, parallel=True)
def simulate_diffusion_p(x, params):
    """
    Simulate a batch from the diffusion model.
    ----------
    INPUT:
    x - np.array of shape (n_batch, n_trials, n_cond)
    params - np.array of shape (n_batch, n_params):
    param index 0 - variability in drift rate
    param index 1 - relative starting point
    param index 2 - variability in starting point
    param index 3 - threshold
    param index 4 - NDT
    param index 6 - variability in NDT
    param index 7 - alpha
    """

    # For each trial
    for j in prange(x.shape[0]):
        x[j, 0] = diffusion_trial(params[0], 0.0, params[2], 0.0,
                                  params[3], params[4], 0.0, params[5],
                                  0.001, 5000)
        x[j, 1] = diffusion_trial(params[1], 0.0, params[2], 0.0,
                                  params[3], params[4], 0.0, params[5],
                                  0.001, 5000)



def simulate_diffusion2c_p(params, n_points=None, n_cond=2, n_trials_min=100, n_trials_max=1000):
    """Simulates batch_size datasets from the full Ratcliff diffusion model."""


    # Sample number of trials, if None given
    if n_points is None:
        n_points = np.random.randint(n_trials_min, n_trials_max+1)
    # Simulate RTs
    X_batch = np.zeros((n_points, n_cond), dtype=np.float32)
    simulate_diffusion_p(X_batch, params)
    return X_batch



if __name__ == "__main__":

    params = np.array([1.1, -2.0, 0.4, 1.5, 0.2, 1.4])
    X_test = simulate_diffusion2c_p(params)
    print(X_test)
