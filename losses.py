import tensorflow as tf
from functools import partial
import numpy as np


def maximum_likelihood_loss(z, log_det_J):
    """
    Computes the ML loss as described by Ardizzone et al. (in press).
    ----------
    Arguments:
    z : tf.Tensor of shape (batch_size, z_dim) -- the output of the final CC block f(x; c, W)
    J : tf.Tensor of shape (batch_size, )      -- the log determinant of the jacobian computed the CC block.

    Output:
    loss : tf.Tensor of shape (,)  -- a single scalar Monte-Carlo approximation of E[ ||z||^2 / 2 - log|det(J)| ]
    """

    return tf.reduce_mean(0.5 * tf.square(tf.norm(z, axis=-1)) - log_det_J)


def heteroscedastic_loss(y_true, y_mean, y_var):
    """
    Computes the heteroscedastic loss for regression.

    ----------
    Arguments:
    y_true : tf.Tensor of shape (batch_size, n_out_dim) -- the vector of true values
    y_mean : tf.Tensor of shape (batch_size, n_out_dim) -- the vector fo estimated conditional means
    y_var  : tf.Tensor of shape (batch_size, n_out_dim) -- the vector of estimated conditional variance
             (alleatoric uncertainty)
    ----------
    Returns:
    loss : tf.Tensor of shape (,) -- a single scalar value representing thr heteroscedastic loss

    """

    logvar = tf.reduce_sum(0.5 * tf.log(y_var), axis=-1)
    squared_error = tf.reduce_sum(0.5 * tf.square(y_true - y_mean) / y_var, axis=-1)
    loss = tf.reduce_mean(squared_error + logvar)
    return loss


def maximum_mean_discrepancy(source_samples, target_samples, weight=1., minimum=0.):
    """
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    ----------

    Arguments:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.

    Returns:
    a scalar tensor representing the MMD loss value.
    """

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=sigmas)
    loss_value = mmd_kernel(source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(minimum, loss_value) * weight
    return loss_value


def kullback_leibler_gaussian(z_mean, z_logvar, beta=1.):
    """
    Computes the KL divergence between a unit Gaussian and an arbitrary Gaussian.

    Arguments:
    z_mean   : tf.Tensor of shape (batch_size, z_dim) -- the means of the Gaussian which will be compared
    z_logvar : tf.Tensor of shape (batch_size, z_dim) -- the log vars of the Gaussian to be compared
    beta     : float -- the factor to weigh the KL divergence with

    Output:
    loss : tf.Tensor of shape (,)  -- a single scalar representing KL( N(z_mu, z_var | N(0, 1) )
    """
    
    loss = 1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
    loss = -0.5 * tf.reduce_sum(loss, axis=-1)
    return beta * tf.reduce_mean(loss)


def kullback_leibler_iaf(z, logqz_x, beta=1.):
    """
    Computes the KL loss for an iaf model.
    """
    
    logpz = -tf.reduce_sum(0.5 * np.log(2*np.pi) + 0.5 * tf.square(z), axis=-1)
    kl = beta * tf.reduce_mean(logqz_x - logpz)
    return kl


def mean_squared_error(y, y_hat):
    """
    Computes the mean squared error between two tensors.

    Arguments:
    z_mean   : tf.Tensor of shape (batch_size, theta_dim) -- the true values
    z_logvar : tf.Tensor of shape (batch_size, theta_dim) -- the predicted values

    Output:
    loss : tf.Tensor of shape (,)  -- the mean squared error
    """

    return tf.losses.mean_squared_error(y, y_hat)


def mean_squared_error(y, y_hat):
    """
    Computes the mean squared error between two tensors.

    Arguments:
    z_mean   : tf.Tensor of shape (batch_size, z_dim) -- the means of the Gaussian which will be compared
    z_logvar : tf.Tensor of shape (batch_size, z_dim) -- the log vars of the Gaussian to be compared

    Output:
    loss : tf.Tensor of shape (,)  -- the mean squared error
    """

    return tf.losses.mean_squared_error(y, y_hat)


def gaussian_kernel_matrix(x, y, sigmas):
    """
    Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
    Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """

    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    

def mmd_kernel(x, y, kernel=gaussian_kernel_matrix):
    """
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
    Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
    """

    loss = tf.reduce_mean(kernel(x, x))
    loss += tf.reduce_mean(kernel(y, y))
    loss -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    loss = tf.where(loss > 0, loss, 0)
    return loss
