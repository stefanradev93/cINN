import tensorflow as tf
import numpy as np
from losses import maximum_likelihood_loss

def apply_gradients(optimizer, gradients, variables, global_step=None):
    """
    Performs one step of the backprop algorithm by updating each tensor in the 'variables' list.
    """
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


def train_loop_active(model, optimizer, data_generator, iterations, batch_size, 
               p_bar=None, clip_value=None, global_step=None, n_smooth=100):
    """
    Utility function to perform the # number of training loops given by the itertations argument.
    ---------

    Arguments:
    model           : tf.keras.Model -- the invertible chaoin with an optional summary net
                                        both models are jointly trained
    optimizer       : tf.train.optimizers.Optimizer -- the optimizer used for backprop
    data_generator  : callable -- a function providing batches of X, theta (data, params)
    iterations      : int -- the number of training loops to perform
    batch_size      : int -- the batch_size used for training
    p_bar           : ProgressBar -- an instance for tracking the training progress
    clip_value      : float       -- the value used for clipping the gradients
    global_step     : tf.EagerVariavle -- a scalar tensor tracking the number of 
                                          steps and used for learning rate decay  
    n_smooth        : int -- a value indicating how many values to use for computing the running ML loss
    ----------

    Returns:
    losses : dict -- a dictionary with the ml_loss and decay
    """
    
    # Prepare a dictionary to track losses
    losses = {
        'ml_loss': [],
        'decay': []
    }
    # Run training loop
    for it in range(1, iterations+1):
        with tf.GradientTape() as tape:
            # Generate data and parameters
            X, theta = data_generator(batch_size)
            # Forward pass
            Z, log_det_J = model(theta, X)
            # Compute total_loss = ML Loss + Regularization loss
            ml_loss = maximum_likelihood_loss(Z, log_det_J)
            decay = tf.add_n(model.losses)
            total_loss = ml_loss + decay 

        # Store losses
        losses['ml_loss'].append(ml_loss.numpy())
        losses['decay'].append(decay.numpy())

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Update progress bar
        running_ml = ml_loss.numpy() if it < n_smooth else np.mean(losses['ml_loss'][-n_smooth:])
        p_bar.set_postfix_str("Iteration: {0},ML Loss: {1:.3f},Running ML Loss: {2:.3f},Regularization Loss: {3:3f}"
        .format(it, ml_loss.numpy(), running_ml, decay.numpy()))
        p_bar.update(1)

    return losses


def train_loop_dataset(model, optimizer, dataset, batch_size, p_bar=None, clip_value=None, 
                       global_step=None, n_smooth=10):
    """
    Utility function to perform a single epoch over a given dataset.
    ---------

    Arguments:
    model           : tf.keras.Model -- the invertible chaoin with an optional summary net
                                        both models are jointly trained
    optimizer       : tf.train.optimizers.Optimizer -- the optimizer used for backprop
    dataset         : iterable -- tf.data.Dataset yielding (X_batch, y_batch) at each iteration
    batch_size      : int -- the batch_size used for training
    p_bar           : ProgressBar -- an instance for tracking the training progress
    clip_value      : float       -- the value used for clipping the gradients
    global_step     : tf.EagerVariavle -- a scalar tensor tracking the number of 
                                          steps and used for learning rate decay  
    n_smooth        : int -- a value indicating how many values to use for computing the running ML loss
    ----------

    Returns:
    losses : dict -- a dictionary with the ml_loss and decay
    """
    
    # Prepare a dictionary to track losses
    losses = {
        'ml_loss': [],
        'decay': []
    }
    # Loop through data
    for bi, (X_batch, theta_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
            # Forward pass
            Z, log_det_J = model(theta_batch, X_batch)
            # Compute total_loss = ML Loss + Regularization loss
            ml_loss = maximum_likelihood_loss(Z, log_det_J)
            decay = tf.add_n(model.losses)
            total_loss = ml_loss + decay 

        # Store losses
        losses['ml_loss'].append(ml_loss.numpy())
        losses['decay'].append(decay.numpy())

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Update progress bar
        running_ml = ml_loss.numpy() if bi < n_smooth else np.mean(losses['ml_loss'][-n_smooth:])
        p_bar.set_postfix_str("Batch: {0},ML Loss: {1:.3f},Running ML Loss: {2:.3f},Regularization Loss: {3:3f}"
        .format(bi, ml_loss.numpy(), running_ml, decay.numpy()))
        p_bar.update(1)

    return losses