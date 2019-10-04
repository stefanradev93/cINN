import tensorflow as tf
import numpy as np
from losses import (maximum_likelihood_loss, kullback_leibler_gaussian, 
                    mean_squared_error, heteroscedastic_loss, kullback_leibler_iaf)
from sklearn.metrics import r2_score


def apply_gradients(optimizer, gradients, variables, global_step=None):
    """
    Performs one step of the backprop algorithm by updating each tensor in the 'variables' list.
    """
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


def train_online_ml(model, optimizer, data_generator, iterations, batch_size,
               p_bar=None, clip_value=None, global_step=None, transform=None, n_smooth=100):
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
    transform       : callable ot None -- a function to transform X and theta, if given
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
            X_batch, theta_batch = data_generator(batch_size)
            # Apply some transformation, if specified
            if transform:
                X_batch, theta_batch = transform(X_batch, theta_batch)

            # Sanity check for non-empty tensors
            if tf.equal(X_batch.shape[0], 0).numpy():
                print('Iteration produced empty tensor, skipping...')
                continue

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
        running_ml = ml_loss.numpy() if it < n_smooth else np.mean(losses['ml_loss'][-n_smooth:])
        p_bar.set_postfix_str("Iteration: {0},ML Loss: {1:.3f},Running ML Loss: {2:.3f},Regularization Loss: {3:.3f}"
        .format(it, ml_loss.numpy(), running_ml, decay.numpy()))
        p_bar.update(1)

    return losses


def train_online_kl(model, optimizer, data_generator, iterations, batch_size, beta, p_bar=None, 
                    clip_value=None, global_step=None, transform=None, beta_max=1.0, beta_step=250, beta_increment=0.01):
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
    transform       : callable ot None -- a function to transform X and theta, if given
    n_smooth        : int -- a value indicating how many values to use for computing the running ML loss
    ----------

    Returns:
    losses : dict -- a dictionary with the ml_loss and decay
    """
    
    # Prepare a dictionary to track losses
    losses = {
        'kl_loss': [],
        'rec_loss': [],
        'decay': []
    }
    # Run training loop
    for it in range(1, iterations+1):
        with tf.GradientTape() as tape:
            # Generate data and parameters
            X_batch, theta_batch = data_generator(batch_size)
            # Apply some transformation, if specified
            if transform:
                X_batch, theta_batch = transform(X_batch, theta_batch)

            # Sanity check for non-empty tensors
            if tf.equal(X_batch.shape[0], 0).numpy():
                print('Iteration produced empty tensor, skipping...')
                continue

            # Forward pass
            z_mean, z_logvar, theta_hat = model(theta_batch, X_batch)

            # Compute losses
            kl = kullback_leibler_gaussian(z_mean, z_logvar, beta)
            rec = mean_squared_error(theta_batch, theta_hat)
            decay = tf.add_n(model.losses)
            total_loss = kl + rec + decay 

        # Store losses
        losses['kl_loss'].append(kl.numpy())
        losses['rec_loss'].append(rec.numpy())
        losses['decay'].append(decay.numpy())

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Increase beta every beta_step iterations
        if (global_step.numpy() + 1) % beta_step == 0:
            tf.assign(beta, min(beta_max, beta.numpy() + beta_increment))

        # Update progress bar
        p_bar.set_postfix_str("Iteration: {0},Rec Loss: {1:.3f},KL Loss: {2:.3f},Regularization Loss: {3:.3f},Beta: {4:.2f}"
        .format(it, rec.numpy(), kl.numpy(), decay.numpy(), beta.numpy()))
        p_bar.update(1)

    return losses


def train_online_iaf(model, optimizer, data_generator, iterations, batch_size, beta, p_bar=None, 
                     clip_value=None, global_step=None, transform=None, beta_max=1.0,
                     beta_step=250, beta_increment=0.01):
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
    transform       : callable ot None -- a function to transform X and theta, if given
    n_smooth        : int -- a value indicating how many values to use for computing the running ML loss
    ----------

    Returns:
    losses : dict -- a dictionary with the ml_loss and decay
    """
    
    # Prepare a dictionary to track losses
    losses = {
        'kl_loss': [],
        'rec_loss': [],
        'decay': []
    }
    # Run training loop
    for it in range(1, iterations+1):
        with tf.GradientTape() as tape:
            # Generate data and parameters
            X_batch, theta_batch = data_generator(batch_size)
            # Apply some transformation, if specified
            if transform:
                X_batch, theta_batch = transform(X_batch, theta_batch)

            # Sanity check for non-empty tensors
            if tf.equal(X_batch.shape[0], 0).numpy():
                print('Iteration produced empty tensor, skipping...')
                continue

            # Forward pass
            theta_hat, z, logqz_x = model(theta_batch, X_batch)
            
            
            # Compute losses
            kl = kullback_leibler_iaf(z, logqz_x, beta)
            rec = mean_squared_error(theta_batch, theta_hat)
            decay = tf.add_n(model.losses)
            total_loss = kl + rec + decay 

        # Store losses
        losses['kl_loss'].append(kl.numpy())
        losses['rec_loss'].append(rec.numpy())
        losses['decay'].append(decay.numpy())

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Increase beta every beta_step iterations
        if (global_step.numpy() + 1) % beta_step == 0:
            tf.assign(beta, min(beta_max, beta.numpy() + beta_increment))

        # Update progress bar
        p_bar.set_postfix_str("Iteration: {0},Rec Loss: {1:.3f},KL Loss: {2:.3f},Regularization Loss: {3:.3f},Beta: {4:.2f}"
        .format(it, rec.numpy(), kl.numpy(), decay.numpy(), beta.numpy()))
        p_bar.update(1)

    return losses


def train_online_heteroscedastic(model, optimizer, data_generator, iterations, batch_size, 
                                 p_bar, transform=None, global_step=None, clip_value=5.):
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
    transform       : callable ot None -- a function to transform X and theta, if given
    ----------

    Returns:
    losses : dict -- a dictionary with the ml_loss and decay
    """
    
    # Prepare a dictionary to track losses
    losses = {
        'h_loss': [],
        'decay': []
    }
    
    for it in range(iterations):

        X_batch, theta_batch = data_generator(batch_size)
        
        # Apply some transformation, if specified
        if transform is not None:
            X_batch, theta_batch = transform(X_batch, theta_batch)
            
        # Sanity check for non-empty tensors
        if tf.equal(X_batch.shape[0], 0).numpy():
            print('Iteration produced empty tensor, skipping...')
            continue

        with tf.GradientTape() as tape:

            # Forward pass
            theta_mean, theta_var = model(X_batch)
            # Compute total loss
            loss = heteroscedastic_loss(theta_batch, theta_mean, theta_var)
            decay = tf.add_n(model.losses)
            total_loss = loss + decay
        
        # Store losses
        losses['h_loss'].append(loss.numpy())
        losses['decay'].append(decay.numpy())

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)

        # Compute reconstruction (rmse)
        rec = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(theta_batch - theta_mean), axis=-1), axis=-1))  
        
        if clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step=global_step)

        # Update p-bar
        p_bar.set_postfix_str("It: {0},Loss:{1:.3f},RMSE.:{2:.3f}".format(
        it, loss.numpy(), rec.numpy()))
        p_bar.update(1)
        
    return losses
    

def train_loop_dataset(model, optimizer, dataset, batch_size, p_bar=None, clip_value=None, 
                       global_step=None, transform=None, n_smooth=10):
    """
    Utility function to perform a single epoch over a given dataset.
    ---------

    Arguments:
    model           : tf.keras.Model -- the invertible chaoin with an optional summary net
                                        both models are jointly trained
    optimizer       : tf.train.optimizers.Optimizer -- the optimizer used for backprop
    dataset         : iterable -- tf.data.Dataset yielding (X_batch, y_batch) at each iteration
    batch_size      : int -- the batch_size used for training
    p_bar           : ProgressBar or None -- an instance for tracking the training progress
    clip_value      : float or None       -- the value used for clipping the gradients
    global_step     : tf.EagerVariavle or None -- a scalar tensor tracking the number of 
                                          steps and used for learning rate decay  
    transform       : callable ot None -- a function to transform X and theta, if given
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
        # Apply transform, if specified
        if transform is not None:
            X_batch, theta_batch = transform(X_batch, theta_batch)

        # Sanity check for non-empty tensors
        if tf.equal(X_batch.shape[0], 0).numpy():
            print('Iteration produced empty tensor, skipping...')
            continue

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
        p_bar.set_postfix_str("Batch: {0},ML Loss: {1:.3f},Running ML Loss: {2:.3f},Regularization Loss: {3:.3f}"
        .format(bi, ml_loss.numpy(), running_ml, decay.numpy()))
        p_bar.update(1)

    return losses


def compute_metrics(model, param_names, simulate_fun, n_test, n_samples_posterior, 
                    p_bar=None, transform=None, n_min=100, n_max=1000):
    """
    Plots a given metric for different numbers of datapoints.
    ---------

    Arguments:
    model           : tf.keras.Model -- the invertible chaoin with an optional summary net
                                        both models are jointly trained
    param_names     : list of strings -- the names of the parameters
    simulate_fun    : callable -- the simulate function
    n_test          : number of test datasets
    n_samples_posterior : number of samples from the approximate posterior
    transform       : callable ot None -- a function to transform X and theta, if given
    n_min           : int -- the minimum number of data points for each dataset
    n_max           : int -- the maximum number of data points for each dataset
    ----------

    Returns:
    ns      : np.array -- the array with time points
    metrics : dict -- a dictionary with the metrics
    """
    
    #Plot NRMSE for all t
    ns = np.arange(n_min, n_max+1)
    metrics = {
        'nrmse': {k: [] for k in param_names},
        'r2': {k: [] for k in param_names},
        'var': {k: [] for k in param_names}
    }
    # For each possible number of data points
    for n_points in ns:
        # Generate data
        X_test, theta_test = simulate_fun(n_test, n_points=n_points)
        if transform is not None:
            X_test, theta_test = transform(X_test, theta_test)
        theta_test = theta_test.numpy()

        # Sample from approx posterior and compute posterior means
        theta_approx = model.sample(X_test, n_samples_posterior, to_numpy=True)
        theta_approx_means = np.mean(theta_approx, axis=0)
        theta_approx_vars = np.var(theta_approx, axis=0, ddof=1)
        
        # --- Plot true vs estimated posterior means on a single row --- #
        for j, name in enumerate(param_names):

            # Compute NRMSE
            rmse = np.sqrt(np.mean( (theta_approx_means[:, j] - theta_test[:, j])**2 ))
            nrmse = rmse / (theta_test[:, j].max() - theta_test[:, j].min())
            # Compute R2
            r2 = r2_score(theta_test[:, j], theta_approx_means[:, j])
            # Compute posterior variance
            var = np.mean(theta_approx_vars[:, j])
            # Add to dict
            metrics['nrmse'][name].append(nrmse)
            metrics['r2'][name].append(r2)
            metrics['var'][name].append(var)
            
        if p_bar is not None:
            p_bar.update(1)
    return ns, metrics