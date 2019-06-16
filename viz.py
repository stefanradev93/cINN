import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score

# Plot settings
plt.rcParams['font.size'] = 14


def plot_true_est_scatter(model, X_test, theta_test, n_samples, 
                          param_names, figsize=(20, 4), show=True, filename=None):
    """Plots a scatter plot with abline of the estimated posterior means vs true values."""

    # Convert true parameters to numpy
    theta_test = theta_test.numpy()

    # Determine figure layout
    if len(param_names) > 6:
        n_col = int(np.ceil(len(param_names) / 2))
        n_row = 2
    else:
        n_col = int(len(param_names))
        n_row = 1

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat
    # Initialize posterior means matrix
    theta_approx_means = model.sample(X_test, n_samples, to_numpy=True).mean(axis=0)
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        
        # Plot analytic vs estimated
        axarr[j].scatter(theta_approx_means[:, j], theta_test[:, j], color='black', alpha=0.4)
        
        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')
        
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (theta_approx_means[:, j] - theta_test[:, j])**2 ))
        nrmse = rmse / (theta_test[:, j].max() - theta_test[:, j].min())
        axarr[j].text(0.2, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axarr[j].transAxes)
        
        # Compute R2
        r2 = r2_score(theta_test[:, j], theta_approx_means[:, j])
        axarr[j].text(0.2, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axarr[j].transAxes)
        
        if j == 0:
            # Label plot
            axarr[j].set_xlabel('Estimated')
            axarr[j].set_ylabel('True')
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
    
    # Adjust spaces
    f.tight_layout()

    if show:
        plt.show()
    
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_scatter.png".format(filename, X_test.shape[1]), dpi=600)


def plot_losses(losses, figsize=(15, 5), show=True):
    """
    Plots ML loss and decay for a given training session.
    ----------

    Arguments:
    losses  : dict -- a dictionary with keys 'ml_loss' and 'decay' containing the portions of the loss.
    figsize : tuple -- the size of the figure to create 
    show    : bool -- a flag indicating whether to call plt.show() or not
    """
    
    f, axarr = plt.subplots(1, 2, figsize=figsize)
    axarr[0].plot(losses['ml_loss'])
    axarr[1].plot(losses['decay'])
    axarr[0].set_title('ML Loss')
    axarr[1].set_title('Decay')
    f.set_title('Course of Loss')

    if show:
        plt.show()


def plot_true_est_posterior(model, n_samples, param_names, n_test=None, data_generator=None, 
                            X_test=None, theta_test=None, figsize=(15, 20), show=True, filename=None):
    """
    Plots approximate posteriors.
    """
    
    if data_generator is None and n_test is None:
        n_test = int(X_test.shape[0])
    elif X_test is None and theta_test is None:
        X_test, theta_test = data_generator(n_test)
    else:
        raise ValueError('Either data_generator and n_test or X_test and y_test should be provided')

    # Convert theta to numpy
    theta_test = theta_test.numpy()

    # Initialize f
    f, axarr = plt.subplots(n_test, len(param_names), figsize=figsize)

    theta_samples = model.sample(X_test, n_samples, to_numpy=True)
    theta_samples_means = theta_samples.mean(axis=0)
    
    # For each row 
    for i in range(n_test):
        
        for j in range(len(param_names)):
            
            # Plot approximate posterior
            sns.distplot(theta_samples[:, i, j], kde=True, hist=True, ax=axarr[i, j], 
                            label='Estimated posterior', color='#5c92e8')
            
            # Plot lines for approximate mean, analytic mean and true data-generating value
            axarr[i, j].axvline(theta_samples_means[i, j], color='#5c92e8', label='Estimated mean')
            axarr[i, j].axvline(theta_test[i, j], color='#e55e5e', label='True')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            
            # Set title of first row
            if i == 0:
                axarr[i, j].set_title(param_names[j])
            
            if i == 0 and j == 0:
                axarr[i, j].legend(fontsize=10)
            
    f.tight_layout()

    # Show, if specified
    if show:
        plt.show()
    
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_density.png".format(filename, X_test.shape[1]), dpi=600)