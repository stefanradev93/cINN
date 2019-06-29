import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score



def plot_true_est_scatter(model, X_test, theta_test, n_samples, param_names, 
                          figsize=(20, 4), show=True, filename=None, font_size=12):
    """Plots a scatter plot with abline of the estimated posterior means vs true values."""


    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Convert true parameters to numpy
    theta_test = theta_test.numpy()

    # Determine figure layout
    if len(param_names) >= 6:
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
        axarr[j].text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes,
                     size=10)
        
        # Compute R2
        r2 = r2_score(theta_test[:, j], theta_approx_means[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=10)
        
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
    f.suptitle('Course of Loss')

    if show:
        plt.show()


def plot_metrics(metrics, ns, param_names, figsize=(12, 4), show=True, 
                 xlabel=r'$n$', filename=None, font_size=12):
    """
    Plots the nrmse and r2 for all parameters
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    
    # Initialize figure
    f, axarr = plt.subplots(1, 2, figsize=figsize)

    for i, metric in enumerate(['nrmse', 'r2']):
        for p in param_names:
            sns.lineplot(ns, metrics[metric][p], label=p, markers=True, dashes=False, ax=axarr[i])
            
        if metric == 'nrmse':
            axarr[i].set_ylabel('NRMSE')
        elif metric == 'r2':
            axarr[i].set_ylabel(r'$R^{2}$')
        axarr[i].set_xlabel(xlabel)
        
        axarr[i].spines['right'].set_visible(False)
        axarr[i].spines['top'].set_visible(False)
        axarr[i].legend(fontsize=12)
    
    f.tight_layout()
        
    if show:
        plt.show()
    
    if filename is not None:
        f.savefig("figures/{}_metrics.png".format(filename), dpi=600, bbox_inches='tight')


def plot_variance(variances, ns, param_names, figsize=(12, 4), show=True, 
                  xlabel=r'$n$', filename=None, tight=True, std=False, font_size=12):
    """
    Plots posterior variances of parameters as a function of the number of time points.
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Initialize figure
    # Determine figure layout
    if len(param_names) >= 6:
        n_col = int(np.ceil(len(param_names) / 2))
        n_row = 2
    else:
        n_col = int(len(param_names))
        n_row = 1

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat


    for i, p in enumerate(param_names):
        
        if std:
            sns.lineplot(ns, np.sqrt(variances[p]), markers=True, dashes=False, ax=axarr[i])
        else:
            sns.lineplot(ns, variances[p], markers=True, dashes=False, ax=axarr[i])
        
        if i == 0:
            axarr[i].set_ylabel('Posterior variance')
            axarr[i].set_xlabel(xlabel)
        axarr[i].set_title(p)
        axarr[i].spines['right'].set_visible(False)
        axarr[i].spines['top'].set_visible(False)

    if tight:
        f.tight_layout()
        
    if show:
        plt.show()
    
    if filename is not None:
        f.savefig("figures/{}_variance.png".format(filename), dpi=600, bbox_inches='tight')


def plot_true_est_posterior(model, n_samples, param_names, n_test=None, data_generator=None, 
                            X_test=None, theta_test=None, figsize=(15, 20), tight=True, 
                            show=True, filename=None, font_size=12):
    """
    Plots approximate posteriors.
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    
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
            axarr[i, j].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axarr[i, j].get_yaxis().set_ticks([])
            
            
            # Set title of first row
            if i == 0:
                axarr[i, j].set_title(param_names[j])       
            
            if i == 0 and j == 0:
                f.legend(loc='lower center', bbox_to_anchor=(0.5, -0.03), shadow=True, ncol=3, fontsize=10, borderaxespad=1)
                #axarr[i, j].legend(fontsize=10)
                
    if tight:
        f.tight_layout()
    f.subplots_adjust(bottom=0.12)
    # Show, if specified
    if show:
        plt.show()
    
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_density.png".format(filename, X_test.shape[1]), dpi=600, bbox_inches='tight')


def plot_sbc(model, n_samples, X_test, theta_test, param_names, 
            figsize=(15, 5), show=True, filename=None, font_size=12):
    """
    Plots the simulation-based posterior checking histograms as advocated by Talts et al. (2018).
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    
    # Prepare figure
    if len(param_names) >= 6:
        n_col = int(np.ceil(len(param_names) / 2))
        n_row = 2
    else:
        n_col = int(len(param_names))
        n_row = 1
    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # Convert theta test to numpy
    theta_test = theta_test.numpy()

    # Sample from approximate posterior
    theta_samples = model.sample(X_test, n_samples, to_numpy=True)

    # Compute ranks (using broadcasting)    
    ranks = np.sum(theta_samples < theta_test, axis=0)

    # Plot histograms
    for j in range(len(param_names)):
        sns.distplot(ranks[:, j], kde=False, ax=axarr[j], rug=True, hist_kws=dict(edgecolor="k", linewidth=1))
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
        if j == 0:
            axarr[j].set_xlabel('Rank statistic')
        axarr[j].get_yaxis().set_ticks([])

    f.tight_layout()
    
    # Show, if specified
    if show:
        plt.show()
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_sbc.png".format(filename, X_test.shape[1]), dpi=600)