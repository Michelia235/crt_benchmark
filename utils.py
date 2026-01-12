import numpy as np
from scipy.stats import norm


def fdr_threshold(pvals, fdr=0.1, method='bhq', reshaping_function=None):
    if method == 'bhq':
        return _bhq_threshold(pvals, fdr=fdr)
    elif method == 'bhy':
        return _bhy_threshold(
            pvals, fdr=fdr, reshaping_function=reshaping_function)
    else:
        raise ValueError(
            '{} is not support FDR control method'.format(method))


def cal_fdp_power(selected, non_zero_index, r_index=False):
    """ Calculate power and False Discovery Proportion

    Parameters
    ----------
    selected: list index (in R format) of selected non-null variables
    non_zero_index: true index of non-null variables
    r_index : True if the index is taken from rpy2 inference

    Returns
    -------
    fdp: (delta-)False Discoveries Proportion
    power: percentage of correctly selected variables over total number of
        non-null variables

    """
    # selected is the index list in R and will be different from index of
    # python by 1 unit

    if selected.size == 0:
        return 0.0, 0.0
    
    n_positives = len(non_zero_index)

    if r_index:
        selected = selected - 1

    true_positive = np.intersect1d(selected, non_zero_index)
    false_positive = np.setdiff1d(selected, true_positive)

    fdp = len(false_positive) / max(1, len(selected))
    power = min(len(true_positive), n_positives) / n_positives

    return fdp, power


def _bhq_threshold(pvals, fdr=0.1):
    """Standard Benjamini-Hochberg for controlling False discovery rate
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if pvals_sorted[i] <= fdr * (i + 1) / n_features:
            selected_index = i
            break
    if selected_index <= n_features:
        return pvals_sorted[selected_index]
    else:
        return -1.0


def _bhy_threshold(pvals, reshaping_function=None, fdr=0.1):
    """Benjamini-Hochberg-Yekutieli procedure for controlling FDR, with input
    shape function.
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    # Default value for reshaping function -- defined in
    # Benjamini & Yekutieli (2001)
    if reshaping_function is None:
        temp = np.arange(n_features)
        sum_inverse = np.sum(1 / (temp + 1))
        return _bhq_threshold(pvals, fdr / sum_inverse)
    else:
        for i in range(n_features - 1, -1, -1):
            if pvals_sorted[i] <= fdr * reshaping_function(i + 1) / n_features:
                selected_index = i
                break
        if selected_index <= n_features:
            return pvals_sorted[selected_index]
        else:
            return -1.0


def _logit(x):
    return np.exp(x) / (1 + np.exp(x))


def _lambda_max(X, y, use_noise_estimate=True):
    """Calculation of lambda_max, the smallest value of regularization parameter in
    lasso program for non-zero coefficient
    """
    n_samples, _ = X.shape

    if not use_noise_estimate:
        return np.max(np.dot(X.T, y)) / n_samples

    norm_y = np.linalg.norm(y, ord=2)
    sigma_0 = (norm_y / np.sqrt(n_samples)) * 1e-3
    sig_star = max(sigma_0, norm_y / np.sqrt(n_samples))

    return np.max(np.abs(np.dot(X.T, y)) / (n_samples * sig_star))


