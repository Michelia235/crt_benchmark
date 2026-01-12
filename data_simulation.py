import numpy as np
from scipy import ndimage
from scipy.linalg import cholesky, toeplitz


def generate_data(n, p, mean=0.0, snr=3.0, rho=0.25, gamma=0.1, smooth_X=0.0,
                  sparsity=0.06, effect=1.0, binary=False, noise_mag=1.0,
                  use_noise_magnitude=False, covariance='toeplitz',
                  decom_method='cholesky', fixed_non_null=False,
                  consecutive=False, verbose=False, seed=None):
    """Function to simulate data

    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of variables
    sparsity : float, optional
        ratio of number of variables with non-zero coefficients over total
        coefficients
    rho : float, optional
        correlation parameter
    effect : float, optional
        signal magnitude, value of non-null coefficients
    decom_method : string, optional
        Decomposition method for covariance matrix, mostly for speedup reason
        as cholesky is very fast compare to default svd choice of
        numpy.random.multivariate_normal default : cholesky.
    fixed_non_null: bool, optional
        if True, non-null index is generated with fixed seed
    seed : None or Int, optional
        random seed for generator

    Returns
    -------
    X : ndarray, shape (n, p)
        Design matrix resulted from simulation
    y : ndarray, shape (n, )
        Response vector resulted from simulation
    beta_true : ndarray, shape (n, )
       Vector of true coefficient value
    non_zero : ndarray, shape (n, )
        Vector of non zero coefficients index

    """
    # Generate mean and covariance matrix of a multivariate normal distribution
    mu = np.full(p, mean)

    beta_true, non_zero = _generate_true_signal(
        p, sparsity=sparsity, effect=effect, consecutive=consecutive,
        fixed_non_null=fixed_non_null, random_state=seed)
    Sigma = _generate_covariance(p, rho=rho, group_label=None,
                                 gamma=None, structure=covariance)

    # Set seed generator
    rng = np.random.RandomState(seed)
    if decom_method == 'svd':
        X = rng.multivariate_normal(mu, Sigma, size=(n))
    elif decom_method == 'cholesky':
        l = cholesky(Sigma)
        X = np.tile(mu, reps=(n, 1)) + np.dot(rng.randn(n, p), l)
    else:
        raise ValueError('Decomposition method not supported')

    # Gaussian smoothing for data matrix
    if isinstance(smooth_X, (float, int)):
        X = np.apply_along_axis(ndimage.filters.gaussian_filter, axis=1,
                                arr=X, **{'sigma': smooth_X})

    # Generate the response from a linear model
    prod_temp = np.dot(X, beta_true)
    if not use_noise_magnitude:
        # SNR formular follows Buhlmann & VdG (2011), pp.104
        noise_mag = np.linalg.norm(prod_temp, ord=2) / (np.sqrt(n) * snr)

    y = prod_temp + noise_mag * rng.normal(size=n)

    if binary:
        pr = _sigmoid(y)
        y = rng.binomial(1, pr)

    if verbose:
        return X, y, beta_true, non_zero, Sigma

    return X, y, beta_true, np.where(beta_true != 0.0)[0]


def _sigmoid(x):
    # Numerical stability sigmoid func
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _generate_true_signal(p, n_groups=None, sparsity=0.06, effect=1.0,
                          consecutive=False, fixed_non_null=False,
                          random_state=None):
    k = int(sparsity * p)
    rng = np.random.RandomState(random_state)

    if n_groups is None:
        if consecutive:
            non_zero = np.arange(k)
        else:
            if fixed_non_null:
                rng_non_zero = np.random.RandomState(0)
                non_zero = rng_non_zero.choice(p, k, replace=False)
            else:
                non_zero = rng.choice(p, k, replace=False)

        beta_true = np.zeros(p)
        beta_true[non_zero] = effect
        beta_true *= rng.choice([-1, 1], size=p)  # coin-flip sign

        return beta_true, non_zero

    # beta_true follows group structure
    if fixed_non_null:
        rng_non_zero = np.random.RandomState(0)
    else:
        rng_non_zero = np.random.RandomState(random_state)
    p_per_group = p // n_groups
    group_label = np.repeat(np.arange(n_groups), p_per_group)
    non_zero_group = rng_non_zero.choice(
        n_groups, size=round(n_groups * sparsity), replace=False)
    non_zero_group_sign = rng_non_zero.choice(
        [-1, 1], size=round(n_groups * sparsity))
    # all non-zero variables inside a group are equal
    beta_true = np.zeros(p)
    for i in range(len(non_zero_group)):
        beta_true[group_label == non_zero_group[i]] =\
            non_zero_group_sign[i] * effect

    return beta_true, non_zero_group, group_label


def _generate_covariance(p, rho, group_label, gamma=0.1, structure='toeplitz'):
    if structure == "toeplitz":
        # Toeplitz covariance matrix
        Sigma = toeplitz(rho ** np.arange(0, p))
    elif structure == "equi_corr":
        # Equicorrelation matrix
        Sigma = np.full((p, p), rho)
        np.fill_diagonal(Sigma, 1.0)
    else:
        raise ValueError('Wrong covariance type.')

    return Sigma
