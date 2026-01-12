import numpy as np
from scipy import stats
from sklearn.covariance import (GraphicalLassoCV, empirical_covariance,
                                ledoit_wolf)
from sklearn.linear_model import (Lasso, LassoCV, LassoLarsCV,
                                  LogisticRegressionCV)
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.preprocessing import StandardScaler


def stat_coefdiff_lasso(X, X_tilde, y, loss='least_square', cv=5,
                         solver='liblinear', n_regu=9, return_coef=False):
    """Calculate test statistic by doing Lasso regression with Cross-validation
    on concatenated design matrix [X X_tilde] to find coefficients
    [beta beta_tilda]. The test statistic is then:

                        W_j =  abs(beta_j) - abs(beta_tilda_j)

    with j = 1, ..., n_features

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        Original design matrix

    X_tilde : 2D ndarray (n_samples, n_features)
        Knockoff design matrix

    y : 1D ndarray (n_samples, )
        Response vector

    loss : str, optional
        if the response vector is continuous, the loss used should be
        'least_square', otherwise
        if the response vector is binary, it should be 'logistic'

    cv : int, optional
        number of cross-validation folds

    solver : str, optional
        solver used by sklearn function LogisticRegressionCV

    n_regu : int, optional
        number of regulation used in the regression problem

    return_coef : bool, optional
        return regression coefficient if set to True

    Returns
    -------
    test_score : 1D ndarray (n_features, )
        vector of test statistic
    """
    if (solver == 'saga') or (loss == 'least_square'):
        n_jobs = 2
    else:
        n_jobs = 1

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])

    if loss == 'least_square':
        clf = LassoCV(n_jobs=n_jobs,
                      max_iter=1000000,
                      cv=cv)
        clf.fit(X_ko, y)
        coef = clf.coef_

    elif loss == 'logistic':
        clf = LogisticRegressionCV(
            penalty='l1', max_iter=1000000,
            solver=solver, cv=cv, n_jobs=n_jobs, tol=1e-8)
        clf.fit(X_ko, y)
        coef = clf.coef_[0]

    else:
        raise ValueError("'loss' must be either 'least_square' or 'logistic'")

    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    if return_coef:
        return test_score, coef
    else:
        return test_score


def model_x_knockoff(X, y, fdr=0.1, offset=1, loss='least_square',
                     method='equi', statistics=stat_coefdiff_lasso,
                     shrink=False, centered=True,
                     cov_estimator='ledoit_wolf', seed=None):
    """Model-X Knockoff inference procedure to control False Discoveries Rate,
    based on Candes et. al. (2018):
    Panning for Gold: Model-X Knockoffs for High-dimensional Controlled Variable Selection
    https://arxiv.org/abs/1610.02351
    
    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        design matrix

    y : 1D ndarray (n_samples, )
        response vector

    fdr : float, optional
        desired controlled FDR level

    offset : int, 0 or 1, optional
        offset to calculate knockoff threshold, offset = 1 is equivalent to
        knockoff+

    loss : str, optional
        if the response vector is continuous, the loss used should be
        'least_square', otherwise
        if the response vector is binary, it should be 'logistic'

    method : str, optional
        knockoff construction methods, at the moment only equi-correlated
        method is available

    statistics : Python function, optional
        method to calculate test score, can be defined manually

    shrink : bool, optional
        whether to shrink the empirical covariance matrix

    centered : bool, optional
        whether to standardize the data before doing the inference procedure

    cov_estimator : str, optional
        method of empirical covariance matrix estimation

    seed : int or None, optional
        random seed used to generate Gaussian knockoff variable

    Returns
    -------
    selected : 1D array, int
        vector of index of selected variables

    test_score : 1D array, (n_features, )
        vector of test statistic

    thres : float
        knockoff threshold

    X_tilde : 2D array, (n_samples, n_features)
        knockoff design matrix
    """

    if centered:
        X = StandardScaler().fit_transform(X)
        if loss == 'least_square':
            y = (y - np.mean(y)) / np.std(y)

    mu, Sigma = _estimate_distribution(
        X, shrink=shrink, cov_estimator=cov_estimator)

    if method == 'equi':
        X_tilde = _knockoff_equi_generated(X, mu, Sigma, seed=seed)
    else:
        raise ValueError("'{}' is not a valid method of knockoff construction."
                         .format(method))

    test_score = statistics(X, X_tilde, y, loss=loss)
    thres = knockoff_threshold(test_score, fdr=fdr, offset=offset)
    selected = np.where(test_score >= thres)[0]

    return selected, test_score, thres, X_tilde


def desparsified_lasso(X, y, centered=True, tol=1e-4, method="lasso", c=0.01,
                       n_jobs=-1):
    """Desparsified Lasso with pvalues; follow van de Geer et al. (2014)
    and Zhang and Zhang (2012)

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        confidence : float, optional
            Confidence level used to compute the confidence intervals.
            Each value should be in the range [0, 1].
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        method : string, optional
            The method for the nodewise lasso: "lasso", "lasso_cv" or
            "zhang_zhang"
        c : float, optional
            Only used if method="lasso". Then alpha = c * alpha_max.
        """

    X = np.asarray(X)
    
    if centered:
        X = StandardScaler().fit_transform(X)

    n_samples, n_features = X.shape

    Z = np.zeros((n_samples, n_features))
    omega_diag = np.zeros(n_features)

    if method == "lasso":

        Gram = np.dot(X.T, X)

        k = c * (1. / n_samples)
        alpha = k * np.max(np.abs(Gram - np.diag(np.diag(Gram))), axis=0)

    elif method == "lasso_cv":

        clf_lasso_loc = LassoCV(tol=tol, n_jobs=n_jobs)

    # Calculating Omega Matrix
    for i in range(n_features):

        if method == "lasso":

            Gram_loc = np.delete(np.delete(Gram, obj=i, axis=0), obj=i, axis=1)
            clf_lasso_loc = Lasso(alpha=alpha[i], precompute=Gram_loc, tol=tol)

        if method == "lasso" or method == "lasso_cv":

            X_new = np.delete(X, i, axis=1)
            clf_lasso_loc.fit(X_new, X[:, i])

            Z[:, i] = X[:, i] - clf_lasso_loc.predict(X_new)

        elif method == "zhang_zhang":

            print("i = ", i)
            X_new = np.delete(X, i, axis=1)
            alpha, z, eta, tau = _lpde_regularizer(X_new, X[:, i])

            Z[:, i] = z

        omega_diag[i] = (n_samples * np.sum(Z[:, i] ** 2) /
                         np.sum(Z[:, i] * X[:, i]) ** 2)

    # Lasso regression
    clf_lasso_cv = LassoCV(n_jobs=n_jobs)
    clf_lasso_cv.fit(X, y)
    beta_lasso = clf_lasso_cv.coef_

    # Estimating the coefficient vector
    beta_bias = y.T.dot(Z) / np.sum(X * Z, axis=0)

    P = ((Z.T.dot(X)).T / np.sum(X * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))

    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    sigma_hat = _reid(X, y)

    zscore = np.sqrt(n_samples) * beta_hat / (sigma_hat * np.sqrt(omega_diag))
    pval = 2 * stats.norm.sf(np.abs(zscore))

    return beta_hat, zscore, pval


def knockoff_threshold(test_score, fdr=0.1, offset=1):
    """Calculate the knockoff threshold based on the procedure stated in the
    article.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        vector of test statistic

    fdr : float, optional
        desired controlled FDR level

    offset : int, 0 or 1, optional
        offset equals 1 is the knockoff+ procedure

    Returns
    -------
    thres : float or np.inf
        threshold level
    """
    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    thres = np.inf
    t_mesh = np.sort(np.abs(test_score[test_score != 0]))
    for i in range(t_mesh.shape[0]):
        false_pos = np.sum(test_score <= -t_mesh[i])
        selected = np.sum(test_score >= t_mesh[i])
        if selected == 0:
            continue
        elif (offset + false_pos) / selected <= fdr:
            thres = t_mesh[i]
            break
    return thres


def _lpde_regularizer(X, y, grid=100, alpha_max=None, kappa_0=0.25,
                      kappa_1=0.5, c_max=0.99, eps=1e-3):

    X = np.asarray(X)
    n_samples, n_features = X.shape

    eta_star = np.sqrt(2 * np.log(n_features))

    z_grid = np.zeros(grid * n_samples).reshape(grid, n_samples)
    eta_grid = np.zeros(grid)
    tau_grid = np.zeros(grid)

    if alpha_max is None:
        alpha_max = np.max(np.dot(X.T, y)) / n_samples

    alpha_0 = eps * c_max * alpha_max
    z_grid[0, :], eta_grid[0], tau_grid[0] = \
        _lpde_regularizer_substep(X, y, alpha_0)

    if eta_grid[0] > eta_star:
        eta_star = (1 + kappa_1) * eta_grid[0]

    alpha_1 = c_max * alpha_max
    z_grid[-1, :], eta_grid[-1], tau_grid[-1] = \
        _lpde_regularizer_substep(X, y, alpha_1)

    alpha_grid = _alpha_grid(X, y, eps=eps, n_alphas=grid)[::-1]
    alpha_grid[0] = alpha_0
    alpha_grid[-1] = alpha_1

    for i, alpha in enumerate(alpha_grid[1:-1], 1):
        z_grid[:, i], eta_grid[i], tau_grid[i] = \
            _lpde_regularizer_substep(X, y, alpha)

    # tol_factor must be inferior to (1 - 1 / (1 + kappa_1)) = 1 / 3 (default)
    index_1 = (grid - 1) - (eta_grid <= eta_star)[-1].argmax()

    tau_star = (1 + kappa_0) * tau_grid[index_1]

    index_2 = (tau_grid <= tau_star).argmax()

    return (alpha_grid[index_2], z_grid[:, index_2], eta_grid[index_2],
            tau_grid[index_2])


def _lpde_regularizer_substep(X, y, alpha):

    clf_lasso = Lasso(alpha=alpha)
    clf_lasso.fit(X, y)

    z = y - clf_lasso.predict(X)
    z_norm = np.linalg.norm(z)
    eta = np.max(np.dot(X.T, z)) / z_norm
    tau = z_norm / np.sum(y * z)

    return z, eta, tau


def _reid(X, y, method="lars", tol=1e-6, max_iter=1e+3):
    """Estimation of noise standard deviation using Reid procedure

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        method : string, optional
            The method for the CV-lasso: "lars" or "lasso"
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    if int(max_iter / 5) <= n_features:
        max_iter = n_features * 5

    if method == "lars":
        clf_lars_cv = LassoLarsCV(max_iter=max_iter, cv=3)
        clf_lars_cv.fit(X, y)
        error = clf_lars_cv.predict(X) - y
        support = sum(clf_lars_cv.coef_ != 0)

    elif method == "lasso":
        clf_lasso_cv = LassoCV(tol=tol, max_iter=max_iter, cv=3)
        clf_lasso_cv.fit(X, y)
        error = clf_lasso_cv.predict(X) - y
        support = sum(clf_lasso_cv.coef_ != 0)

    sigma_hat = np.sqrt((1. / (n_samples - support))
                        * np.linalg.norm(error) ** 2)

    return sigma_hat


def _is_posdef(X, tol=1e-14):
    """Check a matrix is positive definite by calculating eigenvalue of the
    matrix

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples x n_features)
        Matrix to check

    tol : float, optional
        minimum threshold for eigenvalue

    Returns
    -------
    True or False
    """
    eig_value = np.linalg.eigvalsh(X)
    return np.all(eig_value > tol)


def _cov_to_corr(Sigma):
    """Convert covariance matrix to correlation matrix

    Parameters
    ----------
    Sigma : 2D ndarray (n_features, n_features)
        Covariance matrix

    Returns
    -------
    Corr_matrix : 2D ndarray (n_features, n_features)
        Transformed correlation matrix
    """

    features_std = np.sqrt(np.diag(Sigma))
    Scale = np.outer(features_std, features_std)

    Corr_matrix = Sigma / Scale

    return Corr_matrix


def _estimate_distribution(X, shrink=False, cov_estimator='ledoit_wolf'):

    alphas = [1e-3, 1e-2, 1e-1, 1]

    mu = X.mean(axis=0)
    Sigma = empirical_covariance(X)

    if shrink or not _is_posdef(Sigma):

        if cov_estimator == 'ledoit_wolf':
            Sigma_shrink = ledoit_wolf(X, assume_centered=True)[0]

        elif cov_estimator == 'graph_lasso':
            model = GraphicalLassoCV(alphas=alphas)
            Sigma_shrink = model.fit(X).covariance_

        else:
            raise ValueError('{} is not a valid covariance estimated method'
                             .format(cov_estimator))

        return mu, Sigma_shrink

    return mu, Sigma


def _s_equi(Sigma):
    """Estimate diagonal matrix of correlation between real and knockoff
    variables

    Parameters
    ----------
    Sigma : 2D ndarray (n_features, n_features)
        empirical covariance matrix calculated from original design matrix

    Returns
    -------
    1D ndarray (n_features, )
        vector of diagonal values of estimated matrix diag{s}
    """
    n_features = Sigma.shape[0]

    G = _cov_to_corr(Sigma)
    eig_value = np.linalg.eigvalsh(G)
    lambda_min = np.min(eig_value[0])
    S = np.ones(n_features) * min(2 * lambda_min, 1)

    psd = False
    s_eps = 0

    while psd is False:
        # if all eigval > 0 then the matrix is psd
        psd = _is_posdef(2 * G - np.diag(S * (1 - s_eps)))
        if not psd:
            if s_eps == 0:
                s_eps = 1e-08
            else:
                s_eps *= 10

    S = S * (1 - s_eps)

    return S, S * np.diag(Sigma)


def _knockoff_equi_generated(X, mu, Sigma, seed=None):
    """Generate second-order knockoff variables using equi-correlated method.
    Reference: Candes et al. (2016), Barber et al. (2015)

    Parameters
    ----------
    X: 2D ndarray (n_samples, n_features)
        original design matrix

    mu : 1D ndarray (n_features, )
        vector of empirical mean values

    Sigma : 2D ndarray (n_samples, n_features)
        empirical covariance matrix

    Returns
    -------
    X_tilde : 2D ndarray (n_samples, n_features)
        knockoff design matrix
    """

    n_samples, n_features = X.shape

    Diag_s_equi = np.diag(_s_equi(Sigma)[1])
    Sigma_inv_s = np.linalg.solve(Sigma, Diag_s_equi)

    # First part on the RHS of equation 1.4 in Barber & Candes (2015)
    Mu_tilde = X - np.dot(X - mu, Sigma_inv_s)
    # To calculate the Cholesky decomposition later on
    Sigma_tilde = 2 * Diag_s_equi - Diag_s_equi.dot(
        Sigma_inv_s.dot(Diag_s_equi))

    np.random.seed(seed)
    U_tilde = np.random.randn(n_samples, n_features)
    # Equation 1.4 in Barber & Candes (2015)
    X_tilde = Mu_tilde + np.dot(U_tilde, np.linalg.cholesky(Sigma_tilde))

    return X_tilde
