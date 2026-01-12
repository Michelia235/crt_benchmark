import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Lasso, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from utils import fdr_threshold, _logit


def crt_logit(X, y, fdr=0.1, estimated_coef=None, Sigma_X=None, cv=5,
              n_regus=20, max_iter=10000000000000, use_cv=True, refit=False,
              screening=True, screening_threshold=100, centered=True,
              alpha=None, solver='liblinear', fdr_control='bhq', n_jobs=1,
              verbose=False, joblib_verbose=0):
    """Conditional Randomization Test for high-dimensional sparse logistic regression
    See more details at https://arxiv.org/abs/2205.14613

    Parameters
    ----------
    X : 2D ndarray
        Design matrix of size n_samples, n_variables
    y : ndarray
        Binary response, must be either 0 or 1
    fdr : float
        Desired FDR control level

    Returns 

    selected: ndarray
        List of selected indices of variable
    pvals: ndarray
        P-values corresponding to the variable
    Ts: ndarray
        Decorrelated statistics corresponding to the variable
    """
    if centered:
        X = StandardScaler().fit_transform(X)

    _, n_features = X.shape

    if estimated_coef is None:
        clf = LogisticRegressionCV(Cs=np.logspace(-3, 2, n_regus), cv=cv,
                                   tol=1e-4, n_jobs=n_jobs,
                                   fit_intercept=False, random_state=0,
                                   penalty='l1', max_iter=max_iter,
                                   solver=solver)
        clf.fit(X, y)
        coef_X_full = np.ravel(clf.coef_)
    else:
        coef_X_full = estimated_coef
        screening_threshold = 100

    non_selection = np.where(np.abs(coef_X_full) <= np.percentile(
        np.abs(coef_X_full), 100 - screening_threshold))[0]
    coef_X_full[non_selection] = 0.0

    # Screening step -- speed up computation of score function by only running
    # it later on estimated support set
    if screening:
        selection_set = np.setdiff1d(np.arange(n_features), non_selection)

        if selection_set.size == 0:
            if verbose:
                return np.array([]), np.ones(n_features), np.zeros(n_features)
            return np.array([])
    else:
        selection_set = np.arange(n_features)

    if refit and estimated_coef is None and selection_set.size < n_features:
        clf_refit = clone(clf)
        clf_refit.fit(X[:, selection_set], y)
        coef_X_full[selection_set] = np.ravel(clf_refit.coef_)

    # Distillation & calculate score function

    results = Parallel(n_jobs, verbose=joblib_verbose)(
        delayed(_decorrelate_test_score)(
            X, y, idx, coef_X_full, cv=cv, use_cv=use_cv,
            refit=refit, alpha=alpha, n_regus=5, n_jobs=1)
        for idx in selection_set)

    Ts = np.zeros(n_features)
    Ts[selection_set] = np.array(results)
    pvals = np.minimum(2 * stats.norm.sf(np.abs(Ts)), 1)
    threshold = fdr_threshold(pvals, fdr=fdr, method=fdr_control)
    selected = np.where(pvals <= threshold)[0]
    if verbose:
        return selected, pvals, Ts
    return selected


def _decorrelate_test_score(X, y, idx, coef_full, cv=3, n_regus=50,
                            refit=False, alpha=None, use_cv=False,
                            n_jobs=1):
    n_samples, _ = X.shape

    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X
    X_res, fisher_minus_idx = _decorrelate_x(
        X, idx, coef_full, cv=cv, n_regus=n_regus, refit=refit, alpha=alpha,
        use_cv=use_cv, n_jobs=n_jobs)

    # Distill y
    coef_minus_idx = np.delete(np.copy(coef_full), idx)
    eps_res = y - _logit(X_minus_idx.dot(coef_minus_idx))

    Ts = -np.dot(eps_res, X_res) / np.sqrt(n_samples * fisher_minus_idx)

    return Ts


def _decorrelate_x(X, idx, coef_full, cv=3, n_regus=5, alpha=None,
                   refit=False, use_cv=False, n_jobs=1):

    n_samples, n_features = X.shape
    alpha_prime = np.sqrt(np.log(n_features) / n_samples)
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    lasso_weights = \
        np.exp(X.dot(coef_full)) / (1 + np.exp(X.dot(coef_full))) ** 2

    if use_cv:
        alphas = np.logspace(-2, 2, n_regus) * alpha_prime
        param_grid = {'fit_intercept': [False], 'alpha': alphas}
        fit_params = {'sample_weight': 2 * lasso_weights}
        clf = GridSearchCV(Lasso(), param_grid, n_jobs=n_jobs, cv=cv,
                           scoring='r2')
        clf.fit(X_minus_idx, X[:, idx], **fit_params)
        coef_temp = clf.best_estimator_.coef_
    else:
        if alpha is None:
            alpha = alpha_prime
        clf = Lasso(alpha=alpha, fit_intercept=False, random_state=0)
        clf.fit(X_minus_idx, X[:, idx], sample_weight=2 * lasso_weights)
        coef_temp = clf.coef_
        if refit:
            s_hat = np.where(coef_temp != 0)[0]
            if s_hat.size != 0:
                coef_s_hat = \
                    clone(clf).fit(X_minus_idx[:, s_hat], X[:, idx],
                                   sample_weight=2 * lasso_weights).coef_
                coef_temp[s_hat] = coef_s_hat

    X_res = X[:, idx] - X_minus_idx.dot(coef_temp)
    fisher_minus_idx = np.mean(lasso_weights * X[:, idx] * X_res)

    return X_res, fisher_minus_idx
