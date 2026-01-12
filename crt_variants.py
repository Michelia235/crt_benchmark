"""Wrapper of vanilla CRT and HRT, R implementation from
https://github.com/moleibobliu/Distillation-CRT/
"""
from pathlib import Path

import numpy as np
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone

from other_methods import _estimate_distribution
from utils import fdr_threshold, _lambda_max


CRT_FILE = str(Path(__file__).with_suffix('.R'))
# load objects in R
robjects.r(f'''source('{CRT_FILE}')''')
numpy2ri.activate()


def dcrt_zero(X, y, fdr=0.1, estimated_coef=None, Sigma_X=None, cv=5,
              n_regus=20, max_iter=1e4, use_cv=False, refit=False,
              screening=True, screening_threshold=100, centered=True,
              alpha=None, solver='liblinear', fdr_control='bhq', n_jobs=1,
              verbose=False, joblib_verbose=0):
    """D0-CRT following
    Liu et al 2022 - Fast and Powerful Conditional Randomization Testing via Distillation
    https://arxiv.org/abs/2006.03980
    """    
    if centered:
        X = StandardScaler().fit_transform(X)

    _, n_features = X.shape

    if estimated_coef is None:
        clf = LassoCV(cv=cv, n_jobs=n_jobs, n_alphas=n_regus*2, tol=1e-6,
                      fit_intercept=False, random_state=0,
                      max_iter=max_iter)
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
        delayed(_lasso_distillation_residual)(
            X, y, idx, coef_X_full, Sigma_X=Sigma_X, cv=cv,
            use_cv=use_cv, alpha=alpha, n_jobs=1, n_regus=5)
        for idx in selection_set)

    Ts = np.zeros(n_features)
    Ts[selection_set] = np.array(results)
    pvals = np.minimum(2 * stats.norm.sf(np.abs(Ts)), 1)
    threshold = fdr_threshold(pvals, fdr=fdr, method=fdr_control)
    selected = np.where(pvals <= threshold)[0]

    if verbose:
        return selected, pvals, Ts

    return selected


def crt(X, y, n_samplings=100, center=True, method='LASSO', fdr=0.1,
        model='gaussian', verbose=False, n_jobs=1):
    """Conditional Randomization Test  following Candes et al (2018):
    Panning for Gold: Model-X Knockoffs for High-dimensional Controlled Variable Selection
    https://arxiv.org/abs/1610.02351
    """
    if center:
        X = StandardScaler().fit_transform(X)

    n_samples, n_features = X.shape
    mu, Sigma = _estimate_distribution(X)
    crt_smc = robjects.r.CRT_sMC
    results = crt_smc(y.reshape(1, -1), X, Sigma, m=n_samplings, FDR=fdr,
                      model=model, n_jobs=n_jobs)

    if len(results[0]) > 0:
        selected_index = np.array(results[0]) - 1  # R index starting from 1
    else:
        selected_index = np.array([])

    if verbose:
        pvals = results[1]
        return selected_index, pvals

    return selected_index


def hrt(X, y, n_samplings=100, center=True, method='CV', fdr=0.1,
        screening=True, model='gaussian', verbose=False, n_jobs=1):
    """Holdout Randomization Test following Tansey et al (2020):
    The Holdout Randomization Test for Feature Selection in Black Box Models
    https://arxiv.org/abs/1811.00645
    """
    if center:
        X = StandardScaler().fit_transform(X)

    n_samples, n_features = X.shape
    mu, Sigma = _estimate_distribution(X)

    pvl_study = not screening
    hrt_r = robjects.r.HRT
    results = hrt_r(y.reshape(1, -1), X, Sigma, N=n_samplings, FDR=fdr,
                    pvl_study=pvl_study, model_select=method, model=model,
                    n_jobs=n_jobs)

    if len(results[0]) > 0:
        selected_index = np.array(results[0]) - 1  # R index starting from 1
    else:
        selected_index = np.array([])

    if verbose:
        pvals = results[1]
        return selected_index, pvals

    return selected_index


def _x_distillation_lasso(X, idx, Sigma_X=None, cv=3, n_regus=100, alpha=None,
                          use_cv=False, n_jobs=1):

    n_samples = X.shape[0]
    X_minus_idx = np.delete(np.copy(X), idx, 1)

    if Sigma_X is None:
        if use_cv:
            clf = LassoCV(cv=cv, n_jobs=n_jobs, n_alphas=n_regus,
                          random_state=0)
            clf.fit(X_minus_idx, X[:, idx])
            alpha = clf.alpha_
        else:
            if alpha is None:
                alpha = 0.1 * _lambda_max(X_minus_idx, X[:, idx],
                                          use_noise_estimate=False)
            clf = Lasso(alpha=alpha, fit_intercept=False)
            clf.fit(X_minus_idx, X[:, idx])

        X_res = X[:, idx] - clf.predict(X_minus_idx)
        sigma2_X = np.linalg.norm(X_res) ** 2 / n_samples + \
            alpha * np.linalg.norm(clf.coef_, ord=1)

    else:
        Sigma_temp = np.delete(np.copy(Sigma_X), idx, 0)
        b = Sigma_temp[:, idx]
        A = np.delete(np.copy(Sigma_temp), idx, 1)
        coefs_X = np.linalg.solve(A, b)
        X_res = X[:, idx] - np.dot(X_minus_idx, coefs_X)
        sigma2_X = Sigma_X[idx, idx] - np.dot(
            np.delete(np.copy(Sigma_X[idx, :]), idx), coefs_X)

    return X_res, sigma2_X


def _lasso_distillation_residual(X, y, idx, coef_full, Sigma_X=None, cv=3,
                                 n_regus=50, n_jobs=1, use_cv=False,
                                 alpha=None, fit_y=False):
    """Standard Lasso Distillation following Liu et al. (2020) section 2.4. Only
    works for least square loss regression.
    """
    n_samples, _ = X.shape

    X_minus_idx = np.delete(np.copy(X), idx, 1)

    # Distill X
    X_res, sigma2_X = _x_distillation_lasso(X, idx, Sigma_X, cv=cv,
                                            use_cv=use_cv, alpha=alpha,
                                            n_regus=n_regus, n_jobs=n_jobs)

    # Distill Y - calculate residual
    if use_cv:
        clf_null = LassoCV(cv=cv, n_jobs=n_jobs, n_alphas=n_regus,
                           random_state=0)
    else:
        if alpha is None:
            alpha = 0.5 * _lambda_max(X_minus_idx, y,
                                      use_noise_estimate=False)
        clf_null = Lasso(alpha=alpha, fit_intercept=False)

    if fit_y:
        clf_null.fit(X_minus_idx, y)
        coef_minus_idx = clf_null.coef_
    else:
        coef_minus_idx = np.delete(np.copy(coef_full), idx)

    eps_res = y - X_minus_idx.dot(coef_minus_idx)
    sigma2_y = np.mean(eps_res ** 2)

    # T follows Gaussian distribution
    Ts = np.dot(eps_res, X_res) / np.sqrt(n_samples * sigma2_X * sigma2_y)

    return Ts
