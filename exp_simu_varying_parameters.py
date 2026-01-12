"""Reproduce Fig.3 in the main text
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_simulation import generate_data
from utils import cal_fdp_power, fdr_threshold

from crt_logit import crt_logit
from crt_variants import crt, dcrt_zero, hrt
from other_methods import desparsified_lasso, model_x_knockoff


def one_simulation(n, p, fdr, effect, snr, rho, sparsity, seed, covariance,
                   consecutive, fdr_control='bhq'):

    print(f'Iteration: {seed + 1}')
    X, y, _, non_zero_index = \
        generate_data(n, p, snr=snr, rho=rho, sparsity=sparsity, effect=effect,
                      consecutive=consecutive, binary=True,
                      covariance=covariance, seed=seed)

    X = StandardScaler().fit_transform(X)
    result_ko = model_x_knockoff(X, y, fdr=fdr, loss='logistic',
                                 seed=seed*2)

    result_crt_logit = crt_logit(X, y, fdr=fdr, verbose=False,
                                 fdr_control=fdr_control)

    _, _, pval_dlasso = desparsified_lasso(X, y, tol=1e-10)
    threshold = fdr_threshold(pval_dlasso, fdr=fdr)
    result_dlasso = np.where(pval_dlasso <= threshold)[0]

    result_dcrt = dcrt_zero(X, y, fdr=fdr, verbose=False)

    result_hrt = hrt(X, y, model='binomial', n_samplings=5000)
    result_crt = crt(X, y, fdr=fdr, n_samplings=500, model='binomial')

    fdr_dlasso, power_dlasso = cal_fdp_power(result_dlasso, non_zero_index)
    fdr_ko, power_ko = cal_fdp_power(result_ko, non_zero_index)
    fdr_dcrt, power_dcrt = cal_fdp_power(result_dcrt, non_zero_index)
    fdr_crt_logit, power_crt_logit = \
        cal_fdp_power(result_crt_logit, non_zero_index)
    fdr_crt, power_crt = cal_fdp_power(result_crt, non_zero_index)
    fdr_hrt, power_hrt = cal_fdp_power(result_hrt, non_zero_index)

    print(f'dLasso: FDR={fdr_dlasso} | Power={power_dlasso}')
    print(f'KO: FDR={fdr_ko} | Power={power_ko}')
    print(f'CRT: FDR={fdr_crt} | Power={power_crt}')
    print(f'HRT: FDR={fdr_hrt} | Power={power_hrt}')
    print(f'CRT-logit: FDR={fdr_crt_logit} | Power={power_crt_logit}')
    print(f'DCRT: FDR={fdr_dcrt} | Power={power_dcrt}')

    return (
        fdr_dlasso,
        fdr_ko,
        fdr_dcrt,
        fdr_crt_logit,
        fdr_crt,
        fdr_hrt,
        power_dlasso,
        power_ko,
        power_dcrt,
        power_crt_logit,
        power_crt,
        power_hrt,
    )


n_simu = 100
snr, rho, sparsity = 2.0, 0.5, 0.04
rhos = np.arange(0.4, 0.9, 0.1)
snrs = np.arange(1.0, 7.0, 1.0)
sparsities = np.arange(0.03, 0.09, 0.01)
params = [snrs, rhos, sparsities]
n, p, = 400, 600
fdr = 0.1
effect = 2.0
covariance = 'toeplitz'
consecutive = False
fdr_control = 'bhq'

results_rho = np.array(
    [one_simulation(n=n, p=p, covariance=covariance,
                    consecutive=consecutive,
                    fdr_control=fdr_control,
                    fdr=fdr, snr=snr, rho=r,
                    sparsity=sparsity,
                    effect=effect, seed=seed)
     for r in rhos
     for seed in range(n_simu)])

results_snr = np.array(
    [one_simulation(n=n, p=p, fdr=fdr, covariance=covariance,
                    consecutive=consecutive,
                    fdr_control=fdr_control,
                    snr=s, rho=rho, sparsity=sparsity,
                    effect=effect, seed=seed)
     for s in snrs
     for seed in range(n_simu)])

results_sparsity = np.array([
    one_simulation(n=n, p=p, fdr=fdr,
                   covariance=covariance,
                   consecutive=consecutive,
                   fdr_control=fdr_control,
                   snr=snr, rho=rho, sparsity=spr,
                   effect=effect, seed=seed)
    for spr in sparsities
    for seed in range(n_simu)])


plt.style.use('fivethirtyeight')
plt.rcParams["font.family"] = "Roboto Slab"
plt.rcParams["font.size"] = 10

# Mean for FDR and Avg. Power
results_rho = np.array(
    np.split(results_rho, len(rhos), axis=0)).astype(np.float32)
results_snr = np.array(
    np.split(results_snr, len(snrs), axis=0)).astype(np.float32)
results_sparsity = np.array(
    np.split(results_sparsity, len(sparsities), axis=0)).astype(np.float32)

avg = [
    results_snr.mean(1),
    results_rho.mean(1),
    results_sparsity.mean(1),
]

std = [
    results_snr.std(1),
    results_rho.std(1),
    results_sparsity.std(1),
]

METHODS = [
    'dlasso',
    'KO-logit',
    'dCRT',
    'CRT-logit',
    'CRT',
    'HRT',
]

n_methods = len(METHODS)

labels = [
    'snr',
    'rho',
    'sparsity',
]

# Plotting
plt.close()
nrows, ncols = len(labels), 2
fig, ax = plt.subplots(nrows, ncols, figsize=(4.0, 1 + 2 * ncols))

for i in range(len(labels)):
    for j in range(n_methods):
        # FDR
        linestyle = '-'
        marker = ''
        linewidth = 2.0
        alpha = 1.0
        ax[i, 0].grid(linestyle='dotted')
        # ax[i, 0].tick_params(labelsize=14)
        ax[i, 0].plot(params[i], avg[i][:, j],
                      marker=marker,
                      linewidth=linewidth,
                      linestyle=linestyle,
                      alpha=alpha,
                      label=METHODS[j],
                      )

        ax[i, 0].fill_between(params[i], (avg[i][:, j] - std[i][:, j]),
                              (avg[i][:, j] + std[i][:, j]), alpha=.2)

        ax[i, 0].set_xticks(params[i])
        ax[i, 0].set_xticklabels(np.round(params[i], 2), rotation=30)
        # ax[i, 0].set_xticklabels([])
        ax[i, 0].set_xlabel(labels[i])
        ax[i, 0].axhline(y=fdr, linestyle='--', color='k',
                         linewidth=2.0)
        ax[i, 0].set_ylim([-0.02, 0.6])
        ax[i, 0].set_yticks(np.arange(0.0, 0.6, 0.1))
        # Avg. Power
        ax[i, 1].grid(linestyle='dotted')
        # ax[i, 1].tick_params(labelsize=14)
        ax[i, 1].plot(params[i], avg[i][:, j + n_methods],
                      marker=marker,
                      linewidth=linewidth,
                      alpha=alpha,
                      linestyle=linestyle,
                      )
        ax[i, 1].fill_between(
            params[i], (avg[i][:, j + n_methods] - std[i][:, j + n_methods]),
            (avg[i][:, j + n_methods] + std[i][:, j + n_methods]),
            alpha=.2)
        ax[i, 1].set_xticks(params[i])
        ax[i, 1].set_xticklabels(np.round(params[i], 2), rotation=30)
        ax[i, 1].set_xlabel(labels[i])
        ax[i, 1].set_ylim([-0.1, 1.1])
        ax[i, 1].set_yticks(np.arange(0.0, 1.1, 0.2))

        ax[i, 0].set_ylabel('FDR')
        ax[i, 1].set_ylabel('Avg. power')

        if i == 0:
            ax[i, 0].legend(fontsize=6.9, ncol=1)

plt.tight_layout()
plt.show()
