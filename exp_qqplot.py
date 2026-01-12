"""Reproduce Fig.1 and 2 in main text
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot

from crt_logit import crt_logit
from data_simulation import generate_data

from crt_variants import dcrt_zero

plt.rcParams["font.family"] = "Roboto Slab"


def one_simulation(n, p, snr, rho, sparsity, seed, covariance, consecutive):

    # use fixed non-null for easier plotting of null distribution
    X, y, _, _ = generate_data(n, p, snr=snr, rho=rho, sparsity=sparsity,
                               effect=2.0, consecutive=consecutive,
                               binary=True, fixed_non_null=True,
                               covariance=covariance, seed=seed)

    X = StandardScaler().fit_transform(X)

    # disable screening to have correct distribution of the test statistics
    result_dcrt = dcrt_zero(X, y, verbose=True, screening=False)
    result_crt_logit = crt_logit(X, y, verbose=True, screening=False)

    return result_dcrt[-1], result_crt_logit[-1]


n_simu = 1000
snr, rho, sparsity = 3.0, 0.4, 0.06

p = 400
n = 400  # 200, 800, 4000
covariance = 'toeplitz'
consecutive = False

results = np.array([
    one_simulation(n=n, p=p, snr=snr, rho=rho, sparsity=sparsity,
                   covariance=covariance, consecutive=consecutive, seed=seed)
    for seed in range(n_simu)
])

print('Running experiment')
_, _, _, non_zero_index = \
    generate_data(n, p, snr=snr, rho=rho, sparsity=sparsity, effect=2.0,
                  consecutive=consecutive, binary=True, fixed_non_null=True,
                  covariance=covariance, seed=0)

null_index = np.setdiff1d(np.arange(p), non_zero_index)
nulls = results[:, :, null_index]

METHODS = [
    'dCRT',
    'CRT-Logit',
]
n_methods = len(METHODS)

fig, axs = plt.subplots(n_methods, 1, figsize=(3, 3 * n_methods))

# pick a random null index to plot, could be anything else
picked = 6
for j in range(n_methods):
    qqplot(nulls[:, j, picked], dist=stats.norm, line="45", ax=axs[j],
           alpha=1.0, markerfacecolor='blue', markeredgecolor='blue')
    axs[j].set_title(METHODS[j], fontweight='bold')
    axs[j].set_xlim([-3.2, 3.2])
    axs[j].set_ylim([-3.2, 3.2])

fig.show()
