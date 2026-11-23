"""Paired comparisons with uncertainty clustered by announcement quarter."""

import numpy as np
from scipy.stats import t


def relative_reduction(benchmark_loss, model_loss, clusters):
    """Percentage reduction in mean loss with a delta-method 95% interval.

    Standard errors are clustered (CR1) and use a t reference with one fewer
    degree of freedom than the number of clusters.
    """
    d = benchmark_loss - model_loss
    base = benchmark_loss.mean()
    gain = d.mean() / base
    influence = (d - gain * benchmark_loss) / base
    sums = influence.groupby(clusters).sum()
    k = len(sums)
    margin = (t.ppf(0.975, k - 1) * np.sqrt(k / (k - 1) * np.square(sums).sum())
              / len(d))
    return 100 * gain, 100 * (gain - margin), 100 * (gain + margin)


def bootstrap_relative_reduction(paired_totals, draws=20000, seed=42):
    """95% interval for 1 - model/benchmark by resampling whole quarters.

    ``paired_totals`` holds one row per quarter: benchmark total, model total.
    """
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(paired_totals),
                          size=(draws, len(paired_totals)))
    sums = paired_totals[sample].sum(axis=1)
    return np.quantile(100 * (1 - sums[:, 1] / sums[:, 0]), [.025, .975])
