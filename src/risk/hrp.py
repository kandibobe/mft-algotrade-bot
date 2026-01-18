"""
Hierarchical Risk Parity (HRP) Utility
======================================

Implements the HRP algorithm for portfolio weight allocation.
HRP uses machine learning (clustering) to group correlated assets
and allocate risk across them equally.
"""

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

def get_hrp_weights(prices: pd.DataFrame) -> dict[str, float]:
    """
    Calculate asset weights using Hierarchical Risk Parity.

    Args:
        prices: DataFrame of historical prices for assets (columns are symbols)

    Returns:
        Dictionary of {symbol: weight}
    """
    if prices.empty or prices.shape[1] < 2 or len(prices) < 10:
        return {col: 1.0/max(1, prices.shape[1]) for col in prices.columns}

    # 1. Calculate Returns and Covariance
    returns = prices.pct_change().dropna()
    corr = returns.corr()
    cov = returns.cov()

    # 2. Cluster assets
    # Distance metric based on correlation
    dist = ((1 - corr) / 2.0)**0.5
    link = linkage(squareform(dist), method='single')

    # 3. Sort assets by clusters (Quasi-Diagonalization)
    sort_idx = _get_quasi_diag(link)
    sorted_items = corr.columns[sort_idx].tolist()

    # 4. Recursive Bisection to find weights
    weights = pd.Series(1.0, index=sorted_items)
    _recursive_bisection(weights, sorted_items, cov)

    logger.info(f"HRP Weights calculated for {len(sorted_items)} assets.")
    return weights.to_dict()

def _get_quasi_diag(link):
    """Sort items into hierarchical clusters."""
    link = link.astype(int)
    sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_idx.max() >= num_items:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        df0 = sort_idx[sort_idx >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_idx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_idx = pd.concat([sort_idx, df0])
        sort_idx = sort_idx.sort_index()
        sort_idx.index = range(sort_idx.shape[0])
    return sort_idx.tolist()

def _recursive_bisection(weights, items, cov):
    """Allocate weights based on variance parity."""
    if len(items) <= 1:
        return

    # Split items into two clusters
    idx = len(items) // 2
    c1 = items[:idx]
    c2 = items[idx:]

    # Calculate cluster variance
    v1 = _get_cluster_var(c1, cov)
    v2 = _get_cluster_var(c2, cov)

    # Calculate allocation factor
    alpha = 1 - v1 / (v1 + v2)

    # Assign weights
    weights[c1] *= alpha
    weights[c2] *= (1 - alpha)

    # Recurse
    _recursive_bisection(weights, c1, cov)
    _recursive_bisection(weights, c2, cov)

def _get_cluster_var(items, cov):
    """Calculate variance of a cluster."""
    cov_c = cov.loc[items, items]
    # Simplified inverse variance weighting for cluster
    ivp = 1.0 / np.diag(cov_c)
    ivp /= ivp.sum()
    w = ivp.reshape(-1, 1)
    return np.dot(np.dot(w.T, cov_c), w)[0, 0]
