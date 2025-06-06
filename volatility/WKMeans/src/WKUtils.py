import numpy as np
import pandas as pd


def window_lift(series, h1=30, h2=3):
    """
    Convert a 1D time series into overlapping windows and track index ranges.

    Parameters:
    - series: 1D numpy array
    - h1: window size
    - h2: stride

    Returns:
    - windows: list of 1D arrays, each of length h1
    - indices: list of (start, end) tuples for each window
    """
    N = len(series)
    windows = []
    indices = []

    M = (N - (h1 - h2)) // h2  # number of windows

    for m in range(M):
        start = m * h2
        end = start + h1
        if end <= N:
            windows.append(series[start:end])
            indices.append((start, end))

    return windows, indices

def cum_rets(df, return_col='returns', regime_col='cluster', start_idx=1, reset_index=True):
    cumulative_returns = df.groupby(regime_col)[return_col].apply(lambda x: (start_idx + x).cumprod())
    if reset_index:
        cumulative_returns = cumulative_returns.reset_index()
        cumulative_returns['t'] = cumulative_returns.groupby(regime_col).cumcount()
        cumulative_returns = cumulative_returns.pivot(index='t', columns='cluster', values='returns')

    return cumulative_returns


def reconstruct(returns, indices, assignments):
    cumulative_returns = np.cumprod(1 + returns)

    df = pd.DataFrame(data={'returns':returns.values,
                            'cumulative_returns':cumulative_returns,
                            'cluster': np.full(len(returns),np.nan)}, index=returns.index)

    for (start, end), label in zip(indices, assignments):
        df.iloc[start:end, -1] = label

    return df

