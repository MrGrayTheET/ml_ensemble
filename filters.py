
import pandas as pd
import numpy as np


def laguerre_filter(prices, alpha=0.8):
    n = prices.size
    L0, L1, L2, L3, f = [pd.Series(0, index=prices.index)] * 5

    for i in range(1, n):
        p_L0 = L0.iloc[i-1]
        p_L1 = L1.iloc[i-1]
        p_L2 = L2.iloc[i-1]
        p_L3 = L3.iloc[i-1]

        L0.iloc[i] = (1-alpha) * prices[i] + alpha * p_L0
        L1.iloc[i] = -alpha * L0[i] +p_L0 + alpha * p_L1
        L2.iloc[i] = -alpha * L1[i] + p_L1 + alpha * p_L2
        L3.iloc[i] = -alpha * L2[i] + p_L2 + alpha * p_L3

    f = (L0 + (2 * L1) +  (2 * L2) + L3)/6

    return f


def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1/n, adjust=False).mean()
