
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


def kama(price, period=10, period_fast=2, period_slow=30):
    # Efficiency Ratio
    change = abs(price - price.shift(period))
    volatility = (abs(price - price.shift())).rolling(period).sum()
    er = change / volatility

    # Smoothing Constant
    sc_fatest = 2 / (period_fast + 1)
    sc_slowest = 2 / (period_slow + 1)
    sc = (er * (sc_fatest - sc_slowest) + sc_slowest) ** 2

    # KAMA
    kama = np.zeros_like(price)
    kama[period - 1] = price[period - 1]

    for i in range(period, len(price)):
        kama[i] = kama[i - 1] + sc[i] * (price[i] - kama[i - 1])

    kama[kama == 0] = np.nan

    return kama


def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1/n, adjust=False).mean()
