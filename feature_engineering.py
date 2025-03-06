import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta


def get_trading_days(start_year, end_year):
    # Define US market holidays
    trading_days = pd.Series(index=pd.date_range(start_year, end_year))

    trading_days.index = pd.DatetimeIndex(trading_days.index)

    for i in range(start_year, end_year):
        trading_day_count = 0

        current_date = date(i, 1, 1)
        holidays = get_trading_holidays(i)

        while current_date.year == i:
            if current_date.weekday() < 5 and current_date not in holidays:
                trading_day_count += 1
                trading_days.loc[current_date] = trading_day_count

            current_date += timedelta(days=1)

    return trading_days


def get_trading_holidays(year):

    holidays = pd.to_datetime([
        f'{year}-01-01',  # New Year's Day (if on a weekday)
        f'{year}-07-04',  # Independence Day (if on a weekday)
        f'{year}-12-25',  # Christmas Day (if on a weekday)
    ])

    # Handle holidays that fall on weekends (observed days)
    observed_holidays = []
    for holiday in holidays:
        if holiday.weekday() == 5:  # Saturday -> Observed on Friday
            observed_holidays.append(holiday - timedelta(days=1))
        elif holiday.weekday() == 6:  # Sunday -> Observed on Monday
            observed_holidays.append(holiday + timedelta(days=1))
        else:
            observed_holidays.append(holiday)

    holidays = set(observed_holidays)

    # Add variable holidays (MLK Day, Presidents' Day, etc.)
    variable_holidays = [
        pd.Timestamp(f'{year}-01-01') + pd.DateOffset(weekday=2),  # MLK Day (3rd Monday of Jan)
        pd.Timestamp(f'{year}-02-01') + pd.DateOffset(weekday=2),  # Presidents' Day (3rd Monday of Feb)
        pd.Timestamp(f'{year}-05-01') + pd.DateOffset(weekday=2),  # Memorial Day (last Monday of May)
        pd.Timestamp(f'{year}-09-01') + pd.DateOffset(weekday=2),  # Labor Day (1st Monday of Sep)
        pd.Timestamp(f'{year}-11-01') + pd.DateOffset(weekday=4),  # Thanksgiving (4th Thursday of Nov)
    ]

    holidays.update(variable_holidays)

    return holidays


def trade_days(df):
    trading_days = get_trading_days(df.index.year.min(), df.index.year.max())
    common = list(set(df.index.date) & set(trading_days.index)).sort()
    df['tdays'] = trading_days.loc[common]

    return df


def compute_rolling(group, window):
    group.sort_values('datetime')
    group['rolling_mean'] = group['Volume'].rolling(window=window, min_periods=1).mean()
    return group

def rvol(data, by='date',length=10):
    data['datetime'] =  data.index
    data['rolling_mean'] = pd.Series()

    if by == 'date':
        groupby_data = data.index.dayofyear

    if by == 'tdays':
        groupby_data = data['tdays']

    else:
        groupby_data = data.index.time

    rolling_mean = data.groupby(groupby_data, group_keys=False).apply(compute_rolling, window=length)

    return data['Volume'] / rolling_mean['rolling_mean']

def cum_rvol(data, length=5, groupby='month'):
    volume = data['Volume']

    dts = volume.index

    if groupby == 'month':
        cum_volume = volume.groupby(dts.month, sort=False).cumsum()

    prev_mean = lambda days: (
            cum_volume
            .rolling(days, closed='left')
            .mean()
            .reset_index(0, drop=True)  # drop the level with dts.time
        )

    return cum_volume / prev_mean(length)


def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

def tr(data, high='High', low='Low', close='Close', normalized=False):
    df = data.copy()
    df['tr0'] = data[high] - data[low]
    df['tr1'] = np.abs(data[high] - data[close].shift(1))
    df['tr2'] = np.abs(data[low] - data[close].shift(1))
    tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    if normalized:
        return tr /df[close]
    else:
        return tr

def atr(data,high='High', low='Low', close='Close', length=7, normalized=False):
    true_range = tr(data, high, low, close)
    atr = wwma(true_range, length)
    if normalized:
        return atr /data[close]
    else:
        return  atr

def natr(data, high, low, close, length):
    return atr(data, high,low, close, length)/close


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
