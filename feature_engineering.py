import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from filters import wwma

VOL_LOOKBACK = 60
VOL_TARGET = 0.15


def ts_train_test_split(df, test_size=0.2, groups=None):
    """
    Split time series data into training and test sets while preserving sequential order.

    Parameters:
    -----------
    df : pandas.DataFrame
        The time series dataframe to split
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split (0.0 to 1.0)
    groups : str or list, optional
        Column name(s) identifying group(s) to split by. If provided, each group will be
        split separately and then recombined, maintaining sequential order within groups.

    Returns:
    --------
    train_df : pandas.DataFrame
        The training dataset
    test_df : pandas.DataFrame
        The test dataset
    """
    if groups is None:
        # Simple case: split the entire dataframe at once
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        return train_df, test_df

    # Convert single group to list
    if isinstance(groups, str):
        groups = [groups]

    # Get unique values for each group
    group_values = {}
    for group in groups:
        group_values[group] = df[group].unique()

    # Initialize empty dataframes for train and test
    train_dfs = []
    test_dfs = []

    # Helper function to handle nested groups
    def process_groups(df, group_idx=0, group_filters={}):
        if group_idx >= len(groups):
            # Base case: We've filtered down to a specific combination of all groups
            filtered_df = df.copy()
            for k, v in group_filters.items():
                filtered_df = filtered_df[filtered_df[k] == v]

            # Ensure the data is sorted by index (assuming index represents time)
            filtered_df = filtered_df.sort_index()

            # Split this specific group's data
            split_idx = int(len(filtered_df) * (1 - test_size))
            train_dfs.append(filtered_df.iloc[:split_idx])
            test_dfs.append(filtered_df.iloc[split_idx:])
            return

        # Recursive case: process each value of the current group
        current_group = groups[group_idx]
        for value in group_values[current_group]:
            new_filters = group_filters.copy()
            new_filters[current_group] = value
            process_groups(df, group_idx + 1, new_filters)

    # Process all group combinations
    process_groups(df)

    # Concatenate results
    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()

    return train_df, test_df

def reshape_multiindex(df, group_id_column='Ticker', group_name_column='Tickers'):
    """
    Reshape a DataFrame with MultiIndex columns so that:
    - Control groups (second level of columns) become values in a single column
    - Each independent variable (first level of columns) retains its own column
    - A new column identifies each control group with integers
    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame with MultiIndex columns where the first level is the independent variable
        and the second level is the control group.
    group_id_column : str, optional
        Name for the column that will contain the numeric IDs for each control group.
    group_name_column : str, optional
        Name for the column that will contain the names of the control groups.
    Returns:
    --------
    pandas.DataFrame
        A reshaped DataFrame with independent variables as separate columns and
        control groups as values in rows with integer identifiers.
    """
    # Get the unique values for both column levels
    indep_vars = df.columns.get_level_values(0).unique()
    control_groups = df.columns.get_level_values(1).unique()
    # Create a mapping from control groups to integer IDs
    group_to_id = {group: i for i, group in enumerate(control_groups)}
    # Initialize lists to store the data for the new DataFrame
    all_rows = []
    # For each row in the original DataFrame
    for idx, row in df.iterrows():
        for group in control_groups:
            # Create a new row for each control group
            new_row = {
                group_id_column: group_to_id[group],
                group_name_column: group
            }
            # Add values for each independent variable
            for var in indep_vars:
                try:
                    new_row[var] = row[(var, group)]
                except:
                    # Handle case where some combinations might not exist
                    new_row[var] = np.nan
            # Store the original row index
            new_row['Original_Index'] = idx
            all_rows.append(new_row)
    # Create a new DataFrame from the collected rows
    result_df = pd.DataFrame(all_rows)
    # Set a MultiIndex with the original index and the group ID
    result_df = result_df.set_index(['Original_Index', group_id_column])
    return result_df


import pandas as pd
import numpy as np


def stack_groups_vertically(df, group_id_column='Group_ID', group_name_column='Group'):
    """
    Reshape a DataFrame with MultiIndex columns so that control groups are stacked
    vertically one after another, with each group identified by an integer ID.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame with MultiIndex columns where the first level is the independent variable
        and the second level is the control group.
    group_id_column : str, optional
        Name for the column that will contain the numeric IDs for each control group.
    group_name_column : str, optional
        Name for the column that will contain the names of the control groups.

    Returns:
    --------
    pandas.DataFrame
        A reshaped DataFrame with independent variables as separate columns,
        control groups stacked vertically, and group identifiers.
    """
    # Get the unique values for both column levels
    indep_vars = df.columns.get_level_values(0).unique()
    control_groups = df.columns.get_level_values(1).unique()

    # Create a list to hold all the individual group DataFrames
    group_dfs = []

    # Process each control group separately
    for i, group in enumerate(control_groups):
        # Extract columns for this group
        group_data = pd.DataFrame()

        # For each independent variable, extract the corresponding column for this group
        for var in indep_vars:
            group_data[var] = df[(var, group)]

        # Add group identifier columns
        group_data[group_id_column] = i
        group_data[group_name_column] = group

        # Add to our list of DataFrames
        group_dfs.append(group_data)

    # Concatenate all group DataFrames vertically
    result_df = pd.concat(group_dfs, ignore_index=True)

    # Reorder columns to put identifiers first
    cols = [group_id_column, group_name_column] + list(indep_vars)
    result_df = result_df[cols]

    return result_df



def create_labels(prices, future_horizon=1, threshold=0.01):
    """
    Create labels for stock price prediction.

    Args:
        prices (pd.Series): Historical stock prices.
        future_horizon (int): Number of time steps into the future to predict.
        threshold (float): Percentage threshold for labeling.

    Returns:
        pd.Series: Labels (1 for higher, -1 for lower, 0 for flat).
    """
    future_prices = prices.shift(-future_horizon)  # Get future prices
    price_change = (future_prices - prices) / prices  # Calculate percentage change

    # Create labels based on threshold
    labels = np.where(price_change > threshold,2 ,
                      np.where(price_change < -threshold, 0, 1))

    # Drop the last `future_horizon` rows (no future price available)
    labels = labels[:-future_horizon]
    prices = prices[:-future_horizon]  # Align prices with labels

    return pd.Series(labels, index=prices.index)


def calc_returns(price_series, day_offset=1):

    returns = price_series /  price_series.shift(day_offset) - 1.0

    return returns

def calc_daily_vol(returns):
    vol = returns.ewm(span=VOL_LOOKBACK,
                       min_periods=VOL_LOOKBACK).std().ffill()
    return vol


def vol_scaled_returns(returns, daily_vol=pd.Series(None)):

    if not len(daily_vol):
        daily_vol = calc_daily_vol(returns)
    annualized_vol = daily_vol * np.sqrt(252)

    return returns * VOL_TARGET/annualized_vol.shift(1)

def extract_time_features(data, month=True, day = False, dayofweek=False, year=False):
    df = data.copy(deep=True)

    if month:
        df['month'] = df.index.month
    if day:
        df['day'] = df.index.day
    if dayofweek:
        df['weekday'] = df.index.dayofweek
    if year:
        df['year'] = df.index.year

    return df


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


def EWMA_Volatility(rets, lam):
    sq_rets_sp500 = (rets ** 2).values
    EWMA_var = np.zeros(len(sq_rets_sp500))

    for r in range(1, len(sq_rets_sp500)):
        EWMA_var[r] = (1 - lam) * sq_rets_sp500[r] + lam * EWMA_var[r - 1]

    EWMA_vol = np.sqrt(EWMA_var * 250)
    return pd.Series(EWMA_vol, index=rets.index, name="EWMA Vol {}".format(lam))[1:]

class fft_decomp:

    def __init__(self, data):

        self.detrended = False
        self.data = data.dropna()
        self.data['ema'] = self.data.Close.ewm(span=3).mean()

    def kama_detrend(self, period=20, fast=2, slow=30):
        self.detrended_data = (self.data.Close - kama(self.data.Close, period=period, period_fast=fast,
                                                      period_slow=slow))[period:]
        self.detrend_method = 'kama'
        return

    def wwm_detrend(self, period=20):
        self.detrended_data = self.data.Close - wwma(self.data.Close, n=period)
        self.detrend_method = 'wwma'
        return self.detrended_data

    def fft_data(self, detrended=True, d=1, top_n=5, ema=False):
        if detrended:
            if ema:
                data = self.detrended_data.ewm(span=3).mean()
            else:
                data = self.detrended_data

            self.detrended = True
        else:
            if ema:
                data = self.data.Close.ewm(span=3).mean()
            else:
                data = self.data.Close

            self.detrended = False

        self.res = np.fft.fft(data, axis=0)
        N = len(data)

        self.freqs = np.fft.fftfreq(len(self.res), d=1 / d)
        self.magnitude = np.abs(self.res)[:N // 2]
        self.periods = 1 / self.freqs[:N // 2]
        self.periods = self.periods[~np.isinf(self.periods)]
        self.angle = np.angle(self.res)

        peaks, props = find_peaks(self.magnitude, prominence=100)
        peak_periods = self.periods[peaks]
        peak_mags = self.magnitude[peaks]

        sorted_indices = np.argsort(peak_mags)[::-1][:top_n]
        top_periods = peak_periods[sorted_indices]
        top_mags = peak_mags[sorted_indices]
        self.fft_df = pd.DataFrame(top_mags, index=top_periods)
        self.indices = sorted_indices
        self.fft_df['angle'] = np.angle(self.res)[peaks][self.indices]

        return self.fft_df

    def reconstruct_signal(self):
        filtered_data = np.zeros_like(self.res)
        filtered_data[self.indices] = self.res[self.indices]
        filtered_data[-self.indices] = self.res[-self.indices]
        reconstructed_signal = np.fft.ifft(filtered_data).real

        plt.figure(figsize=(12, 8))
        if self.detrended:
            plot_data = self.detrended_data
        else:
            plot_data = self.data.Close

        plt.plot(plot_data.index, plot_data, label='Original')
        plt.plot(plot_data.index, reconstructed_signal, label='Reconstructed')
        plt.legend()
        plt.xlabel("Period")
        plt.ylabel("Close")
        plt.show()
        return

    def extrapolate_signal(self, periods_forward, n_frequencies=5, plot=False, scale_rate=1000):
        if self.detrended:
            data = self.detrended_data
        else:
            data = self.data

        t_extended = np.linspace(0, (len(data) + periods_forward), len(data) + periods_forward, endpoint=False)
        t = np.linspace(0, len(data), len(data), endpoint=False)
        extrapolated_signal = np.zeros_like(t_extended)

        for i in range(n_frequencies):
            extrapolated_signal += self.fft_df[0].iloc[i] * np.sin(
                2 * np.pi * (1 / self.fft_df.index[i]) * t_extended + self.fft_df['angle'].iloc[i])

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(t, data, label="Original Signal")
            plt.plot(t_extended, extrapolated_signal / scale_rate, label='Extrapolated Signal')
            plt.axvline(x=t[-1], color='red', linestyle='dotted', label='Extrapolation Start')
            plt.legend()

        return extrapolated_signal / scale_rate
