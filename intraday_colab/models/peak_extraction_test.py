from sc_loader import sierra_charts as sch
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from intraday_colab import feature_engineering as fe
from intraday_colab.feature_engineering import collect_peak_parameters, obtain_peak_hess
import numba as nb  # For JIT compilation

@nb.njit
def calculate_returns(prices):
    """Numba-optimized return calculation"""
    returns = np.zeros_like(prices)
    for i in range(1, len(prices)):
        returns[i] = prices[i] / prices[i-1] - 1
    return returns


def engineer_delta_feature(tick_data, price_col='price', delta_col='delta', time_col='timestamp'):
    """
    Engineer delta-return feature in 15-minute groups with optimized linear regression

    Parameters:
        tick_data: DataFrame with timestamp, price, and delta columns
        price_col: Name of price column
        delta_col: Name of delta column (askvolume - bidvolume)
        time_col: Name of timestamp column

    Returns:
        DataFrame with intercept and slope for each 15-minute window
    """
    # Convert to numpy for speed
    timestamps = tick_data[time_col].values.astype('datetime64[ns]')
    prices = tick_data[price_col].values.astype(np.float64)
    deltas = tick_data[delta_col].values.astype(np.float64)

    # Create 15-minute groups
    groups = (timestamps.astype('int64') // (15 * 60 * 1e9)).astype(np.int32)
    unique_groups = np.unique(groups)

    # Pre-allocate results
    results = np.zeros((len(unique_groups), 3))  # [group, intercept, slope]

    # Calculate returns once
    returns = calculate_returns(prices)

    # Process each group
    for i, group in enumerate(unique_groups):
        mask = groups == group
        group_deltas = deltas[mask]
        group_returns = returns[mask]

        # Skip if insufficient data
        if len(group_deltas) < 2:
            results[i] = [group, np.nan, np.nan]
            continue

        # High-speed regression (faster than scipy for this case)
        x = group_deltas
        y = group_returns
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.mean((x - x_mean) * (y - y_mean))
        var = np.var(x)

        if var > 1e-10:  # Avoid division by zero
            slope = cov / var
            intercept = y_mean - slope * x_mean
        else:
            slope = np.nan
            intercept = np.nan

        results[i] = [group, intercept, slope]

    # Convert to DataFrame
    result_df = pd.DataFrame(results, columns=['group', 'intercept', 'slope'])
    result_df['timestamp'] = pd.to_datetime(result_df['group'] * 15 * 60 * 1e9)

    return result_df


import pandas as pd
import numpy as np
from scipy.stats import linregress
import datetime as dt

def engineer_delta_features(df, minutes=15, price_col='close',
                            timestamp_col='timestamp', volume_cols=('AskVolume', 'BidVolume'), volume_col = 'Volume', delta_pct=False):
    """
    Engineer delta-return features for specified time windows with proper DataFrame handling

    Parameters:
        df: DataFrame containing market data
        minutes: Window size in minutes (default: 15)
        price_col: Name of price column
        delta_col: Either name of existing delta column OR
                  tuple of (ask_volume, bid_volume) columns to calculate delta
        timestamp_col: Name of timestamp column
        volume_cols: Tuple of (ask_volume, bid_volume) if delta needs calculation

    Returns:
        DataFrame with regression features for each time window
    """
    # Create working copy to preserve input data
    df = df.copy()

    # Calculate delta if needed
    ask_col = volume_cols[0]
    bid_col = volume_cols[1]

    df['delta'] = df[ask_col] - df[bid_col]
    if delta_pct:
        df['delta_pct'] = df['delta']/df[volume_col]
        delta_col = 'delta_pct'
    else:
        delta_col = 'delta'
    # Ensure proper datetime type
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Create time groups
    freq = f'{minutes}T'
    df['time_group'] = df[timestamp_col].dt.floor(freq)

    # Calculate returns
    df['return'] = df.groupby('time_group')[price_col].pct_change()

    # Filter valid rows (need both delta and return)
    valid = df[[delta_col, 'return']].notna().all(axis=1)

    # Group and calculate regression features
    def calculate_slope_intercept(group):
        if len(group) < 2:
            return pd.Series({'intercept': np.nan, 'slope': np.nan})
        slope, intercept, *_ = linregress(group[delta_col], group['return'])
        return pd.Series({'intercept': intercept, 'slope': slope})

    features = (df[valid]
                .groupby('time_group')
                .apply(calculate_slope_intercept)
                .reset_index())

    # Add summary statistics
    stats = (df[valid]
             .groupby('time_group')
             .agg(
        start_price=(price_col, 'first'),
        end_price=(price_col, 'last'),
        mean_delta=(delta_col, 'mean'),
        delta_std=(delta_col, 'std'),
        n_ticks=(delta_col, 'count'),
        total_delta=('delta', 'sum')
    )
             .reset_index())

    # Merge features with statistics
    result = pd.merge(features, stats, on='time_group', how='left')

    # Calculate additional metrics
    result['abs_slope'] = result['slope'].abs()
    result['total_return'] = result['end_price'] / result['start_price'] - 1
    result['minutes'] = minutes

    # Clean up column order
    cols = ['time_group', 'minutes', 'intercept', 'slope',  'total_return', 'total_delta', 'abs_slope',
            'start_price', 'end_price',
            'mean_delta', 'delta_std', 'n_ticks', ]

    return result[cols]

# Example Usage:
if __name__ == "__main__":
    # Generate sample data (replace with your actual tick data)

    loader = sch()

    loader.resample_logic.update({'BidVolume': 'sum', 'AskVolume': 'sum'})

    data = loader.get_chart('NG_F')
    training_data = data.loc['2021':'2022'].copy()
    training_data['timestamp'] = training_data.index
    training_data['hour'] = training_data.index.hour
    training_data['minute'] = training_data.index.minute
    training_data['Value'] = training_data['Close']
    features_df = pd.DataFrame(
        columns=['MM_Intercept', 'MM_Hess', 'average_peak_curvature', 'average_peak_magnitude', 'minute', 'persecond'],
        index=pd.date_range(start=data.index.min(), end=data.index.max(), freq='1min'))

    pk_hess_res = []


    for name, date in training_data.groupby(training_data.index.date):
        for h_name, hour in date.groupby(date.hour):
            hess_dict = obtain_peak_hess(date, groupby='minute')
            features_df.loc[dt.datetime(name.year, name.month, name.day, h_name, hour.index.minute.min()): dt.datetime(name.year, name.month, name.day, h_name, hour.index.minute.max())] = collect_peak_parameters(hess_dict, group_by='minute')


