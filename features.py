import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta


def get_trading_days(start_year, end_year):
    # Define US market holidays
    trading_days = pd.Series(index=pd.date_range(start_year, end_year))

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


def apply_trading_days(df):
    df['TradingDay'] = df.index.map(lambda d: get_trading_days(d.year).get(d, None))
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
            .groupby(dts.time, sort=False)
            .rolling(days, closed='left')
            .mean()
            .reset_index(0, drop=True)  # drop the level with dts.time
        )

    return cum_volume / prev_mean(length)






