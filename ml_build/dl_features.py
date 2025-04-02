import  pywt
import pandas as pd
import numpy as np
import pandas_ta as ta
import seaborn as sns
import yfinance as yf
import datetime as dt
from pywt import waverec, wavedec
import seaborn as sns
from utils import mad


class mbuilder:

    def __init__(self, data, ohlc=['Open', 'High', 'Low', 'Close'], volumes=['Volume', 'BidVolume', 'AskVolume', 'NumberOfTrades']):

        self.data = data
        self.open_col = ohlc[0]
        self.high_col = ohlc[1]
        self.low_col = ohlc[2]
        self.close_col = ohlc[3]
        self.num_trades_col = volumes[3]
        self.closes = self.data[ohlc[3]] # Saving close for later use
        self.logic = {ohlc[0]: 'first',
                 ohlc[1]: 'max',
                 ohlc[2]: 'min',
                 ohlc[3]: 'last',
                 volumes[0]: 'sum',
                 volumes[1]: 'sum',
                 volumes[2]: 'sum',
                 volumes[3]: 'sum'}

    def create_targets(self, period=5, normalize=True):
        self.data[str(period)+'TARGET'] = self.closes.pct_change(-period) # Intended learning target
        return self.data

    def add_SMA(self, period=50):
        self.data[str(period)+'SMA'] = self.closes.rolling(period, min_periods=2).mean()
        self.data[str(period)+'SMA_norm'] = (self.data[self.close_col] - self.data[str(period)+'SMA'])/self.closes

        return self.data

    def add_daily_SMA(self, period=20, offset='-8h', normalize=False, drop_non_normals=True):
        daily_prices = self.data.resample('1d', offset=offset).apply(self.logic)
        self.data[str(period)+'d_'+'SMA'] = daily_prices[self.close_col].rolling(period, min_periods=2).mean()
        self.data = self.data.fillna(method='ffill')
        if normalize:
            self.data[str(period)+'d_'+'SMA'+'_norm'] = (self.data[self.close_col] - self.data[str(period)+'d_'+'SMA'])/self.closes
            if drop_non_normals:
                self.data = self.data.drop(columns=[str(period)+'d_'+'SMA'])
        return self.data

    def add_RSI(self,period=14):
        self.data[str(period)+'RSI'] = ta.rsi(close=self.data[self.close_col],length=period)/self.closes
        return self.data

    def volume_features(self, change_period=1, sma_period=5, volume_col='Volume'):
        self.data[str(change_period) + '_period_' + volume_col + '_pct_change'] = \
            self.data[volume_col].pct_change()

        self.data[volume_col+'_change_'+str(sma_period)+'_SMA'] =\
            self.data[str(change_period) + '_period_' + volume_col + '_pct_change'].rolling(sma_period).mean()
        return self.data

    def bid_ask_vol_features(self, bid_volume="BidVolume", ask_volume="AskVolume", volume="Volume",sma_len=5):
        df = self.data
        df['Bid_chg'] = df[bid_volume] - df[bid_volume].shift(1)
        df['Ask_chg'] = df[ask_volume] - df[ask_volume].shift(1)
        df['Bid%chg'] = (df[bid_volume]).pct_change(-1)
        df['Ask%chg'] = (df[ask_volume]).pct_change(-1)
        df['Delta'] = (df[ask_volume] - df[bid_volume])/df[volume]

        df['Delta_SMA'] = df['Delta'].rolling(sma_len).mean()
        df['avg_size'] = df[volume]/df[self.num_trades_col]

        return self.data

    def prev_day_ret(self, offset=dt.timedelta(hours=-7) ):
        pd_ret = self.closes.resample('1d', offset=offset).last().pct_change(-1)
        self.data['PD_RET'] = pd_ret
        self.data['PD_RET'] = self.data['PD_RET'].fillna(method='ffill')
        return self.data

    def prev_period_ret(self, n_periods=2):
        self.data['prev_ret'] = self.closes.pct_change(-n_periods)
        return self.data
    def supply_demand_zone(self,timeframe='1d', offset=dt.timedelta(hours=-9), normalize=True , drop_non_normals=True):
        pd_prices = self.data.resample(timeframe).apply(self.logic).shift(1)
        new_cols = ['D1', 'D2', 'S1', 'S2']
        high_close = np.where(pd_prices[self.open_col] < pd_prices[self.close_col])
        low_close = np.where(pd_prices[self.open_col] > pd_prices[self.close_col])
        self.data['D1'] = pd.concat([pd_prices[self.open_col].iloc[high_close], pd_prices[self.close_col].iloc[low_close]]).sort_index()
        self.data['D2'] = pd_prices[self.low_col]
        self.data['S1'] = pd.concat([pd_prices[self.close_col].iloc[high_close], pd_prices[self.open_col].iloc[low_close]]).sort_index()
        self.data['S2'] = pd_prices[self.high_col]
        self.data['H-L'] = (self.data[self.high_col] - self.data[self.low_col])/self.closes


        self.data = self.data.fillna(method='ffill')
        if normalize:
            for i in new_cols:
                self.data[str(i)+'_norm'] = ((self.closes) - self.data[i])/self.closes

        if drop_non_normals == True:
            self.data = self.data.drop(columns=new_cols)
        return self.data

    def range_bound(self, timeframe, offset=dt.timedelta(hours=-9)):
        daily_prices = self.data.resample(timeframe, offset=offset).apply(self.logic)
        week_prices = daily_prices.resample('1w').apply(self.logic, offset='12h')
        prev_week_prices = week_prices.shift(1)
        self.pw_prices = prev_week_prices
        daily_prices['PW_High'] = prev_week_prices['High']
        daily_prices['PW_Low'] = prev_week_prices['Low']
        pd_prices = daily_prices.shift(1)
        IB_days = (daily_prices[self.close_col] > pd_prices[self.low_col]) & (daily_prices[self.close_col] < pd_prices[self.high_col])
        within_pw_lvls = (daily_prices[self.high_col] < daily_prices['PW_High']) & (daily_prices[self.low_col] > daily_prices['PW_Low'])

        range_bound_days = daily_prices[IB_days | within_pw_lvls]

        return range_bound_days
    def opening_range(self, range_start='08:30:00', range_end_time='09:30:00', resample_len='1h',offset='30min',normalize=True, drop_non_normals=False):
        start_time = dt.datetime.strptime(range_start, '%H:%M:%S').time()
        range_end = dt.datetime.strptime(range_end_time, '%H:%M:%S').time()
        non_normals = ['OR High', 'OR Low']
        range_df = pd.DataFrame(columns=non_normals, index=self.data.loc[range_end].index)
        opening_range_data = self.data.loc[start_time:range_end].resample(resample_len, offset=offset).apply(self.logic)
        range_df['OR High'] = opening_range_data['High']
        range_df['OR Low'] = opening_range_data['Low']
        self.data['OR High'] = range_df['OR High']
        self.data['OR Low'] = range_df['OR Low']
        self.data['OR vol'] = (self.data['OR High'] - self.data['OR Low'])/self.data[self.close_col]
        self.data = self.data.fillna(method='ffill')
        if normalize:
            self.data['OR High_norm'] = (self.closes - self.data['OR High'])/self.closes
            self.data['OR Low_norm'] = (self.closes - self.data['OR Low'])/self.closes
            self.data = self.data.fillna(method='ffill')
            if drop_non_normals:
                self.data = self.data.drop(columns=non_normals)

        return self.data

    def dt_features(self, month=False, dayofweek=True, day=False, hour=True):

        if month:
            self.data['month'] = self.data.index.month
        if dayofweek:
            self.data['dayofweek'] = self.data.index.dayofweek
        if day:
            self.data['day'] = self.data.index.day
        if hour:
            self.data['hour'] = self.data.index.hour

        return self.data

    def make_time_specific(self, start_time='08:30:00', end_time='15:30:00'):
        start_time = dt.datetime.strptime(start_time, '%H:%M:%S').time()
        end_time = dt.datetime.strptime(end_time, '%H:%M:%S').time()
        self.data = self.data.loc[start_time:end_time]

    def wavelet_transform(self, wavelet, level):
        level = int(level)
        wavelet = wavelet
        coeff = wavedec(self.data[self.close_col], wavelet, level)
        sigma = mad(coeff[-level]) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(self.data[self.close_col])))
        coeff[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeff[1:])
        denoised = waverec(coeff, wavelet)
        if len(self.data) != len(denoised):
            denoised = denoised[:-1]
        self.data['denoised'] = denoised

        return denoised, coeff


    def add_new_series(self, data, columns, names):

        self.data[names] = data[columns].drop_duplicates()
        self.data = self.data.fillna(method='ffill')

        return self.data

class portfolio_features:

    def __init__(self, ticker_list, start_date='2011-01-01', close_col='Adj Close'):
        self.tickers = ticker_list
        self.data = yf.download(self.tickers, start=start_date).dropna()
        self.ret_df = pd.DataFrame(index=self.data.index, columns=list(zip(['weight', 'weighted_ret', 'ret'] * len(self.tickers), self.tickers)))
        self.ret_df.columns = pd.MultiIndex.from_tuples(list(zip(['weight', 'weighted_ret', 'ret'] * len(self.tickers), self.tickers)))

    def market_cap_weight(self, mkt_cap_dict,start_date, end_date):
        mkt_caps = np.fromiter(mkt_cap_dict.values(),dtype=int)
        mkt_cap_weights = mkt_caps/np.sum(mkt_caps)
        weight_dict = dict(zip(mkt_cap_dict.keys(), mkt_cap_weights))
        start = dt.datetime.strptime(start_date, '%Y-%m-%d')
        end = dt.datetime.strptime(end_date, '%Y-%m-%d')


        for i in self.tickers:
            self.ret_df.loc[start:end, ('weight', i)]= weight_dict[i]
            self.ret_df['ret', i ] = self.data['Adj Close', i ].pct_change()

        self.ret_df[list(zip(['weighted_ret'] * len(self.tickers), self.tickers))] = self.ret_df.weight * self.ret_df.ret
        self.ret_df.loc[start:end, 'portfolio_ret'] = self.ret_df['weighted_ret'].sum(axis=1)
        self.ret_df['cumulative_port'] =  (1+self.ret_df['portfolio_ret']).cumprod()
        return self.ret_df

    def equal_weight(self,start_date,end_date):
        weights = np.repeat(1, len(self.tickers))/len(self.tickers)
        weight_dict= dict(zip(self.tickers, weights))
        start = dt.datetime.strptime(start_date, '%Y-%m-%d')
        end = dt.datetime.strptime(end_date, '%Y-%m-%d')
        for i in self.tickers:
            self.ret_df.loc[start:end, ('weight', i)] = weight_dict[i]
            self.ret_df.loc['ret', i ] = self.data['Adj Close', i].pct_change()
        self.ret_df[list(zip(['weighted_ret'] * len(self.tickers), self.tickers))] = self.ret_df.weight * self.ret_df.ret
        self.ret_df['cumulative_port'] = (1+self.ret_df['portfolio_ret']).cumprod()
        return self.ret_df



def feature_corr_map(data):
    corr = data.corr()
    map = sns.heatmap(corr)
    return map
