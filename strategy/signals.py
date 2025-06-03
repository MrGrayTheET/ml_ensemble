
import pandas as pd
import numpy as np


class SignalGen:

    def __init__(self, data, ohlc_cols = ['Open', 'High', 'Low', 'Close'], ATR_col = 'ATR'):
        self.data = data
        self.ATR = data['ATR']
        self.signal_map = {}

    def crossover(self, fast_col, slow_col, signal_name='crossover', stop_col=None, stop_type='crossover', map=False, map_id=None):
        data = self.data
        close = data.Close
        fast = data[fast_col]
        slow = data[slow_col]
        data[signal_name] = np.zeros(len(data))
        data.loc[(fast > slow),signal_name] = 1
        data.loc[(slow > fast),signal_name] = -1

        if stop_col is not None:
            self.sl(close, data, signal_col=signal_name, stop_col=stop_col, stop_type=stop_type, atr_col='ATR')

        self.data = data
        if map:
            self.signal_map.update({signal_name:map_id})


        return

    def hl_mean_reversion(self, high_col, low_col, signal_name='mean_reversion', stop_col=None,hc_col='Close',lc_col='Close', stop_type='ATR', map=False, map_id=None):
        data = self.data
        hc = data[hc_col]
        lc = data[lc_col]
        high = data[high_col]
        low = data[low_col]
        data[signal_name] = np.zeros(len(data))
        data.loc[(hc >= high), signal_name] = -1
        data.loc[(lc <= low), signal_name] = 1
        if stop_col is not None or stop_type=='ATR':
            self.sl(data.Close, data, signal_name, stop_col=stop_col, stop_type=stop_type)
        if map:
            self.signal_map.update({signal_name: map_id})

        self.data = data

        return

    def hl_breakout(self, high_col, low_col, signal_name='breakout', stop_col=None,hc_col='Close',lc_col='Close', stop_type='ATR', map=False, map_id=None):
        data = self.data
        hc = data[hc_col]
        lc = data[lc_col]
        high = data[high_col]
        low = data[low_col]
        data[signal_name] = np.zeros(len(data))
        data.loc[(hc >= high), signal_name] = 1
        data.loc[(lc <= low),signal_name] = -1

        if stop_col is not None:
            self.sl(data.Close, data, signal_name, stop_type=stop_type,stop_col=stop_col)

        if map:
            self.signal_map.update({signal_name: map_id})

        self.data = data
        return

    def percentile_signal(self, signal_col,rolling_lb=120, high_threshold=80, low_threshold=20, signal_name='percentile', stop_col=None, stop_type='crossover', map=False, map_id=None):
        data = self.data
        signal = data[signal_col]
        high_thresh = signal.rolling(rolling_lb).apply(lambda x:np.percentile(x, high_threshold))
        low_thresh = signal.rolling(rolling_lb).apply(lambda x:np.percentile(x, low_threshold))
        data[signal_name] = np.zeros(len(data))
        data.loc[(signal > high_thresh),signal_name] = 1
        data.loc[(signal < low_thresh),signal_name] = -1

        if stop_col is not None:
            self.sl(data.Close, data, signal_name, stop_col, stop_type)
        if map:
            self.signal_map.update({signal_name: map_id})

        self.data = data
        return



    def sl(self, close, data, signal_col, stop_col=['10_high','10_low'], stop_type='crossover', atr_col='ATR',atr_multi=1.0):

        signal = data[signal_col].astype(int)

        if type(stop_col) is not list:
            if stop_type.lower() == 'signal':
                stop = data[stop_col]
                data.loc[(stop == -1) & (signal == 1), signal_col] = 0
                data.loc[(signal == -1) & (stop == 1), signal_col] = 0

                return
            else:
                if stop_type.lower() == 'atr':
                    high_stop = data[stop_col] + data[atr_col] * atr_multi
                    low_stop = data[stop_col] - data[atr_col] * atr_multi
                else: high_stop, low_stop = data[stop_col], data[stop_col]
        else:
            high_stop = data[stop_col[0]]
            low_stop = data[stop_col[1]]
            if stop_type.lower() == 'atr':
                high_stop = high_stop + data[atr_col] * atr_multi
                low_stop = low_stop - data[atr_col] * atr_multi

        data.loc[(signal == 1) & (close < low_stop), signal_col] = 0
        data.loc[(signal == -1) & (close > high_stop), signal_col ] = 0

        return








