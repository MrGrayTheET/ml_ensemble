from numpy import ndarray

from ml_build import torch_dl_2 as tdl
from ml_build.ml_model import ml_model as ml, xgb_params, gbr_params, rfr_params, xgb_clf_params
from ml_build.utils import clean_data, prune_non_builtin
from statsmodels.tsa.seasonal import STL, MSTL, seasonal_decompose
from tests import evaluate_seasonal as es
import matplotlib.pyplot as plt
from feature_engineering import (atr,
                                 tr,
                                 rvol,
                                 get_trading_days,
                                 kama,
                                 vol_scaled_returns,
                                 high_lows,
                                 calc_returns,
                                 calc_daily_vol,
                                 fft_decomp,
                                 stack_groups_vertically,
                                 ts_train_test_split,
                                 plot_two_axes,
                                 hawkes_process,
                                 vol_signal,
                                 macd
                                 )
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from linear_models import multivariate_regression
import ast
import yfinance as yf
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import datetime as dt
import os
from copy import deepcopy
VOL_THRESHOLD = 5


class TrendModel:

    def __init__(self, data, project_dir='F:\\ML\\', vol_scale=True):

        self.y_data = None
        self.x_data = None
        self._features = None
        self.ml = None
        self.model = None
        self.decomp_model = None
        self.data = data.copy(deep=True)
        self.multi_features = []
        self._target = None
        self.mdex = (type(self.data.columns) == pd.MultiIndex)

        if not os.path.isdir(project_dir): os.mkdir(project_dir)

        self.model_info = {
            'Seasonal': {'Features': []},
            'Trend': {'Features': []},
            'Volume': {'Features': []},
            'Volatility': {'Features': []},
            'Custom': {'Features': []},
            'Dir': project_dir,
            'Scaling': {'Vol': True},
            'ML': {'Training_End': 0}
        }
        self.defaults = {'xgb': xgb_params, 'gbr': gbr_params, 'rfr': rfr_params, 'xgbclf': xgb_clf_params,
                         'linear': None, 'linear_l2': None, 'linear_l1': None}

        if self.mdex:
            self.cols = lambda col: [(col, ticker) for ticker in self.data['Close'].columns]
            self.data[[('returns', i) for i in self.data.Close.columns]] = calc_returns(self.data.Close)
            self.data[[('daily_vol', i) for i in self.data.Close.columns]] = calc_daily_vol(self.data.returns)
        else:
            self.data['returns'] = calc_returns(self.data.Close)
            self.data['daily_vol'] = calc_daily_vol(self.data.returns)

        return

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, types):
        features = []
        for i in types:
            features = features + self.model_info[i]['Features']

        self._features = features

    @features.deleter
    def features(self):
        del self._features

    @property
    def target(self, offset=1):
        return self._target

    @target.setter
    def target(self, offset=1):
        self._target = 'target_returns'
        if self.model_info['Scaling']['Vol']:
            if self.mdex:
                self.data[self.cols('target_returns')] = vol_scaled_returns(calc_returns(self.data.Close, offset),
                                                                            self.data.daily_vol).shift(-1)
            else:
                self.data['target_returns'] = vol_scaled_returns(calc_returns(self.data.Close, offset),
                                                                 self.data.daily_vol).shift(-1)
        else:
            self.data['target_returns'] = calc_returns(self.data.Close, offset)

        return

    @target.deleter
    def target(self):
        del self._target

    def find_frequencies(self, detrend_data=False, detrend_method='wwma', detrend_period=25, n_frequencies=25):

        decomp = fft_decomp(self.data)

        if detrend_data:
            if detrend_method == 'wwma':
                decomp.wwm_detrend(detrend_period)
            else:
                decomp.kama_detrend(detrend_period)

        decomp.fft_data(detrended=detrend_data, top_n=n_frequencies)

        return decomp.fft_df

    def seasonal_decomp(self, periods, method="mstl", EMA=False, span=5, plot=False, model='multiplicative',
                        plot_model=False):

        if EMA:
            data = self.data.Close.ewm(span).mean()
        else:
            data = self.data.Close
        if method == 'mstl':
            self.decomp_model = MSTL(self.data.Close, periods=periods).fit()

        elif method == 'stl':
            self.decomp_model = STL(self.data.Close, period=periods).fit()
        else:
            self.decomp_model = seasonal_decompose(self.data.Close, model=model, period=periods)

        self.model_info['Seasonal'].update({'Periods': periods})

        if plot_model:
            self.plot_seasonal_model()

        return

    def plot_seasonal_model(self):
        return self.decomp_model.plot()

    def evaluate_seasonal(self):
        eval_res = es(self.decomp_model, type=type)

        self.model_info['Seasonal'].update({'Eval': eval_res})

        return self.model_info['Seasonal']['Eval']

    def normalize_ma(self, column):
        normalized = (self.data.Close - self.data[column]) / self.data.Close
        return normalized

    def normalize_indicator(self, column):
        normalized = self.data[column] / self.data.Close
        return normalized

    def volume_features(self, volume_ma=True, volume_ma_len=10, sub_ma=True, standardize_volume=True, RVOL=False,
                        RVOL_len=10, cum_rvol=False, cum_rvol_groupby='month'):
        features = []
        if volume_ma:
            self.data[f'volume_{volume_ma_len}'] = self.data.Volume.rolling(volume_ma_len).mean()
            features += [f'volume_{volume_ma_len}']

            if sub_ma:
                self.data['volume_ma'] = self.data.Volume - self.data[f'volume_{volume_ma_len}']
                features += ['volume_ma']

        if standardize_volume:
            self.data['std_volume'] = self.data.Volume / self.data.Volume.rolling(60).std()
            features += ['std_volume']

        self.model_info['Volume'].update({'Features': features})

        return self.data[features]

        return
    def volatility_indicators(self, ATR=False, atr_len=7, TR=False, tr_atr=False, tr_atr_len=12, tr_atr_mean=1,
                              hawkes_vol=False, log_range=False, hawkes_atr_len=20, kappa=0.1, hawkes_binary=False,
                              signal_lb=21, normalize=True, rolling_range=False,
                              rr_hawkes_period='1w', rolling_range_widths=[5, 20], normalize_range=False,
                              normalize_mean_len=61, rr_hawkes=False,rr_mean=52, rr_hawkes_kappa=0.1, rr_signal_lb=21):
        features = []

        if ATR:
            self.data['ATR'] = atr(self.data, length=atr_len, normalized=normalize)
            features += ['ATR']
        if TR:
            self.data['TR'] = tr(self.data, normalized=normalize)
            features += ['TR']
        if tr_atr:
            self.data['tr_ratio'] = (tr(self.data, normalized=False) / atr(self.data, length=tr_atr_len,
                                                                           normalized=False)).rolling(tr_atr_mean).mean()
            features += ['tr_ratio']

        if hawkes_vol:
            candle_range = (self.data.High - self.data.Low)
            norm_atr = atr(self.data, length=hawkes_atr_len, normalized=False)
            norm_range = candle_range / norm_atr
            if log_range:
                norm_range = np.log(norm_range)
            h_vol = hawkes_process(norm_range, kappa)

            if hawkes_binary:
                self.data['hawkes_signal'] = vol_signal(self.data.Close, h_vol, signal_lb)
            else:
                self.data['hawkes_signal'] = h_vol

            features.append('hawkes_signal')


        if rolling_range:
            for i in rolling_range_widths:
                hi = self.data.High.rolling(i).max()
                lo = self.data.Low.rolling(i).min()
                range_ = (hi-lo)
                if normalize_range:
                    mean = range_.rolling(normalize_mean_len).mean()
                    self.data[f'norm_range_{i}x'] = range_/mean
                    features.append(f'norm_range_{i}x')
                else:
                    self.data[f'range_{i}x'] = range_/self.data.Close
                    features.append(f'range_{i}x')
        if rr_hawkes:
            weekly_data = self.data.resample(rr_hawkes_period).ohlc()
            weekly_ohlc = pd.DataFrame()
            weekly_ohlc[['Open','High','Low','Close']] = weekly_data[[('Open','open'), ('High', 'high'), ('Low', 'low'), ('Close', 'close')]]
            weekly_atr = atr(weekly_ohlc, length=rr_mean)
            rs_norm_ranges = (weekly_ohlc.High - weekly_ohlc.Low)/weekly_atr
            weekly_ohlc['rs_norm_range'] = rs_norm_ranges
            wh = hawkes_process(rs_norm_ranges, rr_hawkes_kappa)

            self.data['wh_signal'] = np.zeros(len(self.data))

            if wh.index[-1] not in self.data.index: wh.index = wh.index + dt.timedelta(days=1)


            if hawkes_binary:
                wh_signal = pd.Series(vol_signal(weekly_ohlc.Close, wh, rr_signal_lb),
                                      index = wh.index)

                weekly_ohlc['wh_signal'] = wh_signal

                self.data.wh_signal = wh_signal

            else:
                self.data['wh_signal'] = wh

            self.data.wh_signal = self.data.wh_signal.ffill()

            features.append('wh_signal')

        self.model_info['Volatility'].update({'Features': features})

        return self.data[features]

    def trend_features(self,
                       trend=False, trend_window=1,
                       MACD=False, MACD_params=(12, 26), MACD_cols=['MACD', 'Signal'],
                       KAMA=False, kama_params=[(5, 2, 32)],
                       SMAs=False, sma_lens=None,
                       momentum=False, momentum_lens=None,
                       hls=False, hl_lens=[5, 10, 20],
                       BBands=False, bband_window=20, bband_width=2,
                       normalize_features=True):

        if sma_lens is None:
            sma_lens = [20, 50, 100]

        if momentum_lens is None:
            momentum_lens = [21, 42, 63]

        features = []

        if KAMA:
            for i in kama_params:
                kama_ma = kama(self.data.Close, *i)
                if normalize_features:
                    self.data[f'KAMA_{i[0]}x'] = (self.data.Close - kama_ma) / self.data.Close
                    features.append(f'KAMA_{i[0]}x')
                else:
                    self.data[f'KAMA_{i[0]}'] = kama_ma
                    features.append(f'KAMA_{i[0]}')

        if trend:
            self.data['trend'] = self.decomp_model.trend.values
            self.data['trend_roc'] = self.data.trend.pct_change(trend_window)

            if normalize_features:
                self.data['trend_x_close'] = self.normalize_ma('trend')

                features = features + ['trend_roc', 'trend_x_close']
            else:
                features = features + ['trend', 'trend_roc']

        if MACD:
            self.data[MACD_cols] = macd(self.data, short_term=MACD_params[0], long_term=MACD_params[1],
                                        ret_cols=MACD_cols)

            for i in MACD_cols: features.append(i)

        if SMAs:
            for i in sma_lens:
                self.data[f'SMA_{i}'] = self.data.Close.rolling(i).mean()
                if normalize_features:
                    self.data[f'SMA_{i}x'] = (self.data.Close - self.data[f'SMA_{i}']) / self.data.Close
                    features.append(f'SMA_{i}x')
                else:
                    features.append(f'SMA_{i}')

        if momentum:
            for i in momentum_lens:
                if normalize_features:
                    self.data[f'mom_{i}'] = self.normalized_returns(i)

                else:
                    self.data[f'mom_{i}'] = self.data.Close.diff(i)
                features.append(f'mom_{i}')

        if BBands:
            self.data['bb_ma'] = self.data.Close.rolling(bband_window).mean()
            self.data['bb_upper'] = self.data.bb_ma + (bband_width * self.data.Close.rolling(bband_window).std())
            self.data['bb_lower'] = self.data.bb_ma - (bband_width * self.data.Close.rolling(bband_window).std())

            if normalize_features:
                for i in ['bb_ma', 'bb_upper', 'bb_lower']:
                    self.data[i + '_x'] = self.normalize_ma(i)
                    features.append(i + '_x')
            else:
                features = features + ['bb_ma', 'bb_upper', 'bb_lower']

        if hls:
            df = high_lows(self.data, hl_lens, normalize_features)
            df.columns = df.columns.str.replace('_norm', '_x')
            self.data[df.columns] = df

            features += df.columns.to_list()

        self.model_info['Trend'].update({'Features': features})

        return self.model_info['ML']

    def feature_eval(self, target_period, feat_types=['Trend'], cv=10):

        self.features = feat_types
        self.target = target_period

        reg = multivariate_regression(self.data, self.features, self._target, penalty='l1', alpha=cv)
        feat_df = pd.DataFrame(reg['coefs'], index=self.features, columns=['coefs'])

        returned_features = feat_df[feat_df['coefs'] != 0].to_dict()

        return returned_features

    def normalized_returns(self, offset):
        return (
                (calc_returns(self.data['Close'], offset) / self.data.daily_vol)
                / np.sqrt(offset)
        )

    def train_model(self, target_period=10, feat_types=['Trend'], lasso_selection=False, lasso_cv=5, method='xgb',
                    params=None, train_test_size=0.8,
                    save_file='model_eval.csv', plot=True, high_percentile=65, low_percentile=30, linear_alpha=0.8,
                    save_model=False):

        self.target = target_period

        self.features = feat_types
        if lasso_selection:
            self._features = [*self.feature_eval(8, feat_types=['Trend'], cv=lasso_cv)['coefs'].keys()]

        if params is None: params = self.defaults[method]

        data = self.data[self._features + [self.target]].copy(deep=True).ffill().dropna()
        self.ml = ml(data, self.features, self.target, train_test_size=train_test_size)

        self.model_info['ML'].update({'Train Length': len(self.ml.y_train),
                                      'Test_length': len(self.ml.y_test)})

        if 'linear' not in method:
            self.model = self.train(high_percentile, low_percentile, method, params, plot, save_file)
        else:
            if 'l2' in method:
                res = multivariate_regression(data, self.features, self.target,
                                              penalty='l2',
                                              alpha=linear_alpha,
                                              train_split=True,
                                              train_size=train_test_size
                                              )
            elif 'cv' in method:
                res = multivariate_regression(data, self.features, self.target,
                                              penalty='cv',
                                              alpha=linear_alpha,
                                              train_size=train_test_size,
                                              train_split=True)
            elif 'l1' in method:
                #TBD Add scaler?
                res = multivariate_regression(data, self.features, self.target,
                                              penalty='l1',
                                              alpha=linear_alpha,
                                              train_split=True,
                                              train_size=train_test_size)
            else:
                res = multivariate_regression(data, self.features, self.target,
                                              train_split=True,
                                              train_size=train_test_size)

            res['coefs'] = dict(zip(self.features, res['coefs']))

            self.model = deepcopy(res['model'])
            res

            self.model_info['ML'].update(res)

        print('Model successfully trained!')
        self.data['predictions'] = self.model.predict(self.data[self._features])

        return self.model

    def plot_predictions(self, prediction_col='predictions'):

        true_values = self.ml.y_test.values
        predictions = self.data.predictions.iloc[-self.model_info['ML']['Test_length']:]
        rmse = self.model_info['rmse']
        r2 = self.model_info['r2']

        # Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(true_values, predictions, alpha=0.6)
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--',
                 label='Perfect Prediction')

        # Labels and styling
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction vs True Value')

        # Add metrics as text
        plt.text(0.05, 0.95, f'RMSE = {rmse:.2f}\n$R^2$ = {r2:.2f}',
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        return None

    def train(self, high_percentile, low_percentile, method, params, plot, save_file):
        if method == 'xgb':
            self.model = self.ml.xgb_model(params, evaluate=True, eval_log=self.model_info['Dir'] + save_file,
                                           plot_pred=plot)
        if method == 'gbr':
            self.model = self.ml.tree_model(params, gbr=True, evaluate=True,
                                            eval_log=self.model_info['Dir'] + save_file,
                                            plot_pred=plot)
        if method == 'xgbclf':
            self.model = self.ml.xgb_clf(high_p=high_percentile, low_p=low_percentile, parameter_dict=params)

        self.model_info['ML'].update({'Type': method, 'Hyperparams': params, 'Features':self._features,})

        self.model_info.update({'Features': self._features})

        print(self.model_info)
        return self.model

    def save_model(self, model_name):
        model_dir = self.model_info['Dir'] + model_name + '\\'

        print('Saving to ' + model_dir)

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        if self.ml.model != None:
            self.ml.save_model(file_name=model_name + '_model', save_fp=model_dir)
        elif self.model != None:
            self.ml.model = self.model
            self.ml.save_model(file_name=model_name + '_model', save_fp=model_dir)
        else:
            print('No Model Found')
            return False

        print('Model Saved to ' + model_dir)

        model_params = prune_non_builtin(self.model_info)

        with open(model_dir + 'model_params', 'w') as f:
            f.write(str(model_params))

        training_data = self.data.to_csv(model_dir + 'training_data.csv')

        print('Model Data and parameters saved to ' + model_dir)
        return True

    def load_model(self, model_name, dt_index=False, feature_types=['Trend']):
        model_dir = self.model_info['Dir'] + model_name + '\\'

        self.ml = ml(self.data, [], 'Close')

        print('Loading from ' + model_dir)

        self.model = self.ml.load_model(model_dir + model_name + '_model')

        with open(model_dir + 'model_params', 'r') as f:
            str_data = f.read()
            import re
            s_clean = re.sub(r"'eval_datetime': Timestamp\([^)]+\),? ?", "", str_data)
            s_clean = re.sub(r"'model': Ridge\([^)]*\),?\s*", "", s_clean)

        model_info = ast.literal_eval(s_clean)

        training_data = pd.read_csv(model_dir + 'training_data.csv', index_col=[0])
        if dt_index:
            training_data.index = pd.DatetimeIndex(training_data.index)

        print('Training Data loaded')

        self.data = training_data
        self.model_info = model_info
        try:
            self._features = model_info['Features']

        except:

            try:self.model.feature_names_in_.tolist()

            except: self.features = feature_types


            else: self._features = self.model.feature_names_in_.tolist()
        else:
            self._features = model_info['Features']

        self.x_data = self.data[self._features]
        self.y_data = self.data['target_returns']

        return print('Model Successfully loaded from ' + model_dir)

    def custom_feature(self, series: pd.Series, name: str):
        if type(series.index) == pd.DatetimeIndex:
            self.data[name] = series.loc[self.data.index[0]:self.data.index[-1]]
            self.data[name] = self.data[name].ffill().bfill().dropna()
            self.model_info['Custom']['Features'] += [name]
        else:
            if len(series.index) != len(self.data.index):
                return 'Cannot reindex series of different lengths'
            else:
                self.data[name] = series
                return


class MultiAssetTrend(TrendModel):

    def __init__(self, tickers=None, data=None, project_dir='F:\\ML\\', download_start='2000-01-01',
                 download_end='2025-04-02'):

        self.training_df = None
        if tickers is None:
            if data is None:
                print('No Data, please feed me data or a ticker list')
            else:
                if type(data.columns) == pd.MultiIndex:
                    super().__init__(data, project_dir, vol_scale=True)
                    self.tickers = data['Close'].columns
                elif len(data.columns) > 2:
                    super().__init__(data, project_dir)
                    self.tickers = data.columns

        else:
            data = yf.download(tickers, start=download_start, end=download_end)
            super().__init__(data, project_dir, vol_scale=True)
            self.tickers = tickers

        return

    def multi_trend(self, SMAs=False, sma_lens=None, momentum=False, momentum_lens=None, BBands=False, bband_window=20,
                    bband_width=2, normalize_features=True, **kwargs):
        if sma_lens is None:
            sma_lens = [20, 50, 100]

        if momentum_lens is None:
            momentum_lens = [21, 42, 63]

        features = []

        if SMAs:
            for i in sma_lens:
                self.data[self.cols(f'SMA_{i}')] = self.data.Close.rolling(i).mean()
                if normalize_features:
                    self.data[self.cols(f'SMA_{i}x')] = (self.data.Close - self.data[f'SMA_{i}']) / self.data.Close
                    features.append(f'SMA_{i}x')
                else:
                    features.append(f'SMA_{i}')

        if momentum:
            for i in momentum_lens:
                if normalize_features:
                    self.data[self.cols(f'mom_{i}')] = self.normalized_returns(i)

                else:
                    self.data[self.cols(f'mom_{i}')] = self.data.Close.pct_change(i)
                features.append(f'mom_{i}')

        if BBands:
            self.data[self.cols('bb_ma')] = self.data.Close.rolling(bband_window).mean()
            self.data[self.cols('bb_upper')] = self.data.bb_ma + (
                    bband_width * self.data.Close.rolling(bband_window).std())
            self.data[self.cols('bb_lower')] = self.data.bb_ma - (
                    bband_width * self.data.Close.rolling(bband_window).std())

            if normalize_features:
                for i in ['bb_ma', 'bb_upper', 'bb_lower']:
                    self.data[self.cols(i + '_x')] = self.normalize_ma(i)
                    features.append(i + '_x')

            else:
                features = features + ['bb_ma', 'bb_upper', 'bb_lower']

        self.model_info['Trend'].update({'Features': features})
        print('Selected Trend Features : ')
        print(self.model_info['Trend']['Features'])

        return

    def train_multi(self, target_period=10, feature_types=['Trend'], method='xgbclf', high_percentile=75,
                    low_percentile=25, params=None, plot=False, save_file='xgb_clf_multi', keep_ID=False):

        self.features = feature_types
        self.target = target_period

        self.training_df = stack_groups_vertically(self.data[self._features + [self._target]]).ffill().dropna()
        self.training_df = self.training_df.drop_duplicates()

        if keep_ID:
            self.multi_features = self._features + ['Group_ID']
        else:
            self.multi_features = self._features

        train, test = ts_train_test_split(self.training_df, test_size=0.2, groups=['Group_ID'])
        x_train, y_train = train[self.multi_features], train[self._target]
        x_test, y_test = test[self.multi_features], test[self._target]
        self.ml = ml(self.training_df, self.multi_features, self._target)
        print(f'X Shape:{x_train.shape}')
        self.ml.x_train, self.ml.y_train = x_train, y_train
        self.ml.x_test, self.ml.y_test = x_test, y_test

        if params is None: params = self.defaults[method]

        self.model_info['ML'].update({'Train Length': len(self.ml.y_train),
                                      'Test_length': len(self.ml.y_test)})
        print('Training Data:')
        print(self.ml.x_train.head())

        model = self.train(high_percentile, low_percentile, method, params=params, plot=False, save_file=save_file)

        print('Model successfully trained!')
        print(self.model_info)
        self.training_df['predictions'] = self.model.predict(self.training_df[self.multi_features])

        return model

    def train(self, high_percentile, low_percentile, method, params, plot, save_file):
        if method == 'xgb':
            self.model = self.ml.xgb_model(params, evaluate=True, eval_log=self.model_info['Dir'] + save_file,
                                           plot_pred=plot)
        if method == 'gbr':
            self.model = self.ml.tree_model(params, gbr=True, evaluate=True,
                                            eval_log=self.model_info['Dir'] + save_file,
                                            plot_pred=plot)
        if method == 'xgbclf':
            self.model = self.ml.xgb_clf(high_p=high_percentile, low_p=low_percentile, parameter_dict=params)

        self.model_info['ML'].update({'Type': method, 'Hyperparams': params})

        return self.model


class dl_model(TrendModel):

    def __init__(self, data, project_dir='F:\\ML\\Seasonal\\'):
        super().__init__(data, project_dir)
        self.train_y, self.train_x, self.test_x, self.test_y = [None] * 4
        self.optimizer = None
        self.train_x = None
        self.test_preds = None
        self.loss_func = None
        self.training_preds = None
        self.scaler = None
        self._target = 'Close'
        self.model_info.update({'DL': []})
        self.loss = None

        return

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, types):
        features = []
        for i in types:
            features = features + self.model_info[i]['Features']
            self._features = features

    def torch_model(self, feat_types=['Volume', 'Seasonal', 'Trend'],
                    periods_in=20, periods_out=5, log_file='tdl_model.csv',
                    n_epochs=200, lr=0.0001, hidden_size=2, n_layers=1, loss_func=None, lstm=tdl.lstm,
                    scale_x=True, x_scale_type='standard', scale_y=True, y_scale_type='standard'):

        if loss_func is None:
            self.loss_func = nn.MSELoss()
        else:
            self.loss_func = loss_func()

        self.features = feat_types
        data = self.data[self._features + [self._target]].ffill().copy()
        input_size = len(self._features)

        data = data[~(data.isnull() | data.isna())].dropna(axis=0)

        if scale_y:
            [self.train_x, self.train_y], [self.test_x, self.test_y], self.scaler = clean_data(data, self._features,

                                                                                               self._target,
                                                                                               sequence=True,
                                                                                               periods_in=periods_in,
                                                                                               periods_out=periods_out,
                                                                                               scale_x=scale_x,
                                                                                               x_scale_type=x_scale_type,
                                                                                               to_tensor=True,
                                                                                               scale_y=scale_y,
                                                                                               return_y_scaler=scale_y,
                                                                                               y_scale_type=y_scale_type)
        else:
            [self.train_x, self.train_y], [self.test_x, self.test_y] = clean_data(data, self._features,

                                                                                  self._target,
                                                                                  sequence=True,
                                                                                  periods_in=periods_in,
                                                                                  periods_out=periods_out,
                                                                                  scale_x=scale_x,
                                                                                  x_scale_type=x_scale_type,
                                                                                  to_tensor=True,
                                                                                  scale_y=scale_y,
                                                                                  return_y_scaler=scale_y,
                                                                                  y_scale_type=y_scale_type)

        self.train_x = torch.reshape(self.train_x, (self.train_x.shape[0], periods_in, self.train_x.shape[2]))
        self.test_x = torch.reshape(self.test_x, (self.test_x.shape[0], periods_in, self.test_x.shape[2]))

        if len(self.test_y.shape) > 2:
            self.test_y = self.test_y.mean(dim=2)
            self.train_y = self.train_y.mean(dim=2)

        self.model = lstm(num_classes=periods_out, input_size=input_size, hidden_size=hidden_size,
                          num_layers=n_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.training_preds = self.model(self.train_x)
        self.test_preds = self.model(self.test_x)

        eval_results = self.eval_model()

        return eval_results

    def torch_loop(self, n_epochs=200):
        tdl.training_loop(lstm=self.model, optimizer=self.optimizer, n_epochs=n_epochs, loss_func=self.loss_func,
                          X_train=self.train_x, y_train=self.train_y, X_test=self.test_x, y_test=self.test_y)

        return

    def eval_model(self):
        mse_loss = torch.mean((self.test_preds - self.test_y) ** 2)
        rmse_loss = mse_loss.sqrt()
        std = self.test_y.std()
        mean = self.test_y.mean()

        eval = {'MSE': mse_loss.item(),
                'RMSE': rmse_loss.item(),
                'RMSE/SD': (rmse_loss / std).item(),
                'RMSE/MEAN': (rmse_loss / mean).item()}

        print(eval)

        return eval

    def detach(self, train_data=False, test_data=True):
        ret = {'Train': {},
               'Test': {}
               }
        if train_data:
            ret['Train'].update({'X': self.train_x.detach().cpu().numpy(),
                                 'y': self.train_y.detach().cpu().numpy(),
                                 'predict': self.training_preds.detach().cpu().numpy()})
        if test_data:
            ret['Test'].update({'X': self.test_x.detach().cpu().numpy(),
                                'y': self.test_y.detach().cpu().numpy(),
                                'predict': self.test_preds.detach().cpu().numpy()}
                               )

        return ret

    def returns_plot(self, test_only=True):
        if test_only:
            test_data = self.detach()['Test']
            plt.scatter(test_data['y'], test_data['predict'])
            plt.show()

        return
