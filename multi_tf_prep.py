import yfinance as yf
import datetime as dt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sc_loader import sierra_charts as sch
from feature_engineering import (vsa,
                                 vol_signal, normalize_hls, calc_daily_vol, calc_returns,
                                 hawkes_process, atr, log_returns, kama,
                                 rvol, high_lows, cum_rvol, get_range, rsv)
from ml_build.ml_model import ml_model as ml, xgb_params, gbr_params, rfr_params, xgb_clf_params

from linear_models import multivariate_regression

if os.name == 'nt':
    sc_cfg = 'data_config.toml'
else:
    sc_cfg = '/content/drive/MyDrive/utils/sc_config.toml'

sc = sch(sc_cfg)


class MultiTfModel:

    def __init__(self, data, timeframes=['5min', '10min', '1h', '4h', '1d'], train_test_ratio=0.8,
                 project_dir="F:\\ML\\multi_tf\\"):
        self.features = None
        self.model = None
        if not os.path.isdir(project_dir): os.mkdir(project_dir)

        self.training_df = None
        self.har_df = None
        self.data = data
        self.dfs_dict = {}
        self.feats_dict = {}
        self.train_ratio = train_test_ratio
        for i in timeframes:
            self.dfs_dict.update({i: data.resample(i).apply(sc.resample_logic).ffill().dropna()})
            self.feats_dict.update({i: {'Volatility': [], 'Volume': [], 'Trend': [], 'Temporal': [], 'Additional': []}})

            self.dfs_dict[i]['returns'] = log_returns(data.Close)
        self.model_info = {'Dir': project_dir}

    def volatility_signals(self, timeframe, ATR=False, ATR_length=14, normalize_atr=True,
                           hawkes=False, hawkes_mean=168, kappa=0.1, hawkes_signal_lb=21, hawkes_binary_signal=True,
                           normalize=True, range_width=False, range_length=20, normalize_range_len=63,
                           realized_vol=True, use_semivariance=False, keep_existing_features=False, ):
        if keep_existing_features:
            features = self.feats_dict[timeframe]['Volume']
        else:
            features = []

        data = self.dfs_dict[timeframe]

        if ATR:
            data['ATR'] = atr(data, length=ATR_length, normalized=normalize_atr)
            features.append('ATR')

        if hawkes:
            ranges = data.High - data.Low
            norm_mean = atr(data, length=hawkes_mean)
            vol_data = ranges / norm_mean
            data['hp'] = hawkes_process(vol_data, kappa=kappa)
            if hawkes_binary_signal:
                data['hp_signal'] = vol_signal(data.Close, data.hp, lookback=hawkes_signal_lb)
                features.append('hp_signal')
            else:
                features.append('hp')

        if range_width:
            highs = data.High.rolling(range_length).max()
            lows = data.Low.rolling(range_length).max()
            if normalize:
                data[f'{range_length}_range'] = (highs - lows) / (highs - lows).rolling(normalize_range_len).mean()
                data[f'{range_length}_high_x'] = (highs - data.OHLCAvg) / data.Close.rolling(normalize_range_len).mean()
                data[f'{range_length}_low_x'] = (data.OHLCAvg - lows) / data.Close.rolling(normalize_range_len).mean()
                features += [f'{range_length}_{i}' for i in ['range', 'high_x', 'low_x']]
            else:
                data[f'{range_length}_highs'] = highs
                data[f'{range_length}_lows'] = lows
                features += [f'{range_length}_{i}' for i in ['highs', 'lows']]
        if realized_vol:
            if use_semivariance:
                data[['rsv_neg', 'rsv_pos']] = rsv(data.returns)
                self.dfs_dict['1d'][[f'{timeframe}_rsv_neg', f'{timeframe}_rsv_pos']] = data[['rsv_neg', 'rsv_pos']]

            else:
                data['rv'] = data.returns.groupby(data.index.date).apply(lambda x: np.sqrt(np.sum(x ** 2)))
                self.dfs_dict['1d'][f'{timeframe}_rv'] = data['rv']

            data.rv = data.rv.ffill().dropna()
            features.append('rv')

        self.dfs_dict.update({timeframe: data})
        self.feats_dict[timeframe].update({'Volatility': features})

        return self.dfs_dict[timeframe][features]

    def volume_features(self, timeframe, relative_volume=True, rvol_days=10, cum_rvol=True, delta=False,
                        delta_as_pct=True, VSA=False, vsa_col='Volume', vsa_lb=30,
                        hawkes_vol=False, normalize_vol=False, normalizer_len=30, keep_existing_features=False):
        if keep_existing_features:
            features = self.feats_dict[timeframe]['Volume']
        else:
            features = []

        data = self.dfs_dict[timeframe]

        if relative_volume:
            data[f'rvol_{rvol_days}'] = rvol(data, by='datetime')
            features.append(f'rvol_{rvol_days}')
            data.drop('rolling_mean', axis=1, inplace=True)
        if delta:
            delta_ = data.AskVolume - data.BidVolume
            if delta_as_pct:
                data['delta'] = delta_ / data.Volume
            else:
                data['delta'] = delta_
            features.append('delta')
        if VSA:
            data['VSA'] = vsa(data, vsa_col, vsa_lb)
            features.append('VSA')

        if normalize_vol:
            median = data.Volume.rolling(normalizer_len).median()
            data['norm_vol'] = data.Volume / median

        self.feats_dict[timeframe].update({'Volume': features})
        self.dfs_dict[timeframe] = data

        return data[features]

    def temporal_features(self, timeframe='10min', ranges=True, range_starts=[], range_ends=[], range_rv=True,
                          shift_date=False, keep_existing_features=False):
        data = self.dfs_dict[timeframe]
        if not keep_existing_features:
            features = []
        else:
            features = self.feats_dict[timeframe]['Temporal']

        temp_df = pd.DataFrame(index=pd.Series(data.loc.index.date).unique(), columns=[])

        for i in range(len(range_starts)):

            t_start = range_starts[i]
            t_end = range_ends[i]
            hls = get_range(data, t_start, t_end)
            hls['Close'] = data.loc[t_end].Close
            hls['Open'].loc[t_end] = data['Open'].loc[t_start]
            data[f'{t_end.hour}_H-L'] = (hls['Highs'] - hls['Lows']) / hls['Close']
            data[f'{t_end.hour}_H-C'] = (hls['Highs'] - hls['Close']) / hls['Close']
            data[f'{t_end.hour}_C-L'] = (hls.Close - hls.Lows) / hls.Close
            data[f'{t_end.hour}_Volume'] = data.Volume[t_start:t_end].sum()
            if range_rv:
                data[f'{t_end.hour}_rv'] = np.nan
                rvs = data.returns.loc[t_start:t_end].groupby(data.returns.loc[t_start:t_end].index.date).apply(
                    lambda x: np.sqrt(np.sum(x ** 2)))
                data.loc[t_end, f'{t_end.hour}_rv'] = rvs.values

            features += [f'{t_end.hour}_{col}' for col in ['H-L', 'H-C', 'C-L', 'Volume']]

        self.dfs_dict.update({timeframe: data})
        self.feats_dict[timeframe]['Temporal'] = features

        return

    def trend_indicators(self, timeframe, keep_existing_features=False,
                         SMAs=False, sma_lens=[20, 50, 200], KAMAs=False, kama_params=[(20, 2, 32)],
                         momentum=True, momentum_periods=[24],
                         normalize_features=True):

        if keep_existing_features:
            features = self.dfs_dict[timeframe]['Trend']
        else:
            features = []

        data = self.dfs_dict[timeframe]

        if momentum:
            for i in momentum_periods:
                data[f'momentum_{i}'] = np.log(data.Close) - np.log(data.Close.shift(i))
                features.append(f'momentum_{i}')
        if SMAs:
            for i in sma_lens:
                data[f'SMA_{i}'] = data.Close.rolling(i).mean()
                if normalize_features:
                    data[f'SMA_{i}_x'] = (data.Close - data[f'SMA_{i}']) / data.Close
                    features.append(f'SMA_{i}_x')
                else:
                    features.append(f'SMA_{i}')
        if KAMAs:
            for i in kama_params:
                kama_ma = kama(data.Close, *i)
                if normalize_features:
                    data[f'kama_{i[0]}_x'] = (data.Close - kama_ma) / data.Close
                    features.append(f'kama_{i[0]}_x')
                else:
                    data[f'kama_{i[0]}'] = kama_ma
                    features.append(f'kama_{i[0]}')

        self.dfs_dict[timeframe] = data
        self.feats_dict[timeframe]['Trend'] = features

        return

    def train_HAR(self, scale_data=False, split_data=True, training_size=0.8,
                  regularize=False, cv=3, save_best_only=True, target_horizon=1, add_feature=True):

        rvs = self.dfs_dict['1d'].filter(like='rv').dropna()

        if scale_data:
            self.rv_scaler = StandardScaler()
            rvs = pd.DataFrame(self.rv_scaler.fit_transform(rvs), columns=rvs.columns, index=rvs.index)

        self.har_df = pd.DataFrame(
            columns=pd.MultiIndex.from_product([[col[:-3] for col in rvs.columns], ['rv_d', 'rv_w', 'rv_m', 'rv_t']]),
            index=rvs.index)

        evals = {}
        rmse = []
        best_rmse = 1.5

        for col in rvs.columns:
            rv_d = rvs[col]
            column = col[:-3]

            self.har_df[column] = pd.DataFrame({'rv_d': rv_d.shift(0),
                                                'rv_w': rv_d.shift(0).rolling(5).mean(),
                                                'rv_m': rv_d.shift(0).rolling(22).mean(),
                                                'rv_t': rv_d.shift(-target_horizon)}, index=rvs.index).dropna()
            har_df = self.har_df[column].copy(deep=True)

            df = har_df.ffill().dropna()
            pred_idx = df.index

            # split_ix = int(self.train_ratio * len(rvs))
            # self.models_info.update({'test_idx': split_ix})
            # train_x, test_x = df.iloc[:split_ix, :-1].to_numpy(), df.iloc[split_ix:, :-1].to_numpy()
            # train_y, test_y = df.iloc[:split_ix, -1:].to_numpy(), df.iloc[split_ix:, -1:].to_numpy()
            # regression = sm.OLS(train_y, train_x)

            if regularize:
                results = multivariate_regression(df, X_cols=['rv_d', 'rv_w', 'rv_m'], y_col='rv_t',
                                                  penalty='cv', cv=cv, scaler=self.rv_scaler, train_split=split_data,
                                                  train_size=training_size)

            else:
                results = multivariate_regression(df, X_cols=['rv_d', 'rv_w', 'rv_m'], y_col='rv_t',
                                                  train_split=split_data, train_size=training_size,
                                                  penalty=None)

            df['predictions'] = results['model'].predict(df.drop('rv_t', axis=1))
            df


            if save_best_only:
                rmse = results['rmse']
                tf = column
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_score = tf
                else:
                    del results['model']
            else:
                self.har_df[(column, 'predictions')] = np.nan
                self.har_df.loc[pred_idx, (column, 'predictions')] = df['predictions']

            evals.update({col:results})
        if save_best_only:
            self.har_df[(best_rmse, 'predictions')] = np.nan
            self.har_df.loc[pred_idx, (best_score, 'predictions')] = df['predictions']

            if add_feature:
                self.dfs_dict['1d'].loc[pred_idx]['HAR_preds'] = df['predictions']
                self.feats_dict['1d']['Additional'].append('HAR_preds')




        return evals

    def transfer_feature(self, from_tf, to_tf, feature_name):
        x = self.dfs_dict[from_tf][feature_name]
        y_df = self.dfs_dict[to_tf]
        y_df[feature_name] = x
        self.dfs_dict[to_tf] = y_df
        return y_df

    def prepare_for_training(self, training_tf, target_horizon,
                             vol_normalized_returns=True, normalize_lb=120, additional_tf=None, additional_features=[],
                             train_size=0.8):

        self.training_df = self.dfs_dict[training_tf]
        features = []

        if vol_normalized_returns:
            vol = calc_daily_vol(self.training_df.Close)
            self.training_df['target_returns'] = calc_returns(self.training_df.Close, target_horizon) / vol
        if additional_tf is not None:
            feat_tf = self.dfs_dict[additional_tf]

            for i in additional_features:
                self.training_df[f'{additional_tf}_{i}'] = np.nan
                self.training_df[f'{additional_tf}_{i}'] = feat_tf[i]

            features.append(f'{additional_tf}_{i}')

        self.feats_dict[training_tf]['Additional'] += features
        self.features = sum(self.feats_dict[training_tf].values(), [])

        self.training_df = self.training_df.ffill().dropna()
        self.ml = ml(self.training_df, self.features, 'target_returns', train_test_size=train_size)

        return

    def time_based_target(self, target_tf, training_start, training_end,
                          target_start, target_end, feature_types=[], target_type='Vol', x_feature_classes=4):
        features = sum([self.feats_dict[target_tf][i] for i in feature_types], [])
        data = self.dfs_dict[target_tf].drop_duplicates()

        fh_data = data.loc[training_start:training_end].dropna()

        bh_data = data.loc[target_start:target_end].dropna()
        target_returns = bh_data.groupby(bh_data.index.date).apply(
            lambda x: pd.DataFrame({'close_returns': np.log(x.Close[0]) - np.log(x.Close[-1]),
                                    'vol': np.sqrt(np.sum(x.returns ** 2)),
                                    'range': (np.max(x.High) - np.min(x.Low)) / x.Close[0]}, index=[x.index.date]))

    def train_model(self, method='xgbclf', params=None, high_percentile=85, low_percentile=20,
                    save_file='save_eval.csv', plot=True):
        if method == 'xgb':
            self.model = self.ml.xgb_model(xgb_params, evaluate=True, eval_log=self.model_info['Dir'] + save_file,
                                           plot_pred=plot)
        if method == 'gbr':
            self.model = self.ml.tree_model(gbr_params, gbr=True, evaluate=True,
                                            eval_log=self.model_info['Dir'] + save_file,
                                            plot_pred=plot)
        if method == 'xgbclf':
            self.model = self.ml.xgb_clf(high_p=high_percentile, low_p=low_percentile, parameter_dict=xgb_clf_params)

        self.model_info['ML'].update({'Type': method, 'Hyperparams': params, 'Features': self.features})

        self.model_info.update({'Features': self.features})

