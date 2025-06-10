from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sc_loader import sierra_charts as sch

import feature_engineering
from feature_engineering import (vsa, vol_signal, vol_scaled_returns,
                                 calc_daily_vol, calc_returns, hawkes_process,
                                 atr, log_returns, kama, rvol, get_range,
                                 rsv, historical_rv, ohlc_rs_dict)

from ml_build.ml_model import ml_model as ml, xgb_params, lgb_clf_params, gbr_params, rfr_params, xgb_clf_params
from ml_build.utils import prune_non_builtin
from linear_models import multivariate_regression
from tests import  nan_fraction_exceeds
from volatility.AR import HAR
from itertools import chain
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import pickle
import ast
import os

if os.name == 'nt':
    sc_cfg = 'C:\\Users\\nicho\PycharmProjects\ml_ensembles\data_config.toml'
else:
    sc_cfg = '/content/drive/MyDrive/utils/SC_CFG_FP.toml'

VOL_TARGET = 0.15
resample_dict = ohlc_rs_dict(include_bid_ask=False)


class FeaturePrep:

    def __init__(self, data, intraday_tfs=['5min', '10min', '1h', '4h'],
                 train_test_ratio=0.8, hf_timeframe='5min',
                 vol_scale=True, bid_ask=False,
                 project_dir="F:\\ML\\multi_tf\\", vs_lb=22,
                 rs_offset_hourly='30min', rs_offset_daily=None):
        self.features = None
        self.model = None
        if not os.path.isdir(project_dir): os.mkdir(project_dir)

        self.training_df = None
        self.har_df = None
        self.data = data
        self.dfs_dict = {}
        self.feats_dict = {}
        self.resample_dict = ohlc_rs_dict(include_bid_ask=bid_ask)
        self.train_ratio = train_test_ratio
        self.model_info = {'Dir': project_dir, 'RV_freq': hf_timeframe, 'ML': {}, 'Eval': {}}
        self.dfs_dict['1d'] = data.resample('1d').apply(self.resample_dict).ffill().dropna()
        self.dfs_dict['1d'].index = self.dfs_dict['1d'].index.normalize()
        self.dfs_dict['1d']['returns'] = log_returns(self.dfs_dict['1d'].Close)
        self.dfs_dict['1d']['scaled_returns'] = vol_scaled_returns(self.dfs_dict['1d'].returns, vs_lb)
        vol = calc_daily_vol(self.dfs_dict['1d'].returns, vs_lb).ffill()
        self.ann_vol = vol * np.sqrt(252)

        for i in intraday_tfs + ['1d']:
            if i in ['1h', '4h', '2h']:
                rs_params = {'rule':i, 'offset':rs_offset_hourly}
                vol_lb = 24 / int(i[0]) * vs_lb

            elif i in ['5min', '10min', '30min']:
                rs_params = {'rule':i}
                vol_lb = (60 / int(i[0])) * 24 * vs_lb

            else:
                rs_params = {'rule':i, 'offset':rs_offset_daily}
                vol_lb = vs_lb

            self.dfs_dict[i] = self.data.resample(**rs_params).apply(self.resample_dict)
            self.dfs_dict[i]['log_returns'] = log_returns(self.dfs_dict[i].Close)
            self.dfs_dict[i]['vol'] = calc_daily_vol(self.dfs_dict[i]['log_returns'], vol_lb)
            self.dfs_dict[i].dropna(inplace=True)

            if isinstance(self.dfs_dict[i].index, pd.DatetimeIndex): pass

            else:self.dfs_dict[i].index = pd.to_datetime(self.dfs_dict[i].index)


            #if vol_scale:
            #    self.dfs_dict[i]['scaled_returns'] = self.dfs_dict[i]['returns'] * VOL_TARGET / self.ann_vol

            self.feats_dict.update({i: {'Volatility': [], 'Volume': [], 'Trend': [], 'Temporal': [], 'Additional': []}})

        self.models_dict = {'ann_vol': None, 'trend': None, 'reversal': None}

        return

    def volatility_signals(self, timeframe, ATR=False, ATR_length=14, normalize_atr=True,
                           hawkes=False, hawkes_mean=168, kappa=0.1, hawkes_signal_lb=21, hawkes_binary_signal=True,
                           normalize=True, hl_range=False, range_lengths=None, normalize_range_len=10,
                           lagged_vol=True, vol_lags=None, lagged_semivariance=False, sv_lags=None, average_lags=True,
                           keep_existing_features=False):
        if range_lengths is None:
            range_lengths = [20]
        if sv_lags is None:
            sv_lags = [1, 5]
        if vol_lags is None:
            vol_lags = [1]
        if keep_existing_features:
            features = self.feats_dict[timeframe]['Volatility']
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

        if hl_range:
            normalizer_SMA = data.Close.rolling(normalize_range_len, min_periods=1).mean()
            for len_ in range_lengths:
                hl_data = data[['High', 'Low']].dropna()
                data[f'{len_}_highs'] = data.High.rolling(len_, min_periods=1).max()
                data[f'{len_}_lows'] = data.Low.rolling(len_, min_periods=1).min()
                data[f'{len_}_width'] = data[f'{len_}_highs']  - data[f'{len_}_lows']

                if normalize:
                    hl_feats = [f'{len_}_{i}' for i in ['range_x', 'high_x', 'low_x']]
                    data[hl_feats] = np.full((len(data.index), len(hl_feats)), np.nan)
                    data.loc[:, f'{len_}_range_x'] = data[f'{len_}_width']/data['Close']
                    data.loc[:, f'{len_}_high_x'] = (data[f'{len_}_highs'] - data['Close'])/ data['Close']
                    data.loc[:, f'{len_}_low_x'] = (data['Close'] - data[f'{len_}_lows']) / data['Close']
                    features += [f'{len_}_{i}' for i in ['range_x', 'high_x', 'low_x']]

                else:
                    features += [f'{len_}_{i}' for i in ['high', 'low']]

        if lagged_vol:
            for i in vol_lags:
                data[f'rv_{i}'] = historical_rv(data.returns, i, average=average_lags).ffill()
                features.append(f'rv_{i}')
                self.dfs_dict['1d'][f'rv{timeframe[:1]}_{i}'] = data[f'rv_{i}']
                self.feats_dict['1d']['Volatility'].append(f'rv{timeframe[:1]}_{i}')

        if lagged_semivariance:
            for i in sv_lags:
                data[[f'pos_rsv_{i}', f'neg_rsv_{i}']] = rsv(data.returns, window=i, average=average_lags).ffill()
                self.dfs_dict['1d'][[f'pos_rsv{timeframe[:1]}_{i}', f'neg_rsv{timeframe[:1]}_{i}']] = data[
                    [f'pos_rsv_{i}', f'neg_rsv_{i}']]
                features += [f'pos_rsv_{i}', f'neg_rsv_{i}']
                self.feats_dict['1d']['Volatility'] += [f'pos_rsv{timeframe[:1]}_{i}', f'neg_rsv{timeframe[:1]}_{i}']

        self.dfs_dict.update({timeframe: data.ffill()})
        self.feats_dict[timeframe].update({'Volatility': features})

        return self.dfs_dict[timeframe][features]

    def volume_features(self, timeframe, relative_volume=True, rvol_days=10, cum_rvol=True, delta=False,
                        delta_as_pct=True, VSA=False, vsa_cols=['Volume'], vsa_lb=30,
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
            for vsa_col in vsa_cols:
                data[f'{vsa_col[:3]}_vsa'] = vsa(data, vsa_col, vsa_lb)
                features.append(f'{vsa_col[:3]}_vsa')

        if normalize_vol:
            median = data.Volume.rolling(normalizer_len).median()
            data['norm_vol'] = data.Volume / median

        self.feats_dict[timeframe].update({'Volume': features})
        self.dfs_dict[timeframe] = data

        return data[features]

    def temporal_features(self, timeframe='10min', ranges=True, range_starts=[], range_ends=[], range_rv=True,
                          shift_date=False, keep_existing_features=False, normalize=True):
        data = self.dfs_dict[timeframe]

        if not keep_existing_features:
            features = []
        else:
            features = self.feats_dict[timeframe]['Temporal']

        for i in range(len(range_starts)):
            range_feats = []
            t_start = range_starts[i]
            t_end = range_ends[i]

            hls = get_range(data, t_start, t_end)
            col_time = t_end.strftime("%H%M")

            t_cols = [f'{col_time}_Open',
                      f'{col_time}_High',
                      f'{col_time}_Low',
                      f'{col_time}_Close',
                      f'{col_time}_Return',
                      f'{col_time}_Volume']

            tmp_price = data.loc[t_start:t_end]

            tmp_data = np.zeros((len(hls.index), len(t_cols)))
            tmp_df = pd.DataFrame(index=pd.to_datetime(hls.index), columns=t_cols,
                                  data=tmp_data)

            tmp_df.loc[:, t_cols[0]] = tmp_price['Close'].loc[t_start]
            tmp_df.loc[:, t_cols[1]] = hls['Highs']
            tmp_df.loc[:, t_cols[2]] = hls['Lows']
            tmp_df.loc[:, t_cols[3]] = tmp_price['Close'].loc[t_end]
            tmp_df.loc[t_end, t_cols[-2]] = np.log(tmp_df[f'{t_end.strftime("%H%M")}_Close']) - np.log(tmp_df[f'{t_end.strftime("%H%M")}_Open'])
            range_vol = tmp_price['Volume'].groupby(tmp_price.index.date).sum()
            range_vol.index = tmp_df.index
            tmp_df.loc[t_end,  t_cols[-1]] = range_vol

            if normalize:
                range_feats += [f'{col_time}_H-L', f'{col_time}_H-C',  f'{col_time}_C-L' , t_cols[-1], t_cols[-2]]
                hls['Close'] = tmp_price['Close'].loc[t_start]
                tmp_df[range_feats[0]] = (tmp_df[t_cols[1]] - tmp_df[t_cols[2]]) / tmp_df[t_cols[3]]
                tmp_df[range_feats[1]] = (tmp_df[t_cols[1]] - tmp_df[t_cols[3]]) / tmp_df[t_cols[3]]
                tmp_df[range_feats[2]] = (tmp_df[t_cols[3]] - tmp_df[t_cols[2]]) / tmp_df[t_cols[3]]

            else:
                range_feats += t_cols

            if range_rv:
                range_feats += [f'{col_time}_rv']
                tmp_df[f'{col_time}_rv'] = np.nan
                rv = (tmp_price.returns.groupby(tmp_price.index.date).
                    apply(lambda x: np.sqrt(np.sum(x ** 2))))

                tmp_df.loc[:, f'{col_time}_rv'] = rv.values


            features += range_feats

            data[range_feats] = tmp_df.loc[:, range_feats]



        self.dfs_dict[timeframe][features] = data[features].ffill()
        self.feats_dict[timeframe]['Temporal'] = features

        return

    def trend_indicators(self, timeframe, keep_existing_features=False,
                         SMAs=False, sma_lens=[20, 50, 200], KAMAs=False, kama_params=[(20, 2, 32)],
                         momentum=True, momentum_periods=[24],
                         normalize_features=True, use_scaled_returns=True):

        if keep_existing_features:
            features = self.dfs_dict[timeframe]['Trend']
        else:
            features = []

        data = self.dfs_dict[timeframe]

        if momentum:
            for i in momentum_periods:
                data[f'momentum_{i}'] = np.log(data.Close) - np.log(data.Close.shift(i))
                if use_scaled_returns:
                    data[f'momentum_{i}'] = data[f'momentum_{i}'] * VOL_TARGET / data['vol']
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

    def train_HAR(self, rv_tf, har_type='rv', scale_data=False, training_size=0.8,
                  penalty=None, cv=3, alpha=0.7, target_horizon=1, add_as_feature=True, tf='1d',
                  ):

        har = HAR(self.dfs_dict[rv_tf], horizon=target_horizon)
        har.fit(penalty=penalty,
                scale_data=scale_data,
                model=har_type,
                train_size=training_size,
                alpha=alpha,
                lasso_cv=cv)

        print(har.eval)

        self.vol_model = har
        self.har_df = har.x_df

        if add_as_feature:
            self.dfs_dict[tf][f'HAR-{har_type}_preds'] = self.har_df.preds
            if f'HAR-{har_type}_preds' not in self.feats_dict[tf]['Additional']:
                self.feats_dict[tf]['Additional'].append(f'HAR-{har_type}_preds')

        return self.vol_model

    def transfer_features(self, from_tf, to_tf, feature_names, include_tf_in_col=False, offset=dt.timedelta(minutes=30)):
        X = self.dfs_dict[from_tf][feature_names]
        y_df = self.dfs_dict[to_tf]

        if include_tf_in_col:
            new_names = [f'{to_tf}_{i}' for i in feature_names]
        else:
            new_names = feature_names


        y_df[new_names] = np.nan

        for i in range(len(new_names)):
            if 'd' in from_tf:
                y_df[new_names[i]] = np.nan
                feat = X[feature_names[i]]
                feat.index = feat.index + offset
                y_df.loc[feat[1:].index, new_names[i]] = feat[1:]

            if 'min' in from_tf:
                y_df[new_names[i]] = X[feature_names[i]]

        y_df.ffill(inplace=True)

        if new_names not in self.feats_dict[to_tf]['Additional']:
            self.feats_dict[to_tf]['Additional'] += new_names

        return self.dfs_dict[to_tf][new_names]

    def prepare_for_training(self, training_tf, target_horizon, feature_types=['Volatility', 'Trend', 'Volume'],
                             vol_normalized_returns=False, vol_lb=63, additional_tf=None, additional_features=None,
                             target_vol=False):

        self.training_df = self.dfs_dict[training_tf].copy()

        feats = self.feats_dict[training_tf]

        target_returns = np.log(self.training_df.Close.shift(-target_horizon)) - np.log(self.training_df.Close)

        if vol_normalized_returns:

            target_returns = target_returns * VOL_TARGET / self.training_df.vol

        self.training_df.insert(0, 'target_returns', target_returns)

        if additional_tf is not None:
            self.transfer_features(from_tf=additional_tf, to_tf=training_tf, feature_names=additional_features)

        rows_to_drop = nan_fraction_exceeds(self.training_df, axis=1, threshold =0.5)
        self.training_df = self.training_df[~rows_to_drop]

        cols_to_drop = nan_fraction_exceeds(self.dfs_dict[training_tf], axis=0, threshold=0.9)

        if 'target_returns' in cols_to_drop:
            print(f'More than {0.9 * 100}% of the target column is missing, check inputs')
            raise ValueError

        drop_cols = cols_to_drop[cols_to_drop].index.tolist()

        print(f"Columns {drop_cols} have a high amount of nans, dropping before .dropna()")


        features = list(chain.from_iterable([feats[k] for k in feature_types]))
        self.features = features

        return self.training_df

    def train_model(self, method='xgbclf', train_size=0.8, params=None, high_percentile=85, low_percentile=20,
                    save_file='save_eval.csv', plot=True, linear_cv=5, linear_alpha=0.8):

        if 'linear' in method:
            if method == 'linear_l1':
                scaler = StandardScaler()
                res = multivariate_regression(self.training_df, X_cols=self.features, y_col='target_returns',
                                              cv=linear_cv, penalty='cv', scaler=scaler, train_split=True,
                                              train_size=train_size)
            if 'l2' in method:
                res = multivariate_regression(self.training_df, X_cols=self.features, y_col='target_returns',
                                              alpha=linear_alpha, penalty='l2', train_split=True, train_size=train_size)
            else:
                res = multivariate_regression(self.training_df, X_cols=self.features, y_col='target_returns',
                                              penalty=None, train_split=True, train_size=train_size)
            self.model = res['model']
            self.model_info['ML'].update({
                'Type': method,
                'Hyperparams': {'cv': linear_cv, 'alpha': linear_alpha},
                'Features': self.features,
                'Train_end': res['train_end_idx']
            })
            self.model_info['Eval'] = res

        else:
            self.ml = ml(self.training_df, self.features, 'target_returns', train_test_size=train_size)
            if method == 'xgb':
                self.model = self.ml.xgb_model(xgb_params, evaluate=True, eval_log=self.model_info['Dir'] + save_file,
                                               plot_pred=plot)
            if method == 'gbr':
                self.model = self.ml.tree_model(gbr_params, gbr=True, evaluate=True,
                                                eval_log=self.model_info['Dir'] + save_file,
                                                plot_pred=plot)
            if method == 'xgbclf':
                self.model = self.ml.xgb_clf(high_p=high_percentile, low_p=low_percentile,
                                             parameter_dict=xgb_clf_params)
            if method == 'knn':
                self.model = self.ml.neighbors_clf(1, 10, high_p=high_percentile, low_p=low_percentile)
            if method == 'lgbclf':
                self.model = self.ml.lgb_clf(lgb_clf_params, high_percentile=high_percentile,
                                             low_percentile=low_percentile, num_rounds=1000)

            self.model_info['ML'].update(
                {'Type': method, 'Hyperparams': params, 'Features': self.features, 'Train_end': len(self.ml.x_train)})
            self.model_info['Eval'] = self.ml.eval

        self.model_info.update()

    def save_model(self, model_name):
        model_dir = self.model_info['Dir'] + model_name + '\\'

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        if self.ml.model is not None:
            self.ml.save_model(file_name=model_name + '_model', save_fp=model_dir)
        elif self.model is not None:
            with open(model_dir + model_name + '_model', 'wb') as f:
                pickle.dump(self.model, f)

        else:
            print('No Model Found')
            return False

        print('Model Saved to ' + model_dir)

        model_params = prune_non_builtin(self.model_info)
        model_features = prune_non_builtin(self.feats_dict)

        with open(model_dir + 'model_params', 'w') as f:
            f.write(str(model_params))
        with open(model_dir + 'model_features', 'w') as f:
            f.write(str(model_features))
        with open(model_dir + 'dfs_dict.pkl', 'wb') as f:
            pickle.dump(self.dfs_dict, f)

        training_data = self.data.to_csv(model_dir + 'training_data.csv')

        print('Model Data and parameters saved to ' + model_dir)
        return True

    def load_model(self, model_name):
        model_dir = self.model_info['Dir'] + model_name + '\\'

        if not os.path.isdir(model_dir):
            print(f"Model directory '{model_dir}' not found.")
            return False

        # Load model file
        model_file = os.path.join(model_dir, model_name + '_model')
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            print('Model loaded.')
        else:
            print('Model file not found.')
            return False

        # Load model parameters
        params_path = os.path.join(model_dir, 'model_params')
        with open(params_path, 'r') as f:
            self.model_info = ast.literal_eval(f.read())

        # Load model features
        features_path = os.path.join(model_dir, 'model_features')
        with open(features_path, 'r') as f:
            self.feats_dict = ast.literal_eval(f.read())

        # Load dictionary of DataFrames
        dfs_path = os.path.join(model_dir, 'dfs_dict.pkl')
        with open(dfs_path, 'rb') as f:
            self.dfs_dict = pickle.load(f)

        # Load training data if available
        train_csv_path = os.path.join(model_dir, 'training_data.csv')
        if os.path.exists(train_csv_path):
            self.data = pd.read_csv(train_csv_path, index_col=0, parse_dates=True)
        else:
            self.data = None  # or pd.DataFrame()
            print("Training data not found.")

        print(f"Model and associated data loaded from '{model_dir}'")
        return True


class LoadedModel(FeaturePrep):

    def __init__(self, model_dir, model_name):
        model_dir = model_dir + model_name + '\\'
        train_csv_path = os.path.join(model_dir, 'training_data.csv')
        if os.path.exists(train_csv_path):
            self.data = pd.read_csv(train_csv_path, index_col=0, parse_dates=True)
        else:
            self.data = None  # or pd.DataFrame()
            print("Training data not found.")

        if not os.path.isdir(model_dir):
            print(f"Model directory '{model_dir}' not found.")
            return

            # Load model file
        model_file = os.path.join(model_dir, model_name + '_model.pkl')
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            print('Model loaded.')
        else:
            print('Model file not found.')
            super().__init__(self.data, project_dir=model_dir)

        # Load model parameters
        params_path = os.path.join(model_dir, 'model_params')
        with open(params_path, 'r') as f:
            self.model_info = ast.literal_eval(f.read())

        # Load model features
        features_path = os.path.join(model_dir, 'model_features')
        with open(features_path, 'r') as f:
            self.feats_dict = ast.literal_eval(f.read())

        # Load dictionary of DataFrames
        dfs_path = os.path.join(model_dir, 'dfs_dict.pkl')
        with open(dfs_path, 'rb') as f:
            self.dfs_dict = pickle.load(f)

        # Load training data if available

        print(f"Model and associated data loaded from '{model_dir}'")

        return
