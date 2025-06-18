import sys, os
from pathlib import Path
from time import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from itertools import product, zip_longest
from technical_prep import FeaturePrep as fp
from sc_loader import sierra_charts as sch
from ml_build.utils import TimeSeriesCV as CV
import datetime
from ml_build.ml_model import lgb_optimizer
from linear_models import multivariate_regression
from feature_engineering import get_range
loader = sch()
data = loader.get_chart('ES_F', formatted=True, start_date='2017-01-01')
save_location = 'F:\\'
sys.path.insert(1, os.path.join(sys.path[0], '..'))
cv = CV(5, 190*5,50*5)
data_store = 'data\intraday_algo.h5'
model_path = Path('intraday')

if not model_path.exists():
    model_path.mkdir(parents=True)

# Prepare data and features
momentum_lbs = [1, 5, 21]
range_lbs = [3, 5, 10,]

data.drop('Datetime', axis=1, inplace=True)
data['date'] = data.index.date
transfer_features = []
model_data = fp(data, intraday_tfs=['5min', '1h', '1d'], vol_scale=False, project_dir=model_path)
model_data.train_HAR('5min', har_type='rsv', add_as_feature=True, tf='5min')
model_data.trend_indicators('1d', momentum=True, momentum_periods=momentum_lbs)
model_data.volatility_signals('1d',ATR=True, ATR_length=8, hl_range=True, range_lengths=range_lbs, hawkes=True, hawkes_mean=63, hawkes_signal_lb=21)
model_data.dfs_dict['5min']['hp_signal_1d'] = model_data.dfs_dict['1d']['hp_signal']
model_data.dfs_dict['5min'][[f'momentum_{n}d' for n in [1, 5, 21]]] = model_data.dfs_dict['1d'][[f'momentum_{n}' for n in [1, 5, 21]]]

daily_high_cols, daily_lows_cols = [f'high_{n}d' for n in range_lbs],[f'low_{n}d' for n in range_lbs]
model_data.dfs_dict['5min'][daily_high_cols] = model_data.dfs_dict['1d'][[f'{n}_highs' for n in range_lbs]]
model_data.dfs_dict['5min'][daily_lows_cols] = model_data.dfs_dict['1d'][[f'{n}_lows' for n in range_lbs]]
model_data.dfs_dict['5min'].ffill(inplace=True)

hlc_feats = []
for high_col, low_col in zip(daily_high_cols, daily_lows_cols):
    model_data.dfs_dict['5min'][high_col+'_x'] = (model_data.dfs_dict['5min'][high_col] - model_data.dfs_dict['5min']['Close'])/model_data.dfs_dict['5min'].Close
    model_data.dfs_dict['5min'][low_col+'_x'] = (model_data.dfs_dict['5min']['Close'] - model_data.dfs_dict['5min'][low_col] )/model_data.dfs_dict['5min'].Close
    hlc_feats += [high_col+'_x', low_col+'_x']

transfer_features += [f'momentum_{n}d' for n in [1, 5, 21]] + hlc_feats + ['hp_signal_1d']
model_data.feats_dict['5min']['Additional'] += transfer_features
model_data.temporal_features('5min',
                             range_starts=[datetime.time(0,0), datetime.time(8,30)],
                             range_ends=[datetime.time(8, 30),datetime.time(13, 0)])
model_data.volatility_signals('5min', ATR=True, ATR_length=100, hawkes=True, hawkes_mean=240, hawkes_signal_lb=40, lagged_semivariance=True, sv_lags=[1, 5])
model_data.volume_features('1h', relative_volume=True, rvol_days=10,VSA=False, vsa_cols=['Volume'], vsa_lb=120)
model_data.dfs_dict['5min']['target_data_13'] = get_range(model_data.data, datetime.time(13,0), datetime.time(15,0), normalize=False)['150_Return'].shift(-2, freq='h')
model_data.dfs_dict['5min']['target_data_14'] = get_range(model_data.data, datetime.time(14,0), datetime.time(15, 0))['150_Return'].shift(-2, freq='h')
model_data.transfer_features('1h', '5min', ['rvol_10'], datetime.time(9,30), insert_at=datetime.time(9, 30))
model_data.dfs_dict['5min'].ffill(inplace=True)
dataset = model_data.prepare_for_training('5min', target_horizon=None, feature_types=['Additional', 'Volatility', 'Temporal']).loc[datetime.time(13,0)]
self = lgb_optimizer(dataset, model_data.features[:-1], label_filter='target_data', store_name='lgb_final_cv.h5')
self.set_cv_params(lookaheads=[0, 1], train_lengths=[500, 252], test_lengths=[100,63])
self.run_CV(group_ic_by='date')

int_cols = ['boost_rounds', 'num_leaves', 'train_length', 'test_length','lookahead']
summary = self.evaluate_results(results_store_file='lgb_final_results.h5')

lgb_params = summary[0]['params'].to_dict()
[lgb_params.update({i: int(lgb_params[i])}) for i in int_cols]

lgb_params['lookahead'] = 0
best_rounds = str(lgb_params['boost_rounds'])

test_preds = summary[0]['predictions'][best_rounds]
y_true = summary[0]['predictions'].iloc[:, 0]




