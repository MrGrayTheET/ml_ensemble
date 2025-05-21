from multi_tf_prep import MultiTfModel as MTM, LoadedModel as LM
import pandas as pd
import numpy as np
from sc_loader import sierra_charts as sch
from copy import deepcopy
sc = sch()
df_dict = sc.open_formatted_files()
ticker =df_dict['NG_F']
ticker['OHLCAvg'] = (ticker.Close + ticker.Open + ticker.High + ticker.Low) / 4
tfs = ['5min', '10min' '30min', '1h', '1d']
mtf_es = MTM(ticker, intraday_tfs=tfs[::-1], vol_scale=True, project_dir='F:\\ML\\commodities\\')
daily = mtf_es.dfs_dict['1d']
mtf_es.dfs_dict[tfs[-1]]['OHLCAvg']= (daily.Close + daily.Open + daily.High + daily.Low)/4
mtf_es.volatility_signals(tfs[-1], hawkes=True, hawkes_mean=21, hawkes_signal_lb=10,range_width=True, range_length=10, lagged_vol=True, lag_lengths=[21], keep_existing_features=True)
mtf_es.trend_indicators(tfs[-1], momentum=True, momentum_periods=[1, 5, 22], KAMAs=True, kama_params=[(5,2,32),(20, 2, 32)])
mtf_es.volume_features('1d',relative_volume=False, VSA=True, vsa_col='Volume', vsa_lb=21)
mtf_es.train_HAR('5min', scale_data=True,penalty='cv', har_type='rsv',training_size=0.8, target_horizon=1)
mtf_es.train_HAR('5min', scale_data=False,penalty=None,cv=3, har_type='pd', training_size=0.8, target_horizon=5)
mtf_es.prepare_for_training('1d', target_horizon=5, target_vol=False, feature_types=['Trend', 'Volume', 'Additional'], additional_tf='1d')
mtf_es.train_model(method='xgbclf',linear_alpha=0.5, high_percentile=80, low_percentile=20)
mtf_es.dfs_dict['1d'].to_csv('F:\\charts\\ng_f_2025.csv')

#model = LM('F:\\ML\\commodities\\', 'gc_xgbclf_3d')