from technical_prep import FeaturePrep as MTM
from models.volatility.regimes import GaussianMixture as gm, AggClusters as agg, sort_clusters
from backtest.strategy.signals import SignalGen as sg
from backtest.strategy_test import RegimeBacktester as RBT
import pandas as pd
import numpy as np
from sc_loader import sierra_charts as sch
import datetime as dt
from copy import deepcopy

# Load Data

ticker = pd.read_parquet('C:\\Users\\nicho\PycharmProjects\ml_ensembles\models\es_f.parquet')
ticker['Close'] = ticker['Last']
tfs = ['5min', '10min', '1h']

mtf_es = MTM(ticker, intraday_tfs=tfs, vol_scale=True, project_dir='F:\\ML\\commodities\\')

daily = mtf_es.dfs_dict['1d']
mtf_es.temporal_features('5min', range_starts=[dt.time(0, 30)], range_ends=[dt.time(9, 30)], normalize=True)
mtf_es.transfer_features('5min', '1h', feature_names=mtf_es.feats_dict['5min']['Temporal'])

mtf_es.trend_indicators('1d', SMAs=False, sma_lens=[50, 200], KAMAs=True, momentum=True, momentum_periods=[21,63], kama_params=[(20, 2, 32)])
mtf_es.volatility_signals('5min', lagged_vol=False, vol_lags=[21], lagged_semivariance=True, range_lengths=10, sv_lags=[5])
mtf_es.train_HAR('10min', scale_data=False, penalty=None, cv=3, har_type='rsv', training_size=0.8, target_horizon=5)
mtf_es.train_HAR('5min', scale_data=False,penalty=None,cv=3, har_type='rsv', training_size=0.8, target_horizon=1)
mtf_es.train_HAR('5min', scale_data=True, penalty='cv', har_type='pd')
mtf_es.volume_features('1h', VSA=True, vsa_lb=120,vsa_col='AskVolume', relative_volume=True, cum_rvol=False, normalize_vol=False, normalizer_len=168)
mtf_es.volatility_signals('1h', ATR=True, hawkes=True,lagged_vol=False, hawkes_binary_signal=True, hawkes_mean=168, hawkes_signal_lb=48, kappa=0.2, hl_range=True, range_lengths=[48, 72])
mtf_es.transfer_features('1d', '1h', feature_names=mtf_es.feats_dict['1d']['Additional'])

mtf_es.prepare_for_training(
                            '1h',
                           target_horizon=1,
                           feature_types=['Volatility', 'Volume', 'Additional'],
                           vol_normalized_returns=False,
                           )

feat_df = mtf_es.training_df
feats = mtf_es.features + ['returns']
t_start = dt.time(8, 30)
t_end = dt.time(16, 30)

cluster_model = gm(feat_df,features=feats, n_components=4)

cluster_model.fit(use_pca=True, n_components_pca=3)

# cluster_model.features_df['predictions'] = sort_clusters(cluster_model.features_df, mtf_es.training_df.Close)
cluster_model.visualize_returns()
##cluster_model.features_df.to_parquet('es_f_cluster_2.parquet')
#
# signals = MTM(ticker, intraday_tfs=tfs, vol_scale=False)
# signals.daily_vol = signals.dfs_dict['1d'].Close.ewm(span=60).std()
# signals.trend_indicators('1d', SMAs=False, KAMAs=True, kama_params=[(21, 2,30), (10,2,30), (5,2,30)], momentum=True, momentum_periods=[21,63], normalize_features=False)
# signals.volatility_signals('1d', ATR=True, ATR_length=8, normalize_atr=False, hawkes=True, hawkes_mean=63, kappa=0.1, hawkes_binary_signal=True, hawkes_signal_lb=21, normalize=False, hl_range=True, range_lengths=[5, 20, 50], lagged_vol=False)
# signals.prepare_for_training(training_tf='1d', target_horizon=6, feature_types=['Additional', 'Trend', 'Volatility'])
#
# sg_map = sg(signals.training_df)
# sg_map.data['regime'] = cluster_model.features_df.cluster
# sg_map.crossover('kama_10','kama_21', signal_name='kama_cross',stop_type='signal', stop_col='hp_signal', map=True, map_id=4,)
# sg_map.hl_mean_reversion('5_high', '5_low', stop_col=['5_high', '5_low'], stop_type='crossover', map=True, map_id=4)
# rbt = RBT(sg_map.data, 'Close', regime_col='regime', signal_map=sg_map.signal_map)
# rbt.run()
