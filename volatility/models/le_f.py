from model_prep import TrendModel as tmod
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler as mm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yfinance as yf

ticker = 'LE=F'
le_f = yf.download(ticker, multi_level_index=False)
le_model = tmod(le_f, f'F:\\ML\\Seasonal\\{ticker}\\')

le_model.trend_features(trend=False, SMAs=True, sma_lens=[10],
                        momentum=True, momentum_lens=[21, 10], normalize_features=True,
                        BBands=False, hls=True, hl_lens=[5, 20])

le_model.volatility_indicators(hawkes_vol=True, hawkes_atr_len=21
                               , kappa=0.3, hawkes_binary=True,rolling_range=True,
                               signal_lb=20, rr_hawkes=True, normalize_mean_len=20,
                               rr_mean=52, rr_hawkes_kappa=0.1, rr_signal_lb=12)

le_model.data = le_model.data.ffill().dropna()
le_model.train_model(10, feat_types=['Trend', 'Volatility'],method='linear_l2', train_test_size=0.7, save_file='xgb_momentum_eval_21sma.csv')

le_model.save_model('linear_l2_10')
le_model.train_model(10, feat_types=['Trend', 'Volatility'], method='xgbclf', high_percentile=75, low_percentile=30 )
le_model.save_model('xgb_clf_12')