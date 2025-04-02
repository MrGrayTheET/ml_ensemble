from model_prep import TrendModel as tmod
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler as mm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yfinance as yf

from ml_builder.ml_model import xgb_params
ticker = 'LE=F'
le_f = yf.download(ticker, multi_level_index=False)
le_model = tmod(le_f, f'F:\\ML\\Seasonal\\{ticker}\\')

seasonals = (24, 63, 110)

le_model.seasonal_decomp(periods=seasonals)
le_model.trend_features(trend=True, SMAs=True, sma_lens=[21, 48],
                        momentum=True, momentum_lens=[21, 63], normalize_features=True,
                        BBands=False)

le_model.seasonal_features(normalize_features=True)


le_model.train_model(10, feat_types=['Trend'], save_file='xgb_momentum_eval_21sma.csv')