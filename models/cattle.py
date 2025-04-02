from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tests import adf_test, remove_outliers, eval_model, evaluate_seasonal
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from filters import wwma
from feature_engineering import kama, fft_decomp

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import detrend
from sc_loader import sierra_charts as sc
from ml_builder.ml_model import ml_model as ml, gbr_params, xgb_params
from ml_builder.feature_builder import model_prep as mprep
from screener.indicators import atr, tr

live_cattle = yf.download("LE=F", multi_level_index=False)
feeder_cattle = yf.download('GF=F')

plot_pacf(live_cattle.Close)
seasonal_values = (110, 49,21)

ssmod = fft_decomp(live_cattle)
ssmod.wwm_detrend(50)

ssmod.fft_data(top_n=30, detrended=False)

seasonal_data = MSTL(live_cattle.Close, periods=seasonal_values).fit()
pre_mod = mprep(live_cattle)
seasonal_cols = [f'seasonal_{i}' for i in range(len(seasonal_values))]

for i in range(len(seasonal_values)):
    pre_mod.data[f'seasonal_{i}'] = seasonal_data.seasonal[f'seasonal_{seasonal_values[i]}'].values
    pre_mod.data[f'seasonal_{i}'] = pre_mod.data[f'seasonal_{i}'] / pre_mod.data.Close


pre_mod.create_targets(10)
pre_mod.add_SMA(20)
pre_mod.add_SMA(50)
pre_mod.add_SMA(100)
pre_mod.add_SMA(200)
pre_mod.data['EMA0'] = pre_mod.data.Close.ewm(span=8).mean()
pre_mod.data['EMA1'] = pre_mod.data.Close.ewm(span=15).mean()
pre_mod.data['EMA2'] = pre_mod.data.Close.ewm(span=21).mean()
for i in range(0,3):pre_mod.data[f'EMA{i}_x'] = (pre_mod.data.Close - pre_mod.data[f'EMA{i}'] )/pre_mod.data.Close
pre_mod.data['12TARGET'] = pre_mod.data.Close.pct_change(-12)
pre_mod.data['20TARGET'] = pre_mod.data.Close.pct_change(-20)
pre_mod.data['resid'] = seasonal_data.resid/pre_mod.data.Close
pre_mod.data['Volume_10'] = pre_mod.data.Volume.rolling(10).mean()
pre_mod.data['Volume_20'] = pre_mod.data.Volume.rolling(20).mean()
pre_mod.data['mom_21'] = pre_mod.data.Close.diff(21)
pre_mod.data['mom_42'] = pre_mod.data.Close.diff(42)
pre_mod.data['trend'] = seasonal_data.trend.values

pre_mod.data['kama_20'] = kama(pre_mod.data.Close, 20, 2, 32)
pre_mod.data['kama_5'] = kama(pre_mod.data.Close, 5, 2 , 32)
pre_mod.data['kama_10'] = kama(pre_mod.data.Close, 10, 2, 30)
pre_mod.data['trend_x'] = pre_mod.data.trend.pct_change()
pre_mod.data['trend_x_close'] = (pre_mod.data.Close - pre_mod.data.trend)/pre_mod.data.Close
pre_mod.data['trend_x10'] = pre_mod.data.trend.pct_change(10)
pre_mod.data['nATR'] = atr(pre_mod.data,normalized=True, length=5)
pre_mod.data['TR'] = tr(pre_mod.data)
pre_mod.data['ATR'] = atr(pre_mod.data, length=7)
pre_mod.data['50SMA_chg'] = pre_mod.data['50SMA'].pct_change()
pre_mod.data['TR_ratio'] = (pre_mod.data.TR/pre_mod.data.ATR).ewm(span=3).mean()
pre_mod.data['kama_20'] = kama(pre_mod.data.Close, 20, 2, 32)
pre_mod.data['kama_5'] = kama(pre_mod.data.Close, 5, 2 , 32)
pre_mod.data['kama_10'] = kama(pre_mod.data.Close, 10, 2, 30)
pre_mod.data['resid'] = pre_mod.data['resid']/pre_mod.data.Close
for i in ['kama_20', 'kama_5', 'kama_10']: pre_mod.data[i+'_x'] = (pre_mod.data.Close - pre_mod.data[i])/pre_mod.data.Close

features=[*seasonal_cols,'trend_x','trend_x_close'
    ,'200SMA_norm', '50SMA_norm','resid','mom_42'
                                        ]
le_seasonal = ml(pre_mod.data, features=features, target_column='20TARGET')


le_seasonal.tree_model(parameter_dict=gbr_params, gbr=True)
