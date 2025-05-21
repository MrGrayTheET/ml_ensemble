from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tests import adf_test, remove_outliers, eval_model
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from filters import wwma
import yfinance as yf
import numpy as np
import pandas as pd
from feature_engineering import kama, fft_decomp
from scipy.signal import detrend
from sc_loader import sierra_charts as sc
from tests import evaluate_seasonal
from ml_builder.ml_model import ml_model as ml, gbr_params, xgb_params
from ml_builder.feature_builder import model_prep as mprep
from screener.indicators import atr, tr

gc_f = yf.download('GC=F', multi_level_index=False)
plot_pacf(gc_f.Close)

gc_freq= fft_decomp(gc_f)
gc_freq.wwm_detrend(252)
gc_freq.fft_data(top_n=50)
seasonals = (28,49,62 ,110)
seasonal_model2 = MSTL(gc_f.Close, periods=(seasonals)).fit()
evaluate_seasonal(seasonal_model2)
pre_mod = mprep(gc_f)
pre_mod.data[[f'seasonal_{n}' for n in range(len(seasonals))]] = seasonal_model2.seasonal
pre_mod.data['mom_21'] = pre_mod.data.Close.diff(21).ewm(3).mean()
pre_mod.data['mom_30'] = pre_mod.data.Close.diff(30)
pre_mod.data['mom_63'] = pre_mod.data.Close.diff(63)
pre_mod.data['mom_ewm_1'] = pre_mod.data.mom_30.ewm(span=5).mean()
pre_mod.data['mom_30_x'] = pre_mod.data.mom_30 - pre_mod.data.mom_ewm_1
pre_mod.data['resid'] = seasonal_model2.resid
pre_mod.add_SMA(20)
pre_mod.add_SMA(50)
pre_mod.add_SMA(100)
pre_mod.data['kama_20'] = kama(pre_mod.data.Close, 20, 2, 32)
pre_mod.data['kama_5'] = kama(pre_mod.data.Close, 5, 2 , 32)
pre_mod.data['kama_10'] = kama(pre_mod.data.Close, 10, 2, 30)
pre_mod.data['nATR'] = atr(pre_mod.data, length=7, normalized=True)
pre_mod.data['nATR_2'] = atr(pre_mod.data, length=3, normalized=True)
pre_mod.data['trend'] = seasonal_model2.trend.values
pre_mod.data['trend_x'] = pre_mod.data.trend.pct_change()
pre_mod.data['trend_x2'] = (pre_mod.data.Close - pre_mod.data.trend)/pre_mod.data.Close
pre_mod.data['20TARGET'] = pre_mod.data.Close.pct_change(-20)
pre_mod.data['14TARGET'] = pre_mod.data.Close.pct_change(-14)
pre_mod.data['VOLUME20'] = pre_mod.data.Volume.rolling(20).mean()
features = [*[f'seasonal_{n}' for n in range(len(seasonals))], 'resid','trend_x','trend_x2','mom_30','mom_21', '50SMA_norm', '20SMA_norm', 'kama_10_x', 'kama_5_x', '100SMA_norm']
lt_features = [*[f'seasonal_{n}' for n in range(1,len(seasonals))],'trend_x','trend_x2','mom_63','mom_30', '50SMA_norm', '20SMA_norm', 'kama_20_x', 'kama_10_x', '100SMA_norm']

for n in range(len(seasonals)):
    pre_mod.data[f'seasonal_{n}'] = pre_mod.data[f'seasonal_{n}']/pre_mod.data.Close
pre_mod.data['resid'] = pre_mod.data.resid/pre_mod.data.Close

for i in ['kama_20', 'kama_5', 'kama_10']: pre_mod.data[i+'_x'] = (pre_mod.data.Close - pre_mod.data[i])/pre_mod.data.Close


gc_model = ml(pre_mod.data, features=features, target_column='14TARGET')
lt_model = ml(pre_mod.data, features=lt_features, target_column='20TARGET')
xgb_params.update(dict(reg_lambda=[1,1.5, 2,5], n_estimators=[200,300]))
lt_model.xgb_model(xgb_params, eval_log='F:\\ML\\Seasonal\\GC_F\\model_evals\\xgb_20_eval.csv', save_params=True)
lt_model.tree_model(gbr_params, gbr=True, eval_log='F:\\ML\\Seasonal\\GC_F\\model_evals\\gbr_20_eval.csv')
gc_model.xgb_model(xgb_params, eval_log='F:\\ML\\Seasonal\\GC_F\\model_evals\\xgb_14_eval.csv')
gc_model.tree_model(gbr_params, gbr=True, eval_log='F:\\ML\\Seasonal\\GC_F\\model_evals\\gbr_14_eval.csv')


