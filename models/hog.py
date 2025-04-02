
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
import yfinance as yf
import numpy as np
from ml_build.ml_model import  ml_model as ml, gbr_params
from ml_build.dl_features import mbuilder as mprep
from screener.indicators import atr, tr
from ml_build.utils import clean_data


asset = yf.download('HE=F')
fft_periods = (49, 242)


pre_mod = mprep(asset)
pre_mod.wavelet_transform(wavelet='db2', level=2)
seasonal_model = MSTL(pre_mod.data.denoised, periods=fft_periods).fit()
pre_mod.create_targets(5)
pre_mod.create_targets(10)
pre_mod.create_targets(30)
pre_mod.data['Close_x'] = np.log(pre_mod.data['Close']) - np.log(pre_mod.data['Close'].shift(1))
pre_mod.add_new_series(pre_mod.seasonal, ['seasonal_21','seasonal_49', 'seasonal_242'], ['seasonal_1', 'seasonal_2', 'seasonal_3'])
pre_mod.add_SMA(20)
pre_mod.add_SMA(50)
pre_mod.data['mom_5'] = pre_mod.data.Close.diff(5)
pre_mod.data['mom_10'] = pre_mod.data.Close.diff(10)
pre_mod.data['HL_x'] = (pre_mod.data.High - pre_mod.data.Low)/pre_mod.data.Close
pre_mod.data['resid'] = seasonal_model.resid
pre_mod.data['10TARGET'] = pre_mod.data.Close.pct_change(-10)
pre_mod.data['5TARGET'] = pre_mod.data.Close.pct_change(-5)
pre_mod.data['ATR'] = atr(pre_mod.data, length=7, normalized=True)
pre_mod.data['Volume_MA'] = pre_mod.data.Volume.rolling(10).mean()
pre_mod.data['Volume_MA2'] = pre_mod.data.Volume.rolling(5).mean()
pre_mod.data['Volume_X_2'] = pre_mod.data.Volume - pre_mod.data.Volume_MA2



mt_mod = ml(pre_mod.data, features=['seasonal_1', 'seasonal_2', 'Volume','5RET',
                                     '20SMA_norm', '50SMA_norm','resid', 'mom_10','Volume_MA', 'Volume_MA2'
                                     ], target_column='10TARGET')
mt_mod.tree_model(parameter_dict=gbr_params, gbr=True, params_file='F:\\ML\\Seasonal\\model_params\\mt_params.csv', eval_log='F:\\ML\\Seasonal\\model_evals\\mt_eval.csv')
