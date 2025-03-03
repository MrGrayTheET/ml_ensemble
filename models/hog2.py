from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
import yfinance as yf
from ml_ensemble.filters import kama
import numpy as np
from tests import eval_model
from futures_ml.ml_model import  ml_model as ml, gbr_params, xgb_params
from futures_ml.feature_builder import model_prep as mprep
from screener.indicators import atr, tr

asset = yf.download('HE=F', multi_level_index=False)
new_seasonal = (21,49,122)
pre_mod = mprep(asset)
new_model= MSTL(pre_mod.data.Close, periods=new_seasonal).fit()
pre_mod.add_new_series(new_model.seasonal, ['seasonal_21','seasonal_49', 'seasonal_122'], ['seasonal_1', 'seasonal_2', 'seasonal_3'])
pre_mod.data['mom_5'] = pre_mod.data.Close.diff(5)
pre_mod.data['mom_10'] = pre_mod.data.Close.diff(10)
pre_mod.data['mom_21'] = pre_mod.data.Close.diff(21)
pre_mod.data['mom_10_ewm'] = pre_mod.data.mom_10.ewm(span=3).mean()
pre_mod.data['mom_5_ewm'] = pre_mod.data.mom_5.ewm(span=3).mean()
pre_mod.add_SMA(20)
pre_mod.add_SMA(50)
pre_mod.add_SMA(100)
pre_mod.data['resid'] = new_model.resid
pre_mod.data['trend'] = new_model.trend.values
pre_mod.data['trend_x'] = pre_mod.data.trend.pct_change()
pre_mod.data['10TARGET'] = pre_mod.data.Close.pct_change(-10)
pre_mod.data['15TARGET'] = pre_mod.data.Close.pct_change(-15)
pre_mod.data['Volume_MA'] = pre_mod.data.Volume.rolling(10).mean()
pre_mod.data['Volume_MA2'] = pre_mod.data.Volume.rolling(5).mean()
pre_mod.data['nATR'] = atr(pre_mod.data, normalized=True)
pre_mod.data['5RET'] = pre_mod.data.Close.pct_change(5)
pre_mod.data['kama_10'] = kama(pre_mod.data.Close, 10, 2, 30)
pre_mod.data['kama_10_x'] = pre_mod.data.Close - pre_mod.data.kama_10
pre_mod.data['kama_20_x'] = pre_mod.data.Close - kama(pre_mod.data.Close, 20, 2, 30)
st_mod =ml(pre_mod.data, features=['seasonal_1', 'seasonal_2', 'seasonal_3',
                                  'kama_10_x', 'kama_20_x', '50SMA_norm','100SMA_norm','resid', 'mom_21', 'mom_10', 'Volume_MA2', 'trend_x'
                                    ], target_column='12TARGET')

st_mod.tree_model(parameter_dict=gbr_params, gbr=True,  params_file='F:\\ML\\Seasonal\\model_params\\st_params.csv', eval_log='F:\\ML\\Seasonal\\model_evals\\st_eval.csv')
