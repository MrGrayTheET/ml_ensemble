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
from model_prep import TrendModel as tmod
from ml_builder.ml_model import ml_model as ml, gbr_params, xgb_params
from screener.indicators import atr, tr


df = yf.download('CL=F', multi_level_index=False)
df['Exp'] = df.Close.ewm(span=3).mean()

st_cl = tmod(df, 'F:\\ML\\Seasonal\\CL_F\\')

st_cl.seasonal_decomp(periods=( 86, 49,28, ))
st_cl.seasonal_features(normalize_features=True)
st_cl.trend_features(trend=True, BBands=True, bband_width=2, bband_window=20, normalize_features=True)








