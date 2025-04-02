from model_prep import TrendModel as smod
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import datetime as dt
import os
from ml_builder.ml_model import ml_model as ml, gbr_params, xgb_params

import yfinance as yf
import numpy as np
import pandas as pd

cl_f = yf.download('CL=F', multi_level_index=False, interval='1wk')

cl_f = cl_f[(cl_f.index.date < dt.date(2020, 2, 25)) | (cl_f.index.date> dt.date(2020, 4,30))]

clszn = smod(cl_f, 'F:\\ML\\Seasonal\\CL_F\\')
xgb_params.update(dict(reg_lambda=[1,1.5, 2,5], n_estimators=[200,300,400], subsample=[0.6,0.8]))
plot_pacf(cl_f.Close)
clszn.find_frequencies()
clszn.seasonal_decomp((27, 8, 4) )
clszn.seasonal_features(normalize_features=True)
clszn.trend_features(trend=False, momentum=True, momentum_lens=[5, 8])
clszn.train_model(5,params=xgb_params, method='xgb')