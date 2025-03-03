from ml_ensemble.seasonal_modeling import seasonal_model as smod
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import datetime as dt
import os
import yfinance as yf
import numpy as np
import pandas as pd

cl_f = yf.download('CL=F', multi_level_index=False, interval='1wk')

cl_f = cl_f[(cl_f.index.date < dt.date(2020, 2, 25)) | (cl_f.index.date> dt.date(2020, 4,30))]

clszn = smod(cl_f, 'F:\\ML\\Seasonal\\CL_F\\')

plot_pacf(cl_f.Close)
clszn.find_frequencies()
clszn.seasonal_decomp((27, 8, 4) )
clszn.prep_model(5, trend=True, sma_lens=[5,10, 20], momentum_lens=[5, 8, 20])
clszn.train_model(save_file='xgb_weekly.csv')

