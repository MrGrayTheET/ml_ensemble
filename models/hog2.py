from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
import yfinance as yf
from feature_engineering import kama
from model_prep import TrendModel as tmod

asset = yf.download('HE=F', multi_level_index=False)
new_seasonal = (21,49,122)
