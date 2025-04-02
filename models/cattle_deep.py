from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from feature_engineering import kama
import yfinance as yf
from ml_builder.ml_model import ml_model as ml, gbr_params, xgb_params
from ml_builder.feature_builder import model_prep as mprep
from screener.indicators import atr, tr
from model_prep import dl_model

live_cattle = yf.download("LE=F", multi_level_index=False)

seasonal_values = (110, 49,21)
le_deep = dl_model(live_cattle, "F:\\ML\\Seasonal\\LE_F\\")


le_deep.seasonal_decomp(seasonal_values)
le_deep.seasonal_features(normalize_features=True, residuals=True)
le_deep.trend_features(trend=False, SMAs=True, sma_lens=[50, 120],
                       momentum=True, momentum_lens=[21], BBands=True,
                       normalize_features=False)

le_deep.target = "Close"

le_deep.torch_model(['Seasonal', 'Trend'], 20, 5, 'le_seasonal.csv', scale_y=True, y_scale_type='minmax', scale_x=True, x_scale_type='minmax')

le_deep.torch_loop(40)