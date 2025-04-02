import yfinance as yf
from arch import arch_model
import numpy as np
from ml_builder.utils import clean_data
from ml_builder.torch_dl import ginn_lstm, training_loop

def generate_garch_predictions(returns, forecast_horizon=10):
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, rescale=100)
    fitted_model = garch_model.fit(disp='off')
    forecasts = fitted_model.forecast(horizon=forecast_horizon)
    return np.sqrt(forecasts.variance.values[-1, :])  # Extract volatility forecasts

spy = yf.download('^GSPC', multi_level_index=False)

returns = spy.Close.pct_change()
returns = returns.dropna()
training_data = returns.loc['2010':'2026']


vol = generate_garch_predictions(spy['ret'].dropna(), 5)