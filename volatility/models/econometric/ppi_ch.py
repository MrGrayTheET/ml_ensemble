from linear_models import (get_macro_vars,
                           multivariate_regression,
                           weighted_index, load_csv)
import numpy as np
import pandas as pd
import datetime as dt

us_import_price_weights = {
    "MEXTOT": 0.1576,  # Mexico
    "CANTOT": 0.1427,  # Canada
    "CHNTOT": 0.1091, #China
    "EECTOT":0.0967, #Europe
    "JPNTOT": 0.0375  # Japan
}


_returns = lambda srs: np.log(srs / srs.shift(1))
cpi_yoy = lambda srs: srs / srs.shift(12) - 1
m2_lagged = lambda srs, lags: (srs / srs.shift(12) - 1).shift(lags)
resampled_returns = lambda srs, lag_length: srs.resample('1m').last().pct_change(lag_length)
import_prices = ['CHNTOT', 'JPNTOT', 'CANTOT', 'MEXTOT', 'PPIACO', 'IMPGS']
money_stock = ['M2SL']
market_prices = ['WTISPLC', 'DTWEXBGS']

macro_df = get_macro_vars(import_prices, cpi_yoy)
market_df = get_macro_vars(market_prices, transformation=resampled_returns, args=[12])
market_df.index = market_df.index +dt.timedelta(days=1)
macro_df['M2Stock'] = get_macro_vars(money_stock, m2_lagged, [3])
macro_df['ip_ppi'] = weighted_index(macro_df, us_import_price_weights)
macro_df[['WTI_3M', 'DXY_3M']] = market_df
selected_feats = ['CHNTOT', 'CANTOT', 'MEXTOT', 'M2Stock', 'ip_ppi', 'DXY_3M', 'EECTOT']
macro_df['EECTOT'] = load_csv("C:\\Users\\nicho\\Downloads\EECTOT.csv", transformation=cpi_yoy)
macro_df['PPI_lagged'] = macro_df.PPIACO.shift(-1)
df = macro_df.copy()
df = df.ffill()
df = df.dropna()
res = multivariate_regression(df, X_cols=selected_feats,penalty='l2', alpha=0.7, y_col='PPI_lagged' )


