import yfinance as yf
import numpy as np
from models.volatility.regimes import AggClusters as agg

df = yf.download('^GSPC', start='2009-01-01', end='2025-05-30', multi_level_index=False)
df['returns'] = np.log(df.Close) - np.log(df.Close.shift(1))

cls_mod = agg(df, features=['returns'])
cls_mod.features_df.dropna(inplace=True)
cls_mod.fit(use_pca=False)

