from models.volatility.regimes import WKFi as wkf
import pandas as pd
import numpy as np
from feature_engineering import ohlc_rs_dict

# Load Data

ticker = pd.read_parquet('C:\\Users\\nicho\PycharmProjects\ml_ensembles\gc_f.parq')
ticker['Close'] = ticker['Last']
daily_data = ticker.resample('4h', offset='-8h').apply(ohlc_rs_dict(include_bid_ask=False))

returns = (np.log(daily_data.Close) - np.log(daily_data.Close.shift(1))).dropna().values
window_size = 21
distributions = [returns[i:i + window_size] for i in range(len(returns) - window_size)]

model = wkf(OHLC_data=daily_data.dropna(), k=3, max_iter=20, tol=1e-6, gamma=5.0, mmd_pairs=8)
model.fit_windows(h1=60, h2=15)
df = model.predict_clusters(df=True)
model.visualize_returns()

#plt.figure(figsize=(12, 4))
#plt.plot(df.index, df.Cumulative_ret, label='Returns', alpha=0.5)
#
#for i in range(len(df+window_size)):
#    start = df.index[i-window_size]
#    end = df.index[i]
#    plt.axvspan(start, end, color=f"C{df.Regime.iloc[i+window_size]}", alpha=0.05)
#
#plt.title("WK-means Regime Assignments")
#plt.xlabel("Time")
#plt.ylabel("Return")
#plt.show()
#