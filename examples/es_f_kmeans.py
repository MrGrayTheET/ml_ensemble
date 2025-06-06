from volatility.regimes import WKFi as wkf
import numpy as np
from sc_loader import sierra_charts as sc
from technical_prep import FeaturePrep as fp
# Load Data
loader = sc('C:\\Users\\nicho\PycharmProjects\ml_trading\example_config.toml')
data_= loader.get_chart('ES_F', formatted=True, resample=False, resample_period='4h')
data_h4 = data_.resample('4h').apply(loader.resample_logic).dropna()
returns = (np.log(data_h4.Close) - np.log(data_h4.Close.shift(1))).dropna().values
model = wkf(OHLC_data=data_h4.dropna(), k=3,  max_iter=20, tol=1e-6, gamma=9.0, mmd_pairs=7)
model.fit_windows(h1=60, h2=15)
df = model.predict_clusters(df=True)
model.visualize_returns()
tfs = ['10min', '4h', ]
technicals = fp(data_.dropna(), intraday_tfs=tfs, project_dir='C:\\Users\\nicho\PycharmProjects\ml_trading\examples', rs_offset_hourly='0h')
for i in technicals.dfs_dict.values():
    i = i.dropna()
technicals.trend_indicators('4h',KAMAs=True, kama_params=[(25,2,32), (50,2,32)], SMAs=True, sma_lens=[50], normalize_features=False)
technicals.volatility_signals('4h', ATR=True, ATR_length=20,normalize_atr=False, hawkes=True, hawkes_mean=60, hawkes_binary_signal=True, hawkes_signal_lb=21, lagged_vol=False, normalize=False)
technicals.volume_features('4h', relative_volume=True, VSA=True, vsa_lb=60)
technicals.prepare_for_training('4h', feature_types=['Trend', 'Volume', 'Volatility'], target_horizon=1)
training_df = technicals.dfs_dict['4h'].copy(deep=True)
training_df['cluster'] = df['cluster'].dropna().astype(int)
training_df.dropna(inplace=True)

