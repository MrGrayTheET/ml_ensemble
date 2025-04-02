from cot.chart_data import file_dict as fd, cot_data as cot, sentiment
from dl_features import mbuilder as mp
import ml_model as ml
import pandas as pd
from ml_build.utils import sierra_charts as sc

reader = sc()

es_f = reader.format_sierra(pd.read_csv(fd['ES_F'])).resample('4h').apply(reader.resample_logic)

ohlc = ['Open', 'High', 'Low', 'Last']

es_pre_mod = mp(es_f, ohlc = ohlc)
es_pre_mod.create_targets(period=3)

es_pre_mod.add_new_series(sentiment, ["Bullish", 'Bearish'], ['Bullish','Bearish'])

cot = cot()
futs_noncom_positioning = cot.contract_data('ES_F')[cot.non_commercials]
es_pre_mod.add_new_series(futs_noncom_positioning, cot.non_commercials[2:], ['long_pos', 'short_pos', 'long_chg', 'short_chg', 'pct_long', 'pct_short'])

es_pre_mod.bid_ask_vol_features()
es_pre_mod.add_SMA(20)
es_pre_mod.add_daily_SMA(20, normalize=True)
es_pre_mod.supply_demand_zone()

volume_features = [ 'Delta', 'avg_size', 'BidVolume', 'AskVolume'] # Separate features
supply_demand_features = ['D1_norm', 'S1_norm','H-L']
sma_features = ['20SMA_norm', '20d_SMA_norm']
positioning_feats = ['long_pos', 'short_pos', 'long_chg', 'short_chg', 'pct_long', 'pct_short']
sentiment_feats = ['Bullish', 'Bearish']
technical_feats = sma_features + volume_features + supply_demand_features

all_feats = positioning_feats + sentiment_feats + technical_feats

es_ml_model = ml.ml_model(es_pre_mod.data, all_feats, target_column='3TARGET')

es_ml_model.tree_model(parameter_dict=ml.gbr_params, gbr=True, eval_log='positioning_model')
