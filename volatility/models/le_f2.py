from model_prep import TrendModel as tmod, MultiAssetTrend as mmod
from feature_engineering import stack_groups_vertically as sv
from volatility_models.regime_clustering import RegimeClustering as RC
import yfinance as yf
from xgboost import plot_tree
from ml_build.ml_model import xgb_params, ml_model as ml
from seasonal.arma_garch import model_selection as ms
from feature_engineering import label_reversion


tickers = 'LE=F'

le_f = yf.download(tickers, multi_level_index=False)


le_model = tmod(le_f, f'F:\\ML\\models\\commodities\\')
le_model.volatility_indicators(ATR=8 ,tr_atr=True, tr_atr_len=12)
le_model.trend_features(trend=False, momentum=True, momentum_lens=[8], BBands=True, hls=True, hl_lens=[8])

le_model.data = le_model.data.ffill().dropna()
features = le_model.feature_eval(8, ['Trend', 'Volatility'])
ma_trend = le_model.train_model(8, ['Trend', 'Volatility'], method='linear_l2', linear_alpha=0.7)
le_model.custom_feature(le_model.data['predictions'], 'preds')

le_mid_term = tmod(le_f, f'F:\\ML\\models\\commodities')
le_mid_term.trend_features(trend=False, SMAs=True, sma_lens=[21], momentum=True, momentum_lens=[21], BBands=False, hls=True, hl_lens=[10])
le_mid_term.data = le_mid_term.data.ffill().dropna()

def load_xgb_tuned():
    le_model.load_model('xgb_tuned', feature_types=['Trend'])
    return le_model
def load_xgb_clf():
    le_model.load_model('xgb_momentum_model', feature_types=['Trend'])

    return le_model

def load_xgb_regressor():
    le_model.load_model('xgb_momentum_regressor')
    return le_model

def load_xgb_momentum():
    le_model.load_model('xgb_momentum_2', feature_types=['Trend', 'Custom'])
    return le_model

def load_xgb_clf_6D():
    le_model.load_model('xgb_clf_6d', feature_types=['Trend', 'Custom'])
    return le_model


