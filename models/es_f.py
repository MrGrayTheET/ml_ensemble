from model_prep import TrendModel as tmod
from volatility_models.regime_clustering import RegimeClustering as RC
import yfinance as yf
from xgboost import plot_tree


ticker = 'ES=F'
es_f = yf.download(ticker, multi_level_index=False)
es_models = tmod(es_f, f'F:\\ML\\models\\{ticker}\\')

def load_xgb_clf():
    es_models.load_model('xgb_momentum_model', feature_types=['Trend'])

    return es_models

def load_xgb_regressor():
    es_models.load_model('xgb_momentum_regressor')
    return es_models

def load_xgb_momentum():
    es_models.load_model('xgb_momentum_2', feature_types=['Trend', 'Custom'])
    return es_models

def load_xgb_clf_6D():
    es_models.load_model('xgb_clf_6d', feature_types=['Trend', 'Custom'])
    return es_models


