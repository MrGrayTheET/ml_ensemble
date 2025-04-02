from statsforecast import StatsForecast
from statsforecast.models import GARCH, ARCH
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from statsforecast.models import (
    GARCH,
    ARCH,
    Naive
)
from feature_engineering import calc_daily_vol, EWMA_Volatility
from utilsforecast.losses import mae, mse
import yfinance as yf
import  numpy as np

models = [ARCH(1),
          ARCH(2),
          GARCH(1,1),
          GARCH(1,2),
          GARCH(2,2),
          GARCH(2,1),
          Naive()]

class vol_modeling:

    def __init__(self, tickers=['^GSPC'],lag=1, n_regimes=3, horizon=5, scale_returns=False, start='2009-01-01', end='2025-03-15', interval='1d',):
        data = yf.download(tickers, start, end, multi_level_index=False, interval=interval)
        self.tickers = tickers
        self.h = horizon
        self.regimes = range(0, n_regimes)
        self.model_info = {}
        self.data = self.reshape_df(data, lag)


    def reshape_df(self, data, lags):
        df = data.copy()
        df = df.loc[:, ('Close', self.tickers)]
        df.columns = df.columns.droplevel()  # drop MultiIndex
        df = df.reset_index()
        prices = df.melt(id_vars='Date')
        prices = prices.rename(columns={'Date': 'ds', 'Ticker': 'unique_id', 'value': 'y'})
        prices = prices[['unique_id', 'ds', 'y']]
        prices['rt'] = prices['y'].div(prices.groupby('unique_id')['y'].shift(lags))
        prices['rt'] = np.log(prices['rt']) * 100
        returns = prices[['unique_id', 'ds', 'rt']]
        returns = returns.rename(columns={'rt': 'y'})

        return returns


    def fit_models(self, freq='B', n_jobs=-1, fcast_horizon=5, step_size=5, n_windows=48):
        sf = StatsForecast(
            models=models,
            freq=freq,
            n_jobs=n_jobs
        )
        self.cv_res = sf.cross_validation(h=fcast_horizon, df=self.data, n_windows=n_windows, step_size=step_size)
        self.sf = sf
        return self.cv_res

    def evaluate_model(self, method='mae'):
        self.cv_res.rename(columns={'y': 'actual'}, inplace=True)

        models = self.cv_res.columns.drop(['unique_id', 'ds', 'cutoff', 'actual'])
        mae_cv = mae(self.cv_res, models=models, target_col='actual').set_index('unique_id')
        return  mae_cv.idxmin(axis=1)



start='2009-01-01'
end='2025-03-15'
horizon = 5
lags = 1
interval = '1d'
freq = 'B'
n_jobs = 1


tickers = ['CL=F', 'GC=F', 'NQ=F','6E=F', '6J=F']

vmods = vol_modeling(tickers,horizon=5, interval='1d')

vmods.fit_models(freq='C',n_jobs=1, fcast_horizon=2, step_size=3, n_windows=200)
vmods.evaluate_model()



