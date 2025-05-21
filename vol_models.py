import os.path
import pickle
import pandas as pd
from feature_engineering import log_returns, historical_rv, rsv
from linear_models import multivariate_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from ml_build.utils import save_model
from sc_loader import sierra_charts as sch
from copy import deepcopy
from TSlib.data_provider.data_factory import data_provider

from sklearn.metrics import mean_squared_error, r2_score

sc = sch()
le = sc.get_chart('LE_F')
scaler = StandardScaler()

class HAR:

    def __init__(self, intraday_data: pd.DataFrame, annualize_vol=False, horizon=1):
        """data : pd.Dataframe containing Open, High, Low, Close"""

        self.eval = None
        self.scaler = None
        self.model = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.pd_ = None
        self.rsv_ = None

        self.vols = {}
        self.returns = log_returns(intraday_data['Close'])
        rv_d = historical_rv(returns=self.returns, window=1, annualize=annualize_vol)

        self.rv_ = pd.DataFrame(data={
            'rv_d' :rv_d,
            'rv_w' : rv_d.rolling(5).mean(),
            'rv_m': rv_d.rolling(22).mean(),
            'rv_t': historical_rv(self.returns, horizon, annualize=annualize_vol).shift(-horizon)
        })
        self.daily_data = intraday_data.resample('1d').apply(sc.resample_logic).dropna()
        self.daily_data = self.daily_data.loc[sorted(set(self.rv_.index) & set(self.daily_data.index))]


        self.rv_ = self.rv_[~np.isinf(self.rv_)].ffill().dropna()


        return

    def rsv_model(self):
        sv = rsv(self.returns)
        rs_neg = sv.iloc[:, 0]
        rs_pos = sv.iloc[:, 1]

        self.rsv_ = pd.DataFrame(
             data=dict(rs_neg=rs_neg, rs_pos=rs_pos,
             rs_neg_w=rs_neg.rolling(5).mean(), rs_pos_w=rs_pos.rolling(5).mean(),
             rs_neg_m=rs_neg.rolling(22).mean(), rs_pos_m=rs_pos.rolling(22).mean(), rv_t=self.rv_.rv_t)
                     )
        self.rsv_ = self.rsv_.ffill().dropna()


        return self.rsv_
    def pd_model(self):
        pd_1 = log_returns(self.daily_data.Close)
        pd_5 = log_returns(self.daily_data.Close, 5)
        pd_22 = log_returns(self.daily_data.Close, 22)
        self.pd_ = pd.DataFrame({'r_d': pd_1, 'rv_d':self.rv_.rv_d,
                                 'r_w': pd_5, 'rv_w':self.rv_.rv_w,
                                 'r_m': pd_22, 'rv_w':self.rv_.rv_m,
                                 'rv_t': self.rv_.rv_t})

        self.pd_ = self.pd_.dropna()

        return self.pd_
    def set_target(self, horizon=1):
        rv_t = historical_rv(self.returns, window=horizon, average=True).shift(-horizon)
        self.rv_['rv_t'] = rv_t
        if self.rsv_ is not None:
            self.rsv_['rv_t'] = rv_t

    def fit(self, penalty=None, alpha=0.8, lasso_cv=5,
            scale_data=False, train_size=0.8, model='rv'):
        features = self.rv_.columns[:-1]
        if scale_data:
            self.scaler = scaler

        else:
            self.scaler = None

        if model == 'rsv':
            if self.rsv_ is None:
                self.rsv_model()

            res = multivariate_regression(self.rsv_, X_cols=self.rsv_.columns[:-1], y_col='rv_t',train_split=True,train_size=train_size, scaler=self.scaler,
                                          penalty=penalty, alpha=alpha, cv=lasso_cv, return_data=True)
            self.x_df = res['X']
        if model == 'pd':
            if self.pd_ is None:
                self.pd_model()
            res = multivariate_regression(self.pd_, X_cols=self.pd_.columns[:-1], y_col='rv_t', scaler=self.scaler,
                                          train_split=True, train_size=train_size, penalty=penalty, cv=lasso_cv, alpha=alpha, return_data=True)
            self.x_df =res['X']

        else:

            res = multivariate_regression(df=self.rv_, X_cols=self.rv_.columns[:-1], y_col='rv_t', penalty=penalty,
                                      scaler=self.scaler, train_split=True, train_size=train_size, return_data=True)
            self.x_df = res['X'].copy()

        self.model = deepcopy(res['model'])
        del res['X']
        del res['model']
        self.eval = res
        self.x_df['preds'] = self.predict(self.x_df)

        return res

    def predict(self, X):
        return self.model.predict(X)


    def save_model(self, dir, model_name):

        if not os.path.isdir(dir):
            os.mkdir(dir)
        if not os.path.isdir(dir+model_name):
            os.mkdir(dir+model_name+'\\')

        working_dir = dir+model_name+'\\'
        save_model(self.model, working_dir+model_name)

        os.chdir(working_dir)
        with open(f'{model_name}_params', 'wb') as f:
            pickle.dump(self.eval, f)
    def load_model(self, dir, model_name):
        model_path = os.path.join(dir, model_name)
        with open(os.path.join(model_path, f'{model_name}.pkl'), 'r') as f:
            self.model = pickle.loads(f)
        with open(os.path.join(model_path, f'{model_name}_params'), 'r') as f:
            self.eval = pickle.loads(f)

        return

