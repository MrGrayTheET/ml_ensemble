import os.path
import pickle
import pandas as pd
from feature_engineering import log_returns
import statsmodels.api as sm
from linear_models import multivariate_regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from ml_build.utils import save_model
from arch import arch_model
from tests import eval_model as eval
from sc_loader import sierra_charts as sch
from sklearn.metrics import mean_squared_error, r2_score

scaler = StandardScaler()
def mm_scale(X: pd.DataFrame):
    x_max = X.max()
    x_min = X.min()
    x_scaled = (X - x_min) / (x_max - x_min)
    return x_scaled


def mm_inverse(original_df, scaled_df):
    x_max = original_df.max()
    x_min = original_df.min()
    return scaled_df * (x_max - x_min) + x_min

def r2_sum(data, freq='date'):

    rv_d = data.returns.groupby(data.index.date).apply(lambda x:
                                                           np.sum(x.dropna() ** 2))


    return rv_d

def hls_sum(data):
    range = (data.High - data.Low)/data.Open
    log_rv = range.groupby(data.index.date).apply(lambda x:
                                                        np.sum(x.dropna() ** 2))
    return log_rv

class HAR:

    def __init__(self, data: pd.DataFrame, method='r2_sum'):
        """data : pd.Dataframe containing Open, High, Low, Close"""

        self.model = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.data = data
        self.vols = {}
        self.data['returns'] = log_returns(data['Close'])
        if method == 'hls_sum':
            rv_d = hls_sum(data)
        else:
            rv_d = r2_sum(data)

        self.rv = pd.DataFrame(data={
            'rv_d' :rv_d.shift(1),
            'rv_w' : rv_d.rolling(5).mean(),
            'rv_m': rv_d.rolling(22).mean(),
            'rv_t': rv_d
        })

        self.rv = self.rv[~np.isinf(self.rv)].ffill().dropna()


        return

    def hl_vol(self, high_col='High', low_col='Low'):
        data = self.data
        vol = np.log(data[high_col])- np.log(data[low_col])
        monthly_vol = vol.rolling(21 ).mean()
        weekly_vol = vol.rolling(5).mean()

    def transform(self, train_split_size=0.7, scale=False, target_col='rv_t'):
        if scale:
            data = scaler.fit_transform(self.rv)
            self.scaled = True
        else:
            data = self.rv
        data = sm.add_constant(data)
        # Split train and test sets
        split = int(train_split_size * self.rv.shape[0])
        self.target_col = 'rv_t'
        X = data.drop(target_col, axis=1)
        y = data[[self.target_col]]
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]

        return

    def fit_transform(self,scale=False, train_split_size=0.7, penalty='l1', reg_param=0.1):
        self.transform(train_split_size, scale)
        res = self.fit(penalty=penalty, alpha=reg_param, L1_wt=0)

        return res



    def fit(self, penalty=None, alpha=0.8, L1_wt=0):
        if penalty == 'l1':
            res = sm.OLS(self.y_train, self.X_train).fit_regularized(alpha=0.1, L1_wt=L1_wt)
        else:
            res = sm.OLS(self.y_train, self.X_train).fit()

        features = self.rv.columns
        test_predict = res.predict(self.X_test)
        y_test_vals = ~self.y_test.isna()

        eval_res = eval(self.y_test[y_test_vals], test_predict.loc[y_test_vals.index])
        print(eval_res)
        self.model = res
        self.eval_res = eval_res
        eval_res.update({'Train_End': len(self.X_train), 'Test_Length': len(self.X_test)})
        return res


    def save_model(self, dir, model_name):

        if not os.path.isdir(dir):
            os.mkdir(dir)
        if not os.path.isdir(dir+model_name):
            os.mkdir(dir+model_name+'\\')

        working_dir = dir+model_name+'\\'
        save_model(self.model, working_dir+model_name)

        os.chdir(working_dir)
        with open(f'{model_name}_params', 'wb') as f:
            f.write(self.eval_res)

class VolML:
    def __init__(self, data):

        return


sc = sch()
le = sc.get_chart('LE_F').resample('5min').apply(sc.resample_logic)

har_gold = HAR(le, 'r2_sum')
from sklearn.preprocessing import MinMaxScaler
har_gold.rv = har_gold.rv[~np.isinf(har_gold.rv)].ffill().dropna()
har_gold.transform(scale=False)
har_gold.fit_transform(penalty=None)

