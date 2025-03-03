import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filters import kama, wwma
from ml_ensemble.tests import eval_model, evaluate_seasonal
from scipy.signal import find_peaks
from finance_models import utils
from statsmodels.tsa.seasonal import STL, MSTL, seasonal_decompose
from finance_models.ml_model import ml_model as ml, xgb_params, gbr_params, rfr_params


class fft_decomp:

    def __init__(self, data):

        self.detrended = False
        self.data = data.dropna()
        self.data['ema'] = self.data.Close.ewm(span=3).mean()


    def kama_detrend(self, period=20, fast=2, slow=30):
        self.detrended_data = (self.data.Close - kama(self.data.Close, period=period, period_fast=fast, period_slow=slow))[period:]
        self.detrend_method = 'kama'
        return

    def wwm_detrend(self, period=20):
        self.detrended_data = self.data.Close - wwma(self.data.Close, n=period)
        self.detrend_method = 'wwma'
        return self.detrended_data


    def fft_data(self, detrended=True, d=1, top_n=5, ema=False):
        if detrended:
            if ema:
                data = self.detrended_data.ewm(span=3).mean()
            else:
                data = self.detrended_data

            self.detrended=True
        else:
            if ema:
                data= self.data.Close.ewm(span=3).mean()
            else:
                data = self.data.Close

            self.detrended=False

        self.res = np.fft.fft(data, axis=0)
        N = len(data)

        self.freqs = np.fft.fftfreq(len(self.res), d=1/d)
        self.magnitude = np.abs(self.res)[:N//2]
        self.periods = 1/self.freqs[:N//2]
        self.periods = self.periods[~np.isinf(self.periods)]
        self.angle = np.angle(self.res)

        peaks, props = find_peaks(self.magnitude, prominence=100)
        peak_periods = self.periods[peaks]
        peak_mags = self.magnitude[peaks]

        sorted_indices = np.argsort(peak_mags)[::-1][:top_n]
        top_periods = peak_periods[sorted_indices]
        top_mags = peak_mags[sorted_indices]
        self.fft_df = pd.DataFrame(top_mags, index=top_periods)
        self.indices = sorted_indices
        self.fft_df['angle'] = np.angle(self.res)[peaks][self.indices]


        return self.fft_df
    def reconstruct_signal(self):
        filtered_data = np.zeros_like(self.res)
        filtered_data[self.indices] = self.res[self.indices]
        filtered_data[-self.indices] = self.res[-self.indices]
        reconstructed_signal = np.fft.ifft(filtered_data).real

        plt.figure(figsize=(12, 8))
        if self.detrended:
            plot_data = self.detrended_data
        else:
            plot_data = self.data.Close

        plt.plot(plot_data.index, plot_data, label='Original')
        plt.plot(plot_data.index, reconstructed_signal, label = 'Reconstructed')
        plt.legend()
        plt.xlabel("Period")
        plt.ylabel("Close")
        plt.show()
        return

    def extrapolate_signal(self, periods_forward, n_frequencies=5, plot=False, scale_rate=1000):
        if self.detrended:
            data = self.detrended_data
        else:
            data = self.data



        t_extended = np.linspace(0, (len(data) + periods_forward) , len(data) + periods_forward, endpoint=False)
        t = np.linspace(0, len(data), len(data), endpoint=False)
        extrapolated_signal = np.zeros_like(t_extended)

        for i in range(n_frequencies):
            extrapolated_signal += self.fft_df[0].iloc[i] * np.sin(2 * np.pi * (1/self.fft_df.index[i]) * t_extended + self.fft_df['angle'].iloc[i])

        if plot:
            plt.figure(figsize=(10,5))
            plt.plot(t, data, label="Original Signal")
            plt.plot(t_extended, extrapolated_signal/scale_rate, label='Extrapolated Signal')
            plt.axvline(x=t[-1], color='red', linestyle='dotted', label='Extrapolation Start')
            plt.legend()


        return extrapolated_signal/scale_rate


class seasonal_model:

    def __init__(self, data, project_dir='F:\\ML\\'):

        self.target = None
        self.decomp_model = None
        self.data = data
        self.features = []
        self.model_info = {
                           'Seasonal':{},
                           'ML':{},
                           'Dir':project_dir
                           }

        self.defaults = {'xgb':xgb_params, 'gbr':gbr_params, 'rfr':rfr_params}

        return


    def find_frequencies(self, detrend_data=False, detrend_method='wwma', detrend_period=25, n_frequencies=25):

        decomp = fft_decomp(self.data)

        if detrend_data:
            if detrend_method == 'wwma':
                decomp.wwm_detrend(detrend_period)
            else:
                decomp.kama_detrend(detrend_period)

        decomp.fft_data(detrended=detrend_data, top_n=n_frequencies)

        return decomp.fft_df

    def seasonal_decomp(self, periods, method="mstl", EMA=False, span=5, plot=False, model='multiplicative', plot_model=False):

        if EMA:
            data = self.data.Close.ewm(span).mean()
        else:
            data = self.data.Close
        if method == 'mstl':

            self.decomp_model = MSTL(self.data.Close,periods=periods).fit()

        elif method == 'stl':
            self.decomp_model = STL(self.data.Close, period=periods).fit()

        else:
            self.decomp_model = seasonal_decompose(self.data.Close, model=model, period=periods)

        self.model_info['Seasonal'].update({'Periods':periods})
        if plot_model:
            self.plot_seasonal_model()

        return self.evaluate_seasonal(method)

    def plot_seasonal_model(self):
        return self.decomp_model.plot()

    def evaluate_seasonal(self, type='mstl'):
        if type=='mstl':
           eval_res = evaluate_seasonal(self.decomp_model)
        if type=='stl':
           eval_res  = eval_model(self.decomp_model.observed, self.decomp_model.seasonal.season+self.decomp_model.trend, self.decomp_model.resid)
        else:
            eval_res = {}

        self.model_info['Seasonal'].update({'Eval':{*eval_res}})

        return self.model_info['Seasonal']['Eval']

    def normalize(self, column):
        normalized = (self.data.Close - self.data[column])/self.data.Close
        return normalized


    def prep_model(self, target_period=20, trend=True, SMAs=True, sma_lens=[20, 50, 100], momentum=True, momentum_lens=[21, 42], residuals=True, normalize_features=True):

        self.features = []

        for i in range(len(self.decomp_model.seasonal.columns)):
            self.data[f'seasonal_{i}'] = self.decomp_model.seasonal.iloc[:, i].values
            self.features.append(f'seasonal_{i}')

        if trend:
            self.data['trend'] = self.decomp_model.trend.values
            self.data['trend_roc'] = self.data.trend.pct_change()
            if normalize_features:
                self.data['trend_x_close'] = self.normalize('trend')

                self.features = self.features + ['trend_roc', 'trend_x_close']
            else:
                self.features = self.features + ['trend', 'trend_roc']


        if residuals:
            self.data['resid'] = self.decomp_model.resid
            if normalize_features:
                self.data['resid'] = self.normalize('resid')

            self.features.append('resid')



        if SMAs:
            for i in sma_lens:
                self.data[f'SMA_{i}'] = self.data.Close.rolling(i).mean()
                if normalize_features:
                    self.data[f'SMA_{i}x'] = (self.data.Close - self.data[f'SMA_{i}'])/self.data.Close
                    self.features.append(f'SMA_{i}x')
                else:
                    self.features.append(f'SMA_{i}')
        if momentum:
            for i in momentum_lens:
                self.data[f'mom_{i}'] = self.data.Close.diff(i)
                self.features.append(f'mom_{i}')


        self.data[f'TARGET_{target_period}'] = self.data.Close.pct_change(-target_period)

        self.target = f'TARGET_{target_period}'

        self.model_info['ML'].update({'Features':self.features, 'Target_period':target_period})

        return self.model_info['ML']

    def train_model(self, type='xgb', params=None, save_file='model_eval.csv', plot=True):
        if params is None:
            params = self.defaults[type]

        ml_model = ml(self.data, self.features, self.target)
        self.model_info['ML'].update({'Train Length': len(ml_model.y_train), 'Test_length': len(ml_model.y_test)})

        if type == 'xgb':
            res = ml_model.xgb_model(params, evaluate=True, eval_log=self.model_info['Dir']+save_file, plot_pred=plot)
            self.model = ml_model.model
        if type == 'gbr':
            res = ml_model.tree_model(params,gbr=True, evaluate=True, eval_log=self.model_info['Dir']+save_file, plot_pred=plot)


        self.model_info['ML'].update({'Type': type, 'Eval':res})
        self.model = ml_model.model

        return self.model_info['ML']['Eval']











