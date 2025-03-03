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

class technical_model:

    def __init__(self, data, project_dir='F:\\ML\\'):


        self.model = None
        self.target = None
        self.decomp_model = None
        self.data = data
        self.features = []

        if not os.path.isdir(project_dir): os.mkdir(project_dir)

        self.model_info = {
            'Seasonal': {'Features':[]},
            'Trend': {'Features':[]},
            'Volume': {'Features':[]},
            'Volatility': {'Features':[]},
            'ML': {},
            'Dir': project_dir
        }

        self.defaults = {'xgb': xgb_params, 'gbr': gbr_params, 'rfr': rfr_params}

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

    def seasonal_decomp(self, periods, method="mstl", EMA=False, span=5, plot=False, model='multiplicative',
                        plot_model=False):

        if EMA:
            data = self.data.Close.ewm(span).mean()
        else:
            data = self.data.Close
        if method == 'mstl':
            self.decomp_model = MSTL(self.data.Close, periods=periods).fit()

        elif method == 'stl':
            self.decomp_model = STL(self.data.Close, period=periods).fit()
        else:
            self.decomp_model = seasonal_decompose(self.data.Close, model=model, period=periods)

        self.model_info['Seasonal'].update({'Periods': periods})

        if plot_model:
            self.plot_seasonal_model()

        return

    def plot_seasonal_model(self):
        return self.decomp_model.plot()

    def evaluate_seasonal(self):
        eval_res = es(self.decomp_model, type=type)

        self.model_info['Seasonal'].update({'Eval': eval_res})

        return self.model_info['Seasonal']['Eval']

    def normalize_ma(self, column):
        normalized = (self.data.Close - self.data[column]) / self.data.Close
        return normalized

    def normalize_indicator(self, column):
        normalized = self.data[column] / self.data.Close
        return normalized

    def seasonal_features(self, seasonals=True, residuals=True, normalize_features=False):

        features = []

        if seasonals:
            for i in range(len(self.decomp_model.seasonal.columns)):
                self.data[f'seasonal_{i}'] = self.decomp_model.seasonal.iloc[:, i].values

                if normalize_features:
                    self.data[f'seasonal_{i}'] = self.normalize_indicator(f'seasonal_{i}')

                features.append(f'seasonal_{i}')

        if residuals:
            self.data['resid'] = self.decomp_model.resid

            if normalize_features:
                self.data['resid'] = self.normalize_indicator('resid')

            features.append('resid')

        self.model_info['Seasonal'].update({'Features': features})

        return self.data[features]

    def trend_features(self, trend=True, SMAs=False, sma_lens=None, momentum=False,
                       momentum_lens=None, BBands=False, bband_window=20, bband_width=2, normalize_features=True):

        if sma_lens is None:
            sma_lens = [20, 50, 100]

        if momentum_lens is None:
            momentum_lens = [21, 42]

        features = []

        if trend:
            self.data['trend'] = self.decomp_model.trend.values
            self.data['trend_roc'] = self.data.trend.pct_change()

            if normalize_features:
                self.data['trend_x_close'] = self.normalize_ma('trend')

                features = features + ['trend_roc', 'trend_x_close']
            else:
                features = features + ['trend', 'trend_roc']

        if SMAs:
            for i in sma_lens:
                self.data[f'SMA_{i}'] = self.data.Close.rolling(i).mean()
                if normalize_features:
                    self.data[f'SMA_{i}x'] = (self.data.Close - self.data[f'SMA_{i}']) / self.data.Close
                    features.append(f'SMA_{i}x')
                else:
                    features.append(f'SMA_{i}')

        if momentum:
            for i in momentum_lens:
                self.data[f'mom_{i}'] = self.data.Close.diff(i)
                features.append(f'mom_{i}')

        if BBands:
            self.data['bb_ma'] = self.data.Close.rolling(bband_window).mean()
            self.data['bb_upper'] = self.data.bb_ma + (bband_width * self.data.Close.rolling(bband_window).std())
            self.data['bb_lower'] = self.data.bb_ma - (bband_width * self.data.Close.rolling(bband_window).std())

            if normalize_features:
                for i in ['bb_ma', 'bb_upper', 'bb_lower']:
                    self.data[i + '_x'] = self.normalize_ma(i)
                    features.append(i + '_x')

            else:
                features = features + ['bb_ma', 'bb_upper', 'bb_lower']

        self.model_info['Trend'].update({'Features': features})

        return self.model_info['ML']

    def volume_features(self, volume_data=True, volume_MA=False, ma_lens=[5, 10], ma_diffs=False,
                        rel_vol=True, rel_by='tdays', length=3,
                        cum_vol=False, by='month', cum_len=5):

        features = []

        volume = self.data.Volume

        if volume_data:
            features.append('Volume')

        if volume_MA or ma_diffs:

            for i in ma_lens:
                self.data[f'Volume_{i}'] = volume.rolling(i).mean()
                if volume_MA: features.append(f'Volume_{i}')
                if ma_diffs:
                    self.data[f'Volume_{i}_x'] = (volume - self.data[f'Volume_{i}']) / volume.rolling(65).std()
                    features.append(f'Volume_{i}_x')

        if rel_vol:
            if rel_by == 'tdays':
                trading_days = get_trading_days(self.data.index.year.min(), self.data.index.year.max() + 1)
                self.data['tdays'] = trading_days.loc[self.data.index.date[:-1]]

            self.data[f'rvol_{length}'] = rvol(self.data, length=length, by=rel_by)
            features.append(f'rvol_{length}')

        if cum_vol:
            self.data[f'cum_rvol_{cum_len}'] = cum_rvol(self.data, cum_len)
            features.append(f'cum_rvol_{cum_len}')

        self.model_info['Volume'].update({'Features': features})

        return self.data[features]

    def volatility_features(self, tr_ratio=False, atr_ratio_len=7, ATR=False, ATR_len=5, nATR=True, nATR_len=8,
                            TR=False):

        features = []

        if TR:
            self.data['TR'] = tr(self.data.Close)
        if tr_ratio:
            self.data['tr_ratio'] = tr(self.data) / atr(self.data, length=atr_ratio_len)
            features.append('tr_ratio')

        if ATR:
            self.data['ATR'] = atr(self.data, length=ATR_len)
            features.append('ATR')

        if nATR:
            self.data['nATR'] = atr(self.data, length=nATR_len, normalized=True)
            features.append('nATR')

        self.model_info['Volatility'].update({'Features':features})

        return self.data[features]

    def train_model(self, target_period=10, feat_types=['Volume', 'Seasonal', 'Trend', 'Volatility'], method='xgb', params=None,
                    save_file='model_eval.csv', plot=True):

        self.data[f'TARGET_{target_period}'] = self.data.Close.pct_change(-target_period)
        self.target = f'TARGET_{target_period}'

        self.features = []

        for i in feat_types: self.features = self.features + self.model_info[i]['Features']

        if params is None: params = self.defaults[method]

        data = self.data[self.features + [self.target]].dropna()

        ml_model = ml(data, self.features, self.target)

        self.model_info['ML'].update({'Train Length': len(ml_model.y_train), 'Test_length': len(ml_model.y_test)})

        if method == 'xgb':
            res = ml_model.xgb_model(params, evaluate=True, eval_log=self.model_info['Dir'] + save_file, plot_pred=plot)

        if method == 'gbr':
            res = ml_model.tree_model(params, gbr=True, evaluate=True, eval_log=self.model_info['Dir'] + save_file,
                                      plot_pred=plot)

        self.model_info['ML'].update({'Type': method, 'Eval': res})

        self.model = ml_model.model

        return self.model_info['ML']['Eval']












