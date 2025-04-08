import numpy as np
import pandas as pd
from sc_loader import sierra_charts as scharts
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Union, Optional
from sklearn.mixture import GaussianMixture as gmm


class TimeSeriesFeatureExtractor:
    """
    A class to extract time series features based on the methodology described in Palma et al. (2023b)
    and Parente et al. (2024).
    """

    def __init__(self, df: pd.DataFrame, interval_size: int = 59, ):
        """
        Initialize the feature extractor with the specified interval size.

        Args:
            interval_size: The size of each interval for feature extraction (default: 59 minutes)
        """
        self.df = df
        self.group_col = 'interval_id'
        self.interval_size = interval_size
        self.df[['open', 'high', 'low', 'close', 'volume']] = self.df[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.df['timestamp'] = df.index
        self.df.index = np.arange(0, len(self.df))
        self.df['interval_id'] = self.df.index // interval_size

    def extract_features(self, price_data: pd.DataFrame, proposed_features=True,
                         classical_features=False, custom_features=True) -> pd.DataFrame:
        """
        Extract all features from the price df.

        Args:
            price_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with extracted features
        """
        df = self.df.copy()
        grouped = df.groupby('interval_id')

        original_grouped = pd.DataFrame()

        # Create a DataFrame to store features
        features = pd.DataFrame(index=grouped.indices.keys())

        if proposed_features:
            # Extract proposed features
            features['peak_curvature'] = grouped.apply(self._extract_peak_curvature)
            features['peak_average_magnitude'] = self._calculate_peak_average_magnitude(features['peak_curvature'], df)
            features['percentage_change'] = grouped.apply(self._calculate_percentage_change)

        if classical_features:
            # Extract traditional technical indicators as recommended by Parente et al. (2024)
            features['rsi'] = grouped.apply(self._calculate_rsi)
            features['ultosc'] = grouped.apply(self._calculate_ultimate_oscillator)
            features['close_zscore'] = grouped.apply(self._calculate_zscore, column='close')
            features['volume_zscore'] = grouped.apply(self._calculate_zscore, column='volume')
            features['ma_ratio'] = grouped.apply(self._calculate_ma_ratio)
            features['timestamp'] = grouped['timestamp'].last()
        if custom_features:
            features['vwap_h'] = grouped.apply(self._extract_high_sub_vwap)
            features['vwap_l'] = grouped.apply(self._extract_vwap_sub_low)

        features['close_pct_change'] = grouped.apply(self._calculate_close_pct_change)

        return features

    def _extract_high_sub_vwap(self, group):
        high = np.max(group['High'])
        group_hlc = (group['High'] + group['Low'] + group['Close']) / 3
        group['vwap'] = (group_hlc * group['Volume']).cumsum() / group['Volume'].cumsum()
        vwap_h = high - group['vwap']
        normalized = np.log(vwap_h / group_hlc)
        return normalized[-1]

    def _extract_vwap_sub_low(self, group):
        low = np.min(group['Low'])
        group_hlc = (group['High'] + group['Low'] + group['Close']) / 3
        group['vwap'] = (group_hlc * group['Volume']).cumsum() / group['Volume'].cumsum()
        vwap_l = group['vwap'] - low

        normalized = np.log(vwap_l / group_hlc)

        return normalized[-1]

    def _extract_peak_curvature(self, group: pd.DataFrame) -> float:
        """
        Extract the curvature of peaks in the close price within an interval.

        Args:
            group: DataFrame containing price df for a single interval

        Returns:
            Average curvature of peaks
        """
        close_prices = group['close'].values

        # Find peaks in the close price
        peaks, _ = find_peaks(close_prices)

        if len(peaks) == 0 or np.min(peaks) == 0 or np.max(peaks) == len(close_prices) - 1:
            return 0.0

        # Calculate curvature at each peak
        curvatures = []
        for peak in peaks:
            if peak > 0 and peak < len(close_prices) - 1:
                # Second-order difference as described: y(k-1), y(k), y(k+1)
                curvature = close_prices[peak - 1] - 2 * close_prices[peak] + close_prices[peak + 1]
                curvatures.append(curvature)

        # Return average curvature if available, otherwise 0
        return np.mean(curvatures) if curvatures else 0.0

    def _calculate_peak_average_magnitude(self, peak_curvatures: pd.Series, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the peak average magnitude using a linear model.

        Args:
            peak_curvatures: Series containing peak curvatures for each interval
            df: Original DataFrame with price df

        Returns:
            Series with peak average magnitude for each interval
        """
        # Group df by interval
        grouped = df.groupby('interval_id')

        # Get peak heights for each interval
        peak_heights = grouped['close'].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)

        # Fit linear model: peak_height ~ peak_curvature
        valid_indices = ~np.isnan(peak_curvatures) & ~np.isnan(peak_heights)

        if np.sum(valid_indices) > 1:
            X = peak_curvatures[valid_indices].values.reshape(-1, 1)
            y = peak_heights[valid_indices].values

            # Simple linear regression coefficients
            beta = np.linalg.lstsq(np.hstack([np.ones((X.shape[0], 1)), X]), y, rcond=None)[0]

            # Calculate fitted values
            peak_avg_magnitudes = beta[0] + beta[1] * peak_curvatures
        else:
            # If not enough df for regression, use the peak heights directly
            peak_avg_magnitudes = peak_heights

        return peak_avg_magnitudes

    def _calculate_percentage_change(self, group: pd.DataFrame) -> float:
        """
        Calculate the estimated percentage change as described in the methodology.

        Args:
            group: DataFrame containing price df for a single interval

        Returns:
            Percentage change
        """
        close_prices = group['close'].values
        if len(close_prices) < self.interval_size:
            return 0.0

        # Calculate average of the last 6 values
        last_6_avg = np.mean(close_prices[-6:])

        # Calculate average of all values
        all_avg = np.mean(close_prices)

        # Calculate percentage change
        pct_change = (last_6_avg / all_avg) - 1

        return pct_change

    def _calculate_rsi(self, group: pd.DataFrame, periods: int = 14) -> float:
        """
        Calculate the Relative Strength Index (RSI) for the interval.

        Args:
            group: DataFrame containing price df for a single interval
            periods: Number of periods to use in RSI calculation

        Returns:
            RSI value for the interval
        """
        close_prices = group['close']
        if len(close_prices) <= periods:
            return 50.0  # Default value if not enough df

        # Calculate price changes
        delta = close_prices.diff()

        # Get gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=periods).mean().iloc[-1]
        avg_loss = loss.rolling(window=periods).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_ultimate_oscillator(self, group: pd.DataFrame) -> float:
        """
        Calculate the Ultimate Oscillator for the interval.

        Args:
            group: DataFrame containing price df for a single interval

        Returns:
            Ultimate Oscillator value for the interval
        """
        if len(group) < 7:
            return 50.0  # Default value if not enough df

        # Calculate buying pressure (BP) and true range (TR)
        close = group['close']
        high = group['high']
        low = group['low']

        # Previous close (shifted)
        prev_close = close.shift(1)

        # Calculate buying pressure
        bp = close - np.minimum(low, prev_close)

        # Calculate true range
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # Sum BP and TR for different periods
        bp_sum7 = bp.rolling(window=7).sum().iloc[-1]
        tr_sum7 = tr.rolling(window=7).sum().iloc[-1]

        bp_sum14 = bp.rolling(window=14).sum().iloc[-1]
        tr_sum14 = tr.rolling(window=14).sum().iloc[-1]

        bp_sum28 = bp.rolling(window=28).sum().iloc[-1]
        tr_sum28 = tr.rolling(window=28).sum().iloc[-1]

        # Calculate three averages with different weights
        if tr_sum7 == 0 or tr_sum14 == 0 or tr_sum28 == 0:
            return 50.0

        avg1 = bp_sum7 / tr_sum7
        avg2 = bp_sum14 / tr_sum14
        avg3 = bp_sum28 / tr_sum28

        # Calculate the Ultimate Oscillator
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7

        return uo

    def _calculate_close_pct_change(self, group: pd.DataFrame) -> float:
        """
        Calculate the percentage change in close price over the interval.

        Args:
            group: DataFrame containing price df for a single interval

        Returns:
            Percentage change in close price
        """
        if len(group) < 2:
            return 0.0

        first_price = group['close'].iloc[0]
        last_price = group['close'].iloc[-1]

        return (last_price / first_price) - 1

    def _calculate_zscore(self, group: pd.DataFrame, column: str, window: int = None) -> float:
        """
        Calculate the Z-Score of a given column.

        Args:
            group: DataFrame containing price df for a single interval
            column: Column to calculate Z-Score for
            window: Optional rolling window size for Z-Score calculation

        Returns:
            Z-Score value at the end of the interval
        """
        if len(group) < 2:
            return 0.0

        series = group[column]

        if window is None or window >= len(series):
            # Use the whole series
            mean = series.mean()
            std = series.std()
        else:
            # Use a rolling window
            mean = series.rolling(window=window).mean().iloc[-1]
            std = series.rolling(window=window).std().iloc[-1]

        if std == 0:
            return 0.0

        # Z-Score of the last value
        return (series.iloc[-1] - mean) / std

    def _calculate_ma_ratio(self, group: pd.DataFrame, short_window: int = 10, long_window: int = 30) -> float:
        """
        Calculate the Moving Average Ratio (short MA / long MA).

        Args:
            group: DataFrame containing price df for a single interval
            short_window: Short moving average window
            long_window: Long moving average window

        Returns:
            MA Ratio at the end of the interval
        """
        close_prices = group['close']

        if len(close_prices) < long_window:
            return 1.0  # Default value if not enough df

        # Calculate short and long moving averages
        short_ma = close_prices.rolling(window=short_window).mean().iloc[-1]
        long_ma = close_prices.rolling(window=long_window).mean().iloc[-1]

        if long_ma == 0:
            return 1.0

        # Calculate MA Ratio
        ma_ratio = short_ma / long_ma

        return ma_ratio

    def predict_next_value(self, price_data: pd.DataFrame, model=None) -> float:
        """
        Predict the next value (at t=60) based on features extracted from t=59.

        Args:
            price_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            model: Pre-trained model for prediction (if None, returns features only)

        Returns:
            Predicted value or features for t=60
        """
        # Extract features
        features = self.extract_features(price_data)

        # Get features for the last interval
        last_features = features.iloc[-1].to_dict()

        if model is None:
            return last_features
        else:
            # Use the provided model to make prediction
            prediction = model.predict([list(last_features.values())])[0]
            return prediction


import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def normalize_x(x_data, method='both'):
    """Apply minmax and/or z-score normalization"""
    x = x_data.values.reshape(-1, 1)
    results = {}

    if method in ('minmax', 'both'):
        minmax = MinMaxScaler().fit_transform(x).flatten()
        results['x_minmax'] = minmax

    if method in ('zscore', 'both'):
        zscore = StandardScaler().fit_transform(x).flatten()
        results['x_zscore'] = zscore
    df = pd.DataFrame(results)

    return df


class PeakExtractor(TimeSeriesFeatureExtractor):
    def __init__(self, df, interval=59, prominence=0.7):
        """
        Initialize financial movement pex

        Parameters:
        - df: DataFrame containing price df
        - price_col: column name for price df
        - group_col: column name for grouping (e.g., ticker, asset class)
        - date_col: column name for datetime index
        """
        super().__init__(df, interval_size=interval)
        if 'Close' in self.df.columns:
            pass
        else:
            self.df['close'] = self.df.Last

        self.price_col = 'close'
        self.prominence = prominence

    def calculate_movement_features(self):
        """Calculate all movement-inspired features for each group"""
        features = []

        for name, group in tqdm(self.df.groupby(self.group_col), desc="Calculating features"):
            prices = group[self.price_col]

            # Basic transformations
            returns = prices.pct_change().dropna()
            log_returns = np.log(prices / prices.shift(1)).dropna()

            # Movement-inspired features
            flat_periods = (returns.abs() < 0.001).sum()  # Idle time equivalent
            total_movement = np.abs(prices.diff()).sum()  # Path length equivalent, normalized
            volatility = log_returns.std() * np.sqrt(252)

            # Direction changes (angle shift equivalent)
            if len(prices) >= 3:
                diffs = prices.diff().dropna()
                angles = np.arctan2(diffs, 1)
                angle_changes = np.abs(angles.diff().dropna())
                total_angle_shift = angle_changes.sum() * (180 / np.pi)
            else:
                total_angle_shift = np.nan

            # Peak features
            peak_features = self._calculate_peak_features(prices)

            features.append({
                self.group_col: name,
                'interval_id': group['interval_id'],
                'flat_periods': flat_periods,
                'total_movement': total_movement,
                'volatility': volatility,
                'total_angle_shift': total_angle_shift,
                **peak_features
            })

        return pd.DataFrame(features)

    def _calculate_peak_features(self, prices, ):
        """Calculate peak-related features"""
        prices = prices.values
        peaks, props = signal.find_peaks(prices, prominence=self.prominence)
        troughs, _ = signal.find_peaks(-prices, prominence=self.prominence)

        if len(peaks) == 0:
            return {
                'n_peaks': 0,
                'avg_peak_height': np.nan,
                'avg_peak_curvature': np.nan,
                'peak_height_std': np.nan,
                'peak_curvature_std': np.nan
            }

        # Calculate curvatures (2nd derivative approximation)
        curvatures = []
        for peak in peaks:
            if peak > 0 and peak < len(prices) - 1:
                curv = prices[peak - 1] - 2 * prices[peak] + prices[peak + 1]
                curvatures.append(curv)
            else:
                curvatures.append(np.nan)

        # Calculate heights (prominence)
        heights = props['prominences']

        return {
            'n_peaks': len(peaks),
            'avg_peak_height': np.nanmean(heights),
            'avg_peak_curvature': np.nanmean(curvatures),
            'peak_height_std': np.nanstd(heights),
            'peak_curvature_std': np.nanstd(curvatures)
        }

    def normalize_features(self, features_df, methods=['minmax', 'z_score']):
        """Normalize features using specified methods"""
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        df = features_df.copy()

        scaled_df = pd.DataFrame(columns=features_df.columns)

        scaler_1 = MinMaxScaler()
        scaler_2 = StandardScaler()

        scaled_df[numeric_cols] = scaler_1.fit_transform(df[numeric_cols])
        scaled_df[numeric_cols] = scaler_2.fit_transform(scaled_df[numeric_cols])

        return scaled_df

    def fit_mixed_effects(self, peak_df, endog_col='height', exogs=['x_minmax', 'x_zscore'], exog_re='curvature'):
        """

        :type exogs: object
        """
        model_df = peak_df[[self.group_col, endog_col, exog_re] + exogs].copy().dropna()
        model = sm.MixedLM(
            endog=model_df['height'],
            exog=sm.add_constant(model_df[[exogs[0], exogs[1]]]),
            groups=model_df['Group'],
        )
        if type(model) == Tuple:
            model = model[0]

        return model.fit()

    def calculate_peak_magnitude_features(self):
        """
        Calculate peak magnitude features for each interval:
        1. Collect peaks and their curvatures
        2. Fit linear model: peak_height ~ curvature
        3. Calculate percentage change using last 6 vs all values
        """
        peak_features = []

        for interval_id, group in self.df.groupby('interval_id'):
            close_prices = group['close'].values

            # 1. Find peaks and calculate curvatures
            peaks, props = find_peaks(close_prices, prominence=self.prominence)
            heights = []
            curvatures = []

            for peak in peaks:
                if 0 < peak < len(close_prices) - 1:
                    # Calculate curvature (2nd order difference)
                    curvature = close_prices[peak - 1] - 2 * close_prices[peak] + close_prices[peak + 1]
                    heights.append(props['prominences'][peaks.tolist().index(peak)])
                    curvatures.append(curvature)

            # 2. Fit linear model if we have peaks
            if len(heights) > 1:
                X = np.array(curvatures).reshape(-1, 1)
                y = np.array(heights)
                model = sm.OLS(y, sm.add_constant(X)).fit()
                avg_magnitude = model.params[0]  # Intercept represents baseline magnitude
            else:
                avg_magnitude = np.mean(heights) if heights else np.nan

            # 3. Calculate percentage change (last 6 vs all)
            if len(close_prices) >= 6:
                last_6_avg = np.mean(close_prices[-6:])
                all_avg = np.mean(close_prices)
                pct_change = (last_6_avg / all_avg) - 1
            else:
                pct_change = np.nan

            peak_features.append({
                'interval_id': interval_id,
                'peak_avg_magnitude': avg_magnitude,
                'peak_pct_change': pct_change,
                'n_peaks': len(peaks)
            })

        return pd.DataFrame(peak_features)

    def extract_features(self, gam_feats_normalized=True, lm_feats_normalized=False, **kwargs):
        """Complete analysis pipeline
        :param **kwargs:
        """
        # 1. Calculate features
        self.results = {}
        features_df = self.calculate_movement_features()
        print("\nCalculating peak magnitude features...")
        features_df = features_df.ffill().dropna()
        peak_magnitude_features = self.calculate_peak_magnitude_features()
        features_df[['peak_avg_magnitude', 'peak_pct_change']] = peak_magnitude_features[
            ['peak_avg_magnitude', 'peak_pct_change']]
        self.results.update({'features': features_df})

        # 2. Normalize features

        normalized_df = self.normalize_features(features_df.select_dtypes(include=[np.number]))
        self.results.update({'normalized_features': normalized_df})

        # 4. Prepare peak df for mixed-effects modeling
        print("\nPreparing peak df for mixed-effects modeling...")
        peak_data = []
        for name, group in tqdm(self.df.groupby(self.group_col), desc="Processing peaks"):
            prices = group[self.price_col].values
            peaks, props = signal.find_peaks(prices, prominence=self.prominence)

            for i, peak in enumerate(peaks):
                if 0 < peak < len(prices) - 1:
                    height = props['prominences'][i]
                    curvature = prices[peak - 1] - 2 * prices[peak] + prices[peak + 1]
                    peak_data.append({
                        self.group_col: name,
                        'height': height,
                        'curvature': curvature,
                        'peak_position': peak
                    })

        peak_df = pd.DataFrame(peak_data)

        normalized_x = normalize_x(peak_df['curvature'])

        x_df = pd.concat([peak_df, normalized_x], axis=1)
        self.results.update({'peak_data': x_df})

        # 4. Fit mixed-effects model
        if len(x_df) > 0:
            print("\nFitting mixed-effects model...")
            self.me_model = self.fit_mixed_effects(
                x_df,
            )
            # Merge with existing features

            features_df[['random_effects', 'random_slopes']] = pd.DataFrame.from_dict(me_model.random_effects,
                                                                                      orient='ihdex')
        else:
            me_model = None
            print("No peaks detected for mixed-effects modeling")

        self.results.update({
            'features': features_df,
            'normalized_features': normalized_df,
            'peak_data': peak_df,
            'mixed_effects_model': me_model,
            'peak_magnitude_features': peak_magnitude_features
        })

        return self.results

    def predict(self, use_normalized_features=True, n_steps=None):
        peak_df = self.results['peak_data']
        if use_normalized_features:
            df = self.results['normalized_features']
        else:
            df = self.results['features']

        X = df.select_dtypes(include=[np.number]).ffill().dropna()
        if n_steps is None:
            n_steps = len(X)

        height_prediction = self.results['mixed_effects_model'].predict(self.results['peak_data'].iloc[:n_steps])


def obtain_peak_hess(data):
    """Extract peaks and curvature from financial time series

    This function extracts peaks and curvature
    from a given financial time series by applying
    second-order differentiation to identified peaks.
    A peak is defined as a local maximum or minimum
    point in the time series, determined by comparing
    the value at each point to the values of the immediately
    preceding and following observations.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        The selected region of the time series
        that the features will be extracted.

    Returns
    -------
    dict
        A dictionary containing the selected
        the peaks, peak indices and the raw
        data.
    """

    dat = data.groupby('Minute')
    peak = {name: [] for name in dat.groups}
    peak_index = {name: [] for name in dat.groups}
    hess = {name: [] for name in dat.groups}

    for name, group in dat:
        peaks, _ = find_peaks(group['Value'])
        peak[name] = group['Value'].iloc[peaks].tolist()
        peak_index[name] = peaks.tolist()

    for name, group in dat:
        for k in range(len(peak_index[name])):
            if len(peak_index[name]) == 1 and peak_index[name][0] == 0:
                hess[name].append(0)
            else:
                idx = peak_index[name][k]
                if idx > 0 and idx < len(group) - 1:
                    hess_val = \
                        np.diff(np.diff(group['Value'].iloc[idx-1:idx+2]))
                    hess[name].append(hess_val[0] if len(hess_val) > 0 else 0)
                else:
                    hess[name].append(0)

    peak_hess = []
    for name in dat.groups:
        peak_hess.append(pd.DataFrame({'Minute': float(name),
                                       'peak': peak[name],
                                       'hess': hess[name]}))

    peak_hess = pd.concat(peak_hess, ignore_index=True)

    res = {'result': peak_hess,
           'dat': dat,
           'peak': peak,
           'peak_index': peak_index}

    return res


def collect_peak_parameters(peak_hess_data):
    """Linear model' slope and intercept for peak and curvature

    This function obtains the Linear model' slope and intercept
    of a linear model fit using the number of peaks and the
    curvature obtained by the financial time series.

    Parameters
    ----------
    peak_hess_data : dict
                  A dictionary containing the selected
                  the peaks, peak indices and the raw
                  data.


    Returns
    -------
    pandas.core.frame.DataFrame
        A data frame containing the linear models' slope
        and intercept based on the number of peaks and
        curvature extracted from the financial time series.

    Examples
    --------
    data = pd.DataFrame({
    'Minute': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    'Value': [1, 2, 3, 2, 1, 2, 3, 5, 3, 2, 2, 4, 6, 4, 2]
    })

    peak_hess_data = obtain_peak_hess(data)
    peak_parameters = collect_peak_parameters(peak_hess_data)

    print("Peak and Hess Data:")
    print(peak_hess_data['result'])
    print("\nPeak Parameters:")
    print(peak_parameters)
    """
    peak_curvature_parameters = peak_hess_data['result'].groupby('Minute').agg(
        average_peak_magnitude=('peak', 'mean'),
        average_peak_curvature=('hess', 'mean')
    ).reset_index()

    n_peaks = [len(peaks) for peaks in peak_hess_data['peak'].values()]
    peaks_perseccond_data = \
        pd.DataFrame({'Minute': list(peak_hess_data['peak'].keys()),
                                          'n_peaks': n_peaks})
    peaks_perseccond_data['persecond'] = \
        peaks_perseccond_data['n_peaks'] / len(list(peak_hess_data['dat'])[0])

    Mixed_Models_data = peak_hess_data['result'].groupby('Minute').apply(
        lambda x: pd.Series({
            'MM_Intercept': np.polyfit(x['hess'], x['peak'], 1)[1],
            'MM_Hess': np.polyfit(x['hess'], x['peak'], 1)[0]
        })
    ).reset_index()

    peak_parameters_data = pd.merge(
    Mixed_Models_data,
    peak_curvature_parameters[[
        'Minute',
        'average_peak_magnitude',
        'average_peak_curvature'
    ]],
    on='Minute')
    peak_parameters_data = pd.merge(peak_parameters_data,
                                    peaks_perseccond_data[['Minute',
                                                           'persecond']],
                                                           on='Minute')

    return peak_parameters_data

