import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Union, Optional
from sklearn.mixture import GaussianMixture as gmm




class TimeSeriesFeatureExtractor:
    """
    A class to extract time series features based on the methodology described in Palma et al. (2023b)
    and Parente et al. (2024).
    """

    def __init__(self, interval_size: int = 59):
        """
        Initialize the feature extractor with the specified interval size.

        Args:
            interval_size: The size of each interval for feature extraction (default: 59 minutes)
        """
        self.interval_size = interval_size

    def extract_features(self, price_data: pd.DataFrame, proposed_features=True,
                         classical_features=False) -> pd.DataFrame:
        """
        Extract all features from the price data.

        Args:
            price_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with extracted features
        """
        self.data = price_data
        # Make a copy to avoid modifying the original data
        df = self.data.copy()
        df[['open', 'high', 'low', 'close', 'volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df['timestamp'] = df.index

        df.index = np.arange(0, len(df))
        # Group data into intervals
        df['interval_id'] = df.index // self.interval_size
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

        features['close_pct_change'] = grouped.apply(self._calculate_close_pct_change)

        return features

    def _extract_peak_curvature(self, group: pd.DataFrame) -> float:
        """
        Extract the curvature of peaks in the close price within an interval.

        Args:
            group: DataFrame containing price data for a single interval

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
            df: Original DataFrame with price data

        Returns:
            Series with peak average magnitude for each interval
        """
        # Group data by interval
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
            # If not enough data for regression, use the peak heights directly
            peak_avg_magnitudes = peak_heights

        return peak_avg_magnitudes

    def _calculate_percentage_change(self, group: pd.DataFrame) -> float:
        """
        Calculate the estimated percentage change as described in the methodology.

        Args:
            group: DataFrame containing price data for a single interval

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
            group: DataFrame containing price data for a single interval
            periods: Number of periods to use in RSI calculation

        Returns:
            RSI value for the interval
        """
        close_prices = group['close']
        if len(close_prices) <= periods:
            return 50.0  # Default value if not enough data

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
            group: DataFrame containing price data for a single interval

        Returns:
            Ultimate Oscillator value for the interval
        """
        if len(group) < 7:
            return 50.0  # Default value if not enough data

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
            group: DataFrame containing price data for a single interval

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
            group: DataFrame containing price data for a single interval
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
            group: DataFrame containing price data for a single interval
            short_window: Short moving average window
            long_window: Long moving average window

        Returns:
            MA Ratio at the end of the interval
        """
        close_prices = group['close']

        if len(close_prices) < long_window:
            return 1.0  # Default value if not enough data

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
