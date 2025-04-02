import pandas as pd
from backtesting import Strategy, Backtest
import numpy as np
from backtesting.test import SMA
from models.es_f import load_xgb_regressor, load_xgb_clf, es_models, load_xgb_clf_6D, load_xgb_momentum

ES_regressor = load_xgb_regressor()
ES_clf = load_xgb_clf()

TRAIN_INDEX = ES_regressor.model_info['ML']['Train Length']

def EMA(values, n):
    return pd.Series(values).ewm(span=n).mean()

class MlRegressorStrategy(Strategy):
    """
    Backtesting strategy using a trained regression model (e.g., RandomForestRegressor).

    Parameters:
        model: Trained regression model.
        buy_threshold: Prediction value above which we go long.
        sell_threshold: Prediction value below which we go short.
        dynamic_threshold: If True, dynamically computes percentiles for thresholds.
    """
    trailing_stop: pd.Series
    model_cls = ES_regressor
    model = ES_regressor.model
    buy_threshold = 0.01  # Default fixed threshold for entering long positions
    sell_threshold = -0.01  # Default fixed threshold for entering short positions
    dynamic_threshold = True  # Set to True to use percentile-based thresholds
    trailing_sl = True
    feature_list = ES_regressor._features
    SMA_len = 50
    sell_hold_buy = [0, 1, 2]

    def init(self):
        """Initialize strategy with features for the ML model."""

        # Compute dynamic thresholds if enabled
        if self.dynamic_threshold:
            self.buy_threshold, self.sell_threshold = self.compute_dynamic_thresholds()
        if self.trailing_sl:
            self.trailing_ma = self.I(SMA, self.data.Close, self.SMA_len)

    def next(self):
        """Generate trading signals using the trained regression model and execute trades."""
        if self.model is None:
            raise ValueError("No trained regression model provided to strategy.")
        if len(self.data) < TRAIN_INDEX:
            return

        # Get latest data point
        current_features = self.data.df[self.feature_list].iloc[-2:]

        # Make prediction (continuous value)
        prediction = self.model.predict(current_features)[-1]

        # Convert prediction into trade signals

        if prediction >= self.buy_threshold and not self.position:
            self.buy()
        elif prediction <= self.sell_threshold and not self.position:
            self.sell()

        if self.position and self.trailing_sl:
            trailing_stop = self.trailing_ma[-1]
            if self.position.is_long and self.data.Close[-1] < trailing_stop:
                self.position.close()
            elif self.position.is_short and self.data.Close[-1] > trailing_stop:
                self.position.close()

        elif self.position and not self.trailing_sl:
            if self.position.is_long and prediction == self.sell_hold_buy[0]:
                self.position.close()
            elif self.position.is_short and prediction == self.sell_hold_buy[2]:
                self.position.close()



    def calculate_rsi(self, series, period=14):
        """Calculate the RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def compute_dynamic_thresholds(self):
        """Dynamically computes buy/sell thresholds based on past model predictions."""
        past_predictions = self.model.predict(self.model_cls.x_data)
        buy_threshold = np.percentile(past_predictions, 70)  # 75th percentile for buy
        sell_threshold = np.percentile(past_predictions, 30)  # 25th percentile for sell
        return buy_threshold, sell_threshold

class MlClfStrategy(Strategy):
    """
    Backtesting strategy using a trained regression model (e.g., RandomForestRegressor).

    Parameters:
        model: Trained regression model.
        buy_threshold: Prediction value above which we go long.
        sell_threshold: Prediction value below which we go short.
        dynamic_threshold: If True, dynamically computes percentiles for thresholds.
    """
    trailing_stop: pd.Series
    model_cls = ES_clf
    model = ES_clf.model
    buy_threshold = 0.01  # Default fixed threshold for entering long positions
    sell_threshold = -0.01  # Default fixed threshold for entering short positions
    dynamic_threshold = False  # Set to True to use percentile-based thresholds
    trailing_sl = False
    feature_list = ES_clf._features
    SMA_len = 21
    sell_hold_buy = [0, 1, 2]

    def init(self):
        """Initialize strategy with features for the ML model."""

        # Compute dynamic thresholds if enabled
        if self.dynamic_threshold:
            self.buy_threshold, self.sell_threshold = self.compute_dynamic_thresholds()
        if self.trailing_sl:
            self.trailing_ma = self.I(EMA, self.data.Close, self.SMA_len)

    def next(self):
        """Generate trading signals using the trained regression model and execute trades."""
        if self.model is None:
            raise ValueError("No trained regression model provided to strategy.")
        if len(self.data) < TRAIN_INDEX:
            return

        # Get latest data point
        current_features = self.data.df[self.feature_list].iloc[-2:]

        # Make prediction (continuous value)
        prediction = self.model.predict(current_features)[-1]

        # Convert prediction into trade signals

        if prediction == self.sell_hold_buy[2] and not self.position:
            self.buy()
        elif prediction == self.sell_hold_buy[0] and not self.position:
            self.sell()

        if self.position and self.trailing_sl:
            trailing_stop = self.trailing_ma[-1]
            if self.position.is_long and self.data.Close[-1] < trailing_stop:
                self.position.close()
            elif self.position.is_short and self.data.Close[-1] > trailing_stop:
                self.position.close()

        elif self.position and not self.trailing_sl:
            if self.position.is_long and prediction == self.sell_hold_buy[0]:
                self.position.close()
            elif self.position.is_short and prediction == self.sell_hold_buy[2]:
                self.position.close()



    def calculate_rsi(self, series, period=14):
        """Calculate the RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def compute_dynamic_thresholds(self):
        """Dynamically computes buy/sell thresholds based on past model predictions."""
        past_predictions = self.model.predict(self.model_cls.x_data)
        buy_threshold = np.percentile(past_predictions, 75)  # 75th percentile for buy
        sell_threshold = np.percentile(past_predictions, 30)  # 25th percentile for sell
        return buy_threshold, sell_threshold


bt = Backtest(ES_clf.data , MlClfStrategy)
bt.run()
bt.plot()
