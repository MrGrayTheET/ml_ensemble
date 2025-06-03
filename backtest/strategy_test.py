import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import pandas as pd

class SignalBacktester:
    def __init__(self, data: pd.DataFrame, price_col: str, signal_col: str, return_col: str = None):
        """
        data: DataFrame containing price and signal
        price_col: name of the price column (used to compute returns if return_col not given)
        signal_name: name of the column with trading signals (1, 0, -1)
        return_col: (optional) if returns are precomputed
        """
        self.data = data.copy()
        self.price_col = price_col
        self.signal_col = signal_col
        self.return_col = return_col or "returns"
        self._prepare_returns()

    def _prepare_returns(self):
        if self.return_col not in self.data.columns:
            self.data[self.return_col] = self.data[self.price_col].pct_change().fillna(0)

    def run(self):
        """Runs the backtest and returns a DataFrame with strategy performance."""
        self.data['strategy_returns'] = self.data[self.signal_col].shift(1).fillna(0) * self.data[self.return_col]
        self.data['cumulative_market'] = (1 + self.data[self.return_col]).cumprod()
        self.data['cumulative_strategy'] = (1 + self.data['strategy_returns']).cumprod()
        return self.data

    def plot(self):
        """Plot the cumulative returns."""
        import matplotlib.pyplot as plt
        self.data[['cumulative_market', 'cumulative_strategy']].plot(figsize=(12, 6))
        plt.title("Backtest: Strategy vs. Market")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class RegimeBacktester:
    def __init__(self, data: pd.DataFrame, price_col: str, regime_col: str, signal_map: dict, return_col: str = None):
        """
        data: DataFrame with prices, regimes, and signal columns
        price_col: name of price column used to compute returns
        regime_col: name of column defining the current regime
        signal_map: dict mapping regime name to signal column, e.g. {'bull': 'signal_risk_on', 'bear': 'signal_risk_off'}
        return_col: optional name for existing returns column; otherwise computed from price_col
        """
        self.data = data.copy()
        self.price_col = price_col
        self.regime_col = regime_col
        self.signal_map = signal_map
        self.return_col = return_col or "returns"
        self._prepare_returns()

    def _prepare_returns(self):
        if self.return_col not in self.data.columns:
            self.data[self.return_col] = self.data[self.price_col].pct_change().fillna(0)

    def _build_active_signal_column(self):
        """Construct the active strategy signal based on current regime."""
        active_signals = pd.Series(index=self.data.index, dtype=float)

        for signal_col, regime in self.signal_map.items():
            mask = (self.data[self.regime_col] == regime)
            active_signals[mask] = self.data.loc[mask, signal_col]

        self.data["active_signal"] = active_signals.fillna(0)

    def run(self):
        """Run backtest using regime-based signals."""
        self._build_active_signal_column()
        self.data["strategy_returns"] = self.data["active_signal"].shift(1).fillna(0) * self.data[self.return_col]
        self.data["cumulative_market"] = (1 + self.data[self.return_col]).cumprod()
        self.data["cumulative_strategy"] = (1 + self.data["strategy_returns"]).cumprod()
        return self.data

    def plot(self):
        """Plot the cumulative returns."""
        self.data[["cumulative_market", "cumulative_strategy"]].plot(figsize=(12, 6))
        plt.title("Regime-Based Strategy vs. Market")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.tight_layout()
        plt.show()