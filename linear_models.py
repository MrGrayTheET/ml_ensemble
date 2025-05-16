from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge ,LassoCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import os
from fredapi import Fred
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

fredkeyfile = "F:\\Macro\\fredapi.txt"
with open(fredkeyfile) as fredkey:
    key = fredkey.read()

fred = Fred(key)

def weighted_index(df: pd.DataFrame, weights: dict, base_value: float = 100.0,
                   normalize: bool = True) -> pd.Series:
    """
    Create a weighted index from a DataFrame and a weight dictionary.

    Parameters:
        df (pd.DataFrame): DataFrame with columns representing assets (e.g., ETF prices or returns).
        weights (dict): Dictionary of {column_name: weight}. Weights don't need to sum to 1.
        base_value (float): Starting value of the index if normalized.
        normalize (bool): Whether to normalize each series to start at 1 before weighting.

    Returns:
        pd.Series: The weighted index as a time series.
    """
    # Filter and align only columns present in both df and weights
    valid_assets = [col for col in weights if col in df.columns]
    missing_assets = [col for col in weights if col not in df.columns]

    ret_df = pd.DataFrame()

    if not valid_assets:
        raise ValueError("None of the weights match the DataFrame columns.")

    if missing_assets:
        print(f"Warning: These tickers were not found in the DataFrame and will be ignored: {missing_assets}")

    # Normalize weights
    weight_sum = sum(weights[col] for col in valid_assets)
    normalized_weights = {col: weights[col] / weight_sum for col in valid_assets}

    df_sub = df[valid_assets].copy()

    # Normalize price series to start at 1 if needed
    if normalize:
        df_sub = df_sub / df_sub.iloc[0]

    # Apply weights
    ret_df['returns'] = df_sub.mul([normalized_weights[col] for col in df_sub.columns], axis=1).sum(axis=1)
    ret_df['index'] = (ret_df['returns'] * base_value).cumsum()

    # Scale to base value
    return ret_df['returns']


def load_csv(fp, transformation=None, args=None):
    series = pd.read_csv(fp, index_col=[0], date_format='%Y-%m-%d')
    if transformation is None:
        return series
    elif args is None:
        transformed_series = transformation(series)
    else:
        transformed_series = transformation(series, *args)

    return transformed_series



def get_macro_vars(macro_vars: list, transformation=None, args=None, resample=False):
    if args is None:
        args = []
    macro_df = pd.DataFrame()
    if transformation is None:
        transformation = lambda x: x

    for var in macro_vars:
        series = fred.get_series(var)
        if args is None:
            transformed_series = transformation(series)
        else:
            transformed_series = transformation(series, *args)
        macro_df[var] = transformed_series

    return macro_df


def linreg(df, x_col, y_col, lags=3, penalty=None):
    data_df = df.copy().ffill().dropna()
    x = sm.add_constant(data_df[x_col])[lags:]
    y = data_df[y_col].shift(lags)[lags:]

    ols = sm.OLS(y, x)
    stats = ols.fit()

    return stats


def correlations(data, heatmap=True, feature_correlation=False, ticker=None):
    correlation_matrix = data.corr()

    if heatmap:
        plt.figure(figsize=(14, 6))
        hmap = sns.heatmap(correlation_matrix)
        plt.title('Correlation Betweeen stocks')
        plt.show()

    return correlation_matrix


def multivariate_regression(df: pd.DataFrame, X_cols: list, y_col: str, penalty=None, alpha=0.8, train_split=False,scaler=None,
                            train_size=0.8,cv=3, params=None):
    reg_df = df.copy().ffill().dropna()

    if scaler is not None:
        reg_df = scaler.fit_transform(df)

    X = reg_df[X_cols].values
    y = reg_df[y_col].values

    if  (penalty == 'cv'):
        regressor = LassoCV(cv=cv, random_state=42)
    elif penalty == 'l1':
        regressor = Lasso(alpha=alpha, random_state=42)
    elif penalty == 'l2':
        regressor = Ridge(alpha=alpha, random_state=42)
    else:
        regressor = LinearRegression()

    if train_split:
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_size)
    else:
        train_X, train_y, test_X, test_y = X, y, X, y

    model = regressor.fit(train_X, train_y)
    test_preds = model.predict(test_X)
    train_preds = model.predict(train_X)

    res = {
        'coefs': model.coef_,
        'train_mse': mean_squared_error(train_y,train_preds),
        'train_rmse': root_mean_squared_error(train_y, train_preds),
        'test_mse': mean_squared_error(test_y,test_preds),
        'rmse':root_mean_squared_error(test_y, test_preds),
        'data_std':y.std(),
        'rmse/sd': root_mean_squared_error(test_y, test_preds)/y.std(),
        'r2': model.score(test_X, test_y),
        'model': model

    }
    print(res)

    return res


class MacroModel:

    def __init__(self, tickers=[], data_dir='F:\\factors\\', loading_method='yf', model_name='factor_example',
                 interval='1mo', start='2020-01-01', end='2025-04-09'):
        self.macro_df = None
        self.new_col = lambda col_name: [(col_name, ticker) for ticker in tickers]
        self.ticker_ohlc = lambda ticker: tuple(zip([ticker * 4], ['Open', 'High', 'Low', 'Close']))

        if loading_method == 'sc':
            import sc_loader as sc
            loader = sc()
            self.load = loader.open_formatted_files
            self.data = self.load()
            columns = pd.MultiIndex.from_product([['Open', 'High', 'Low', 'Close'], tickers])
            df = pd.DataFrame(columns=columns, index=pd.date_range(start, end, freq=interval))




        else:
            self.load = yf.download
            if (interval.endswith('w')):
                yf_interval = '1wk'

            elif interval.endswith('min'):
                yf_interval = interval.strip('in')

            else:
                yf_interval = interval

            self.data = self.load(tickers, start=start, end=end, interval=yf_interval)
            self.cols = lambda col_name: [(col_name, ticker) for ticker in tickers]

        if not data_dir.endswith('\\'):
            self.dir = data_dir + '\\'
        else:
            self.dir = data_dir

        self.data[self.cols('returns')] = np.log(self.data.Close.values / self.data.Close.shift(1).values)

        if not os.path.isdir(data_dir):
            os.mkdir(data_dir + '\\')
            os.mkdir(data_dir + model_name)
        elif not os.path.isdir(data_dir + model_name):
            os.mkdir(data_dir + model_name)
        else:
            overwrite = input(
                f"Directory {self.dir} already contains a model named {model_name}\n Would you like to overwrite? Y/N").strip().lower()
            if overwrite != 'y':
                pass
            else:
                exit()

        return

    def price_index(self, weight_dict: dict, type='Fred', data=None, normalize=False):
        series_names = [*weight_dict.keys()]
        if type == 'Fred':
            _df = get_macro_vars(series_names, transformation=lambda x:np.log(x/x.shift(1)))
        else:
            _df = self.load(series_names)

        index_df = weighted_index(_df, weight_dict, normalize=normalize)

        return index_df

    def benchmark_regression(self, feature_col=None, benchmark_ticker='^GSPC'):
        if feature_col is None:
            y = yf.download(benchmark_ticker, start=self.data.index[0], end=self.data.index[-1])['Close']
        else:
            y = self.data[feature_col]

        for i, col in self.data['returns'].columns:
            x = self.data['returns'][col]

        return
