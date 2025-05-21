from sklearn.linear_model import LinearRegression, Lasso, Ridge ,LassoCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt


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



def correlations(data, heatmap=True, feature_correlation=False, ticker=None):
    correlation_matrix = data.corr()

    if heatmap:
        plt.figure(figsize=(14, 6))
        hmap = sns.heatmap(correlation_matrix)
        plt.title('Correlation Betweeen stocks')
        plt.show()

    return correlation_matrix


def multivariate_regression(df: pd.DataFrame, X_cols: list, y_col: str, penalty=None, alpha=0.8, train_split=False,scaler=None,
                            train_size=0.8,cv=3, params=None, return_data=False):
    reg_df = df.copy().ffill().dropna()

    if scaler is not None:
        reg_df = pd.DataFrame(scaler.fit_transform(df), columns=reg_df.columns, index=reg_df.index)

    X = reg_df[X_cols].values
    y = reg_df[y_col].values

    if penalty == 'cv':
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
        'train_end_idx': len(train_y),
        'r2': model.score(test_X, test_y),
        'model': model

    }
    if return_data:
        res['X'] = reg_df[X_cols]


    return res
