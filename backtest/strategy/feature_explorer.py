import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from seaborn import heatmap as hmap

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def linear_eval(df, y_col, features=None,penalty='cv',split=True,train_size=0.8, **params):
    data = df[features + [y_col]].dropna()

    if features is None:
        features = df.columns[:-1]

    if penalty == 'cv':
        Linreg=LassoCV(**params)
        scaler = StandardScaler()
        scaled_data = scaler.fit(data)
        data = pd.DataFrame(data=scaled_data, columns=data.columns, index=data.index)
    else:
        Linreg = LinearRegression(**params)

    X = data[features]
    y = data[y_col]
    if split:
        train_idx = int(0.8 * len(X))
        train_x, train_y, test_x, test_y = X[:train_idx], y[:train_idx], X[train_idx:],y[train_idx:]
    else:
        train_x, train_y, test_x, test_y = (X, y) * 2
    # Create a new DF to keep dates in line while removing NAs

    Linreg.fit(train_x, train_y)
    train_preds = Linreg.predict(train_x)
    test_preds = Linreg.predict(test_x)

    res = {
        'coefs':Linreg.coef_,
        'test_score':Linreg.score(test_x, test_y),
        'training_mse':mse(train_y, train_preds),
        'test_mse':mse(test_y, test_preds),
        'train_rmse': rmse(train_y,  train_preds),
        'test_rmse': rmse(test_y, test_preds),
        }

    return res











