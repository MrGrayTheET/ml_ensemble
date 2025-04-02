import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import torch
import pickle
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error,
                             r2_score,
                             accuracy_score,
                             log_loss)


def clean_data(data, feats, target_col, sequence=False, periods_in=50, periods_out=20,
               train_split=True, train_size=0.80, scale_x=True, scale_y=False, x_scale_type='standard', y_scale_type='minmax',
               minmax_settings=(0, 1),
               to_tensor=False, return_y_scaler=True):
    # Cleans data and converts to array
    cols = feats + [target_col]
    data = data[cols]
    x_data = data[feats]
    y_data = data[target_col] # Target data

    if scale_x:
        if x_scale_type == 'minmax':

            scale = MinMaxScaler(minmax_settings)

        else:
            scale = StandardScaler()

        scale.fit(x_data)
        x_scaler = scale
        x_data = scale.transform(x_data)

        if scale_y:
            if y_scale_type == 'minmax':
                y_scaler = MinMaxScaler(minmax_settings)
            else:
                y_scaler = StandardScaler()

            y_scaler.fit(data[[target_col]].values)
            y_data = y_scaler.fit_transform(data[[target_col]].values)

    if sequence:
        X, y = [], []  # Sequences for deep learning arrays
        seq_x, seq_y = [], []

        for i in range(len(x_data)):
            end_idx = periods_in + i
            out_end_idx = end_idx + periods_out - 1
            if out_end_idx > len(x_data): break
            seq_x, seq_y = x_data[i:end_idx], y_data[end_idx - 1:out_end_idx]
            if (len(seq_y[np.isnan(seq_y)] > 0) | len(seq_x[np.isnan(seq_x)] > 0)):continue
            X.append(seq_x), y.append(seq_y)


        y_arr = np.array(y)
        X_arr =  np.array(X)

    else:
        X_arr, y_arr = x_data, y_data  # Return previous data

    if train_split:
        train_len = round(len(X_arr) * train_size)  # Split arrays for train/test
        x_train = X_arr[:train_len]
        y_train = y_arr[:train_len]
        x_test = X_arr[train_len:]
        y_test = y_arr[train_len:]

        if to_tensor:  # Tensors for deep learning
            x_train = torch.Tensor(x_train)
            x_test = torch.Tensor(x_test)
            y_train = torch.Tensor(y_train)
            y_test = torch.Tensor(y_test)


        if scale_y and return_y_scaler:
            return [x_train, y_train], [x_test, y_test], y_scaler

        else:
            return [x_train, y_train], [x_test, y_test]

    else:

        return X_arr, y_arr


def plot_predictions(train_predict, test_predict, y_train, y_test, y_scaler):
    train_pred, test_pred = [], []
    train_true, test_true = [], []
    train_predictions = y_scaler.inverse_transform(train_predict.data.numpy())
    test_predictions = y_scaler.inverse_transform(test_predict.data.numpy())
    train_y = y_scaler.inverse_transform(y_train.data.numpy())
    test_y = y_scaler.inverse_transform(y_test.data.numpy())

    for i in range(len(train_predictions)):
        train_pred.append(train_predictions[i][0])
        train_true.append(train_y[i][0])

    for i in range(len(test_predictions)):
        test_pred.append(test_predictions[i][0])
        test_true.append(test_y[i][0])

    fig, ax= plt.subplots(2)
    ax[0].plot(train_pred, label='predicted')
    ax[0].plot(train_true, label='Actual')
    ax[1].plot(test_true, label='Actual')
    ax[1].plot(test_pred, label='Predicted')
    plt.show()


def evaluate_model(test_predict, y_test, features, log_file='rfr_log.csv', sorted_features=False, ):

    pred_std_dev = np.std(y_test)

    evaluation_results = {
            'eval_datetime': [pd.Timestamp.today()],
            'r2_score': [r2_score(y_test, test_predict)],
            'mse': [mean_squared_error(y_test, test_predict)],
            'rmse': [mean_squared_error(y_test, test_predict,squared=False)],
            'rmse/sd': [mean_squared_error(y_test, test_predict, squared=False) / pred_std_dev],
            'mape': [mean_absolute_percentage_error(y_test, test_predict) / 1000000],
            'sorted_features': sorted_features,
            'features': [features]
                                    }
    log_data(evaluation_results, log_file)

    print(evaluation_results)

    return evaluation_results

def create_labels(returns, long_p=60, short_p=20):
    if type(returns) is pd.Series:
        returns = returns[~returns.isna()]
    else:pass
    upper_p = np.percentile(returns, long_p)
    low_p = np.percentile( returns,short_p)

    y = np.repeat(1, len(returns))
    y[returns > upper_p] = 2
    y[returns < low_p] = 0
    if type(returns) is pd.Series:
        y = pd.Series(y, index=returns.index)

    return y




def save_model(model, name):
    """
    Save a scikit-learn model to disk using pickle.

    Parameters:
    -----------
    model : object
        The scikit-learn model to save
    name : str
        The name to use for the saved model file (without extension)

    Returns:
    --------
    str
        Path to the saved model file
    """
    # Create filename with .pkl extension
    filename = f"{name}.pkl"

    # Save the model to disk
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model successfully saved as '{filename}'")
    return filename


def load_model(name):
    """
    Load a scikit-learn model from disk using pickle.

    Parameters:
    -----------
    name : str
        The name of the model file to load (without extension)

    Returns:
    --------
    object
        The loaded scikit-learn model

    Raises:
    -------
    FileNotFoundError
        If the model file doesn't exist
    """
    # Create filename with .pkl extension
    filename = f"{name}.pkl"

    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file '{filename}' not found")

    # Load the model from disk
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    print(f"Model successfully loaded from '{filename}'")
    return model

def log_data(evaluation_results, log_file):
    if not os.path.isfile(log_file):
        log_df = pd.DataFrame(columns=evaluation_results.keys()).set_index('eval_datetime')

    else:
        log_df = pd.read_csv(log_file, index_col=['eval_datetime'], infer_datetime_format=True)
    log_df.loc[pd.Timestamp.now()] = evaluation_results
    log_df.to_csv(log_file)


def evaluate_clf(test_predictions,test_probs, y_test, sorted_features ,log_file='clf_log.csv'):

    evaluation_res = {
        'eval_datetime': [pd.Timestamp.today()],
        'accuracy_score':[accuracy_score(y_test, y_pred=test_predictions)],
        'log_loss':[log_loss(y_test, test_probs, labels=[0,1,2])]

    }
    log_data(evaluation_res, log_file)

    return evaluation_res



class sierra_charts:

    def __init__(self):
        self.resample_logic = {'Open'  : 'first',
         'High'  : 'max',
         'Low'   : 'min',
         'Last' : 'last',
         'Volume': 'sum',
         'BidVolume':'sum',
         'AskVolume': 'sum',
         'NumberOfTrades':'sum'}


    def format_sierra(self, df, date_col='Date', time_col='Time'):
        df.columns = df.columns.str.replace(' ', '')
        datetime_series = df[date_col].astype(str)+df[time_col].astype(str)
        datetime_series = pd.to_datetime(datetime_series, format='mixed')
        df = df.drop(columns=[date_col, time_col]).set_index(datetime_series)
        return df


def mad(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def format_qt_dts(str_series):
    time_format = '%m/%d/%Y %H:%M:%S %p'
    dt_series = pd.to_datetime(str_series, format='%m/%d/%Y %I:%M:%S %p -06:00',)
    return dt_series
