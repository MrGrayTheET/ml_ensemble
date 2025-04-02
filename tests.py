import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller as adf
from ml_build.utils import evaluate_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import zscore
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adf(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    return dfoutput

def remove_outliers(df, std=3):
    return df[(np.abs(zscore(df)) < std).all(axis=1)]
def eval_model(actual, predicted, residuals):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    # Residual diagnostics
    lb_pvalue = acorr_ljungbox(residuals, lags=[10], return_df=True)["lb_pvalue"].values[0]

    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R² Score": r2, "Ljung-Box p-value": lb_pvalue
    }

def evaluate_seasonal(model, type='MSTL'):
    if type == 'MSTL':
        seasonals = model.seasonal.sum(axis=1)
    else:
        seasonals = model.seasonal

    evaluation = eval_model(model.observed,seasonals+model.trend, model.resid)

    return evaluation
