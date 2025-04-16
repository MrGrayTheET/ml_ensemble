import pandas as pd
import numpy as np
from scipy.signal import find_peaks


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
