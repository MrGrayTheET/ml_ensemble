import sys;

import pandas as pd

sys.path.append('C:\\Users\\nicho\PycharmProjects\ml_ensembles\TSlib')
from torch.utils.data import DataLoader, Dataset
import os
import numpy
import pandas
from TSlib.data_provider.data_factory import data_provider

data_root_dir = 'F:\\charts\\'
tslib_data_dir = os.path.join(data_root_dir, 'learning_data')


def tslib_preprocess(data: pd.DataFrame, output_dir, output_prefix, clean=True):
    cols = data.columns
    if 'date' not in cols:
        if type(data.index) != pd.DatetimeIndex:
            if data.select_dtypes(include=['datetime']).columns == 0:
                'Please add date index as date column'
            else:
                data.insert(0, 'date', value = data.select_dtypes(include=['datetime']).iloc[:, 0])
        else:
            data.insert(0,'date', value=data.index.normalize())
    if clean:
        data = data.ffill().dropna()

    data.reset_index(drop=True)
    fp = os.path.join(tslib_data_dir, output_dir, output_prefix+'_raw.csv')
    data.to_csv(fp)

    return fp

class Seq_model:
    #Class built for running TimesNet model

    def __init__(self, params=None, pred_len=24, seq_len=48,
                 label_len = 24, num_workers=1, top_k=3, features='S',):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.features = features
        self.num_workers = num_workers
        self.task_name = 'short_term_forecast'
        self.root_path = tslib_data_dir
        self.top_k = top_k

        return
