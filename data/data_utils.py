import numpy as np
import pandas as pd
import sys; sys.path.append('C:\\Users\\nicho\PycharmProjects\ml_ensembles\TSlib')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import argparse as ap
from torch.utils.data import Dataset, DataLoader
from TSlib.models.Autoformer import Model as AF
from TSlib.data_provider.data_loader import Dataset_Custom
from TSlib.data_provider.data_factory import data_provider

def create_time_features(index):
    return pd.DataFrame({
        "hour": index.hour / 23.0,
        "dayofweek": index.dayofweek / 6.0,
        "day": index.day / 31.0,
        "month": index.month / 12.0
    }, index=index)



train_args = ap.Namespace(
                    batch_size=1,
                    freq='m',
                    data='custom',
                    root_path='F:\\charts\\learning_data\\timesnet\\',
                    embed=0,
                    task_name='short_term_forecast',
                    data_path='gc_f2.csv',
                    size=None,
                    seq_len=48,
                    pred_len=48,
                    label_len=24,
                    num_workers=1,
                    seasonal_patterns=None,
                    features='MS',
                    augmentation_ratio=-1,
                    target='target_returns',
                    seed=42)


train_dataset, train_dataloader = data_provider(train_args, 'train')

def create_windows(data, time_features, seq_len, label_len, pred_len, target_col=0):
    """
    Efficiently returns arrays of encoder/decoder inputs and target values for forecasting.

    Parameters:
        data: pd.DataFrame or np.ndarray, shape [T, D]
        time_features: pd.DataFrame or np.ndarray, shape [T, T_feats]
        Returns:
            X_enc, X_mark_enc, X_dec, X_mark_dec: [N, L, D]
            y: [N, pred_len, 1]
    """
    data = np.asarray(data)
    time_features = np.asarray(time_features)

    samples = len(data) - seq_len - pred_len
    x_enc_arr = np.empty((samples, seq_len, data.shape[1]))
    x_mark_enc_arr = np.empty((samples, seq_len, time_features.shape[1]))

    x_dec_arr = np.empty((samples, label_len + pred_len, data.shape[1]))
    x_mark_dec_arr = np.empty((samples, label_len + pred_len, time_features.shape[1]))

    y_arr = np.empty((samples, pred_len, 1))  # target column only

    for i in range(samples):
        s = i
        e = s + seq_len
        l = e - label_len
        p = e + pred_len

        x_enc_arr[i] = data[s:e]
        x_mark_enc_arr[i] = time_features[s:e]

        x_dec_arr[i] = data[l:p]
        x_mark_dec_arr[i] = time_features[l:p]

        y_arr[i] = data[e:p, target_col].reshape(-1, 1)

    return x_enc_arr, x_mark_enc_arr, x_dec_arr, x_mark_dec_arr, y_arr

class TimeSeriesDataset(Dataset):
    def __init__(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y):
        """
        Initializes the dataset from pre-windowed numpy arrays.

        Parameters:
            x_enc: [N, seq_len, D] - encoder input
            x_mark_enc: [N, seq_len, T_feats] - encoder time features
            x_dec: [N, label_len + pred_len, D] - decoder input
            x_mark_dec: [N, label_len + pred_len, T_feats] - decoder time features
            y: [N, pred_len, 1] - target values
        """
        self.x_enc = torch.tensor(x_enc, dtype=torch.float32)
        self.x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32)
        self.x_dec = torch.tensor(x_dec, dtype=torch.float32)
        self.x_mark_dec = torch.tensor(x_mark_dec, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x_enc.shape[0]

    def __getitem__(self, idx):
        return (
            self.x_enc[idx],
            self.x_mark_enc[idx],
            self.x_dec[idx],
            self.x_mark_dec[idx],
            self.y[idx]
        )


class Config:

    def __init__(self, seq_len, label_len, pred_len, batch_size, df):
        self.task_name = 'long_term_forecast'
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = df.shape[1]
        self.dec_in = df.shape[1]
        self.c_out = df.shape[1]
        self.d_model = 64
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.1
        self.n_heads = 4
        self.d_ff = 256
        self.e_layers = 2
        self.d_layers = 1
        self.moving_avg = 25
        self.factor = 1
        self.activation = 'gelu'

class preprocess:

    def __init__(self, data:pd.DataFrame, config:classmethod, scaler='minmax', scale_data= True, target_col='Close', ):
        if scale_data:
            if scaler == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
                
            scaled_data = self.scaler.fit_transform(data)
            self.data = pd.DataFrame(scaled_data, columns=data.columns, index = data.index)
        else:
            self.data = data
        if config != None:
            self.config = config

        self.tgt_col_idx = data.columns.tolist().index(target_col)

    def autoformer_prep( self, seq_len, label_len, pred_len, batch_size=8):
        self.model = AF
        x_enc, x_mark_enc, x_dec, x_mark_dec, y = create_windows(self.data,
                                                                 self.data.index,
                                                                 seq_len,
                                                                 label_len,
                                                                 pred_len,
                                                                 self.tgt_col_idx)
        model_dataset = TimeSeriesDataset(x_enc, x_mark_enc, x_dec, x_mark_dec, y)
        self.loader = DataLoader(model_dataset, batch_size=batch_size, shuffle=False)

        return self.loader

    def autoformer_train(self, epochs=3, lr=0.001):
        train_autoformer(self.model, self.loader, epochs, lr)

def train_autoformer(model, train_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")

    criterion.to(device)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x_enc, x_mark_enc, x_dec, x_mark_dec, target in train_loader:
            batch = [x_enc, x_mark_enc, x_dec, x_mark_dec, target]
            batch = [b.to(device) for b in batch]

            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")