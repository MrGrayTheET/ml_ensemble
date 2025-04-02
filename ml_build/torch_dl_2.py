import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ml_build.utils import clean_data

from torch.utils.data import DataLoader, TensorDataset

class lstm(nn.Module):

    def __init__(self, num_classes, input_size,hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        hid_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cel_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        output, (hn, cn) = self.lstm(x, (hid_0, cel_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        return self.fc_2(out)




class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.2):
            super(LSTMModel, self).__init__()
            output_size = num_classes
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layers with dropout in between
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)

            # Dropout layer
            self.dropout = nn.Dropout(dropout_prob)

            # Fully connected layers
            self.fc1 = nn.Linear(hidden_size, 128)
            self.batch_norm = nn.BatchNorm1d(128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, output_size)

        def forward(self, x):
            # Initialize hidden state and cell state
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))

            # Apply dropout
            out = self.dropout(out[:, -1, :])  # Taking the last time step

            # Fully connected layers with batch norm and activation
            out = self.fc1(out)
            out = self.batch_norm(out)
            out = self.relu(out)
            out = self.fc2(out)

            return out


def training_loop(X_train,y_train,X_test,y_test, n_epochs, lstm, optimizer, loss_func, verbosity=5):
    lstm.train()
    device = torch.device('cpu')

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i in range(X_train.shape[0]):
            input_sample = X_train[i].unsqueeze(0).to(device)  # shape: (1, 50, 19)
            target_sample = y_train[i].unsqueeze(0).to(device)

            optimizer.zero_grad()
            output = lstm(input_sample)
            loss = loss_func(output, target_sample)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /=X_train.shape[0]

        test_pred = lstm(X_test)
        test_loss = loss_func(test_pred, y_test)

        if epoch % verbosity == 0:

            print(f'Epoch: {epoch} / {n_epochs}, Loss: {epoch_loss:.4f} test_loss: {test_loss.item()}')
            
        if epoch == n_epochs:

            lstm.eval()


            continue
        
        

        return lstm(X_train), lstm(X_test)

def sharpe_loss(predictions, returns):
    mean_return = torch.mean(predictions)
    var_return = torch.var(predictions, unbiased=False)
    sharpe_ratio = mean_return / (torch.sqrt(var_return) + 1e-8)  # Avoid division by zero
    return -torch.sqrt(torch.tensor(252.0)) * sharpe_ratio

def plot_final_prediction(test_x, test_y, y_scaler, lstm):
    '''
        test_predict : LSTM prediction result
         test_true: torch tensor containing true values
         '''
    forecast = lstm(test_x[-1].unsqueeze(0))
    forecast = y_scaler.inverse_transform(forecast.detach().numpy())
    forecast = forecast[0].tolist()
    final_true = test_y[-1].detach().numpy()
    final_true = y_scaler.inverse_transform(final_true.reshape(1,-1))
    final_true = final_true[0].tolist()
    plt.plot(final_true, label='Actual')
    plt.plot(forecast, label='Predicted')

