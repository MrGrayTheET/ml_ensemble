import yfinance as yf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as gmm
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from statsmodels.tsa.regime_switching import markov_autoregression

class RegimeClustering:

    def __init__(self, df, features=[], type='gmm', n_components=3):
        self.train_X = None
        if len(features) == 0: self.selected_features = df.columns
        else:
            self.selected_features = features

        if type == 'gmm':
            self.model = self.gmm = gmm(
            n_components=n_components,
            covariance_type='full',
            random_state=42
        )

        self.n_components = n_components

        self.features_df = df[self.selected_features]



    def fit(self, use_pca=True, n_components_pca=4, split=True, split_length=0.8):
        self.scaler = StandardScaler()
        self.selected_features = None
        # Scale features
        if use_pca:
            X = self.pca_transform(n_components_pca,)
        else:
            X = self.scaler.fit_transform(self.features_df)
        if split:
            split_index = int(len(X) * split_length)
            self.train_X, self.test_X = X[:split_index], X[split_index:]
            self.gmm.fit(self.train_X)

        else:
            self.gmm.fit(X)
            self.test_X, self.train_X = X, X


        # Predict cluster labels
        labels = self.gmm.predict(X)
        probabilities = self.gmm.predict_proba(X)

        # Add cluster labels and probabilities to the features dataframe
        self.features_df['cluster'] = labels
        for i in range(self.n_components):
            self.features_df[f'prob_cluster_{i}'] = probabilities[:, i]

        return self.features_df, labels

    def pca_transform(self, n_components_pca):
        X = self.scaler.fit_transform(self.features_df)
        self.pca = PCA(n_components=min(n_components_pca, X.shape[1]))
        X = self.pca.fit_transform(X)
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        return X

    def plot_feature_by_regime(self, feature, label_col='cluster'):
        df = self.features_df
        plt.figure(figsize=(14, 6))
        for cluster in sorted(df[label_col].unique()):
            clustered_data = df[df[label_col] == cluster]
            plt.plot(clustered_data.index, clustered_data[feature], '.', label=f'Regime {cluster}')
        plt.title(f'{feature} by Regime')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_regimes(self, test_ticker, train_test=False):

        df = self.features_df

        df['dt'] = df.index
        mean_interval = df.dt.diff().mean()
        n_labels = df.clusters.max()

        if ((mean_interval > dt.timedelta(days=0)) and
                (mean_interval < dt.timedelta(days=3))):
            validation_data = yf.download(test_ticker, interval='1d', start=df.dt[0],
                                          end=df.dt[-1])

        elif ((mean_interval > dt.timedelta(days=20)) and
              (mean_interval < dt.timedelta(days=35))):

            validation_data = yf.download(test_ticker, interval='1mo', start=df.dt[0],
                                          end=df.dt[-1])

        elif (((mean_interval) < dt.timedelta(days=95)) &
              (mean_interval > dt.timedelta(days=45))):
            validation_data = yf.download(test_ticker, start=df.dt[0], end=df.dt[-1],
                                          interval='3mo')

        else:
            validation_data = yf.download(test_ticker, interval='5d')

        val_data = validation_data['Close'].ffill().dropna()

        if len(df) > len(val_data):
            df = df.loc[val_data.index[0]:val_data.index[-1]]

        reg_0_mean = val_data.loc[df['clusters'] == 0].mean()[0]
        reg_2_mean = val_data.loc[df['clusters'] == df.clusters.max()].mean()[0]

        colors = ['green', 'yellow', 'red']

        if n_labels > 2:
            colors = ['green', 'blue', 'yellow', 'orange', 'red', 'magenta']

        if reg_0_mean < reg_2_mean:
            colors = colors[::-1]

        for i in range(0, df['clusters'].max() + 1):
            val_reg = val_data.loc[df['clusters'] == i]
            plt.scatter(val_reg.index, val_reg, color=colors[i])

        if train_test:
            line_idx = self.train_X.index[-1]
            plt.axvline(x=line_idx, color='cyan')

        plt.show()

        return

