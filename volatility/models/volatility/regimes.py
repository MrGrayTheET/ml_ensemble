import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as gmm
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from WKMeans.src.WKMean import WKMeans
from WKMeans.src.WKUtils import window_lift, reconstruct


def sort_clusters(clusters_df: pd.DataFrame, close_data: pd.Series):
    clusters_df['returns'] = np.log(close_data) - np.log(close_data.shift(1))
    cluster_means = clusters_df.groupby('cluster')['returns'].mean()
    cluster_ranks = cluster_means.rank(method='dense', ascending=False)
    predictions = clusters_df['cluster'].map(cluster_ranks).astype(int) - 1

    return predictions


def cum_rets(df, return_col='returns', regime_col='cluster', start_idx=1, reset_index=True):
    cumulative_returns = df.groupby(regime_col)[return_col].apply(lambda x: (start_idx + x).cumprod())
    if reset_index:
        cumulative_returns = cumulative_returns.reset_index()
        cumulative_returns['t'] = cumulative_returns.groupby(regime_col).cumcount()
        cumulative_returns = cumulative_returns.pivot(index='t', columns='cluster', values='returns')

    return cumulative_returns


def plot_by_regime(df, return_col='returns', regime_col='cluster', start=1):
    cum_ret = cum_rets(df, return_col, regime_col, start, reset_index=True)

    for regime in cum_ret.columns:
        plt.plot(cum_ret.index, cum_ret[regime], label=f'Cluster {regime}')

    plt.title('Cumulative Return by cluster')
    plt.xlabel('Time Index')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return


class SKCluster:
    def __init__(self, df, features, n_components=3):
        self.model = None
        self.selected_features = features
        self.features_df = df[self.selected_features]
        self.n_components = n_components

    def _fit(self, use_pca=True, n_components_pca=4, split=True, split_length=0.8, probabilities=True):

        self.scaler = StandardScaler()
        # Scale features
        if use_pca:
            X = self.pca_transform(n_components_pca, )
        else:
            X = self.scaler.fit_transform(self.features_df)
        if split:
            split_index = int(len(X) * split_length)
            self.train_X, self.test_X = X[:split_index], X[split_index:]
            self.model.fit(self.train_X)

        else:
            self.model.fit(X)
            self.test_X, self.train_X = X, X

        # Predict cluster labels
        labels = self.model.predict(X)

        # Add cluster labels and probabilities to the features dataframe
        self.features_df['cluster'] = labels

        # Probabilities when available
        if probabilities:
            probs = self.model.predict_proba(X)
            for i in range(self.n_components):
                self.features_df[f'prob_cluster_{i}'] = probs[:, i]

        return self.features_df, labels

    def pca_transform(self, n_components_pca):
        X = self.scaler.fit_transform(self.features_df)
        self.pca = PCA(n_components=min(n_components_pca, X.shape[1]))
        X = self.pca.fit_transform(X)
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        return X

    def visualize_returns(self):
        df = self.features_df
        cum_return_by_reg = cum_rets(df)
        df['cum_return_'] = (1 + df['returns']).cumprod()
        fig, ax = plt.subplots(2, 1)

        for i in range(0, df['cluster'].max() + 1):
            ret_cur_reg = df.loc[df.cluster == i, 'cum_return_']
            ax[0].scatter(ret_cur_reg.index, ret_cur_reg, label=f'cluster_{i}')

        ax[0].set_title('Cumulative Returns')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Cumulative Market Returns')
        ax[0].legend()

        # Plot cumulative returns by cluster (lines)
        for regime in cum_return_by_reg.columns:
            ax[1].plot(cum_return_by_reg.index, cum_return_by_reg[regime], label=f'Cluster {regime}')

        ax[1].set_title('Cumulative Return by Cluster')
        ax[1].set_xlabel('Time Index')
        ax[1].set_ylabel('Cumulative Return')
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

        return

    def get_cluster_statistics(self) -> pd.DataFrame:
        """
        Calculate statistics for each cluster.

        Args:
            features_df: DataFrame with cluster assignments

        Returns:
            DataFrame with cluster statistics
        """
        # Group by cluster and calculate statistics
        features_df = self.features_df
        cluster_stats = features_df.groupby('cluster').model(['mean', 'std', 'min', 'max', 'count'])
        X = self.scaler.transform(features_df[self.selected_features])

        # Calculate BIC and AIC
        cluster_stats.loc['model_metrics', ('cluster', 'count')] = len(features_df)
        cluster_stats.loc['model_metrics', ('cluster', 'mean')] = self.model.n_components
        cluster_stats.loc['model_metrics', ('cluster', 'std')] = 0
        cluster_stats.loc['model_metrics', ('cluster', 'min')] = self.model.bic(X)
        cluster_stats.loc['model_metrics', ('cluster', 'max')] = self.model.aic(X)

        return cluster_stats



class GaussianMixture(SKCluster):

    def __init__(self, df, features=[], n_components=3, cov_type='full', random_state=42, tol=1e-4, **params):
        super().__init__(df, features, n_components)
        if features == []:
            self.selected_features = df.columns
        else:
            pass

        self.random_state = random_state
        self.tol = tol
        self.model = gmm(n_components,
                         covariance_type=cov_type,
                         random_state=random_state,
                         tol=tol,
                         **params)

    def find_optimal_clusters(self, max_components: int = 10):
        """
        Find the optimal number of clusters using BIC and AIC.

        Args:
            price_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            features: List of feature names to use for clustering
            max_components: Maximum number of components to try

        Returns:
            Dictionary with BIC and AIC values for each number of components
        """
        # Extract and select features
        features_df = self.features_df[self.selected_features]

        # Handle missing values
        features_df = features_df.fillna(0)

        # Scale features
        X = self.scaler.fit_transform(features_df)

        # Try different numbers of components
        n_components_range = range(1, max_components + 1)
        models = [gmm(n_components=n, covariance_type='full', random_state=self.random_state)
                  for n in n_components_range]

        # Calculate BIC and AIC
        bic_values = []
        aic_values = []

        for model in models:
            model.fit(X)
            bic_values.append(model.bic(X))
            aic_values.append(model.aic(X))

        # Find optimal number of components
        best_bic = np.argmin(bic_values) + 1
        best_aic = np.argmin(aic_values) + 1

        return {
            'n_components_range': list(n_components_range),
            'bic_values': bic_values,
            'aic_values': aic_values,
            'best_bic': best_bic,
            'best_aic': best_aic
        }

    def get_cluster_statistics(self) -> pd.DataFrame:
        """
        Calculate statistics for each cluster.

        Args:
            features_df: DataFrame with cluster assignments

        Returns:
            DataFrame with cluster statistics
        """
        # Group by cluster and calculate statistics
        features_df = self.features_df
        cluster_stats = features_df.groupby('cluster').model(['mean', 'std', 'min', 'max', 'count'])
        X = self.scaler.transform(features_df[self.selected_features])

        # Calculate BIC and AIC
        cluster_stats.loc['model_metrics', ('cluster', 'count')] = len(features_df)
        cluster_stats.loc['model_metrics', ('cluster', 'mean')] = self.model.n_components
        cluster_stats.loc['model_metrics', ('cluster', 'std')] = 0
        cluster_stats.loc['model_metrics', ('cluster', 'min')] = self.model.bic(X)
        cluster_stats.loc['model_metrics', ('cluster', 'max')] = self.model.aic(X)

        return cluster_stats

    def plot_regimes(self, test_data, train_test=False):
        df = self.features_df.copy()
        df['dt'] = df.index

        # Align validation data
        validation_data = test_data.loc[df.index.date.min():df.index.date.max()]
        val_data = validation_data['Close'].ffill().dropna()

        if len(df) > len(val_data):
            df = df.loc[val_data.index[0]:val_data.index[-1]]

        # Compute cumulative returns
        cum_returns = (1 + val_data.pct_change().fillna(0)).cumprod()

        # Estimate volatility for each regime
        df['volatility'] = self.data.loc[df.index, 'returns'].rolling(window=10).std().fillna(method='bfill')
        regime_vols = df.groupby('cluster')['volatility'].mean()

        # Normalize volatilities to [0, 1] for colormap
        norm = mcolors.Normalize(vmin=regime_vols.min(), vmax=regime_vols.max())
        cmap = plt.cm.Reds  # Choose a red gradient (low vol = light, high vol = dark)

        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cum_returns.index, cum_returns.values, color='black', label='Cumulative Return')

        # Draw shaded regime spans
        last_label = None
        start_idx = None

        for i in range(len(df)):
            label = df['cluster'].iloc[i]
            if label != last_label:
                if start_idx is not None:
                    start = df.index[start_idx]
                    end = df.index[i]
                    color = cmap(norm(regime_vols[last_label]))
                    ax.axvspan(start, end, color=color, alpha=0.3)
                start_idx = i
                last_label = label

        # Final region
        if start_idx is not None:
            color = cmap(norm(regime_vols[last_label]))
            ax.axvspan(df.index[start_idx], df.index[-1], color=color, alpha=0.3)

        # Optional: vertical line for train/test split
        if train_test and hasattr(self, 'train_X'):
            split_index = df.iloc[len(self.train_X)].name
            ax.axvline(x=split_index, color='cyan', linestyle='--', label='Train/Test Split')

        # Finalize plot
        ax.set_title('Cumulative Return with Volatility-Colored Regimes')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_cum_rets(self):
        return plot_by_regime(self.features_df)

    def fit(self, use_pca=True, n_components_pca=4, split=True, split_length=0.8):
        return self._fit(use_pca, n_components_pca, split, split_length, probabilities=True)


class AggClusters(SKCluster):

    def __init__(self, df, features=None, n_components=3):
        if features is None:
            self.selected_features = df.columns

        super().__init__(df, features, n_components)
        self.model = AgglomerativeClustering(n_clusters=self.n_components, metric='euclidean', linkage='ward')
        self.model.predict = self.model.fit_predict
        self.model.predict_proba = lambda x: np.nan

    def fit(self, use_pca=False,n_components_pca=3, split=False, **kwargs):
        return self._fit(use_pca, n_components_pca, split, probabilities=False, **kwargs)



class WKFi(WKMeans):

    def __init__(self, OHLC_data: pd.DataFrame, k: int, max_iter=50, tol=1e-4, gamma=1.0, mmd_pairs=10, sample_size=50):
        self.idxs = None
        self.data = OHLC_data.copy(deep=True).dropna()
        self.data['returns'] = (np.log(self.data.Close) - np.log(self.data.Close.shift(1))).dropna()
        super().__init__(k, max_iter, tol, gamma, mmd_pairs, sample_size)
        self.distributions = None
        self.split = False
        self.data.dropna(inplace=True)
        return

    def fit_windows(self, h1=30, h2=3, data=None, split=False, training_length=0.8):
        self.split = split
        if data is not None:
            self.data = data
            self.data['returns'] = np.log(self.data.Close) - np.log(self.data.Close.shift(1))
        self.distributions, self.idxs = window_lift(self.data.returns, h1, h2)
        self.fit(self.distributions)
        return self

    def predict_clusters(self, data=None, df=True):
        assignments = self.predict(self.distributions)
        if not df:
            return assignments
        else:
            results = reconstruct(self.data.returns, self.idxs, assignments)
            self.data['cluster'] = results['cluster']

        return self.data

    def plot_regimes(self, train_test=False):
        df = self.data.copy()
        df['dt'] = df.index

        # Align validation data
        val_data = df['Close'].ffill().dropna()

        # Compute cumulative returns
        cum_returns = (1 + val_data.pct_change().fillna(0)).cumprod()

        # Estimate volatility for each regime
        df['volatility'] = self.data.loc[df.index, 'returns'].rolling(window=10).std().fillna(method='bfill')
        regime_vols = df.groupby('cluster')['volatility'].mean()

        # Normalize volatilities to [0, 1] for colormap
        norm = mcolors.Normalize(vmin=regime_vols.min(), vmax=regime_vols.max())
        cmap = plt.cm.Reds  # Choose a red gradient (low vol = light, high vol = dark)

        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cum_returns.index, cum_returns.values, color='black', label='Cumulative Return')

        # Draw shaded regime spans
        last_label = None
        start_idx = None

        for i in range(len(df)):
            label = df['cluster'].iloc[i]
            if label != last_label:
                if pd.isna(label):
                    continue
                if start_idx is not None:
                    start = df.index[start_idx]
                    end = df.index[i]
                    color = cmap(norm(regime_vols[last_label]))
                    ax.axvspan(start, end, color=color, alpha=0.3)
                start_idx = i
                last_label = label

        # Final region
        if start_idx is not None:
            color = cmap(norm(regime_vols[last_label]))
            ax.axvspan(df.index[start_idx], df.index[-1], color=color, alpha=0.3)

        # Optional: vertical line for train/test split
        if train_test and hasattr(self, 'train_X'):
            split_index = df.iloc[len(self.train_X)].name
            ax.axvline(x=split_index, color='cyan', linestyle='--', label='Train/Test Split')

        # Finalize plot
        ax.set_title('Cumulative Return with Volatility-Colored Regimes')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def visualize_returns(self):
        df = self.data.copy()
        df = df[df['cluster'].notna()].copy()  # Drop NaN clusters

        # Calculate cumulative returns
        df['cum_return_'] = (1 + df['returns']).fillna(0).cumprod()

        # Compute rolling volatility per row
        df['volatility'] = self.data.loc[df.index, 'returns'].rolling(window=10).std().fillna(method='bfill')

        # Compute mean volatility per cluster
        regime_vols = df.groupby('cluster')['volatility'].mean()

        # Normalize volatilities for colormap scaling
        norm = mcolors.Normalize(vmin=regime_vols.min(), vmax=regime_vols.max())
        cmap = plt.cm.Reds  # Use red colormap for volatility

        # Setup plots
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # Plot cumulative return
        ax[0].plot(df.index, df['cum_return_'], color='black', label='Cumulative Return')

        # Regime-colored spans based on volatility
        last_label = None
        start_idx = None
        for i in range(len(df)):
            label = df['cluster'].iloc[i]
            if label != last_label:
                if start_idx is not None:
                    start = df.index[start_idx]
                    end = df.index[i]
                    color = cmap(norm(regime_vols[last_label]))
                    ax[0].axvspan(start, end, color=color, alpha=0.3)
                start_idx = i
                last_label = label

        # Final span
        if start_idx is not None:
            start = df.index[start_idx]
            end = df.index[-1]
            color = cmap(norm(regime_vols[last_label]))
            ax[0].axvspan(start, end, color=color, alpha=0.3)

        ax[0].set_title('Cumulative Return with Volatility-Colored Regimes')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Cumulative Market Returns')
        ax[0].legend()
        ax[0].grid(True)

        # Plot cumulative return by cluster (assumes cum_rets returns a DataFrame)
        cum_return_by_reg = cum_rets(df)
        for regime in cum_return_by_reg.columns:
            ax[1].plot(cum_return_by_reg.index, cum_return_by_reg[regime], label=f'Cluster {regime}')

        ax[1].set_title('Cumulative Return by Cluster')
        ax[1].set_xlabel('Time Index')
        ax[1].set_ylabel('Cumulative Return')
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()
