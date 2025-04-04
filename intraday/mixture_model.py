import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sc_loader import sierra_charts as sc
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional

# Import the TimeSeriesFeatureExtractor class that we created earlier
from intraday.feature_engineering import TimeSeriesFeatureExtractor


class TimeSeriesGMMClustering:
    """
    A class to perform Gaussian Mixture Model clustering on time series features.
    """

    proposed_features = ['peak_curvature', 'peak_average_magnitude', 'percentage_change', 'close_pct_change']
    classical_features = ['rsi', 'close_pct_change', 'close_zscore', 'volume_zscore', 'ma_ratio']


    def __init__(self, feature_extractor=None, n_components=3, random_state=42):
        """
        Initialize the GMM clustering model.

        Args:
            feature_extractor: An instance of TimeSeriesFeatureExtractor (if None, a new one is created)
            n_components: Number of Gaussian mixture components (clusters)
            random_state: Random seed for reproducibility
        """
        self.data = None

        self.feature_extractor = feature_extractor or TimeSeriesFeatureExtractor()
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.selected_features = None
        self.pca = None

    def extract_and_select_features(self, price_data: pd.DataFrame,
                                    proposed_features=True,
                                    classical=False,
                                    selected_features=None) -> pd.DataFrame:
        """
        Extract features and select the ones to use for clustering.

        Args:
            price_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            selected_features: List of feature names to use for clustering (if None, use all features)

        Returns:
            DataFrame with selected features
            :param classical_features:
            :param proposed_features:
            :param selected_features:
        """
        # Extract all features
        features = self.feature_extractor.extract_features(price_data, proposed_features=proposed_features, classical_features=classical)
        self.data = self.feature_extractor.data

        # Select features for clustering
        if selected_features is None:
            # Default to using all features
            if proposed_features and not classical:
                self.selected_features = self.proposed_features
            elif classical and not proposed_features:
                self.selected_features = self.classical_features
        else:
            # Make sure all requested features exist
            available_features = features.columns.tolist()
            self.selected_features = [f for f in selected_features if f in available_features]

            if len(self.selected_features) < len(selected_features):
                missing = set(selected_features) - set(self.selected_features)
                print(f"Warning: Some requested features are not available: {missing}")

            if not self.selected_features:
                raise ValueError("No valid features selected for clustering")

        # Return selected features
        return features[self.selected_features]

    def fit(self, price_data: pd.DataFrame, proposed_features=True, classical_features=False, selected_features: List[str] = None,
            use_pca: bool = False, n_components_pca: int = 2) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit the GMM model to the extracted and selected features.

        Args:
            price_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            selected_features: List of feature names to use for clustering
            use_pca: Whether to apply PCA dimensionality reduction before clustering
            n_components_pca: Number of PCA components if use_pca is True

        Returns:
            Tuple of (features_df, cluster_labels)
        """
        # Extract and select features
        features_df = self.extract_and_select_features(price_data, proposed_features,classical=classical_features, selected_features=None)

        # Handle missing values
        features_df = features_df.fillna(0)

        # Scale features
        X = self.scaler.fit_transform(features_df)

        # Apply PCA if requested
        if use_pca:
            self.pca = PCA(n_components=min(n_components_pca, X.shape[1]))
            X = self.pca.fit_transform(X)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")

        # Fit GMM model
        self.gmm.fit(X)

        # Predict cluster labels
        labels = self.gmm.predict(X)
        probabilities = self.gmm.predict_proba(X)

        # Add cluster labels and probabilities to the features dataframe
        features_df['cluster'] = labels
        for i in range(self.n_components):
            features_df[f'prob_cluster_{i}'] = probabilities[:, i]

        return features_df, labels

    def predict(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            price_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            Array of cluster labels
        """
        # Extract and select features
        features_df = self.extract_and_select_features(price_data, selected_features=[])

        # Handle missing values
        features_df = features_df.fillna(0)

        # Scale features
        X = self.scaler.transform(features_df)

        # Apply PCA if it was used during fitting
        if self.pca is not None:
            X = self.pca.transform(X)

        # Predict cluster labels
        return self.gmm.predict(X)

    def get_cluster_statistics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each cluster.

        Args:
            features_df: DataFrame with cluster assignments

        Returns:
            DataFrame with cluster statistics
        """
        # Group by cluster and calculate statistics
        cluster_stats = features_df.groupby('cluster').agg(['mean', 'std', 'min', 'max', 'count'])

        # Calculate BIC and AIC
        cluster_stats.loc['model_metrics', ('cluster', 'count')] = len(features_df)
        cluster_stats.loc['model_metrics', ('cluster', 'mean')] = self.gmm.n_components
        cluster_stats.loc['model_metrics', ('cluster', 'std')] = 0
        cluster_stats.loc['model_metrics', ('cluster', 'min')] = self.gmm.bic(
            self.scaler.transform(features_df[self.selected_features].fillna(0)))
        cluster_stats.loc['model_metrics', ('cluster', 'max')] = self.gmm.aic(
            self.scaler.transform(features_df[self.selected_features].fillna(0)))

        return cluster_stats

    def visualize_clusters(self, features_df: pd.DataFrame, feature1: str = None,
                           feature2: str = None, use_pca: bool = True,
                           title: str = "GMM Clusters") -> plt.Figure:
        """
        Visualize the clusters in a 2D scatter plot.

        Args:
            features_df: DataFrame with features and cluster assignments
            feature1: First feature to plot (x-axis)
            feature2: Second feature to plot (y-axis)
            use_pca: Whether to use PCA for visualization (overrides feature1/feature2)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Prepare data for plotting
        if use_pca and self.pca is not None:
            # Use PCA components for plotting
            X = self.scaler.transform(features_df[self.selected_features].fillna(0))
            X_pca = self.pca.transform(X)
            x = X_pca[:, 0]
            y = X_pca[:, 1]
            x_label = "PCA Component 1"
            y_label = "PCA Component 2"
        else:
            # Use specified features or default to first two
            if feature1 is None or feature2 is None:
                if len(self.selected_features) < 2:
                    raise ValueError("Need at least two features for visualization")
                feature1 = self.selected_features[0]
                feature2 = self.selected_features[1]

            x = features_df[feature1]
            y = features_df[feature2]
            x_label = feature1
            y_label = feature2

        # Get cluster labels
        labels = features_df['cluster']

        # Plot each cluster with a different color
        unique_labels = sorted(labels.unique())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            ax.scatter(x[mask], y[mask], color=colors[i], alpha=0.7,
                       label=f'Cluster {label}', edgecolors='k', s=50)

        # Plot cluster centers if using PCA
        if use_pca and self.pca is not None:
            # Transform cluster centers to PCA space
            centers = self.gmm.means_
            if self.pca.n_components_ >= 2:
                centers_pca = self.pca.transform(centers)
                ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, marker='X',
                           color='black', alpha=0.8, label='Cluster Centers')

        # Add labels and legend
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        return fig

    def find_optimal_clusters(self, price_data: pd.DataFrame, selected_features: List[str] = None,
                              max_components: int = 10) -> Dict:
        """
        Find the optimal number of clusters using BIC and AIC.

        Args:
            price_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            selected_features: List of feature names to use for clustering
            max_components: Maximum number of components to try

        Returns:
            Dictionary with BIC and AIC values for each number of components
        """
        # Extract and select features
        features_df = self.extract_and_select_features(price_data, selected_features, selected_features=[])

        # Handle missing values
        features_df = features_df.fillna(0)

        # Scale features
        X = self.scaler.fit_transform(features_df)

        # Try different numbers of components
        n_components_range = range(1, max_components + 1)
        models = [GaussianMixture(n_components=n, covariance_type='full', random_state=self.random_state)
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

    def visualize_optimal_clusters(self, results: Dict) -> plt.Figure:
        """
        Visualize BIC and AIC values for different numbers of components.

        Args:
            results: Dictionary returned by find_optimal_clusters

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(results['n_components_range'], results['bic_values'], label='BIC', marker='o')
        ax.plot(results['n_components_range'], results['aic_values'], label='AIC', marker='s')

        ax.axvline(results['best_bic'], color='r', linestyle='--', alpha=0.5,
                   label=f'Best BIC: {results["best_bic"]}')
        ax.axvline(results['best_aic'], color='g', linestyle='--', alpha=0.5,
                   label=f'Best AIC: {results["best_aic"]}')

        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Information Criterion')
        ax.set_title('BIC and AIC for Different Numbers of Components')
        ax.legend()
        plt.tight_layout()

        return fig

    def analyze_time_series_by_cluster(self, price_data: pd.DataFrame,
                                       features_df: pd.DataFrame, resample_dict=None, resample_period='30min', cluster_sample_method='mean') -> Dict[int, pd.DataFrame]:
        """
        Group time series data by cluster and analyze patterns.

        Args:
            price_data: Original price data
            features_df: DataFrame with features and cluster assignments

        Returns:
            Dictionary mapping cluster IDs to DataFrames with price data
        """
        # Add interval_id to price_data

        if type(price_data.index) == pd.DatetimeIndex:
            price_data['idx'] = np.arange(0, len(price_data))
            price_data['interval_id'] = price_data.idx // self.feature_extractor.interval_size
          
        
        # Create a mapping from interval_id to cluster
        interval_to_cluster = features_df['cluster'].to_dict()
                                           
        else: 
            price_data['interval_id'] = price_data.index // self.feature_extractor.interval_size

        # Optional resampling
        
        if resample_dict is not None:
            resample_dict.update({'interval_id':'last'})
            price_data = price_data.resample(resample_period).apply(resample_dict)
        
     
        # Add cluster to price_data
        price_data['cluster'] = price_data['interval_id'].map(interval_to_cluster)


        # Group by cluster
        cluster_groups = {}
        for cluster in features_df['cluster'].unique():
            cluster_groups[cluster] = price_data[price_data['cluster'] == cluster].copy()

        return cluster_groups, price_data



