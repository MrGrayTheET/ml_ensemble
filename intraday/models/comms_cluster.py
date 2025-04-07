from sc_loader import sierra_charts as schart
import matplotlib.pyplot as plt
from intraday.mixture_model import TimeSeriesGMMClustering
from intraday.feature_engineering import TimeSeriesFeatureExtractor as ext,  PeakExtractor as pe
from intraday.mixture_model import TimeSeriesGMMClustering
import pandas as pd
import numpy as np

sc = schart()
intraday_es = sc.get_chart('ES_F')
intraday_cl = sc.get_chart('CL_F')
intraday_ng = sc.get_chart('NG_F')
gc_f = sc.format_chart(sc.unformatted_charts +'gc_25.csv')


clustering = TimeSeriesGMMClustering(
        feature_extractor=pe,
        n_components=4
    )

resampling_logic = {
    'Open': 'first',        # Opening price of the resampled interval
    'High': 'max',          # Highest price
    'Low': 'min',           # Lowest price
    'Close': 'last',        # Closing price
    'Volume': 'sum',        # Total traded volume
}
sample_data = gc_f.ffill().dropna()

# Fit model with selected features

selected_features  = clustering.proposed_features
features_with_clusters, labels = clustering.fit(
    sample_data,
    proposed_features=True,
    classical_features=False,
    selected_features=selected_features,
    use_pca=True
)

print("Features with cluster assignments:")
print(features_with_clusters.head())

# Get cluster statistics
cluster_stats = clustering.get_cluster_statistics()
print("\nCluster statistics:")
print(cluster_stats)

sample_data[['open', 'low', 'high', 'close', 'volume']] = sample_data[['Open','Low', 'High','Close', 'Volume']]

# Group time series by cluster
cluster_groups, price_data = clustering.analyze_time_series_by_cluster(sample_data, features_with_clusters,resampling_logic)


# Find optimal number of clusters
optimal_results = clustering.find_optimal_clusters(
    sample_data,
    selected_features=selected_features,
    max_components=7
)

print("\nOptimal number of clusters:")
print(f"Best BIC: {optimal_results['best_bic']}")
print(f"Best AIC: {optimal_results['best_aic']}")

print("\nTime series patterns by cluster:")
for cluster, data in cluster_groups.items():
    print(f"Cluster {cluster}: {len(data)} data points")
    if len(data) > 0:
        print(f"  Average close change: {data['Close'].pct_change().mean():.5f}")



def plot_clusters(cluster_groups, n_clusters, labels, cluster_stats=None, subplots=False):


    colors = ['blue','cyan', 'green', 'yellow', 'red', 'orange']
    for i in range(n_clusters):

        cluster_i = cluster_groups[i].ffill().dropna()
        plt.scatter(cluster_i.index, cluster_i['Close'], color=colors[i])



