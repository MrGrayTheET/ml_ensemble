from sc_loader import sierra_charts as schart
import matplotlib.pyplot as plt
from intraday.mixture_model import TimeSeriesGMMClustering
from intraday.feature_engineering import TimeSeriesFeatureExtractor as ext
from intraday.mixture_model import TimeSeriesGMMClustering
import pandas as pd
import numpy as np

sc = schart()
intraday_es = sc.get_chart('ES_F')
intraday_cl = sc.get_chart('CL_F')
intraday_ng = sc.get_chart('NG_F')

extractor = ext()

clustering = TimeSeriesGMMClustering(
        feature_extractor=extractor,
        n_components=4
    )

sample_data = intraday_cl.ffill().dropna()
resampled = intraday_cl.resample('1h').apply(sc.resample_logic)

# Fit model with selected features

selected_features  = clustering.proposed_features
features_with_clusters, labels = clustering.fit(
    sample_data,
    proposed_features=True,
    classical_features=False,
    selected_features=selected_features,
    use_pca=False
)

print("Features with cluster assignments:")
print(features_with_clusters.head())

# Get cluster statistics
cluster_stats = clustering.get_cluster_statistics(features_with_clusters)
print("\nCluster statistics:")
print(cluster_stats)

sample_data[['open', 'low', 'high', 'close', 'volume']] = sample_data[['Open','Low', 'High','Close', 'Volume']]

# Group time series by cluster
cluster_groups = clustering.analyze_time_series_by_cluster(sample_data, features_with_clusters)


# # Find optimal number of clusters
# optimal_results = clustering.find_optimal_clusters(
#     sample_data,
#     selected_features=selected_features,
#     max_components=7
# )
#
# print("\nOptimal number of clusters:")
# print(f"Best BIC: {optimal_results['best_bic']}")
# print(f"Best AIC: {optimal_results['best_aic']}")

print("\nTime series patterns by cluster:")
for cluster, data in cluster_groups.items():
    print(f"Cluster {cluster}: {len(data)} data points")
    if len(data) > 0:
        print(f"  Average close price: {data['close'].mean():.2f}")

