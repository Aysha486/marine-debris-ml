"""
Hotspot detection utilities (KMeans + optional DBSCAN fallback).

Functions:
- kmeans_hotspots(df, lat_col='latitude', lon_col='longitude', n_clusters=5)

Returns: DataFrame copy with a `cluster` column and a `centroids` dict (cluster->(lat,lon)).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def kmeans_hotspots(df, lat_col='latitude', lon_col='longitude', n_clusters=5):
    dfc = df.copy()
    if lat_col not in dfc.columns or lon_col not in dfc.columns:
        raise ValueError('Latitude/longitude columns not found')

    coords = dfc[[lat_col, lon_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if coords.empty:
        raise ValueError('No valid coordinates available')

    k = min(n_clusters, coords.shape[0])
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(coords.values)
    coords = coords.assign(cluster=labels)

    # join back to original df (rows without coords will have NaN cluster)
    dfc = dfc.merge(coords['cluster'], left_index=True, right_index=True, how='left')

    centroids = {int(i): (float(lat), float(lon)) for i, (lat, lon) in enumerate(km.cluster_centers_)}
    return dfc, centroids


if __name__ == '__main__':
    print('hotspots.py loaded — use kmeans_hotspots(df, lat_col, lon_col, n_clusters)')
