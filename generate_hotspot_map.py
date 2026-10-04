"""Generate an interactive Folium map showing hotspots.

Saves `hotspots_map.html` in the project root.

Usage:
    python generate_hotspot_map.py

Requires: folium, branca (install with `pip install folium branca`)
"""
import os
import sys
import pandas as pd

try:
    import folium
    from branca.colormap import linear
except Exception:
    print('Missing dependency: please `pip install folium branca`')
    raise

from mdri import compute_mdri
from hotspots import kmeans_hotspots


def load_data():
    # prefer normalized file if present
    candidates = ['data_cleaned_normalized.csv', 'data_cleaned.csv']
    for p in candidates:
        if os.path.exists(p):
            print(f'Loading {p}')
            return pd.read_csv(p)
    raise FileNotFoundError('No cleaned data file found; run imputation.ipynb first')


def make_map(df, lat_col='latitude', lon_col='longitude', mdri_col='mdri', n_clusters=6, out='hotspots_map.html'):
    dfc = df.copy()
    # ensure numeric coords
    dfc[lat_col] = pd.to_numeric(dfc.get(lat_col), errors='coerce')
    dfc[lon_col] = pd.to_numeric(dfc.get(lon_col), errors='coerce')
    dfc = dfc.dropna(subset=[lat_col, lon_col])
    if dfc.empty:
        raise ValueError('No valid coordinates found in data')

    # compute mdri if missing
    if mdri_col not in dfc.columns:
        print('`mdri` column not found — computing via mdri.compute_mdri')
        try:
            dfc = compute_mdri(dfc)
        except Exception as e:
            print('Failed to compute mdri:', e)

    # cluster and get centroids
    try:
        dfc_clustered, centroids = kmeans_hotspots(dfc, lat_col=lat_col, lon_col=lon_col, n_clusters=n_clusters)
    except Exception as e:
        raise RuntimeError(f'Hotspot clustering failed: {e}')

    # build folium map
    center_lat = dfc[lat_col].mean()
    center_lon = dfc[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3)

    # color scale based on mdri
    mdri_vals = pd.to_numeric(dfc_clustered.get(mdri_col, pd.Series([])), errors='coerce')
    if not mdri_vals.dropna().empty:
        cmap = linear.YlOrRd_09.scale(mdri_vals.min(), mdri_vals.max())
    else:
        cmap = linear.YlOrRd_09.scale(0, 1)

    # add points
    for _, r in dfc_clustered.iterrows():
        lat = r[lat_col]
        lon = r[lon_col]
        if pd.isna(lat) or pd.isna(lon):
            continue
        mdv = r.get(mdri_col, None)
        color = cmap(mdv) if mdv is not None and not pd.isna(mdv) else '#3186cc'
        popup = folium.Popup(html=folium.IFrame(str(r.to_dict()), width=300, height=150))
        folium.CircleMarker(location=[lat, lon], radius=3, color=color, fill=True, fill_opacity=0.7, popup=popup).add_to(m)

    # add centroids
    for cid, (clat, clon) in centroids.items():
        folium.Marker([clat, clon],
                      icon=folium.Icon(color='red', icon='star'),
                      popup=f'Centroid: cluster {cid}').add_to(m)

    # add legend (colorbar)
    try:
        cmap.caption = 'MDRI'
        m.add_child(cmap)
    except Exception:
        pass

    m.save(out)
    print(f'Saved hotspot map to {out}')


def main():
    df = load_data()
    out = 'hotspots_map.html'
    make_map(df, n_clusters=6, out=out)


if __name__ == '__main__':
    main()
