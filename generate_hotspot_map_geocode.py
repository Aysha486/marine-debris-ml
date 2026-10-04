"""Generate hotspot map using country centroids obtained via Nominatim geocoding.

This script:
- loads `data_cleaned_normalized.csv` (or `data_cleaned.csv`)
- computes `mdri` if missing using `compute_mdri`
- aggregates mdri per country
- geocodes each country name using Nominatim (cached to `country_centroids_geocoded.csv`)
- saves `hotspots_map_geocode.html` with markers sized/colored by mdri

Usage:
    python generate_hotspot_map_geocode.py

Notes:
- Respects Nominatim by sleeping 1s between requests. Requires internet access.
"""
import os
import time
import requests
import pandas as pd

try:
    import folium
    from branca.colormap import linear
except Exception:
    print('Missing packages: install with `pip install folium branca requests`')
    raise

from mdri import compute_mdri

DATA_FILES = ['data_cleaned_normalized.csv', 'data_cleaned.csv']
CACHE_FILE = 'country_centroids_geocoded.csv'
OUT_HTML = 'hotspots_map_geocode.html'
# limit geocoding to top N countries by mdri to avoid long runs
TOP_N = 50


def load_data():
    for p in DATA_FILES:
        if os.path.exists(p):
            print(f'Loading {p}')
            return pd.read_csv(p)
    raise FileNotFoundError('No cleaned data file found; run imputation.ipynb first')


def geocode_country(name, session=None):
    # Use Nominatim OSM
    if session is None:
        session = requests.Session()
    url = 'https://nominatim.openstreetmap.org/search'
    params = {'q': name, 'format': 'json', 'limit': 1, 'countrycodes': ''}
    headers = {'User-Agent': 'maraine-hotspot-script/1.0 (contact: no-reply@example.com)'}
    resp = session.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    item = data[0]
    return float(item['lat']), float(item['lon'])


def load_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame(columns=['country', 'lat', 'lon'])


def save_cache(df):
    df.to_csv(CACHE_FILE, index=False)


def build_map(country_mdri, centroids, out=OUT_HTML):
    # center map at weighted mean if possible
    valid = centroids.dropna(subset=['lat', 'lon']).copy()
    if valid.empty:
        raise ValueError('No centroids available')
    center_lat = valid['lat'].mean()
    center_lon = valid['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)

    # merge mdri
    merged = centroids.merge(country_mdri, on='country', how='left')
    vals = pd.to_numeric(merged['mdri'].fillna(0), errors='coerce')
    if not vals.dropna().empty:
        cmap = linear.YlOrRd_09.scale(vals.min(), vals.max())
    else:
        cmap = linear.YlOrRd_09.scale(0, 1)

    for _, r in merged.iterrows():
        lat = r['lat']
        lon = r['lon']
        if pd.isna(lat) or pd.isna(lon):
            continue
        mdv = r.get('mdri', 0)
        color = cmap(mdv) if mdv and not pd.isna(mdv) else '#3186cc'
        radius = 4 + (mdv or 0) * 1000
        popup_html = f"<b>{r['country']}</b><br>mdri: {mdv:.3e}"
        folium.CircleMarker(location=[lat, lon], radius=6, color=color, fill=True, fill_opacity=0.8,
                            popup=folium.Popup(popup_html, max_width=300)).add_to(m)

    try:
        cmap.caption = 'MDRI'
        m.add_child(cmap)
    except Exception:
        pass

    m.save(out)
    print(f'Saved map to {out}')


def main():
    df = load_data()

    # compute mdri if not present
    if 'mdri' not in df.columns:
        print('Computing mdri...')
        df = compute_mdri(df)

    # aggregate mdri per country (mean)
    # prefer `country` or `COUNTRY / LOCATION`
    if 'country' in df.columns:
        country_col = 'country'
    elif 'COUNTRY / LOCATION' in df.columns:
        country_col = 'COUNTRY / LOCATION'
    else:
        raise ValueError('No country column found')

    country_mdri = df.groupby(country_col)['mdri'].mean().reset_index().rename(columns={country_col: 'country'})

    # load or build centroids cache
    cache = load_cache()
    session = requests.Session()
    # choose top N countries by mdri for geocoding (skip ones already cached)
    top_countries = country_mdri.sort_values('mdri', ascending=False).head(TOP_N)['country'].tolist()
    to_geocode = [c for c in top_countries if c not in list(cache['country'])]

    if to_geocode:
        print(f'Geocoding {len(to_geocode)} countries (this may take a while)')
    for i, c in enumerate(to_geocode):
        try:
            print(f'Geocoding ({i+1}/{len(to_geocode)}): {c}')
            res = geocode_country(c, session=session)
            if res:
                lat, lon = res
                cache = pd.concat([cache, pd.DataFrame([{'country': c, 'lat': lat, 'lon': lon}])], ignore_index=True)
            else:
                cache = pd.concat([cache, pd.DataFrame([{'country': c, 'lat': None, 'lon': None}])], ignore_index=True)
        except Exception as e:
            print(f'Failed to geocode {c}: {e}')
            cache = pd.concat([cache, pd.DataFrame([{'country': c, 'lat': None, 'lon': None}])], ignore_index=True)
        save_cache(cache)
        time.sleep(1.1)

    # build map
    build_map(country_mdri, cache, out=OUT_HTML)


if __name__ == '__main__':
    main()
