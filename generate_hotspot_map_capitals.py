"""Generate hotspot map using country capitals as centroids.

Workflow:
- load cleaned data
- compute mdri if missing
- fetch country->capital list from GitHub
- map your `country` names to capitals (use `country_normalization_map.json` if present)
- geocode capitals via Nominatim (cached to `country_capital_centroids.csv`)
- build `hotspots_map_capitals.html`
"""
import os
import time
import json
import argparse
import requests
import pandas as pd

try:
    import folium
    from branca.colormap import linear, LinearColormap
except Exception:
    print('Install required packages: pip install folium branca requests')
    raise

from mdri import compute_mdri

CAPITALS_URL = 'https://raw.githubusercontent.com/icyrockcom/country-capitals/master/data/country-list.csv'
CACHE_FILE = 'country_capital_centroids.csv'
OUT_HTML = 'hotspots_map_capitals.html'
# default top N; can be overridden by CLI arg
DEFAULT_TOP_N = 20


def load_data():
    for p in ['data_cleaned_normalized.csv', 'data_cleaned.csv']:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError('No cleaned data file found')


def fetch_capitals_csv():
    print('Fetching capitals list...')
    r = requests.get(CAPITALS_URL, timeout=30)
    r.raise_for_status()
    text = r.text
    # CSV has headers: "country","capital","type"
    df = pd.read_csv(pd.io.common.StringIO(text))
    df['country_key'] = df['country'].str.strip().str.lower()
    df['capital'] = df['capital'].str.replace('\n', ' ', regex=False)
    return df


def load_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame(columns=['country','capital','lat','lon'])


def save_cache(df):
    df.to_csv(CACHE_FILE, index=False)


def geocode(query, session=None):
    if session is None:
        session = requests.Session()
    url = 'https://nominatim.openstreetmap.org/search'
    headers = {'User-Agent': 'maraine-hotspot-script/1.0 (contact: no-reply@example.com)'}
    params = {'q': query, 'format': 'json', 'limit': 1}
    r = session.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return float(data[0]['lat']), float(data[0]['lon'])


def build_map(merged, out=OUT_HTML):
    valid = merged.dropna(subset=['lat','lon'])
    if valid.empty:
        raise ValueError('No centroids to map')
    center_lat = valid['lat'].mean()
    center_lon = valid['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles='CartoDB positron')
    vals = pd.to_numeric(merged['mdri'].fillna(0), errors='coerce')
    
    # Create custom color palette: light blue for low MDRI, dark red for high MDRI
    # This ensures colors are visible on light map background
    cmap = LinearColormap(
        colors=['#e6f2ff', '#99ccff', '#4da6ff', '#ff9999', '#ff6666', '#ff3333', '#cc0000', '#990000'],
        vmin=vals.min() if not vals.dropna().empty else 0,
        vmax=vals.max() if not vals.dropna().empty else 1,
        caption='MDRI (Marine Debris Risk Index)'
    )

    for _, r in merged.iterrows():
        if pd.isna(r['lat']) or pd.isna(r['lon']):
            continue
        mdv = r.get('mdri', 0)
        color = cmap(mdv) if mdv and not pd.isna(mdv) else '#4da6ff'
        popup = folium.Popup(f"<b>{r['country']}</b><br>capital: {r['capital']}<br>mdri: {mdv:.3e}", max_width=300)
        folium.CircleMarker(location=[r['lat'], r['lon']], radius=8, color=color, fill=True, fill_opacity=0.9, 
                          weight=2, popup=popup).add_to(m)

    m.add_child(cmap)
    
    out = os.path.normpath(out)
    m.save(out)
    print(f'[OK] Saved map to {out}')


def main(top_n=DEFAULT_TOP_N):
    df = load_data()
    if 'mdri' not in df.columns:
        print('Computing mdri...')
        df = compute_mdri(df)

    # pick country column
    country_col = 'country' if 'country' in df.columns else 'COUNTRY / LOCATION'
    country_mdri = df.groupby(country_col)['mdri'].mean().reset_index().rename(columns={country_col:'country'})
    top_countries = country_mdri.sort_values('mdri', ascending=False).head(top_n)

    capitals_df = fetch_capitals_csv()
    cache = load_cache()
    session = requests.Session()

    results = []
    for _, row in top_countries.iterrows():
        cname = str(row['country']).strip()
        key = cname.lower()
        # try direct match in capitals_df
        cap_row = capitals_df[capitals_df['country_key'] == key]
        capital = None
        if not cap_row.empty:
            capital = cap_row.iloc[0]['capital']
        else:
            # try normalization map if available
            norm_file = 'country_normalization_map.json'
            if os.path.exists(norm_file):
                with open(norm_file,'r',encoding='utf-8') as f:
                    norm_map = json.load(f)
                if cname in norm_map:
                    std = norm_map[cname]
                    cap_row = capitals_df[capitals_df['country_key'] == std.strip().lower()]
                    if not cap_row.empty:
                        capital = cap_row.iloc[0]['capital']

        # if we have a capital, check cache
        lat = lon = None
        if capital:
            cached = cache[(cache['country'] == cname) | (cache['capital'] == capital)]
            if not cached.empty:
                lat = cached.iloc[0]['lat']
                lon = cached.iloc[0]['lon']
            else:
                try:
                    print(f'Geocoding capital {capital} for country {cname}')
                    res = geocode(f"{capital}, {cname}", session=session)
                    if res:
                        lat, lon = res
                        cache = pd.concat([cache, pd.DataFrame([{'country': cname, 'capital': capital, 'lat': lat, 'lon': lon}])], ignore_index=True)
                        save_cache(cache)
                except Exception as e:
                    print(f'Failed to geocode {capital} ({cname}): {e}')
                time.sleep(1.1)

        results.append({'country': cname, 'capital': capital if capital else '', 'lat': lat, 'lon': lon, 'mdri': row['mdri']})

    merged = pd.DataFrame(results)
    build_map(merged, out=OUT_HTML)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate hotspot map using capitals (geocodes top-N countries)')
    parser.add_argument('--top', type=int, default=DEFAULT_TOP_N, help='Top N countries by mdri to geocode')
    parser.add_argument('--all', action='store_true', help='Geocode all countries (overrides --top)')
    args = parser.parse_args()
    if args.all:
        # set a large top value; script will limit to available countries
        main(top_n=99999)
    else:
        main(top_n=args.top)
