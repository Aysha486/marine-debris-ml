"""Generate interactive 3D globe map with enhanced visualization using Plotly."""
import os
import pandas as pd
import plotly.graph_objects as go
from mdri import compute_mdri


def create_3d_globe_map(df, top_n=50, output_file='hotspots_globe_3d.html'):
    """Create an interactive 3D globe visualization of marine debris hotspots."""
    
    print('[*] Creating 3D globe map with top {} countries...'.format(top_n))
    
    # Ensure MDRI is computed
    if 'mdri' not in df.columns:
        print('[*] Computing MDRI...')
        df = compute_mdri(df)
    
    # Get country column
    country_col = 'country' if 'country' in df.columns else 'COUNTRY / LOCATION'
    
    # Load capital coordinates
    if not os.path.exists('country_capital_centroids.csv'):
        print('[!] Warning: country_capital_centroids.csv not found')
        return
    
    capitals_df = pd.read_csv('country_capital_centroids.csv')
    capitals_df['country'] = capitals_df['country'].astype(str).str.strip().str.upper()
    
    # Compute MDRI by country
    country_mdri = df.groupby(country_col, as_index=True)['mdri'].agg(['mean', 'count']).reset_index()
    country_mdri.columns = [country_col, 'mdri', 'count']
    country_mdri.rename(columns={country_col: 'country'}, inplace=True)
    country_mdri = country_mdri.sort_values('mdri', ascending=False).head(top_n).reset_index(drop=True)
    country_mdri['country'] = country_mdri['country'].astype(str).str.strip().str.upper()
    
    # Merge with capital coordinates
    merged = country_mdri.merge(
        capitals_df[['country', 'lat', 'lon']], 
        on='country', 
        how='inner'
    )
    
    if len(merged) == 0:
        print('[!] No matching countries found with capital coordinates')
        return
    
    print('[*] Found {} countries with coordinates'.format(len(merged)))
    
    # Normalize MDRI for color scale (0-1)
    mdri_min = merged['mdri'].min()
    mdri_max = merged['mdri'].max()
    merged['mdri_normalized'] = (merged['mdri'] - mdri_min) / (mdri_max - mdri_min + 1e-6)
    
    # Create hover text with detailed information
    merged['hover_text'] = (
        '<b>' + merged['country'] + '</b><br>' +
        'MDRI: ' + merged['mdri'].round(4).astype(str) + '<br>' +
        'Risk Level: ' + merged['mdri_normalized'].apply(
            lambda x: 'HIGH' if x >= 0.66 else ('MEDIUM' if x >= 0.33 else 'LOW')
        ) + '<br>' +
        'Records: ' + merged['count'].astype(int).astype(str)
    )
    
    # Create 3D globe figure
    fig = go.Figure()
    
    # Add hotspot markers
    fig.add_trace(go.Scattergeo(
        lon=merged['lon'].tolist(),
        lat=merged['lat'].tolist(),
        mode='markers',
        marker=dict(
            size=merged['mdri_normalized'] * 20 + 5,
            color=merged['mdri'].tolist(),
            colorscale='Reds',
            showscale=True,
            colorbar=dict(
                title='MDRI<br>Risk Index',
                thickness=15,
                len=0.7,
                x=1.02
            ),
            line=dict(width=2, color='rgba(26, 26, 46, 0.8)'),
            opacity=0.9,
            sizemode='diameter'
        ),
        text=merged['hover_text'].tolist(),
        hovertemplate='%{text}<extra></extra>',
        name='Hotspots'
    ))
    
    # Update layout for enhanced 3D appearance
    fig.update_layout(
        title=dict(
            text='<b>Marine Debris Hotspot Analysis - 3D Global Map</b><br><sub>MDRI Risk Index by Country</sub>',
            font=dict(size=18, color='#1a1a2e'),
            x=0.5,
            xanchor='center'
        ),
        geo=dict(
            projection=dict(type='orthographic'),
            bgcolor='rgba(200, 230, 255, 0.3)',
            coastlinecolor='rgba(100, 150, 200, 0.5)',
            landcolor='rgba(240, 240, 240, 0.8)',
            lakecolor='rgba(150, 200, 255, 0.4)',
            showland=True,
            showcountries=True,
            countrycolor='rgba(150, 150, 150, 0.3)',
            showocean=True,
            oceancolor='rgba(100, 180, 255, 0.2)',
            showlakes=True,
            framecolor='rgba(50, 50, 50, 0.3)',
            framewidth=1,
            showframe=True
        ),
        width=1400,
        height=800,
        hovermode='closest',
        paper_bgcolor='rgba(240, 245, 250, 1)',
        font=dict(family='Arial, sans-serif', size=12, color='#1a1a2e'),
        margin=dict(l=0, r=200, t=100, b=50)
    )
    
    fig.add_annotation(
        text=(
            '<b>Color Scale:</b> Red = High Risk | Yellow = Medium Risk | Cyan = Low Risk<br>'
            '<b>Top {} Countries by MDRI</b> | Bubble size represents risk magnitude'.format(top_n)
        ),
        xref='paper', yref='paper',
        x=0.5, y=-0.05,
        showarrow=False,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(100, 100, 100, 0.5)',
        borderwidth=1,
        font=dict(size=11, color='#1a1a2e'),
        align='center'
    )
    
    fig.write_html(output_file)
    print('[OK] 3D globe map saved to {}'.format(output_file))
    return fig


def create_mercator_globe_map(df, top_n=50, output_file='hotspots_globe_mercator.html'):
    """Create an interactive Mercator projection map."""
    
    print('[*] Creating Mercator globe map with top {} countries...'.format(top_n))
    
    # Ensure MDRI is computed
    if 'mdri' not in df.columns:
        print('[*] Computing MDRI...')
        df = compute_mdri(df)
    
    # Get country column
    country_col = 'country' if 'country' in df.columns else 'COUNTRY / LOCATION'
    
    # Load capital coordinates
    if not os.path.exists('country_capital_centroids.csv'):
        print('[!] Warning: country_capital_centroids.csv not found')
        return
    
    capitals_df = pd.read_csv('country_capital_centroids.csv')
    capitals_df['country'] = capitals_df['country'].astype(str).str.strip().str.upper()
    
    # Compute MDRI by country
    country_mdri = df.groupby(country_col, as_index=True)['mdri'].agg(['mean', 'count']).reset_index()
    country_mdri.columns = [country_col, 'mdri', 'count']
    country_mdri.rename(columns={country_col: 'country'}, inplace=True)
    country_mdri = country_mdri.sort_values('mdri', ascending=False).head(top_n).reset_index(drop=True)
    country_mdri['country'] = country_mdri['country'].astype(str).str.strip().str.upper()
    
    # Merge with capital coordinates
    merged = country_mdri.merge(
        capitals_df[['country', 'lat', 'lon']], 
        on='country', 
        how='inner'
    )
    
    if len(merged) == 0:
        print('[!] No matching countries found')
        return
    
    print('[*] Found {} countries with coordinates'.format(len(merged)))
    
    # Normalize MDRI
    mdri_min = merged['mdri'].min()
    mdri_max = merged['mdri'].max()
    merged['mdri_normalized'] = (merged['mdri'] - mdri_min) / (mdri_max - mdri_min + 1e-6)
    
    # Create hover text
    merged['hover_text'] = (
        '<b>' + merged['country'] + '</b><br>' +
        'MDRI: ' + merged['mdri'].round(4).astype(str) + '<br>' +
        'Risk: ' + merged['mdri_normalized'].apply(
            lambda x: 'HIGH' if x >= 0.66 else ('MEDIUM' if x >= 0.33 else 'LOW')
        ) + '<br>' +
        'Records: ' + merged['count'].astype(int).astype(str)
    )
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lon=merged['lon'].tolist(),
        lat=merged['lat'].tolist(),
        mode='markers+text',
        marker=dict(
            size=merged['mdri_normalized'] * 25 + 8,
            color=merged['mdri'].tolist(),
            colorscale='Hot',
            showscale=True,
            colorbar=dict(
                title='MDRI Index',
                thickness=15,
                len=0.7,
                x=1.02
            ),
            line=dict(width=2.5, color='rgba(26, 26, 46, 0.9)'),
            opacity=0.95,
            sizemode='diameter'
        ),
        text=merged['country'].tolist(),
        textposition='top center',
        textfont=dict(size=9, color='#1a1a2e'),
        hovertext=merged['hover_text'].tolist(),
        hoverinfo='text',
        name='Hotspots'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Marine Debris Hotspots - Mercator Projection</b><br><sub>Global Risk Distribution Map</sub>',
            font=dict(size=18, color='#1a1a2e'),
            x=0.5,
            xanchor='center'
        ),
        geo=dict(
            projection=dict(type='mercator'),
            bgcolor='rgba(220, 240, 255, 0.4)',
            coastlinecolor='rgba(80, 120, 180, 0.6)',
            landcolor='rgba(230, 235, 240, 0.8)',
            lakecolor='rgba(150, 200, 255, 0.5)',
            showland=True,
            showcountries=True,
            countrycolor='rgba(100, 100, 100, 0.3)',
            showocean=True,
            oceancolor='rgba(100, 180, 255, 0.3)',
            showlakes=True,
            framecolor='rgba(50, 50, 50, 0.2)',
            framewidth=1,
            showframe=True
        ),
        width=1400,
        height=700,
        hovermode='closest',
        paper_bgcolor='rgba(240, 245, 250, 1)',
        font=dict(family='Arial, sans-serif', size=12, color='#1a1a2e'),
        margin=dict(l=0, r=200, t=100, b=50)
    )
    
    fig.add_annotation(
        text=(
            '<b>Risk Levels:</b> Red = Extreme | Orange = High | Yellow = Medium | Blue = Low<br>'
            '<b>Hotspots:</b> Top {} countries | Bubble size = Risk magnitude'.format(top_n)
        ),
        xref='paper', yref='paper',
        x=0.5, y=-0.05,
        showarrow=False,
        bgcolor='rgba(255, 255, 255, 0.85)',
        bordercolor='rgba(100, 100, 100, 0.5)',
        borderwidth=1,
        font=dict(size=11, color='#1a1a2e'),
        align='center'
    )
    
    fig.write_html(output_file)
    print('[OK] Mercator globe map saved to {}'.format(output_file))
    return fig


if __name__ == '__main__':
    print('[*] Loading data...')
    
    for file_path in ['data_cleaned.csv', 'data_cleaned_normalized.csv']:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print('[OK] Loaded {} records from {}'.format(len(df), file_path))
            break
    else:
        print('[!] No cleaned data file found')
        exit(1)
    
    create_3d_globe_map(df, top_n=50, output_file='hotspots_globe_3d.html')
    create_mercator_globe_map(df, top_n=50, output_file='hotspots_globe_mercator.html')
    
    print('[OK] Globe map generation complete!')
