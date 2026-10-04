import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from pathlib import Path
import joblib
import streamlit.components.v1 as components
import subprocess
from mdri import compute_mdri
from hotspots import kmeans_hotspots
from forecast_future_debris import DebrisForecastEngine
from forecast_visualizations import ForecastVisualizer
import plotly.graph_objects as go
from login_page import show_login_page
from model_comparison import show_model_comparison_page
from country_model_comparison import show_country_model_comparison

try:
    from gis_dashboard import render_advanced_hotspot_dashboard
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False
try:
    from unified_forecast_engine import UnifiedForecastEngine
    import torch
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# Check if user is logged in
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Initialize page state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'dashboard'

st.set_page_config(page_title='Marine Debris MDRI Dashboard', layout='wide')

# If not logged in, show login page
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# Route to model comparison page if selected
if st.session_state.current_page == 'model_comparison':
    show_model_comparison_page()
    st.stop()

# Route to country model comparison page if selected
if st.session_state.current_page == 'country_comparison':
    show_country_model_comparison()
    st.stop()

# Main app starts here (user is logged in)
st.title('Hybrid Marine Debris Prediction & Risk Mapping')

# Global style: ocean-inspired palette, higher contrast, highlighted sidebar
st.markdown(
        """
        <style>
            :root {
                --ocean-900: #0d1b2a;
                --ocean-700: #1b263b;
                --ocean-500: #274c77;
                --mint-400: #52b788;
                --sand-100: #f6f7fb;
            }
            body {
                background:
                    linear-gradient(180deg, rgba(240,244,248,0.92), rgba(232,238,244,0.94)),
                    url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=2000&q=80') center/cover fixed;
                color: #0b1220;
            }
            .stApp { background: transparent; }
            /* Main content shell for readability */
            .block-container {
                background: rgba(255,255,255,0.92);
                border: 1px solid rgba(12,26,42,0.08);
                box-shadow: 0 10px 28px rgba(0,0,0,0.12);
                border-radius: 18px;
                padding: 1.6rem 1.8rem;
                backdrop-filter: blur(6px);
            }
            /* Card look for sections */
            .glass-card {
                background: rgba(255,255,255,0.96);
                border: 1px solid rgba(12,26,42,0.08);
                box-shadow: 0 8px 22px rgba(0,0,0,0.10);
                border-radius: 16px;
                padding: 1.25rem;
            }
            /* Dataframe wrapper */
            .stDataFrame, .element-container iframe {
                border-radius: 12px !important;
                overflow: hidden;
                box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            }
            /* Buttons */
            .stButton > button {
                border-radius: 10px;
                font-weight: 700;
                background: linear-gradient(135deg, var(--mint-400), #4cc9f0);
                border: none;
                color: #0b1220;
                box-shadow: 0 8px 20px rgba(76, 201, 240, 0.35);
            }
            .stButton > button:hover { transform: translateY(-1px); }
            .stButton > button:active { transform: translateY(0); }
            /* Sidebar styling & highlighting */
            [data-testid="stSidebar"] {
                background: rgba(13,27,42,0.92);
                border-right: 1px solid rgba(255,255,255,0.12);
            }
            [data-testid="stSidebar"] *,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span {
                color: #f8fbff !important;
                font-weight: 700;
            }
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4 {
                color: #ffffff !important;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            /* Tabs */
            .stTabs [data-baseweb="tab"] { color: #dce8ff; }
            .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #0b1220; font-weight: 700; }
            /* Headers */
            h1, h2, h3, h4 { color: #0b1220; text-shadow: none; }
            .metric-label { color: #1b263b; font-weight: 600; }
            .hero-title { font-size: 2.3rem; font-weight: 800; margin: 0; color: #0b1220; }
            .hero-sub { color: #1f2d3d; margin-top: 0.4rem; }
            .hero-badge {
                display: inline-block; padding: 0.35rem 0.75rem; border-radius: 999px;
                background: #52b788; color: #0b1220; font-weight: 800; font-size: 0.9rem;
                box-shadow: 0 4px 10px rgba(0,0,0,0.10);
            }
            /* Inputs text color fix */
            input, textarea, select { color: #0b1220 !important; background: #ffffff !important; }
            label, p, span { color: #0b1220 !important; }
            .stTabs [data-baseweb="tab"] { color: #1b263b; }
            .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #0b1220; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Hero header
hero = st.container()
with hero:
    col_h1, col_h2 = st.columns([2,1])
    with col_h1:
        st.markdown('<div class="hero-badge">Secure Workspace</div>', unsafe_allow_html=True)
        st.markdown('<p class="hero-title">Hybrid Marine Debris Prediction & Risk Mapping</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">AI-driven forecasts • MDRI analytics • 3D globe visualizations</p>', unsafe_allow_html=True)
    with col_h2:
        if st.session_state.get('username'):
            st.metric('Active User', st.session_state.username)
        st.metric('Data Source', 'data_cleaned.csv')
        # Logout button
        if st.button('🚪 Logout', use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

# Add logout button in sidebar
with st.sidebar:
    st.write(f"👤 Logged in as: **{st.session_state.username}**")
    if st.button('🚪 Logout', type='secondary'):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
    st.divider()

DATA_PATH = r"data_cleaned.csv"

@st.cache_data
def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    return df


def attach_country_centroids(df):
    """Add latitude/longitude columns by merging country centroids when missing."""
    if {'latitude', 'longitude'}.issubset(df.columns):
        return df

    centroids_path = os.path.join(os.path.dirname(__file__), 'country_capital_centroids.csv')
    if not os.path.exists(centroids_path):
        return df

    try:
        centroids = pd.read_csv(centroids_path)
    except Exception:
        return df

    if centroids.empty:
        return df

    centroids = centroids.dropna(subset=['lat', 'lon'])

    centroids = centroids.copy()
    centroids['country_key'] = centroids['country'].astype(str).str.strip().str.upper()

    # normalization map is optional; fall back to uppercase country name
    norm_map = {}
    norm_file = os.path.join(os.path.dirname(__file__), 'country_normalization_map.json')
    try:
        if os.path.exists(norm_file):
            with open(norm_file, 'r', encoding='utf-8') as f:
                norm_map = json.load(f)
    except Exception:
        norm_map = {}

    df_out = df.copy()
    country_col = 'COUNTRY / LOCATION' if 'COUNTRY / LOCATION' in df_out.columns else 'country'

    def make_key(val):
        raw = str(val).strip()
        upper = raw.upper()
        mapped = norm_map.get(upper, norm_map.get(raw, raw))
        return str(mapped).strip().upper()

    df_out['country_key'] = df_out[country_col].apply(make_key)

    merged = df_out.merge(
        centroids[['country_key', 'lat', 'lon']],
        on='country_key',
        how='left'
    )
    merged = merged.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    merged.drop(columns=['country_key'], inplace=True, errors='ignore')
    return merged
    return merged

st.sidebar.header('Settings')
cluster_count = st.sidebar.slider('Hotspot clusters (KMeans)', 2, 12, 5)
compute_mdri_btn = st.sidebar.button('Compute MDRI')

df = load_data()
if df.empty:
    st.warning(f"`{DATA_PATH}` not found or empty. Run `imputation.ipynb` to create it.")
else:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader('Data preview')
    st.dataframe(df.head(200))

    st.sidebar.markdown('---')
    st.sidebar.header('MDRI & Hotspots')

    if compute_mdri_btn:
        st.info('Computing MDRI...')
        try:
            df_mdri = compute_mdri(df)
            df_mdri = attach_country_centroids(df_mdri)
            
            # Scale MDRI from 0-1 to 0-100 for better visualization
            df_mdri['mdri'] = df_mdri['mdri'] * 100
            
            # Store results in session state for persistence
            st.session_state['mdri_computed'] = True
            st.session_state['df_mdri'] = df_mdri
            st.session_state['cluster_count'] = cluster_count
            
        except Exception as e:
            st.error(f'Error computing MDRI or hotspots: {e}')
            st.session_state['mdri_computed'] = False
    
    # Display MDRI results if computed (persistent)
    if st.session_state.get('mdri_computed', False) and 'df_mdri' in st.session_state:
        df_mdri = st.session_state['df_mdri']
        cluster_count = st.session_state.get('cluster_count', 5)
        
        st.success('✅ MDRI Computed Successfully!')
        
        # MDRI Visualization Section
        st.markdown("---")
        st.header("📊 MDRI Risk Index Analysis")
        
        # 1. Top Risk Countries - Horizontal Bar Chart
        st.subheader("🔴 Top 20 Highest Risk Countries")
        
        # Get unique countries only and sort by MDRI
        df_mdri_unique = df_mdri.groupby('COUNTRY / LOCATION')['mdri'].max().reset_index()
        top_countries = df_mdri_unique.nlargest(20, 'mdri').copy()
        
        # Define risk levels with colors based on percentiles for better distribution
        mdri_values = df_mdri_unique['mdri'].values
        p75 = np.percentile(mdri_values, 75)
        p50 = np.percentile(mdri_values, 50)
        p25 = np.percentile(mdri_values, 25)
        
        # Add Risk Level to top_countries using same logic
        def assign_risk_level(mdri):
            if mdri >= p75:
                return 'Critical'
            elif mdri >= p50:
                return 'High'
            elif mdri >= p25:
                return 'Medium'
            else:
                return 'Low'
        
        top_countries['Risk Level'] = top_countries['mdri'].apply(assign_risk_level)
        
        # Attach risk levels for later use
        df_mdri_unique['Risk Level'] = df_mdri_unique['mdri'].apply(assign_risk_level)
        df_mdri['Risk Level'] = df_mdri['mdri'].apply(assign_risk_level)
        
        # Map colors for top countries
        color_map = {'Critical': '#d32f2f', 'High': '#f57c00', 'Medium': '#fbc02d', 'Low': '#388e3c'}
        top_countries['Color'] = top_countries['Risk Level'].map(color_map)
        
        fig_top = go.Figure()
        fig_top.add_trace(go.Bar(
            y=top_countries['COUNTRY / LOCATION'],
            x=top_countries['mdri'],
            orientation='h',
            marker=dict(
                color=top_countries['Color'],
                line=dict(color='white', width=1)
            ),
            text=top_countries['mdri'].round(1),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>MDRI: %{x:.1f}<br>Risk: %{customdata}<extra></extra>',
            customdata=top_countries['Risk Level']
        ))
        
        fig_top.update_layout(
            title=f"Top 20 Countries by Marine Debris Risk Index ({len(top_countries)} shown)",
            xaxis_title="MDRI Score (0-100)",
            yaxis_title="Country",
            height=800,
            showlegend=False,
            yaxis={'categoryorder':'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig_top, use_container_width=True)
        
        st.info(f"📊 **Percentile-based Risk Thresholds:** Critical ≥{p75:.1f} | High ≥{p50:.1f} | Medium ≥{p25:.1f} | Low <{p25:.1f}")
        
        # Risk Level and MDRI score distribution charts removed per request.
        
        # 4. Detailed Data Table with Search
        st.subheader("🗂️ Detailed MDRI Data")
        
        search_country = st.text_input("🔍 Search by country name:", "")
        
        if search_country:
            filtered_df = df_mdri_unique[df_mdri_unique['COUNTRY / LOCATION'].str.contains(search_country, case=False, na=False)]
        else:
            filtered_df = df_mdri_unique
        
        display_df = filtered_df.sort_values('mdri', ascending=False)[['COUNTRY / LOCATION', 'mdri', 'Risk Level']].copy()

        # Map risk levels to colors
        risk_colors = {
            'Critical': '#d32f2f',  # red
            'High': '#f57c00',      # orange
            'Medium': '#fbc02d',    # yellow
            'Low': '#388e3c'        # green
        }

        # Build HTML table with colored rows (color only MDRI and Risk Level columns)
        html_table = '<table style="width:100%; border-collapse: collapse;"><tr style="background-color: #f0f0f0;"><th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Country</th><th style="padding: 8px; text-align: left; border: 1px solid #ddd;">MDRI Score</th><th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Risk Level</th></tr>'
        
        for idx, row in display_df.iterrows():
            risk_level = row['Risk Level']
            color = risk_colors.get(risk_level, '#ffffff')
            mdri_val = f"{row['mdri']:.2f}"
            country = row['COUNTRY / LOCATION']
            html_table += f'<tr><td style="padding: 8px; border: 1px solid #ddd;">{country}</td><td style="padding: 8px; border: 1px solid #ddd; background-color: {color}; color: white;">{mdri_val}</td><td style="padding: 8px; border: 1px solid #ddd; background-color: {color}; color: white;">{risk_level}</td></tr>'
        
        html_table += '</table>'
        
        st.markdown(html_table, unsafe_allow_html=True)

        # hotspots - Interactive map with Folium
        if {'latitude','longitude'}.issubset(df_mdri.columns):
            st.subheader('🗺️ Interactive Hotspot Map')
            st.info('KMeans clustering shows marine debris concentration zones')
            try:
                df_hot, _centroids = kmeans_hotspots(df_mdri, lat_col='latitude', lon_col='longitude', n_clusters=cluster_count)
                
                # Create Folium map centered on global center
                import folium
                from folium.plugins import MarkerCluster, HeatMap
                
                # Center map on mean coordinates
                center_lat = df_mdri['latitude'].mean()
                center_lon = df_mdri['longitude'].mean()
                
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=3,
                    tiles='OpenStreetMap'
                )
                
                # Add clustered markers with MDRI info
                marker_cluster = MarkerCluster().add_to(m)
                
                # Add individual hotspot markers with color coding
                risk_color_map = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
                
                for idx, row in df_hot.iterrows():
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                        risk_level = row.get('Risk Level', 'Unknown')
                        color = risk_color_map.get(risk_level, 'gray')
                        mdri_val = row.get('mdri', 0)
                        country = row.get('COUNTRY / LOCATION', 'Unknown')
                        
                        # Create popup with detailed info
                        popup_text = f"""
                        <b>{country}</b><br>
                        MDRI: {mdri_val:.2f}<br>
                        Risk: {risk_level}
                        """
                        
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=max(3, min(15, mdri_val / 5)),  # Size based on MDRI
                            popup=folium.Popup(popup_text, max_width=200),
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.7,
                            weight=2,
                            opacity=0.9
                        ).add_to(m)
                
                # Display map
                from streamlit_folium import st_folium
                st_folium(m, width=1400, height=600)
                
            except ImportError:
                st.warning('⚠️ Folium or streamlit-folium not installed. Using basic map instead.')
                df_coords = df_hot.rename(columns={'latitude':'lat','longitude':'lon'})[['lat','lon']].dropna()
                if not df_coords.empty:
                    st.map(df_coords)
            except Exception as ex:
                st.warning(f'Hotspot mapping failed: {ex}')
        else:
            st.warning('No `latitude`/`longitude` columns found for hotspot mapping. Consider running `generate_hotspot_map_capitals.py` to cache centroids.')

    st.sidebar.markdown('---')
    st.write('Use the sidebar to compute MDRI and view hotspots.')

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Place Advanced Analytics block at the very bottom of the sidebar ---
    st.sidebar.markdown('---')
    st.sidebar.header('📊 Advanced Analytics')

    # Polished spacing and layout
    st.sidebar.markdown('<div style="margin-bottom: 6px"></div>', unsafe_allow_html=True)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button('🤖 Model Comparison', use_container_width=True, key='nav_model_compare_bottom'):
            st.session_state.current_page = 'model_comparison'
            st.rerun()
    with col2:
        if st.button('📈 Dashboard', use_container_width=True, key='nav_dashboard_bottom'):
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    # Add Country Model Comparison button
    st.sidebar.markdown('<div style="margin-top: 8px"></div>', unsafe_allow_html=True)
    if st.sidebar.button('🌍 Country Model Comparison', use_container_width=True, key='nav_country_compare'):
        st.session_state.current_page = 'country_comparison'
        st.rerun()

    # Slight bottom padding to avoid flush against viewport edge
    st.sidebar.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)

    # Final spacing before placing Advanced Analytics at the very bottom
    st.sidebar.markdown('<div style="height: 8px"></div>', unsafe_allow_html=True)

    # --- Inference section: upload CSV or paste a single-row JSON and predict ---
    st.markdown('---')
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header('Inference (use trained pipeline)')
    pipeline_path = os.path.join(os.path.dirname(__file__), 'inference_pipeline.joblib')

    @st.cache_data
    def load_pipeline(path):
        if os.path.exists(path):
            return joblib.load(path)
        return None

    pipe = load_pipeline(pipeline_path)
    if pipe is None:
        st.warning('`inference_pipeline.joblib` not found. Run `build_inference_pipeline.py` first.')
    else:
        st.success('Loaded inference pipeline')

        # Option A: upload a CSV of raw rows
        uploaded = st.file_uploader('Upload CSV for batch prediction', type=['csv'])
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                preds = pipe.predict(df_in)
                df_out = df_in.copy()
                df_out['prediction'] = preds
                st.write('Predictions (first 20 rows)')
                st.dataframe(df_out.head(200))
                csv = df_out.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions CSV', csv, file_name='predictions.csv', mime='text/csv')
            except Exception as e:
                st.error(f'Batch prediction failed: {e}')

        # Option B: paste a single JSON-like row (key:value) or CSV row
        st.markdown('Or paste a single-row JSON/CSV-like dict (e.g. {"year":2020, "PEOPLE":1000, ...})')
        text = st.text_area('Paste single-row JSON here', height=120)
        if st.button('Predict from pasted row'):
            if not text.strip():
                st.error('No input provided')
            else:
                try:
                    # try to parse as JSON
                    import json
                    row = json.loads(text)
                    df_row = pd.DataFrame([row])
                    pred = pipe.predict(df_row)
                    st.write('Prediction:')
                    st.write(float(pred[0]))
                except Exception as e:
                    st.error(f'Failed to parse/predict pasted row: {e}')

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Hotspot 3D Globe Map (Interactive) ---
    st.markdown('---')
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header('🌍 Marine Debris Hotspots - 3D Interactive Globe')

    st.write('Explore marine debris risk across the globe with an interactive 3D visualization. Bubble size and color represent MDRI risk levels.')
    
    col1, col2 = st.columns([2, 1])
    with col1:
        map_style = st.radio('Map Projection:', ['3D Globe (Orthographic)', 'Mercator Projection'], horizontal=True)
    with col2:
        top_countries = st.number_input('Top N countries to display', min_value=10, max_value=150, value=50)

    if st.button('Generate 3D Globe Map'):
        proj_dir = Path(__file__).resolve().parent
        generator = proj_dir / 'generate_3d_globe_map.py'
        
        if not generator.exists():
            st.error(f'Generator script missing: {generator.name}')
        else:
            st.info('Generating 3D globe map... this may take a moment.')
            with st.spinner('Creating visualization...'):
                try:
                    proc = subprocess.run([sys.executable, str(generator)], cwd=str(proj_dir), capture_output=True, text=True, timeout=300)
                    out = proc.stdout or ''
                    err = proc.stderr or ''
                    
                    # Store generation result in session state
                    st.session_state['globe_map_generated'] = True
                    st.session_state['globe_map_output'] = out
                    st.session_state['globe_map_error'] = err
                    
                except subprocess.TimeoutExpired:
                    st.error('Generation timed out. Try reducing the number of countries.')
                    st.session_state['globe_map_generated'] = False
                except Exception as e:
                    st.error(f'Failed to generate map: {e}')
                    st.session_state['globe_map_generated'] = False
    
    # Display generation status if map was generated (persistent)
    if st.session_state.get('globe_map_generated', False):
        st.success('3D globe map generated successfully!')
        if st.session_state.get('globe_map_error'):
            with st.expander('View generation details'):
                st.code(st.session_state.get('globe_map_output', '') + st.session_state.get('globe_map_error', ''))
    
    # Prefer 3D globe maps
    globe_maps = [
        ('hotspots_globe_3d.html', '3D Globe (Orthographic)'),
        ('hotspots_globe_mercator.html', 'Mercator Projection'),
        ('hotspots_map_capitals.html', 'Capital Cities Map'),
        ('hotspots_map.html', 'Standard Hotspot Map')
    ]
    
    selected_map = None
    if map_style == '3D Globe (Orthographic)':
        for mfile, desc in globe_maps:
            if mfile == 'hotspots_globe_3d.html':
                path = Path(__file__).resolve().parent / mfile
                if path.exists():
                    selected_map = path
                    break
    else:
        for mfile, desc in globe_maps:
            if mfile == 'hotspots_globe_mercator.html':
                path = Path(__file__).resolve().parent / mfile
                if path.exists():
                    selected_map = path
                    break
    
    # Fallback to any available map
    if selected_map is None:
        for mfile, desc in globe_maps:
            path = Path(__file__).resolve().parent / mfile
            if path.exists():
                selected_map = path
                break

    if selected_map is None:
        st.warning('No 3D globe map found. Click "Generate 3D Globe Map" above to create one.')
    else:
        st.write(f'**Displaying:** {selected_map.name}')
        try:
            with open(selected_map, 'r', encoding='utf-8') as fh:
                html = fh.read()
            components.html(html, height=800, scrolling=False)
            st.download_button('Download map (HTML)', html.encode('utf-8'), file_name=selected_map.name, mime='text/html')
        except Exception as e:
            st.error(f'Failed to load map: {e}')

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Future Debris Forecasting Section ---
    st.markdown('---')
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header('🔮 Future Debris Forecasting')
    
    # Model selection
    st.subheader('Model Selection')
    st.info('Using LightGBM (classical ML) for forecasts. Deep learning options have been removed for simplicity and stability.')
    st.write('Generate future debris forecasts using the LightGBM model. Select countries and forecast horizon.')

    # Initialize forecast engine first (before any UI that references it)
    @st.cache_resource
    def get_forecast_engine():
        try:
            return DebrisForecastEngine('data_cleaned.csv', 'inference_pipeline.joblib')
        except Exception as e:
            st.error(f'Failed to load forecast engine: {e}')
            return None

    engine = get_forecast_engine()

    if engine is not None:
        # User-defined horizon
        years_ahead = st.number_input('Years ahead to forecast', min_value=1, max_value=10, value=3)
        
        # Optional target year selector (needs engine to know max year)
        target_year = None
        target_year_enabled = st.checkbox('Specify target year instead')
        if target_year_enabled:
            max_known_year = int(engine.data['year'].max()) if not engine.data.empty else 2023
            target_year = st.number_input('Target future year', min_value=max_known_year + 1, max_value=max_known_year + 20, value=max_known_year + int(years_ahead))

        # Get available countries (synthetic fallback will cover sparse histories)
        available_countries = engine.get_available_countries(min_years=1)
        
        # Country selection interface
        st.subheader('Select Countries to Forecast')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            forecast_mode = st.radio(
                'Forecast mode:',
                ['Top N countries by debris volume', 'Select specific countries', 'All available countries']
            )
        
        with col2:
            if forecast_mode == 'Top N countries by debris volume':
                top_n_forecast = st.number_input('Top N countries:', min_value=1, max_value=50, value=10)
                # Get top countries
                country_totals = engine.data.groupby('COUNTRY / LOCATION')['Total Items Collected'].sum()
                top_countries = country_totals.nlargest(top_n_forecast).index.tolist()
                selected_countries = [c for c in top_countries if c in available_countries]
            elif forecast_mode == 'Select specific countries':
                selected_countries = st.multiselect(
                    'Choose countries:',
                    options=available_countries,
                    default=available_countries[:5] if len(available_countries) >= 5 else available_countries
                )
            else:  # All countries
                selected_countries = available_countries
                st.info(f'Will forecast all {len(selected_countries)} countries (synthetic fallback for sparse histories)')
        
        st.write(f'**Selected: {len(selected_countries)} countries**')
        st.caption('If a country has insufficient history, synthetic data will be injected automatically before forecasting.')
        
        # Generate forecasts button
        if st.button('🚀 Generate Forecasts', type='primary'):
            if len(selected_countries) == 0:
                st.error('Please select at least one country')
            else:
                with st.spinner(f'Generating forecasts for {len(selected_countries)} countries...'):
                    try:
                        # Generate forecasts
                        combined = engine.forecast_multiple_countries(
                            selected_countries,
                            save_individual=True,
                            years_ahead=int(years_ahead) if not target_year_enabled else None,
                            target_year=int(target_year) if target_year is not None else None,
                            min_years=3
                        )
                        
                        if not combined.empty:
                            # Store in session state immediately
                            st.session_state['forecast_generated'] = True
                            st.session_state['forecast_data'] = combined
                            st.session_state['forecast_error'] = None
                        else:
                            st.session_state['forecast_generated'] = False
                            st.session_state['forecast_error'] = 'No forecast data generated'
                            
                    except Exception as e:
                        st.session_state['forecast_generated'] = False
                        st.session_state['forecast_error'] = str(e)
                        import traceback
                        st.session_state['forecast_traceback'] = traceback.format_exc()
        
        # Display forecast results (persistent)
        if st.session_state.get('forecast_generated', False):
            combined = st.session_state.get('forecast_data')
            if combined is not None and not combined.empty:
                st.success(f'✓ Generated forecasts for {combined["country"].nunique()} countries')
                
                # Display summary statistics
                st.subheader('Forecast Summary')
                final_year = combined['year'].max()
                final_data = combined[combined['year'] == final_year].sort_values('predicted_debris', ascending=False)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Countries Forecasted', combined['country'].nunique())
                with col2:
                    st.metric('Forecast Years', f"{combined['year'].min()} - {combined['year'].max()}")
                with col3:
                    total_predicted = final_data['predicted_debris'].sum()
                    st.metric(f'Total Predicted ({final_year})', f'{total_predicted:,.0f}')
                
                # Top 10 table
                st.subheader(f'Top 10 Predicted Debris Volumes ({final_year})')
                top10 = final_data.head(10)[['country', 'predicted_debris', 'lower_ci', 'upper_ci']].copy()
                top10.columns = ['Country', 'Predicted Debris', 'Lower CI', 'Upper CI']
                top10['Predicted Debris'] = top10['Predicted Debris'].apply(lambda x: f'{x:,.0f}')
                top10['Lower CI'] = top10['Lower CI'].apply(lambda x: f'{x:,.0f}')
                top10['Upper CI'] = top10['Upper CI'].apply(lambda x: f'{x:,.0f}')
                st.dataframe(top10, use_container_width=True)
                
                # Download combined CSV
                csv = combined.to_csv(index=False).encode('utf-8')
                st.download_button(
                    '📥 Download All Forecasts (CSV)',
                    csv,
                    file_name='future_forecasts_combined.csv',
                    mime='text/csv'
                )
                if 'data_source' in combined.columns:
                    source_summary = combined[['country', 'data_source']].drop_duplicates()['data_source'].value_counts()
                    st.caption('Data sources used (real / mixed / synthetic):')
                    st.write(source_summary.to_dict())
        
        # Display any errors
        if st.session_state.get('forecast_error'):
            st.error(f'Forecast generation failed: {st.session_state["forecast_error"]}')
            if 'forecast_traceback' in st.session_state:
                st.code(st.session_state['forecast_traceback'])
        
        # Visualization section (if forecasts exist)
        if 'forecast_data' in st.session_state or os.path.exists('future_forecasts_combined.csv'):
            st.markdown('---')
            st.subheader('📊 Forecast Visualizations')
            
            # Load forecast data
            if 'forecast_data' in st.session_state:
                forecast_df = st.session_state['forecast_data']
            else:
                forecast_df = pd.read_csv('future_forecasts_combined.csv')
            
            # Visualization tabs
            tab_list = ['📈 Country Trends', '🗺️ Forecast Map', '📊 Comparison', '📋 Full Data']
            if GIS_AVAILABLE:
                tab_list.append('🗺️ Advanced GIS Hotspots')
            
            tabs = st.tabs(tab_list)
            tab1, tab2, tab3, tab4 = tabs[0], tabs[1], tabs[2], tabs[3]
            
            with tab1:
                st.write('View historical + forecast trends for individual countries')
                
                available_forecast_countries = sorted(forecast_df['country'].unique().tolist())
                selected_viz_country = st.selectbox('Select country:', available_forecast_countries)
                
                if st.button('Generate Trend Plot'):
                    try:
                        viz = ForecastVisualizer('data_cleaned.csv', 'future_forecasts_combined.csv')
                        fig = viz.plot_country_forecast(selected_viz_country)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f'Failed to generate plot: {e}')
            
            with tab2:
                st.write('Interactive map showing forecast by country')
                
                forecast_years = sorted(forecast_df['year'].unique().tolist())
                selected_map_year = st.selectbox('Select forecast year:', forecast_years, index=len(forecast_years)-1)
                
                if st.button('Generate Forecast Map'):
                    with st.spinner('Creating map...'):
                        try:
                            viz = ForecastVisualizer('data_cleaned.csv', 'future_forecasts_combined.csv')
                            m = viz.create_forecast_map(year=selected_map_year, save_path='forecast_map_temp.html')
                            
                            # Display map
                            with open('forecast_map_temp.html', 'r', encoding='utf-8') as f:
                                map_html = f.read()
                            components.html(map_html, height=600, scrolling=True)
                            
                            # Download button
                            st.download_button(
                                '📥 Download Map (HTML)',
                                map_html.encode('utf-8'),
                                file_name=f'forecast_map_{selected_map_year}.html',
                                mime='text/html'
                            )
                        except Exception as e:
                            st.error(f'Failed to generate map: {e}')
            
            with tab3:
                st.write('Compare forecasts across multiple countries')
                
                compare_countries = st.multiselect(
                    'Select countries to compare:',
                    options=available_forecast_countries,
                    default=available_forecast_countries[:5] if len(available_forecast_countries) >= 5 else available_forecast_countries
                )
                
                if st.button('Generate Comparison Plot') and len(compare_countries) > 0:
                    try:
                        viz = ForecastVisualizer('data_cleaned.csv', 'future_forecasts_combined.csv')
                        fig = viz.create_forecast_comparison(countries=compare_countries)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f'Failed to generate comparison: {e}')
            
            with tab4:
                st.write('Complete forecast data table')
                st.dataframe(forecast_df, use_container_width=True)
                
                # Per-country performance metrics
                if os.path.exists('per_country_cv_evals.csv'):
                    st.subheader('Model Performance (Cross-Validation)')
                    cv_df = pd.read_csv('per_country_cv_evals.csv')
                    
                    # Filter to forecasted countries
                    cv_filtered = cv_df[cv_df['country'].isin(forecast_df['country'].unique())]
                    if not cv_filtered.empty:
                        st.dataframe(cv_filtered[['country', 'n_years', 'cv_mae_mean', 'cv_rmse_mean', 'n_folds']], 
                                   use_container_width=True)
            
            # Advanced GIS Hotspot Analysis Tab
            if GIS_AVAILABLE and len(tabs) > 4:
                with tabs[4]:
                    render_advanced_hotspot_dashboard()
    
    else:
        st.warning('Forecast engine not available. Ensure `data_cleaned.csv` and `inference_pipeline.joblib` exist.')

    st.markdown('</div>', unsafe_allow_html=True)

