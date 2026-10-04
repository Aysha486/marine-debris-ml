"""
Advanced GIS Hotspot Analysis Dashboard Component
Integrates DBSCAN, HDBSCAN, and KDE Analysis
For use with Streamlit app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from advanced_hotspot_analysis import AdvancedHotspotAnalyzer, HDBSCAN_AVAILABLE
import folium
from streamlit_folium import st_folium

def render_advanced_hotspot_dashboard():
    """Render Advanced GIS Hotspot Analysis Dashboard"""
    
    st.header("🗺️ Advanced GIS Hotspot Analysis")
    st.markdown("""
    Advanced geospatial clustering and visualization of marine debris hotspots.
    Combining multiple algorithms to identify risk zones.
    """)
    
    # Initialize analyzer
    @st.cache_resource
    def load_analyzer():
        return AdvancedHotspotAnalyzer()
    
    analyzer = load_analyzer()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🎯 Analysis Settings")
        
        methods = ["DBSCAN", "KDE Heatmap"]
        if HDBSCAN_AVAILABLE:
            methods.insert(1, "HDBSCAN")
        analysis_type = st.radio(
            "Select Clustering Method:",
            methods
        )
        
        # Get available countries
        countries = sorted(analyzer.df['COUNTRY / LOCATION'].unique())
        selected_country = st.selectbox("Select Country:", countries)
        
        # Year range
        all_years = sorted(analyzer.df['year'].dt.year.unique())
        year_range = st.slider(
            "Year Range:",
            min_value=min(all_years),
            max_value=max(all_years),
            value=(min(all_years), max(all_years)),
            step=1
        )
        
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📍 {selected_country} - {analysis_type}")
    
    with col2:
        st.metric("Dataset Records", len(analyzer.df))
    
    # Analysis execution
    try:
        # Prepare data
        country_data = analyzer.prepare_spatial_data(
            country=selected_country,
            year_range=year_range
        )
        
        st.success(f"✓ Loaded {len(country_data)} debris records for {selected_country}")
        
        if analysis_type == "DBSCAN":
            render_dbscan_analysis(analyzer, country_data, selected_country)
        
        elif analysis_type == "HDBSCAN":
            render_hdbscan_analysis(analyzer, country_data, selected_country)
        
        elif analysis_type == "KDE Heatmap":
            render_kde_analysis(analyzer, country_data, selected_country)
        
        # Temporal Evolution removed per UI request
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("This may be due to insufficient data for the selected country/year range.")


def render_dbscan_analysis(analyzer, country_data, country):
    """Render DBSCAN clustering analysis"""
    
    st.subheader("DBSCAN Clustering")
    st.markdown("""
    **Density-Based Spatial Clustering of Applications with Noise**
    - Good for irregular coastlines and non-convex clusters
    - Parameters: eps (neighborhood radius), min_samples (core points)
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        eps = st.slider("EPS (neighborhood radius):", 0.05, 0.5, 0.15, 0.05)
    with col2:
        min_samples = st.slider("Min Samples:", 1, 10, 3, 1)
    
    # Apply DBSCAN, with synthetic fallback if insufficient data
    with st.spinner("🔍 Applying DBSCAN clustering..."):
        clustered_data = analyzer.dbscan_clustering(
            country_data,
            eps=eps,
            min_samples=min_samples
        )

        if clustered_data is None:
            # Fallback to synthetic dataset when records are insufficient
            import os
            synth_path = os.path.join(os.path.dirname(__file__), 'data_synthetic_all_countries.csv')
            if os.path.exists(synth_path):
                synth_df = pd.read_csv(synth_path)
                synth_df = synth_df[synth_df['COUNTRY / LOCATION'] == country]
                if len(synth_df) >= max(min_samples, 5):
                    st.info("Using synthetic data fallback for clustering")
                    clustered_data = analyzer.dbscan_clustering(
                        synth_df,
                        eps=eps,
                        min_samples=min_samples
                    )

        if clustered_data is None:
            st.error("Insufficient data to perform DBSCAN clustering for this selection.")
            st.info("Try a broader year range or select a different country.")
            return

        # Get country center coordinates for map
        coords = analyzer._get_country_coordinates(country_data)
        center = None
        if coords is not None and len(coords) > 0:
            center = [coords[:, 0].mean(), coords[:, 1].mean()]
        
        # Create map
        dbscan_map = analyzer.create_dbscan_map(clustered_data, center=center)
        st_folium(dbscan_map, width=1200, height=600)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    n_clusters = len(set(clustered_data['dbscan_cluster']) - {-1})
    n_noise = (clustered_data['dbscan_cluster'] == -1).sum()
    
    with col1:
        st.metric("Clusters Found", n_clusters)
    with col2:
        st.metric("Noise Points", n_noise)
    with col3:
        st.metric("Total Points", len(clustered_data))
    
    # Download option
    st.download_button(
        label="📥 Download DBSCAN Results (CSV)",
        data=clustered_data.to_csv(index=False),
        file_name=f"dbscan_{country}.csv",
        mime="text/csv"
    )


def render_hdbscan_analysis(analyzer, country_data, country):
    """Render HDBSCAN clustering analysis"""

    if not HDBSCAN_AVAILABLE:
        st.warning("⚠️ HDBSCAN not available. Install with: pip install hdbscan")
        return
    
    st.subheader("HDBSCAN Clustering")
    st.markdown("""
    **Hierarchical Density-Based Spatial Clustering**
    - Advanced: multi-scale density analysis
    - Better handling of varying density clusters
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        min_cluster_size = st.slider("Min Cluster Size:", 2, 20, 5, 1)
    with col2:
        min_samples = st.slider("Min Samples:", 1, 10, 5, 1)
    
    # Apply HDBSCAN
    with st.spinner("🔍 Applying HDBSCAN clustering..."):
        clustered_data = analyzer.hdbscan_clustering(
            country_data,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
        if clustered_data is not None:
            # Rename HDBSCAN column to dbscan_cluster for compatibility with map function
            clustered_data['dbscan_cluster'] = clustered_data['hdbscan_cluster']
            
            # Get country center coordinates for map
            coords = analyzer._get_country_coordinates(country_data)
            center = None
            if coords is not None and len(coords) > 0:
                center = [coords[:, 0].mean(), coords[:, 1].mean()]
            
            # Create map
            hdbscan_map = analyzer.create_dbscan_map(clustered_data, center=center)
            st_folium(hdbscan_map, width=1200, height=600)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            n_clusters = len(set(clustered_data['hdbscan_cluster']) - {-1})
            n_noise = (clustered_data['hdbscan_cluster'] == -1).sum()
            
            with col1:
                st.metric("Clusters Found", n_clusters)
            with col2:
                st.metric("Noise Points", n_noise)
            with col3:
                st.metric("Total Points", len(clustered_data))
            
            # Download option
            st.download_button(
                label="📥 Download HDBSCAN Results (CSV)",
                data=clustered_data.to_csv(index=False),
                file_name=f"hdbscan_{country}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No clusters found or insufficient data for HDBSCAN")


def render_kde_analysis(analyzer, country_data, country):
    """Render KDE Heatmap analysis"""
    
    st.subheader("Kernel Density Estimation (KDE)")
    st.markdown("""
    **Smooth Probability Distribution Heatmap**
    - Visualizes debris density as continuous surface
    - Better for understanding spatial patterns
    """)
    
    bandwidth = st.slider("Bandwidth:", 0.05, 0.5, 0.1, 0.05)
    
    # Apply KDE
    with st.spinner("🔥 Generating KDE heatmap..."):
        kde_result = analyzer.kde_heatmap(country_data, bandwidth=bandwidth)
        
        # Get country center coordinates for map
        coords = analyzer._get_country_coordinates(country_data)
        center = None
        if coords is not None and len(coords) > 0:
            center = [coords[:, 0].mean(), coords[:, 1].mean()]
        
        # Create map
        kde_map = analyzer.create_kde_map(kde_result, center=center)
        st_folium(kde_map, width=1200, height=600)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Grid Size", "50 × 50")
    with col2:
        st.metric("Data Points", len(country_data))
    with col3:
        # Extract density from kde_result dictionary
        density_range = kde_result['density']
        st.metric("Density Range", f"{density_range.min():.2f} - {density_range.max():.2f}")


def render_temporal_analysis(analyzer, country):
    """Render Temporal Evolution analysis"""
    
    st.subheader("Spatiotemporal Evolution")
    st.markdown("""
    **Hotspot Evolution Over Time**
    - Analyze how debris patterns change year-by-year
    - Identify emerging hotspots and migration patterns
    """)
    
    with st.spinner("⏱️ Analyzing temporal patterns..."):
            temporal_results = analyzer.spatiotemporal_analysis(country)
            if not temporal_results:
                st.warning("⚠️ Insufficient temporal data for this country")
                return
        
            # Create summary table
            summary_data = []
            for year in sorted(temporal_results.keys()):
                data = temporal_results[year]
                n_clusters = len(set(data['dbscan_cluster']) - {-1}) if 'dbscan_cluster' in data.columns else 0
                summary_data.append({
                    'Year': year,
                    'Records': len(data),
                    'Clusters': n_clusters,
                    'Avg Debris': data['DEBRIS AMOUNT'].mean() if 'DEBRIS AMOUNT' in data.columns else 0
                })
        
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
            # Visualization
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(summary_df.set_index('Year')['Records'])
            with col2:
                st.line_chart(summary_df.set_index('Year')['Clusters'])
        
            # Download option
            st.download_button(
                label="📥 Download Temporal Summary (CSV)",
                data=summary_df.to_csv(index=False),
                file_name=f"temporal_summary_{country}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    render_advanced_hotspot_dashboard()
