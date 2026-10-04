"""
Forecast Visualization Module
Creates interactive plots and maps for future debris forecasts.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
import json
import os

class ForecastVisualizer:
    """Creates visualizations for debris forecasts."""
    
    def __init__(self, historical_data_path: str, forecast_data_path: str):
        """
        Initialize with historical and forecast data.
        
        Args:
            historical_data_path: Path to cleaned historical CSV
            forecast_data_path: Path to combined forecast CSV
        """
        self.historical = pd.read_csv(historical_data_path)
        self.forecast = pd.read_csv(forecast_data_path)
        
        # Load country centroids if available
        self.centroids = self._load_centroids()
    
    def _load_centroids(self):
        """Load geocoded country centroids."""
        paths = ['country_capital_centroids.csv', 'country_centroids_geocoded.csv']
        for p in paths:
            if os.path.exists(p):
                return pd.read_csv(p)
        return pd.DataFrame()
    
    def plot_country_forecast(self, country: str, save_path: str = None) -> go.Figure:
        """
        Create time series plot with historical data and forecast.
        
        Args:
            country: Country name
            save_path: Optional path to save HTML
        
        Returns:
            Plotly figure object
        """
        # Get historical data
        hist = self.historical[self.historical['COUNTRY / LOCATION'] == country].copy()
        hist = hist.sort_values('year')
        
        # Get forecast data
        fcast = self.forecast[self.forecast['country'] == country].copy()
        fcast = fcast.sort_values('year')
        
        if len(hist) == 0:
            raise ValueError(f"No historical data found for {country}")
        
        # Create figure
        fig = go.Figure()
        
        # Historical line
        fig.add_trace(go.Scatter(
            x=hist['year'],
            y=hist['Total Items Collected'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        if len(fcast) > 0:
            # Forecast line
            fig.add_trace(go.Scatter(
                x=fcast['year'],
                y=fcast['predicted_debris'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=fcast['year'].tolist() + fcast['year'].tolist()[::-1],
                y=fcast['upper_ci'].tolist() + fcast['lower_ci'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,127,14,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Marine Debris Forecast: {country}',
            xaxis_title='Year',
            yaxis_title='Total Items Collected',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def create_forecast_comparison(self, countries: list = None, save_path: str = None) -> go.Figure:
        """
        Create comparison plot of forecasts for multiple countries.
        
        Args:
            countries: List of countries (if None, show top 10 by final forecast)
            save_path: Optional path to save HTML
        
        Returns:
            Plotly figure
        """
        if countries is None:
            # Get top 10 by final year forecast
            final_year = self.forecast['year'].max()
            top = self.forecast[self.forecast['year'] == final_year].nlargest(10, 'predicted_debris')
            countries = top['country'].tolist()
        
        fig = go.Figure()
        
        for country in countries:
            fcast = self.forecast[self.forecast['country'] == country].sort_values('year')
            if len(fcast) > 0:
                fig.add_trace(go.Scatter(
                    x=fcast['year'],
                    y=fcast['predicted_debris'],
                    mode='lines+markers',
                    name=country,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title='Future Debris Forecasts: Country Comparison',
            xaxis_title='Year',
            yaxis_title='Predicted Debris (Total Items)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.9)')
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def create_forecast_map(self, year: int = None, save_path: str = 'forecast_map.html') -> folium.Map:
        """
        Create interactive map showing predicted debris by country.
        
        Args:
            year: Forecast year (if None, use final year)
            save_path: Path to save HTML map
        
        Returns:
            Folium map object
        """
        if year is None:
            year = self.forecast['year'].max()
        
        # Filter forecast for specified year
        year_data = self.forecast[self.forecast['year'] == year].copy()
        
        if len(self.centroids) == 0:
            print("Warning: No centroids available, map will be limited")
            # Create simple table-based map
            m = folium.Map(location=[20, 0], zoom_start=2)
        else:
            # Merge with centroids (handle different column names)
            centroid_cols = self.centroids.columns.tolist()
            lat_col = 'lat' if 'lat' in centroid_cols else 'latitude'
            lon_col = 'lon' if 'lon' in centroid_cols else 'longitude'
            
            year_data = year_data.merge(
                self.centroids[['country', lat_col, lon_col]].rename(columns={lat_col: 'latitude', lon_col: 'longitude'}), 
                on='country', 
                how='left'
            )
            
            # Create map centered on mean location
            center_lat = year_data['latitude'].mean() if 'latitude' in year_data else 20
            center_lon = year_data['longitude'].mean() if 'longitude' in year_data else 0
            m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles='OpenStreetMap')
            
            # Normalize debris for marker sizing
            max_debris = year_data['predicted_debris'].max()
            min_debris = year_data['predicted_debris'].min()
            
            # Add markers for each country
            for _, row in year_data.iterrows():
                if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                    # Size proportional to debris
                    size = 5 + 30 * (row['predicted_debris'] - min_debris) / (max_debris - min_debris + 1)
                    
                    # Color based on magnitude (bright and dark contrasting palette)
                    if row['predicted_debris'] > max_debris * 0.7:
                        # High: Bright Magenta/Hot Pink with dark outline
                        color = '#1a1a2e'  # Dark navy for outline
                        fill_color = '#FF006E'  # Bright hot pink
                    elif row['predicted_debris'] > max_debris * 0.4:
                        # Medium: Bright Yellow with dark outline
                        color = '#1a1a2e'  # Dark navy for outline
                        fill_color = '#FFD60A'  # Bright yellow
                    else:
                        # Low: Bright Cyan with dark outline
                        color = '#1a1a2e'  # Dark navy for outline
                        fill_color = '#00D9FF'  # Bright cyan
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=size,
                        popup=f"<b>{row['country']}</b><br>"
                              f"Year: {year}<br>"
                              f"Predicted: {row['predicted_debris']:,.0f}<br>"
                              f"CI: [{row['lower_ci']:,.0f}, {row['upper_ci']:,.0f}]",
                        tooltip=row['country'],
                        color=color,
                        weight=2,
                        fill=True,
                        fillColor=fill_color,
                        fillOpacity=0.8
                    ).add_to(m)
            
            # Add legend
            legend_html = f'''
            <div style="position: fixed; top: 10px; right: 10px; width: 220px; 
                        background-color: rgba(255, 255, 255, 0.95); border:3px solid #1a1a2e; z-index:9999; 
                        font-size:14px; padding: 12px; border-radius: 8px;">
                <p style="margin: 0 0 8px 0; font-weight: bold;">Forecast Year: {year}</p>
                <p style="margin: 6px 0;"><i class="fa fa-circle" style="color:#FF006E; font-size:16px;"></i> <b>High</b> (&gt; {max_debris*0.7:,.0f})</p>
                <p style="margin: 6px 0;"><i class="fa fa-circle" style="color:#FFD60A; font-size:16px;"></i> <b>Medium</b> ({max_debris*0.4:,.0f} - {max_debris*0.7:,.0f})</p>
                <p style="margin: 6px 0;"><i class="fa fa-circle" style="color:#00D9FF; font-size:16px;"></i> <b>Low</b> (&lt; {max_debris*0.4:,.0f})</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(save_path)
        print(f"Saved forecast map to {save_path}")
        
        return m
    
    def create_summary_dashboard(self, save_path: str = 'forecast_dashboard.html'):
        """
        Create comprehensive dashboard with multiple views.
        
        Args:
            save_path: Path to save HTML dashboard
        """
        # Get final year data
        final_year = self.forecast['year'].max()
        final_data = self.forecast[self.forecast['year'] == final_year].sort_values('predicted_debris', ascending=False)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Top 15 Countries - Forecast {final_year}',
                'Forecast Trend (Top 5)',
                'Growth Rate by Country',
                'Uncertainty Range'
            ),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'box'}]]
        )
        
        # 1. Top countries bar chart
        top15 = final_data.head(15)
        fig.add_trace(
            go.Bar(x=top15['country'], y=top15['predicted_debris'], name='Forecast', marker_color='#ff7f0e'),
            row=1, col=1
        )
        
        # 2. Trend lines for top 5
        top5_countries = final_data.head(5)['country'].tolist()
        for country in top5_countries:
            country_fcast = self.forecast[self.forecast['country'] == country].sort_values('year')
            fig.add_trace(
                go.Scatter(x=country_fcast['year'], y=country_fcast['predicted_debris'], 
                          mode='lines+markers', name=country),
                row=1, col=2
            )
        
        # 3. Growth rate (first vs last year)
        first_year = self.forecast['year'].min()
        first_data = self.forecast[self.forecast['year'] == first_year]
        growth = final_data.merge(first_data[['country', 'predicted_debris']], on='country', suffixes=('_final', '_first'))
        growth['pct_change'] = ((growth['predicted_debris_final'] - growth['predicted_debris_first']) / 
                                 (growth['predicted_debris_first'] + 1)) * 100
        top_growth = growth.nlargest(10, 'pct_change')
        fig.add_trace(
            go.Bar(x=top_growth['country'], y=top_growth['pct_change'], name='% Change', marker_color='#2ca02c'),
            row=2, col=1
        )
        
        # 4. Uncertainty box plot for top 10
        top10 = final_data.head(10)['country'].tolist()
        for country in top10:
            country_fcast = self.forecast[self.forecast['country'] == country]
            fig.add_trace(
                go.Box(y=country_fcast['predicted_debris'], name=country, boxmean='sd'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text=f"Marine Debris Forecast Dashboard ({self.forecast['year'].min()} - {final_year})",
            template='plotly_white'
        )
        
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(title_text='Year', row=1, col=2)
        fig.update_xaxes(tickangle=-45, row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=2)
        
        fig.update_yaxes(title_text='Predicted Debris', row=1, col=1)
        fig.update_yaxes(title_text='Predicted Debris', row=1, col=2)
        fig.update_yaxes(title_text='% Growth', row=2, col=1)
        fig.update_yaxes(title_text='Predicted Debris', row=2, col=2)
        
        fig.write_html(save_path)
        print(f"Saved dashboard to {save_path}")
        
        return fig


def main():
    """CLI for generating forecast visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize debris forecasts')
    parser.add_argument('--historical', type=str, default='data_cleaned.csv', help='Historical data CSV')
    parser.add_argument('--forecast', type=str, default='future_forecasts_combined.csv', help='Forecast CSV')
    parser.add_argument('--country', type=str, help='Plot single country')
    parser.add_argument('--map-year', type=int, help='Year for forecast map')
    parser.add_argument('--dashboard', action='store_true', help='Create full dashboard')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.forecast):
        print(f"Error: Forecast file not found: {args.forecast}")
        print("Run forecast_future_debris.py first to generate forecasts")
        return
    
    viz = ForecastVisualizer(args.historical, args.forecast)
    
    if args.country:
        viz.plot_country_forecast(args.country, save_path=f'forecast_plot_{args.country.replace(" ", "_")}.html')
    
    if args.map_year:
        viz.create_forecast_map(year=args.map_year)
    elif not args.country:
        viz.create_forecast_map()
    
    if args.dashboard:
        viz.create_summary_dashboard()
    
    if not args.country and not args.map_year and not args.dashboard:
        # Default: create comparison and map
        viz.create_forecast_comparison(save_path='forecast_comparison.html')
        viz.create_forecast_map()
        print("\nGenerated:")
        print("- forecast_comparison.html")
        print("- forecast_map.html")


if __name__ == '__main__':
    main()
