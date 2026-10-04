"""
Country-Specific Model Comparison
Compare ML models for a selected country's data
Shows how different models perform on country-specific patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def show_country_model_comparison():
    st.markdown("""
    Compare how different ML models perform on data from a specific country.
    This helps identify which algorithm works best for predicting marine debris in different regions.
    """)
    
    # Load data
    data_paths = [
        "data_cleaned.csv",
        "data.xlsx"
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)
                break
            except Exception as e:
                st.warning(f"Error loading {path}: {e}")
    
    if df is None:
        st.error("❌ No data found! Please ensure data_cleaned.csv exists.")
        return
    
    # Country selection
    st.sidebar.markdown("---")
    st.sidebar.header("🌍 Country Selection")
    
    if 'COUNTRY / LOCATION' not in df.columns:
        st.error("Column 'COUNTRY / LOCATION' not found in dataset")
        return
    
    countries = sorted(df['COUNTRY / LOCATION'].dropna().unique())
    
    selected_country = st.sidebar.selectbox(
        "Select Country:",
        options=countries,
        help="Choose a country to analyze model performance"
    )
    
    # Filter data by country
    country_df = df[df['COUNTRY / LOCATION'] == selected_country].copy()
    
    st.info(f"📊 **Selected Country:** {selected_country}")
    st.info(f"📈 **Records Available:** {len(country_df)} data points")
    
    # Check if enough data
    if len(country_df) < 10:
        st.warning(f"⚠️ Only {len(country_df)} records available for {selected_country}. Need at least 10 records for reliable comparison.")
        st.info("💡 Tip: Try selecting a country with more historical data.")
        return
    
    # Show data preview
    with st.expander("📋 View Country Data Preview"):
        st.dataframe(country_df.head(20), use_container_width=True)
    
    # Target column
    TARGET = 'Total Items Collected'
    
    if TARGET not in country_df.columns:
        st.error(f"Target column '{TARGET}' not found")
        return
    
    # Prepare data
    raw_cols = [c for c in country_df.columns if c.endswith('_raw')]
    country_df = country_df.drop(columns=raw_cols, errors='ignore')
    country_df = country_df[~country_df[TARGET].isna()].copy()
    
    if len(country_df) < 10:
        st.warning(f"After cleaning, only {len(country_df)} valid records remain. Need at least 10.")
        return
    
    X = country_df.drop(columns=[TARGET, 'COUNTRY / LOCATION'], errors='ignore')
    y = country_df[TARGET].astype(float)
    
    # Select features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Settings")
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    # Train button
    if st.sidebar.button("🚀 Run Model Comparison", type="primary", use_container_width=True):
        
        with st.spinner(f"Training 5 models on {selected_country} data..."):
            
            # Train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            except Exception as e:
                st.error(f"Error splitting data: {e}")
                return
            
            st.success(f"✅ Train: {len(X_train)} samples | Test: {len(X_test)} samples")
            
            # Build preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Ensure dense output for HistGradientBoosting compatibility
            try:
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='Missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
            except TypeError:
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='Missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
                ])
            
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
            
            # Train models
            results = {}
            models_to_train = {
                'Dummy (Mean)': DummyRegressor(strategy='mean'),
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
                'HistGradientBoosting': HistGradientBoostingRegressor(random_state=random_state)
            }
            
            if LIGHTGBM_AVAILABLE:
                models_to_train['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=random_state, verbose=-1)
            
            progress_bar = st.progress(0)
            total_models = len(models_to_train)
            
            for idx, (name, model) in enumerate(models_to_train.items()):
                try:
                    # Create pipeline
                    pipe = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Train
                    pipe.fit(X_train, y_train)
                    
                    # Predict
                    train_pred = pipe.predict(X_train)
                    test_pred = pipe.predict(X_test)
                    
                    # Calculate metrics
                    results[name] = {
                        'mae_train': mean_absolute_error(y_train, train_pred),
                        'mae_test': mean_absolute_error(y_test, test_pred),
                        'rmse_test': np.sqrt(mean_squared_error(y_test, test_pred)),
                        'r2_score': r2_score(y_test, test_pred),
                        'predictions': test_pred,
                        'actual': y_test.values
                    }
                    
                except Exception as e:
                    st.warning(f"Model {name} failed: {e}")
                
                progress_bar.progress((idx + 1) / total_models)
            
            st.success(f"✅ Trained {len(results)} models successfully!")
            
            # Store results in session state
            st.session_state[f'country_comparison_{selected_country}'] = results
    
    # Display results if available
    if f'country_comparison_{selected_country}' in st.session_state:
        results = st.session_state[f'country_comparison_{selected_country}']
        
        st.markdown("---")
        st.header(f"📊 Results for {selected_country}")
        
        # 1. Metrics Comparison
        st.subheader("1️⃣ Model Performance Metrics")
        
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'MAE (Train)': [results[m]['mae_train'] for m in results.keys()],
            'MAE (Test)': [results[m]['mae_test'] for m in results.keys()],
            'RMSE (Test)': [results[m]['rmse_test'] for m in results.keys()],
            'R² Score': [results[m]['r2_score'] for m in results.keys()]
        })
        
        st.dataframe(metrics_df.style.highlight_min(subset=['MAE (Test)', 'RMSE (Test)'], color='lightgreen')
                                      .highlight_max(subset=['R² Score'], color='lightgreen'), 
                    use_container_width=True)
        
        # 2. Visual Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### MAE Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=metrics_df['Model'],
                y=metrics_df['MAE (Test)'],
                marker_color='lightblue',
                text=metrics_df['MAE (Test)'].round(1),
                textposition='outside'
            ))
            fig.update_layout(
                title=f"Test MAE by Model ({selected_country})",
                xaxis_title="Model",
                yaxis_title="MAE (Lower is Better)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### R² Score Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=metrics_df['Model'],
                y=metrics_df['R² Score'],
                marker_color='lightgreen',
                text=metrics_df['R² Score'].round(3),
                textposition='outside'
            ))
            fig.update_layout(
                title=f"R² Score by Model ({selected_country})",
                xaxis_title="Model",
                yaxis_title="R² (Higher is Better)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 3. Best Model Recommendation
        st.subheader("2️⃣ Best Model Recommendation")
        
        best_model = metrics_df.loc[metrics_df['MAE (Test)'].idxmin(), 'Model']
        best_mae = metrics_df.loc[metrics_df['MAE (Test)'].idxmin(), 'MAE (Test)']
        best_r2 = metrics_df.loc[metrics_df['MAE (Test)'].idxmin(), 'R² Score']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Model", best_model)
        with col2:
            st.metric("📉 Test MAE", f"{best_mae:.2f}")
        with col3:
            st.metric("📈 R² Score", f"{best_r2:.3f}")
        
        st.info(f"""
        **💡 Recommendation for {selected_country}:**
        - Best performing model: **{best_model}**
        - Expected error: ±{best_mae:.1f} items
        - Model explains {best_r2*100:.1f}% of variance in debris collection
        """)
        
        # 4. Predictions vs Actual
        st.subheader("3️⃣ Predictions vs Actual Values")
        
        fig = go.Figure()
        
        # Add actual values
        actual_values = results[list(results.keys())[0]]['actual']
        x_indices = list(range(len(actual_values)))
        
        fig.add_trace(go.Scatter(
            x=x_indices,
            y=actual_values,
            mode='markers',
            name='Actual',
            marker=dict(size=10, color='red', symbol='circle')
        ))
        
        # Add predictions for each model
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for idx, (model_name, color) in enumerate(zip(results.keys(), colors)):
            fig.add_trace(go.Scatter(
                x=x_indices,
                y=results[model_name]['predictions'],
                mode='markers',
                name=model_name,
                marker=dict(size=8, color=color, symbol='x')
            ))
        
        fig.update_layout(
            title=f"Model Predictions vs Actual ({selected_country})",
            xaxis_title="Test Sample Index",
            yaxis_title="Total Items Collected",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. Download Results
        st.subheader("4️⃣ Download Results")
        
        csv = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {selected_country} Model Comparison CSV",
            data=csv,
            file_name=f"{selected_country.replace(' ', '_')}_model_comparison.csv",
            mime='text/csv'
        )
    
    # Back button
    st.markdown("---")
    if st.button("⬅️ Back to Dashboard", use_container_width=True):
        st.session_state.current_page = 'dashboard'
        st.rerun()
