"""
Model Comparison Page
Compares different ML models and algorithms on the same dataset
Shows accuracy metrics, predictions, and visual comparisons
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
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """Load the cleaned dataset for model comparison"""
    data_paths = [
        r"c:\Users\Anki_tha\OneDrive\Desktop\maraine_related_project data\project\data_cleaned.csv",
        r"c:\Users\Anki_tha\OneDrive\Desktop\maraine_related_project data\project\data.xlsx"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    return pd.read_csv(path)
                else:
                    return pd.read_excel(path)
            except Exception as e:
                st.warning(f"Error loading {path}: {e}")
    
    st.error("No data found! Please run imputation.ipynb first.")
    return None

def load_best_pipeline():
    """Load the best trained pipeline"""
    pipeline_path = r"c:\Users\Anki_tha\OneDrive\Desktop\maraine_related_project data\project\best_pipeline.joblib"
    
    if os.path.exists(pipeline_path):
        try:
            return joblib.load(pipeline_path)
        except Exception as e:
            st.warning(f"Could not load pipeline: {e}")
    return None

def prepare_data_for_modeling(df):
    """Prepare data for modeling"""
    TARGET = 'Total Items Collected'
    
    if TARGET not in df.columns:
        st.error(f"Target column '{TARGET}' not found in dataset")
        return None, None, None, None
    
    # Drop raw copies and rows with missing target
    raw_cols = [c for c in df.columns if c.endswith('_raw')]
    df_clean = df.drop(columns=raw_cols, errors='ignore')
    df_clean = df_clean[~df_clean[TARGET].isna()].copy()
    
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET].astype(float)
    
    # Select numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return X, y, numeric_cols, categorical_cols

def build_preprocessing_pipeline(numeric_cols, categorical_cols):
    """Build data preprocessing pipeline"""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Ensure dense output for compatibility with HistGradientBoosting
    try:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    except TypeError:
        # Fallback for older scikit-learn versions that use 'sparse' instead of 'sparse_output'
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')
    
    return preprocessor

def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate multiple models"""
    from sklearn.pipeline import Pipeline
    
    models_dict = {
        'Dummy (Mean Baseline)': DummyRegressor(strategy='mean'),
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    
    # Try to load LightGBM
    try:
        from lightgbm import LGBMRegressor
        models_dict['LightGBM'] = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
    except ImportError:
        pass
    
    # Try to load HistGradientBoosting
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        models_dict['Hist Gradient Boosting'] = HistGradientBoostingRegressor(random_state=42)
    except ImportError:
        pass
    
    results = {}
    predictions = {}
    
    with st.spinner('Evaluating models...'):
        for model_name, model in models_dict.items():
            try:
                # Build pipeline
                pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                # Train
                pipe.fit(X_train, y_train)
                
                # Predict
                y_pred_train = pipe.predict(X_train)
                y_pred_test = pipe.predict(X_test)
                
                # Calculate metrics
                mae_train = mean_absolute_error(y_train, y_pred_train)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                r2_test = r2_score(y_test, y_pred_test)
                
                results[model_name] = {
                    'MAE_Train': mae_train,
                    'MAE_Test': mae_test,
                    'RMSE_Test': rmse_test,
                    'R2_Score': r2_test,
                    'Model': pipe
                }
                
                predictions[model_name] = {
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'y_train': y_train,
                    'y_test': y_test
                }
                
            except Exception as e:
                st.warning(f"Error training {model_name}: {str(e)}")
    
    return results, predictions

def create_metrics_comparison_chart(results):
    """Create comparison chart for metrics"""
    df_results = pd.DataFrame({
        model: {
            'MAE (Train)': results[model]['MAE_Train'],
            'MAE (Test)': results[model]['MAE_Test'],
            'RMSE (Test)': results[model]['RMSE_Test'],
            'R² Score': results[model]['R2_Score']
        } for model in results.keys()
    }).T.reset_index()
    
    df_results.rename(columns={'index': 'Model'}, inplace=True)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    metrics = ['MAE (Train)', 'MAE (Test)', 'RMSE (Test)']
    for metric in metrics:
        fig.add_trace(go.Bar(
            x=df_results['Model'],
            y=df_results[metric],
            name=metric,
            hovertemplate='<b>%{x}</b><br>' + metric + ': %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Model Comparison: Error Metrics (Lower is Better)',
        barmode='group',
        xaxis_title='Model',
        yaxis_title='Error Value',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig, df_results

def create_r2_comparison_chart(results):
    """Create R² score comparison chart"""
    models = list(results.keys())
    r2_scores = [results[m]['R2_Score'] for m in models]
    
    colors = ['#0ea5e9' if r2 > 0 else '#ef4444' for r2 in r2_scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=r2_scores,
            marker_color=colors,
            text=[f'{r2:.4f}' for r2 in r2_scores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>R² Score: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Model Comparison: R² Score (Higher is Better)',
        xaxis_title='Model',
        yaxis_title='R² Score',
        template='plotly_white',
        height=500,
        hovermode='x',
        showlegend=False
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Baseline")
    
    return fig

def create_prediction_comparison_chart(predictions, num_samples=20):
    """Create actual vs predicted comparison"""
    fig = go.Figure()
    
    # Get test predictions
    sample_idx = np.random.choice(len(predictions[list(predictions.keys())[0]]['y_test']), 
                                  min(num_samples, len(predictions[list(predictions.keys())[0]]['y_test'])), 
                                  replace=False)
    
    y_true = predictions[list(predictions.keys())[0]]['y_test'].iloc[sample_idx].values
    
    for model_name in predictions.keys():
        y_pred = predictions[model_name]['y_pred_test'][sample_idx]
        
        fig.add_trace(go.Scatter(
            x=np.arange(len(y_true)),
            y=y_pred,
            mode='markers+lines',
            name=model_name,
            hovertemplate='Sample: %{x}<br>Prediction: %{y:.2f}<extra></extra>'
        ))
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=np.arange(len(y_true)),
        y=y_true,
        mode='markers',
        name='Actual Values',
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='Sample: %{x}<br>Actual: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Model Predictions vs Actual (Random {num_samples} Samples)',
        xaxis_title='Sample Index',
        yaxis_title='Total Items Collected',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_residual_chart(predictions):
    """Create residual comparison for test set"""
    fig = go.Figure()
    
    for model_name in predictions.keys():
        y_true = predictions[model_name]['y_test']
        y_pred = predictions[model_name]['y_pred_test']
        residuals = y_true - y_pred
        
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name=model_name,
            hovertemplate='Prediction: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Perfect Fit")
    
    fig.update_layout(
        title='Residual Plot: Prediction Error Distribution',
        xaxis_title='Predicted Value',
        yaxis_title='Residual (Actual - Predicted)',
        template='plotly_white',
        height=500,
        hovermode='closest'
    )
    
    return fig

def create_metrics_table(results):
    """Create detailed metrics table"""
    data = []
    for model_name, metrics in results.items():
        data.append({
            'Model': model_name,
            'MAE (Train)': f"{metrics['MAE_Train']:.4f}",
            'MAE (Test)': f"{metrics['MAE_Test']:.4f}",
            'RMSE (Test)': f"{metrics['RMSE_Test']:.4f}",
            'R² Score': f"{metrics['R2_Score']:.4f}"
        })
    
    return pd.DataFrame(data)

def show_model_comparison_page():
    """Main model comparison page"""
    st.set_page_config(page_title='Model Comparison', layout='wide')
    
    # Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0ea5e9 0%, #38bdf8 50%, #67e8f9 100%);
                padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>🤖 Model Comparison & Accuracy Analysis</h1>
        <p style='color: white; margin: 5px 0 0 0;'>Compare different machine learning models on the same marine debris dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    st.info("Loading dataset and preparing models for comparison...")
    df = load_cleaned_data()
    
    if df is None:
        st.stop()
    
    # Prepare data
    X, y, numeric_cols, categorical_cols = prepare_data_for_modeling(df)
    
    if X is None:
        st.stop()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.success(f"✅ Data loaded successfully!")
    st.markdown(f"""
    **Dataset Overview:**
    - Total Samples: {len(df):,}
    - Training Samples: {len(X_train):,}
    - Test Samples: {len(X_test):,}
    - Features: {X.shape[1]}
    - Target Variable: Total Items Collected
    """)
    
    # Build preprocessor
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # Evaluate models
    results, predictions = evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    
    if not results:
        st.error("Could not evaluate any models!")
        st.stop()
    
    # ===== SECTION 1: Metrics Comparison =====
    st.markdown("---")
    st.subheader("📊 1. Error Metrics Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_metrics, df_metrics = create_metrics_comparison_chart(results)
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        fig_r2 = create_r2_comparison_chart(results)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # ===== SECTION 2: Detailed Metrics Table =====
    st.markdown("---")
    st.subheader("📈 2. Detailed Performance Metrics")
    
    metrics_table = create_metrics_table(results)
    st.dataframe(
        metrics_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Model': st.column_config.TextColumn('Model', width='large'),
            'MAE (Train)': st.column_config.NumberColumn('MAE (Train)', format="%.4f"),
            'MAE (Test)': st.column_config.NumberColumn('MAE (Test)', format="%.4f"),
            'RMSE (Test)': st.column_config.NumberColumn('RMSE (Test)', format="%.4f"),
            'R² Score': st.column_config.NumberColumn('R² Score', format="%.4f")
        }
    )
    
    st.markdown("""
    **Metric Definitions:**
    - **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values (lower is better)
    - **RMSE (Root Mean Squared Error):** Square root of average squared errors (penalizes larger errors more)
    - **R² Score:** Proportion of variance explained by the model (1.0 = perfect, 0 = baseline, negative = worse than baseline)
    """)
    
    # ===== SECTION 3: Prediction Comparison =====
    st.markdown("---")
    st.subheader("🎯 3. Predictions Comparison (Test Set)")
    
    fig_pred = create_prediction_comparison_chart(predictions, num_samples=30)
    st.plotly_chart(fig_pred, use_container_width=True)
    
    st.caption("Visualization showing predicted vs actual values for random test samples. The diamond points represent actual values.")
    
    # ===== SECTION 4: Residual Analysis =====
    st.markdown("---")
    st.subheader("📉 4. Residual Analysis")
    
    fig_residual = create_residual_chart(predictions)
    st.plotly_chart(fig_residual, use_container_width=True)
    
    st.markdown("""
    **Residual Interpretation:**
    - **Points near the red line:** Model predictions are close to actual values
    - **Points far from the red line:** Model makes significant prediction errors
    - **Horizontal spread:** If residuals are randomly distributed, the model is unbiased
    - **Vertical pattern:** If residuals increase with prediction value, model has heteroscedasticity
    """)
    
    # ===== SECTION 5: Model Ranking =====
    st.markdown("---")
    st.subheader("🏆 5. Model Ranking & Recommendations")
    
    # Create ranking dataframe
    ranking_data = []
    for i, (model_name, metrics) in enumerate(sorted(results.items(), 
                                                       key=lambda x: x[1]['MAE_Test']), 1):
        ranking_data.append({
            'Rank': i,
            'Model': model_name,
            'MAE (Test)': f"{metrics['MAE_Test']:.4f}",
            'R² Score': f"{metrics['R2_Score']:.4f}",
            'Recommendation': '⭐' * min(6 - i, 5)
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    
    # Best model analysis
    best_model_name = min(results.keys(), key=lambda x: results[x]['MAE_Test'])
    best_metrics = results[best_model_name]
    
    st.success(f"""
    **🏅 Best Model: {best_model_name}**
    
    - Test MAE: {best_metrics['MAE_Test']:.4f}
    - Test RMSE: {best_metrics['RMSE_Test']:.4f}
    - R² Score: {best_metrics['R2_Score']:.4f}
    
    This model provides the best balance of accuracy and generalization for predicting marine debris collection totals.
    """)
    
    # ===== SECTION 6: Model Insights =====
    st.markdown("---")
    st.subheader("💡 6. Model Insights & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Characteristics:**
        
        1. **Dummy (Mean Baseline)**
           - Predicts average value for all samples
           - Useful as a minimum baseline
           - Poor performance indicates other models are learning patterns
        
        2. **Linear Regression**
           - Assumes linear relationship between features and target
           - Fast to train, interpretable
           - May underfit if relationships are non-linear
        
        3. **Random Forest**
           - Ensemble of decision trees
           - Captures non-linear patterns well
           - Generally good generalization
        """)
    
    with col2:
        st.markdown("""
        **Key Observations:**
        
        📌 **Training vs Test Performance:**
        - Large gap indicates overfitting
        - Small gap indicates good generalization
        
        📌 **R² Score Interpretation:**
        - Score > 0.8: Excellent fit
        - Score 0.5-0.8: Good fit
        - Score < 0.5: Weak fit
        
        📌 **Error Magnitude:**
        - Consider actual scale of target variable
        - Is error acceptable for practical use?
        """)
    
    if 'LightGBM' in results:
        st.info("""
        **LightGBM Model:** Fast gradient boosting model optimized for speed and accuracy.
        Typically provides excellent performance for structured data.
        """)
    
    # ===== SECTION 7: Download Results =====
    st.markdown("---")
    st.subheader("📥 7. Download Comparison Results")
    
    # Prepare download data
    metrics_download = metrics_table.copy()
    csv_buffer = metrics_download.to_csv(index=False)
    
    st.download_button(
        label="📊 Download Metrics as CSV",
        data=csv_buffer,
        file_name="model_comparison_metrics.csv",
        mime="text/csv",
        key="download_metrics"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='background: #f0f9ff; padding: 15px; border-radius: 10px; border-left: 4px solid #0ea5e9;'>
        <p style='margin: 0;'><strong>ℹ️ Note:</strong> This comparison helps identify the best-performing model for your specific dataset. 
        The best model may vary depending on your use case, data quality, and business requirements.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Back button
    st.markdown("---")
    if st.button("← Back to Dashboard", key="back_to_dashboard"):
        st.session_state.current_page = "dashboard"
        st.rerun()
