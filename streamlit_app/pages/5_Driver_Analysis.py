import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import sys
sys.path.append('..')
from utils.data_loader import load_combined_metrics

st.set_page_config(page_title="Driver Analysis", page_icon="📈", layout="wide")

st.markdown('<style>.main-header {font-size: 2.5rem; font-weight: 700; color: #e74c3c;}</style>', unsafe_allow_html=True)
st.markdown('<p class="main-header">📈 Emissions Driver Analysis</p>', unsafe_allow_html=True)

st.markdown("""
## Understanding What Drives Emissions

Use machine learning to identify which operational factors have the strongest impact on emissions.

**Analysis Methods:**
- **Random Forest**: Identifies non-linear relationships and feature importance
- **SHAP Values**: Explains individual predictions and feature contributions
- **What-If Scenarios**: Model the impact of operational changes
""")

st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load combined emissions and operational metrics"""
    df = load_combined_metrics()
    
    # Remove rows with missing operational data
    df = df.dropna(subset=['electricity_mwh', 'water_consumption_gallons', 'pue'])
    
    return df

try:
    data = load_data()
    
    if len(data) == 0:
        st.error("No data available. Please generate the database first.")
        st.stop()
    
    st.success(f"✅ Loaded {len(data):,} records with operational metrics")
    
    # Sidebar - Model Configuration
    st.sidebar.header("🎛️ Analysis Configuration")
    
    # Target variable selection
    target_options = {
        'Total Emissions': 'total_emissions',
        'Scope 1': 'scope1_tonnes',
        'Scope 2 (Market)': 'scope2_market_tonnes',
        'Scope 3': 'scope3_tonnes'
    }
    
    target_name = st.sidebar.selectbox(
        "Target Variable",
        options=list(target_options.keys()),
        help="Which emission category to analyze"
    )
    
    target_col = target_options[target_name]
    
    # Feature selection
    st.sidebar.subheader("📊 Available Features")
    
    feature_options = {
        'Electricity Consumption (MWh)': 'electricity_mwh',
        'Water Consumption (gallons)': 'water_consumption_gallons',
        'PUE (Power Usage Effectiveness)': 'pue',
        'Carbon-Free Energy %': 'cfe_pct',
        'Renewable Energy (MWh)': 'renewable_electricity_mwh',
        'Waste Generated (tons)': 'waste_generated_tons',
    }
    
    selected_features = st.sidebar.multiselect(
        "Select Features",
        options=list(feature_options.keys()),
        default=list(feature_options.keys())[:4],
        help="Choose which operational metrics to include in the model"
    )
    
    if len(selected_features) == 0:
        st.warning("⚠️ Please select at least one feature")
        st.stop()
    
    # Map selected features to column names
    feature_cols = [feature_options[f] for f in selected_features]
    
    # Prepare data
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    # Remove any remaining NaN values
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) == 0:
        st.error("No valid data after removing missing values")
        st.stop()
    
    # Split data
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Model Type",
        ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"],
        help="Choose the machine learning algorithm"
    )
    
    st.markdown("---")
    
    # Train models
    st.markdown("### 🤖 Model Training & Performance")
    
    with st.spinner("Training model..."):
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        elif model_choice == "Linear Regression":
            model = LinearRegression()
        else:  # Ridge
            model = Ridge(alpha=1.0)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    st.success("✅ Model trained successfully!")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test R² Score", f"{test_r2:.3f}", help="Proportion of variance explained (higher is better)")
    
    with col2:
        st.metric("Test MAE", f"{test_mae:,.0f} tonnes", help="Mean Absolute Error")
    
    with col3:
        st.metric("Test RMSE", f"{test_rmse:,.0f} tonnes", help="Root Mean Squared Error")
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs Predicted
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=y_test,
            y=y_pred_test,
            mode='markers',
            name='Test Set',
            marker=dict(size=8, color='#3498db', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_pred.update_layout(
            title=f'Actual vs Predicted {target_name}',
            xaxis_title=f'Actual {target_name} (tonnes CO₂e)',
            yaxis_title=f'Predicted {target_name} (tonnes CO₂e)',
            height=400,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig_pred, width = 'stretch)
    
    with col2:
        # Residuals
        residuals = y_test - y_pred_test
        
        fig_res = go.Figure()
        
        fig_res.add_trace(go.Scatter(
            x=y_pred_test,
            y=residuals,
            mode='markers',
            marker=dict(size=8, color='#e74c3c', opacity=0.6)
        ))
        
        fig_res.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig_res.update_layout(
            title='Residual Plot',
            xaxis_title=f'Predicted {target_name} (tonnes CO₂e)',
            yaxis_title='Residuals',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_res, width = 'stretch)
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 📊 Feature Importance")
    st.markdown("Which operational factors have the strongest impact on emissions?")
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig_importance = go.Figure(go.Bar(
            x=feature_importance_df['Importance'],
            y=feature_importance_df['Feature'],
            orientation='h',
            marker_color='#3498db',
            text=feature_importance_df['Importance'].apply(lambda x: f"{x:.3f}"),
            textposition='outside'
        ))
        
        fig_importance.update_layout(
            title='Feature Importance (Random Forest)',
            xaxis_title='Importance Score',
            yaxis_title='',
            height=max(300, len(selected_features) * 50),
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig_importance, width = 'stretch)
        
        # Insights
        top_feature = feature_importance_df.iloc[0]
        st.info(f"""
        **Key Insight:** {top_feature['Feature']} is the strongest driver of {target_name.lower()}, 
        accounting for {top_feature['Importance']*100:.1f}% of the model's predictive power.
        """)
        
    else:
        # Linear models - use coefficients
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            
            # Standardize to get comparable coefficients
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            model_scaled = model.__class__(**model.get_params())
            model_scaled.fit(X_scaled, y_train)
            
            coef_df = pd.DataFrame({
                'Feature': selected_features,
                'Coefficient': np.abs(model_scaled.coef_)
            }).sort_values('Coefficient', ascending=False)
            
            fig_coef = go.Figure(go.Bar(
                x=coef_df['Coefficient'],
                y=coef_df['Feature'],
                orientation='h',
                marker_color='#3498db',
                text=coef_df['Coefficient'].apply(lambda x: f"{x:.0f}"),
                textposition='outside'
            ))
            
            fig_coef.update_layout(
                title='Standardized Coefficients',
                xaxis_title='Absolute Coefficient Value',
                yaxis_title='',
                height=max(300, len(selected_features) * 50),
                template='plotly_white',
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_coef, width = 'stretch)
    
    st.markdown("---")
    
    # SHAP Analysis
    st.markdown("### 🔍 SHAP Analysis - Explainable AI")
    st.markdown("SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions.")
    
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        with st.spinner("Computing SHAP values..."):
            # Sample data for SHAP (computational efficiency)
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # SHAP summary plot
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Mean absolute SHAP values
                mean_shap = np.abs(shap_values).mean(axis=0)
                
                shap_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Mean |SHAP|': mean_shap
                }).sort_values('Mean |SHAP|', ascending=False)
                
                fig_shap = go.Figure(go.Bar(
                    x=shap_df['Mean |SHAP|'],
                    y=shap_df['Feature'],
                    orientation='h',
                    marker_color='#e74c3c',
                    text=shap_df['Mean |SHAP|'].apply(lambda x: f"{x:.0f}"),
                    textposition='outside'
                ))
                
                fig_shap.update_layout(
                    title='Mean Absolute SHAP Values',
                    xaxis_title='Average Impact on Prediction',
                    yaxis_title='',
                    height=max(300, len(selected_features) * 50),
                    template='plotly_white',
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig_shap, width = 'stretch)
            
            with col2:
                st.markdown("**SHAP Interpretation:**")
                st.markdown("""
                - **Higher values** = Feature has stronger impact
                - Shows **actual contribution** to predictions
                - More accurate than simple feature importance
                - Accounts for feature interactions
                """)
                
                top_shap = shap_df.iloc[0]
                st.success(f"""
                **Top Driver:** {top_shap['Feature']}
                
                Average impact: {top_shap['Mean |SHAP|']:.0f} tonnes CO₂e
                """)
        
        st.info("""
        **💡 Actionable Insight:** Focus operational improvements on the top 2-3 drivers for maximum emissions reduction impact.
        """)
    else:
        st.info("SHAP analysis is only available for tree-based models (Random Forest, Gradient Boosting)")
    
    st.markdown("---")
    
    # What-If Scenario Analysis
    st.markdown("### 🎯 What-If Scenario Analysis")
    st.markdown("Model the impact of operational changes on emissions.")
    
    st.markdown("**Adjust the sliders below to see predicted emissions impact:**")
    
    # Create scenario inputs
    cols = st.columns(3)
    scenario_values = {}
    
    # Create mapping for display names
    reverse_feature_map = {v: k for k, v in feature_options.items()}
    
    for idx, feature in enumerate(feature_cols):
        col_idx = idx % 3
        display_name = reverse_feature_map.get(feature, feature)
        
        with cols[col_idx]:
            avg_value = X[feature].mean()
            min_value = X[feature].min()
            max_value = X[feature].max()
            
            # Fix error with step=0 and min == max
            if max_value > min_value:
                step_size = float((max_value - min_value) / 100)
            else:
                # If constant value, create a small range around it
                step_size = 0.01
                if min_value == 0:
                    min_value = 0.0
                    max_value = 1.0
                else:
                    # Create ±10% range around the constant value
                    min_value = float(min_value * 0.9)
                    max_value = float(max_value * 1.1)

            scenario_values[feature] = st.slider(
                display_name,
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(avg_value),
                step=step_size,
                format="%.2f"
            )
    
    # Predict scenario
    scenario_df = pd.DataFrame([scenario_values])
    scenario_prediction = model.predict(scenario_df)[0]
    
    # Compare to baseline
    baseline_prediction = model.predict(X_test.mean().values.reshape(1, -1))[0]
    difference = scenario_prediction - baseline_prediction
    pct_change = (difference / baseline_prediction) * 100
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Baseline Prediction",
            f"{baseline_prediction:,.0f} tonnes",
            help="Average facility emissions based on typical operational metrics"
        )
    
    with col2:
        st.metric(
            "Scenario Prediction",
            f"{scenario_prediction:,.0f} tonnes",
            delta=f"{difference:+,.0f} tonnes ({pct_change:+.1f}%)",
            delta_color="inverse"
        )
    
    with col3:
        if difference < 0:
            st.success(f"✅ Reduction of {abs(difference):,.0f} tonnes CO₂e")
        else:
            st.error(f"⚠️ Increase of {difference:,.0f} tonnes CO₂e")
    
    # Example scenarios
    st.markdown("---")
    st.markdown("### 💡 Example Efficiency Scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Scenario 1: PUE Improvement**
        - Improve PUE from 1.10 → 1.08
        - Expected: 2-5% emissions reduction
        - Investment: HVAC optimization, cooling upgrades
        """)
    
    with col2:
        st.markdown("""
        **Scenario 2: Renewable Energy**
        - Increase renewable MWh by 20%
        - Expected: 15-20% Scope 2 reduction
        - Investment: Additional PPAs, on-site solar
        """)
    
    with col3:
        st.markdown("""
        **Scenario 3: Water Efficiency**
        - Reduce water consumption 30%
        - Expected: Indirect cooling energy savings
        - Investment: Closed-loop cooling, efficiency tech
        """)

except Exception as e:
    st.error(f"Error loading data or running analysis: {e}")
    st.info("Please ensure the database exists and contains emissions data with operational metrics.")
    import traceback
    st.code(traceback.format_exc())